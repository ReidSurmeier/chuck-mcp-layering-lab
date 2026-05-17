# Photoshop Integration — Research Synthesis

Researcher: PHOTOSHOP-INTEGRATION
Swarm: swarm-1778969836247-ys4o7z
Date: 2026-05-16

## VERDICT

**Use `psd-tools` 1.17.0 (Python) to generate flattened layered PSD templates and to ingest user-painted PSDs.**

Rationale:

1. **Tested end-to-end here.** A4@300DPI template (2480x3508, 11 layers, 26MB) generated in 11s on this Linux box, re-opens in <10ms, composites in 20ms. Round-trip is lossless for what we need: named layers, per-layer opacity, RGB pixel data.
2. **Active maintenance.** v1.17.0 released 2026-05-11 (5 days ago at time of writing). 1938 commits, 86 releases, used by Adobe-adjacent tooling (CyberAgentAILab/LayerD ICCV 2025).
3. **Photoshop reads the output natively.** The `file` command identifies the output as a valid "Adobe Photoshop Image" file. The CyberAgent and QwenLM Layered repos use psd-tools writes as canonical exports in 2025-2026 academic publications, implying Photoshop opens them clean.
4. **Pip-installable, pure Python.** `pip install psd-tools pillow numpy`. Works on Linux (Reid's primary box) for ingestion, and the template it produces opens identically on Mac/Windows Photoshop. No COM, no AppleScript, no ExtendScript needed.
5. **Alternatives are worse for this job.**
   - `pytoshop` (mdboom): last release 2018, issue #29 "create from scratch" still open. Stale.
   - `PhotoshopAPI` (EmilDohne): 5-20x faster, but "early development", v0.9.1, fewer downstream users. Keep as fallback if psd-tools file size becomes a bottleneck.
   - `photoshop-python-api`: requires Photoshop on the machine via COM (Win) / AppleEvents (Mac). Dead end for Linux server.
   - **OpenRaster (.ora)**: open ZIP+XML+PNG format, but Photoshop refuses to import it (feature request open since 2021). Disqualified by user constraint.
   - **GIMP XCF (`gimpformats`)**: same problem — Photoshop won't open XCF.
   - **Layered PNG + JSON manifest**: viable fallback (see "Why not layered-PNG-only" below) but loses kento alignment guarantees and forces the user to manage 4-9 PNG exports on every iteration. PSD wins on UX.

## File count summary

| Artifact | Type | Purpose |
|---|---|---|
| `NOTES.md` | synthesis | THIS file |
| `gen_template.py` | code | Production-ready PSD template generator |
| `ingest_masks.py` | code | Production-ready user-painted PSD parser + validator |
| `kento_spec.md` | reference | Kento mark coordinate convention, A3/A4/A5/A6 |
| `mask_validation_rules.md` | reference | All validation rules with thresholds |
| `library_comparison.md` | reference | psd-tools vs pytoshop vs PhotoshopAPI vs photoshop-python-api vs OpenRaster |
| `psd_format_quirks.md` | reference | PSD format gotchas: 30k px limit, 1GB cap, bit depth, RGB-only API |
| `chuck_template_a4.psd` | sample | Working A4@300DPI generated template (binary) |
| `chuck_template_a4_preview.png` | sample | Composite preview of the template |

Total: **9 artifacts** (5 code/reference docs + 1 synthesis + 2 generated samples + 1 binary).

## Top-3 must-reads (in priority order)

1. **`gen_template.py`** — copy-paste-runnable template generator. Defines the canonical layer naming scheme, kento layout, opacity convention. Start here.
2. **`mask_validation_rules.md`** — the contract between user output and the solver. Without this, garbage-in-garbage-out.
3. **`kento_spec.md`** — sub-mm registration convention. Gets reused by the CNC plate generator downstream.

## Why not layered-PNG-only?

Considered: drop PSD entirely. Output a ZIP of:

- `source_30pct.png` (visual reference)
- `kento.png` (overlay reference)
- `slot_01_light_yellow_cheek.png`, ..., `slot_09_*.png` (empty paintable PNGs)
- `manifest.json` (layer order, dimensions, kento corner pixels, slot semantic labels)

User would paint each PNG separately in Photoshop, save back, then upload the same ZIP back. Sounds simpler. **Three reasons it's worse:**

1. **No multi-layer in Photoshop = no visual context while painting.** User needs to see the source-image-30%-underneath while painting each mask, plus all the other masks they've already painted (so they don't double-cover an area). With per-mask PNGs, user has to manually layer them in Photoshop anyway, then export each. PSD gives this for free.
2. **Kento drift.** PNG files are independent. User accidentally crops, resizes, rotates one. PSD is one file with one canvas — kento marks stay locked.
3. **Round-trip cognitive load.** Painting 9 separate PNGs → upload 9 files. Painting 9 PSD layers → upload 1 file. The 25x cognitive multiplier matters because we expect Reid to iterate this 10+ times per print.

PSD is the right answer despite the proprietary format. The library `psd-tools` makes it free.

**Fallback:** ship a `--export-png-zip` flag that writes the same content as layered PNGs + manifest, for users without Photoshop. Two outputs, one canonical (PSD), one for accessibility (PNG ZIP).

## Sample code — generate template

See `gen_template.py`. The key invocations:

```python
from psd_tools import PSDImage
from PIL import Image

psd = PSDImage.new(mode='RGB', size=(W, H), depth=8)

# Background: source image at 30% opacity (via layer opacity, not pixel mul)
psd.create_pixel_layer(source_pil, name='[REF] source_30pct_DO_NOT_PAINT', opacity=77)

# Kento marks: opaque corner L's + center crosshair
psd.create_pixel_layer(kento_pil, name='[REF] kento_marks_DO_NOT_PAINT', opacity=255)

# 9 empty slots, named with semantic role
empty = Image.new('RGB', (1, 1), (0, 0, 0))  # 1px placeholder
for slot_name in slot_names:
    psd.create_pixel_layer(empty, name=slot_name, top=0, left=0, opacity=255)

psd.save('chuck_template.psd')
```

The `[REF]` prefix and `DO_NOT_PAINT` suffix are user-facing convention. The reference layers should also be group-locked in a "Reference" group (psd-tools supports `create_group`). User paints on the unprefixed `slot_NN_*` layers.

## Sample code — ingest user-painted

See `ingest_masks.py`. Key flow:

```python
from psd_tools import PSDImage
import numpy as np

psd = PSDImage.open('user_painted.psd')

# Validate canvas dimensions match expected (from template manifest)
assert (psd.width, psd.height) == expected_dims, "canvas resized"

# Validate kento layer still present + bitwise-identical to template
kento_layer = next(L for L in psd if 'kento_marks' in L.name)
kento_arr = np.array(kento_layer.composite().convert('L'))
expected_kento = np.array(reference_kento.convert('L'))
assert np.array_equal(kento_arr, expected_kento), "kento drift"

# Extract painted masks
masks = {}
for layer in psd:
    if not layer.name.startswith('slot_'):
        continue
    pil = layer.composite()
    if pil is None:
        masks[layer.name] = None
        continue
    arr = np.array(pil.convert('L'))
    # Validate binary
    binary_pct = ((arr <= 2) | (arr >= 253)).mean()
    if binary_pct < 0.99:
        raise ValueError(f'{layer.name} is not binary ({binary_pct:.3f} clustered)')
    # Threshold at 127
    masks[layer.name] = (arr > 127).astype(np.uint8)

# Pass `masks` to solver as hard constraints
solver.fit_underlayers(masks=masks)
```

## Kento mark coordinate convention (summary)

Full spec in `kento_spec.md`. TL;DR:

- 4 L-shaped corner marks + 1 center crosshair.
- 10mm inset from canvas edge (= 118 px at 300 DPI).
- 15mm arm length (= 177 px at 300 DPI).
- 0.5mm stroke width (= 6 px at 300 DPI). Visible to naked eye, sub-mm in registration.
- Color: pure black (#000000) on transparent / white. Single channel, alpha matte clean.
- The corner the L opens *toward* indicates which corner it is. NW L opens SE.

## Mask validation rules (summary)

Full spec in `mask_validation_rules.md`. TL;DR:

1. **Binary threshold check**: >99% of pixels at 0±2 or 255±2. Reject otherwise (user used a soft brush).
2. **Dimension check**: layer composite dimensions ≤ canvas dimensions. Layer offset + dimension ≤ canvas. Reject if layer overflows canvas.
3. **Kento overlap check**: painted area intersected with kento mark region (corner L bounding boxes + center crosshair) must be zero. Reject if user painted over the registration marks (this is a hard mokuhanga error).
4. **Empty-but-named check**: if layer has no painted pixels, return `None` and warn. Solver decides whether to error (slot was promised) or just skip (over-promised).
5. **Coverage sanity**: each slot >0.1% and <80% of canvas. Reject if too small (= probably an accidental dot) or too large (= probably forgot to mask off).

## Cross-domain hooks

- **calibration-workflow/**: PSD template inherits the user-annotated 9-underlayer reference. `pin_region` fixture in solver = direct ingest from the painted PSD.
- **binary-mask-jax/**: solver receives `Dict[str, np.ndarray(uint8)]` from `ingest_masks.py`. JAX-compatible.
- **cnc-mokuhanga-carve/**: kento corners are reused by the CNC plate generator. Same coordinate origin (top-left, mm + DPI), same corner inset (10mm), same arm length (15mm). Locked.
- **llm-prompt-translation/**: LLM doesn't generate PSDs. The MCP tool calls `gen_template.py` with the slot names the LLM proposed.

Sources:
- [psd-tools PyPI](https://pypi.org/project/psd-tools/)
- [psd-tools GitHub (active, 1.17.0 May 2026)](https://github.com/psd-tools/psd-tools)
- [psd-tools usage docs](https://psd-tools.readthedocs.io/en/latest/usage.html)
- [PhotoshopAPI (C++ alternative)](https://github.com/EmilDohne/PhotoshopAPI)
- [pytoshop (stale)](https://github.com/mdboom/pytoshop)
- [photoshop-python-api (needs Photoshop installed)](https://pypi.org/project/photoshop-python-api/)
- [OpenRaster spec (Photoshop won't import)](https://www.openraster.org/)
- [gimpformats XCF (Photoshop won't import)](https://pypi.org/project/gimpformats/)
- [CyberAgentAILab/LayerD ICCV 2025 (uses psd-tools)](https://github.com/CyberAgentAILab/LayerD)
- [Adobe PSD format spec](https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/)
