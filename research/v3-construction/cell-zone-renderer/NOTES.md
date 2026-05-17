# cell-zone-renderer — v3 rendering subsystem

Research swarm CELL-ZONE-RENDERER, 2026-05-16.

Domain: cell-zone → SVG plate, cell-zone → α-map → pull image, pull
image accumulation → proof state.

Companion to:
- `docs/v2-design-locked-2026-05-16.md` (canonical design contract)
- `docs/reconstruction-plan-2026-05-16.md` (the diagnosis that this
  subsystem exists to fix)
- v23 backend code in `backend/services/v23/stages/s3b_cell_graph.py`
  (where SNIC cell-zones originate) and `s6b_jigsaw_organize.py`
  (where plate role assignments happen)

## File map

```
research/v3-construction/cell-zone-renderer/
├── cz_types.py                       # CellZone, Plate, Pull, ProofState dataclasses
├── plate_renderer.py                 # render_plate_svg + render_plate_preview
├── pull_renderer.py                  # render_pull (α-map K-M overprint, PRIVATE α-maps)
├── proof_state_assembler.py          # assemble_proof_sheet (8-up) + assemble_plate_and_pull_sheet
├── contact_sheet_renderer.py         # render_all_blocks_contact_sheet (replaces v13 sheet)
├── test_renderers.py                 # synthetic 27-plate Emma harness — runs the full pipeline
├── test_plate_not_composite.py       # the hard validator + adversarial check
└── out/                              # produced artifacts (see Test outputs below)
```

Tested with Python 3.12 + the venv at
`/home/reidsurmeier/src/chuck-mcp-layering-lab/.venv-renderer/`
(svgwrite 1.4.3, shapely 2.1.2, scikit-image 0.26, PIL 12.2, numpy 2.4.5).

## What I built — 4 working renderer modules

### 1. `plate_renderer.render_plate_svg(plate, output_path, mirror=True)`

Mirrored vector plate for CNC handoff. Three nested layers:

1. **Wood-grain ground**: a chained `feTurbulence` + `feColorMatrix`
   filter that maps stretched fractal noise (`baseFrequency="0.014
   0.0022"`, fractalNoise) to maple-plywood tones, plus a second
   higher-frequency `turbulence` pass for the fine grain dots. The
   seed varies per block_id so the 27-plate contact sheet doesn't
   look photocopied.

2. **Cell-zone inked region**: the union of all `CellZone.polygon`
   objects on this plate, closed (dilate+erode) by `mill_radius_px`
   to fill in cell-cell seams, then opened by half that radius to
   kill 1-pixel spurs. Final geometry survives tool-radius offset
   on the CNC. Filled with the plate's pigment color at
   `cfg.pigment_alpha * plate.opacity` opacity so it reads as wet
   paint, not vector decal.

3. **Kento marks**: kagi (L-shape, two thin rects) at the print's
   bottom-right corner; hikitsuke (single bar) along the bottom edge
   ~45% across from center. Both mirrored along with the rest of
   the plate so the carved jig and the SVG agree on which corner
   the L sits at. The marks are a visual *reference* — physical
   registration is done by the soto-kento jig (`docs/v2-design-
   locked-2026-05-16.md`).

svgwrite is initialized with `profile="tiny"` because the default
profile rejects multi-spaced `feColorMatrix values` strings.

### 2. `plate_renderer.render_plate_preview(plate, output_path)`

Raster PNG of the same plate for the web UI block grid. Pure
numpy + PIL — no SciPy dependency. The wood ground is built from
three layered noise components:

- Long warped horizontal grain stripes (high y-freq, warped x).
- Mid-frequency color blotches (low-pass random field).
- Fine speckle (1px noise blurred 0.5px).

Mixes between `wood_base` (light maple) and `wood_dark` (warm
walnut grain). Pigment region is painted on a transparent overlay
with subtle alpha noise so brushed ink doesn't look like plastic.

### 3. `pull_renderer.render_pull(plate, prev_proof_state, opacity)`

The ONE function where α-maps live, per design Q26.

1. Rasterize plate's cell-zones to a soft Gaussian-edged α-map (the
   "wet brush" boundary).
2. Multiply α by `plate.opacity * opacity * ink_density`, clip to [0,1].
3. Composite onto `prev_proof_state` using porter-duff source-over:
   `out = prev * (1 - α) + pigment * α`.
4. Return NORMAL orientation — mirror only lives on the plate SVG.

Two performance tricks let this run 132 pulls at 2048×2048 in ~20s:

- **α-map cache**: keyed by `(id(plate), out_size, src_size,
  softness)`, so each plate is rasterized once per session even
  if it contributes 5 pulls.
- **Masked composite**: we only touch pixels where the α-map is
  non-trivially > 0, instead of doing the per-pixel multiply across
  the full 2048×2048 buffer. ~7× speedup over naïve composite.

For V2-onwards K-M physics, swap the call to
`backend.services.v23.core.forward_render_jax` — signature is
identical.

### 4. `proof_state_assembler.assemble_proof_sheet(pulls, checkpoint_indices)`

4×2 grid laid out exactly like
`/srv/woodblock-share/Examples/Screenshot 2026-05-14 at 6.13.05
PM.png`. Each cell is a Lanczos-downsampled cumulative-pull image,
labelled `after pull NN`. Default checkpoints `[4, 8, 12, 16, 20,
24, 64, 132]` — the locked-design 7 plus a final.

Also exports `assemble_plate_and_pull_sheet(plate_previews,
pull_previews)` — each cell stacks a plate preview above its
cumulative-pull-after thumbnail, exactly the
Woodblock-print-process.png didactic layout.

### 5. `contact_sheet_renderer.render_all_blocks_contact_sheet(plates)`

The artifact that replaces `LATEST_CHUCK_MCP_ALL_BLOCKS_CONTACT_
SHEET.png`. 7×4 grid (28 cells, 27 plates + 1 blank), each cell is
`render_plate_preview` output. By construction, every cell shows
isolated jigsaw regions on wood-grain ground — not a faded full-
face residual. The v13 failure mode is impossible here because
`render_plate_preview` only paints the polygons in
`plate.cell_zones`; everything else stays wood ground.

## Plate-not-composite verification strategy

`test_plate_not_composite.py` implements the validator as a hard gate.

```python
plate_not_composite_score = 1.0 - (final_resemblance + coverage) / 2
REJECT if score < 0.6
```

Where:

- **final_resemblance**: of the final composite's "interesting"
  pixels (pixels far from the median paper color, threshold 0.30 in
  RGB distance), how many are also "inked" on this plate? A v13
  residual covers ~85% of the final's interesting pixels; a real
  Emma plate covers ~3-10%.

- **coverage**: how much of the plate's surface is inked? Mapped
  through a smooth ramp: `frac <= 0.20` → 0 (definitely sparse
  plate), `frac >= 0.55` → 1 (definitely a residual composite).

The "inked" mask is detected by perpendicular distance to the
`wood_base ↔ wood_dark` color gradient in RGB. Wood-grain noise
varies brightness along that gradient and reads as ground; real
pigment ink shifts hue OFF that gradient and reads as ink. This
detector handles all 27 of our test pigments correctly, including
pale yellows that overlap heavily with wood tones.

**Adversarial check**: we synthesize a fake v13-style residual by
blending the final composite at 55% with paper-white at 45%, then
score it. Result:

```
Real plates: 27/27 PASS (score 0.946-1.000)
v13 residual: score 0.133 → CORRECTLY REJECTED
```

The design-doc threshold `> 0.6` appears reversed — the doc says
"reject if > 0.6" but a high score = more plate-like (less
composite-like) under the formula. I'm treating that as a typo
and using `>= 0.6 PASS / < 0.6 FAIL` because that matches the v13
failure mode the validator is supposed to catch. **This is
flagged for downstream reviewer attention** — if the design doc
intent was the literal text, the formula needs to be sign-flipped.

The plate_renderer enforces this property by construction: only
the polygons in `plate.cell_zones` are painted. The only way the
validator could fail on a renderer output is if the cell-zone
assignment upstream gave one plate >55% of the image area's
worth of cells — which is itself an upstream bug, exactly what the
plate-not-composite gate is designed to catch.

## Performance budget

Target (from task brief): full proof sheet render < 30s for 132 pulls
+ 27 plates at 2048×2048.

Measured (on Linux dev machine):

```
[1/6] generate synthetic 27-plate plan     0.02s
[2/6] render 27 plate SVG + preview PNGs   3.67s
[3/6] render 132 cumulative pulls         20.11s   <-- main loop
[4/6] proof-state 8-up sheet               0.79s
[5/6] plate-and-pull pair sheet            2.10s
[6/6] all-blocks contact sheet             2.55s
                                          -------
Total                                     ~29.2s   PASSES <30s budget
```

Just the proof sheet (`pulls + assemble`) ≈ 21s. Just the contact
sheet ≈ 6s. Both well within 30s.

Optimizations applied:
- α-map cache: 34s → 26s for the pull loop (each plate rasterized
  once instead of N times).
- Masked-pixel composite: 26s → 20s for the pull loop (skip
  unaffected pixels).

If this is still too slow for the web UI's responsiveness, two more
levers:
- Render previews at 1024×1024 (currently 2048×2048) — 4× speedup.
- Switch to JAX `forward_render_jax` on GPU — likely 10×+ speedup
  with the same API.

## Top 3 must-reads

1. `docs/reconstruction-plan-2026-05-16.md` — the 257-line
   diagnostic that defines the plate-not-composite distinction.
   Every design decision in this folder traces back to it.

2. `web_potrace_selinger_2003.md` + `web_morphological_area_opening
   _vincent_1993.md` (in `research/papers/vectorization-cnc/`) —
   the closing/opening morphology that protects the SVG geometry
   from mill-radius offset. We implement the buffer(+r).buffer(-r).
   buffer(-r/2).buffer(+r/2) form via shapely in
   `plate_renderer._merge_cell_zones`.

3. https://art-design-glossary.musabi.ac.jp/uchi-kento-soto-kento/
   + https://mlyon.com/2017/two-small-woodcuts-using-a-wedged-
   registration-jig/ — kento mark geometry. Kagi at print bottom-
   right, hikitsuke along bottom ~45% across. Mirrored on the
   plate because the block prints backwards.

Honorable mentions:
- CSS-Tricks "Creating Patterns With SVG Filters" — the exact
  `feTurbulence baseFrequency="0.1 0.01" + feColorMatrix` snippet
  I adapted for the wood-grain SVG filter.
- `Screenshot 2026-05-10 at 5.49.23 PM.png` (Emma's mother plate
  photo) — the visual target for plate previews.
- `Screenshot 2026-05-14 at 6.13.05 PM.png` (Chuck Close 8-up
  proof grid) — the visual target for the proof state assembler.

## Test outputs (under `out/`)

Generated by `test_renderers.py --size 2048 --preview-size 512
--n-pulls 132 --n-cells 400`:

```
out/
├── all_blocks_contact_sheet.png           # 7×4 grid — replaces v13 sheet
├── proof_state_sheet.png                  # 4×2 — Chuck Close reference layout
├── plate_and_pull_sheet.png               # 4×2 of plate-above-pull pairs
├── plates/
│   ├── block_01.svg ... block_27.svg     # mirrored SVG with wood-ground + kento
│   └── block_01.preview.png ... block_27.preview.png
└── pulls/
    └── pull_004.png ... pull_132.png      # checkpoint cumulative pulls
```

The plate-not-composite validator can be run standalone:

```
.venv-renderer/bin/python test_plate_not_composite.py
```

## Integration handoff

When this module is wired into the v3 backend pipeline:

1. **Input**: a `Plate` object whose `cell_zones` come from
   `s3b_cell_graph` (SNIC superpixels) after assignment by the
   block-partition stage. The plate must already have its pigment,
   role, and opacity fields populated by the planner.

2. **`render_plate_svg`** is called once per plate during the
   `export_carving_files` MCP tool. Output goes to
   `~/cnc-carving-jobs/emma-<date>/plates/block_NN.svg`.

3. **`render_plate_preview`** is called once per plate during
   `plan_emma_print` to populate the web UI's 27-block grid.

4. **`render_pull`** is called 132× during `plan_emma_print`,
   walking the locked print order. The α-map cache lives for the
   duration of one plan generation; call `clear_alpha_cache()`
   between plans.

5. **`assemble_proof_sheet`** produces the 8-up checkpoint sheet
   for the validation UI's "side-by-side vs Hokusai/Chuck Close
   reference" panel.

6. **`render_all_blocks_contact_sheet`** produces the file at
   `proofs/all_blocks_plate_contact_sheet.png` in the carving job
   folder. This is the artifact that should not look like the
   failing v13 sheet — and by construction (plate-not-composite
   validator-gated), it cannot.

## Open items for downstream review

- **Design doc threshold sign flip**: confirm `< 0.6` is the correct
  reject direction. (My reading: yes; doc says `> 0.6` which I
  believe is a typo. Validator works either way — invert
  `REJECT_THRESHOLD` comparison if needed.)
- **JAX renderer port**: when forward_render_jax is K-M-calibrated
  in V2, swap into `render_pull`. The function signature is the
  contract.
- **Wood-tone palette**: `_DEFAULT_CFG.wood_base/wood_dark` are
  maple-plywood values. The real Mike Lyon blocks are shina-faced
  birch plywood, slightly paler. Tune once Reid confirms which
  plywood he's actually using.
- **Kento mark exact mm geometry**: V1 puts the marks at relative
  `0.04 × short_dim` for the kagi. Mike Lyon's exact wedged-jig
  dimensions are not in his public posts; will need email follow-
  up or measurement off a real block once one is carved.
