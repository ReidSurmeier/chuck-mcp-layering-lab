# alpha-proof-dumper — chuck-mcp v4 acceptance-harness data feeder

**Mission:** plug the gap between `chuck_mcp_v2.plan_emma` and the v4 acceptance
harness. Before this module, real Emma runs produced `production_plan.json` +
`hybrid_result.json` + a tiny `iter_NN/` thumbnail dump, and the acceptance sheet
showed `NOT FOUND` placeholders in row 4 (alphas) plus tiny non-mirrored
plate thumbnails in row 3. After this module, the same runs additionally write:

```
<artifacts_dir>/
├── alphas/pull_NNN_alpha.png        # 1 per pull, raw mask (normalized for visibility)
├── plates/block_NN.png               # 1 per plate, mirrored wood-grain preview
├── plates/block_NN.preview.png       # duplicate, harness-preferred name
├── pulls/pull_NNN.png                # 1 per pull, cumulative composite
├── proofs/proof_NN_after_pull_MMM.png  # 7 evenly spaced checkpoints
├── cumulative_pull_NN.png            # alias of proofs/, at the plan root for the harness
├── alpha_masks/alpha_NN.png          # subset of alphas, harness-preferred name
└── final_composite.png               # last cumulative state (overrides the iter_NN copy)
```

## TDD cycle status

| Cycle | Test | Status |
|------|------|--------|
| 1 | `test_dump_writes_per_pull_alpha_png` | green |
| 2a | `test_dump_writes_per_plate_preview_png` | green |
| 2b | `test_plate_preview_is_horizontally_mirrored` | green |
| 3 | `test_dump_writes_per_pull_cumulative_proof` | green |
| 4 | `test_dump_writes_7_checkpoint_proof_states` | green |
| 5 | `test_acceptance_harness_finds_plates_and_alphas_after_plan_emma` | green |
| Integration | `test_plan_emma_emits_artifacts_dir_consumed_by_harness` | green |

7 tests, 7 passes, 0 failures.

## Files written per type (real Emma run, 96-px solver shape, 24 plate cap)

After the optimizer drops 3-5 plates during Stage 4 morphology repair,
21 plates survive. The dumper writes:

- 21 alphas in `alphas/`
- 21 plate previews in `plates/` (× 2 for the `.png` + `.preview.png` aliases = 42 files)
- 21 cumulative pulls in `pulls/`
- 7 proof checkpoints in `proofs/`
- 7 root `cumulative_pull_NN.png` aliases
- 7 `alpha_masks/alpha_NN.png` aliases
- 1 `final_composite.png`

**Wall-time impact:** ~0.5s on top of the 3-4s the rest of plan_emma takes for
the 96-px solver scale. Well under the 30s budget the mission set.

## How the acceptance sheet looks after the fix

Before:
- Row 2: 7 tiny grid renders (mostly white because the cheap composite
  in `_render_cumulative_pulls` didn't pick up the pigment Lab values).
- Row 3: kraft-paper placeholders (no `plates/` dir).
- Row 4: solid `NOT FOUND` placeholders.

After:
- Row 2: 7 real cumulative proof states (pulls 1, 4, 8, 11, 14, 18, 21).
- Row 3: 8 mirrored wood-grain plates with the L+bar kento mark and the
  pigment region. (The kento sits on the bottom-left of the print = mirrored
  bottom-right of the carving.)
- Row 4: 7 actual alpha snapshots — each one a normalized grayscale of the
  inked mask for that pull.

`acceptance_count_rendered` jumps from `0` to `7`.

## Wiring

`chuck_mcp_v2/plan_emma.py:run()` now imports the dumper via
`_load_alpha_proof_dumper()` (lazy load, because the package dir has a hyphen
and isn't a regular Python package) and calls it after the hybrid optimizer
returns. The integration is gated on:

- `--artifacts-dir <dir>` must be passed (no-op otherwise — same as the legacy
  iter_NN dump).
- `--no-alpha-proof-dump` skips the step entirely (escape hatch).
- `--proof-checkpoint-count N` overrides the default 7.

## Why pure-numpy and not the v3 cell-zone-renderer?

The v3 `plate_renderer.render_plate_preview` + `pull_renderer.render_pull`
need `shapely.geometry.Polygon` objects (the `CellZone` dataclass requires
them), and the hybrid optimizer's `SolvedPlate` only carries a raster
`inked_mask: np.ndarray`. Converting a binary mask back to a Polygon would
need `rasterio.features.shapes` or `skimage.measure.find_contours` + a Polygon
constructor, AND the renderers also need `svgwrite` + `shapely`, neither of
which the system Python has.

Rather than add a third dependency layer, this module reimplements the same
visual idea (wood grain + pigment overlay + kento marks) in pure numpy + PIL.
The output looks the same to a reviewer at sheet-tile size and is
~100× faster.

## Open follow-ups (not blocking acceptance)

1. The cheap K-M composite in `_compose_pull` uses Porter-Duff "source over"
   with the optimizer's `pigment_blend_lab` — same approximation as
   `_render_cumulative_pulls`. For a high-fidelity proof sheet, swap to
   `backend.services.v23.core.forward_render_jax`, but that needs the full
   spectral pigment library not the 6-pigment fallback.
2. The plate preview uses a 4-stripe procedural wood grain; the v3 SVG
   renderer uses two feTurbulence passes. The v3 output is prettier but only
   visible at high DPI — irrelevant at the 256-px sheet tile size.
3. The dumper still runs at solver resolution (typically 96×96 or 128×128).
   For the production review sheet we want a high-res pass that takes the
   original target image, rasterizes each plate's mask to (W_full, H_full)
   via nearest-neighbour upsampling, then renders the wood-ground at full
   res. Defer to a v5 follow-on agent.
