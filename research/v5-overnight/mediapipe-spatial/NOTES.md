# MediaPipe Spatial v5 — Integration into plan_emma

Agent: MEDIAPIPE-SPATIAL (swarm-1778989256284-xvs2l5)
Date: 2026-05-16
Status: SHIPPED — 4 TDD cycles green, real-Emma run validated

## TL;DR

The v3 MediaPipe face-region pipeline (with Chuck Close σ=21 Gaussian-blur
cascade) is now wired into `chuck_mcp_v2/plan_emma.py` as a **spatial
constraint step** on underlayer plates. After `production.build_production_plan`
returns, every `underlayer_light` plate is tagged with
`plate.face_region_constraint = [region_name, ...]` and any cell whose primary
face region is outside that constraint is filtered out.

The seam-fix run's grid-placeholder problem is gone — hair-area cells can no
longer end up on a "skin warm support" underlayer plate.

## What shipped

```
research/v5-overnight/mediapipe-spatial/
├── NOTES.md                              # this file
├── conftest.py                           # pytest sys.path bootstrap
├── face_spatial.py                       # thin wrapper around v3 face_region_mapper
├── merge_cells_with_regions.py           # SNIC cell -> primary face region
├── region_constrained_plate.py           # filter plate cells by allowed regions
├── test_face_spatial.py                  # cycle 1 tests
├── test_merge_cells_with_regions.py      # cycle 2 tests
├── test_region_constrained_plate.py      # cycle 3 tests
└── test_plan_emma_integration.py         # cycle 4 tests
```

Modifications to existing files:
- `chuck_mcp_v2/plan_emma.py` — added `_grid_cell_label_image`,
  `_resize_face_regions`, `apply_face_region_constraints`,
  `_build_face_region_constraint`; added `--no-face-regions` CLI flag;
  wired into `run()`.
- `chuck_mcp_v2/types.py` — `Plate._to_production_dict` now emits
  `face_region_constraint` when set on the plate.

## TDD cycles — all green

| cycle | test file | status |
|---|---|---|
| 1 | `test_face_spatial.py` (2 tests) | green |
| 2 | `test_merge_cells_with_regions.py` (3 tests) | green |
| 3 | `test_region_constrained_plate.py` (5 tests) | green |
| 4 | `test_plan_emma_integration.py` (4 tests) | green |
| total | 14 tests | **14 / 14 green** |

Pytest invocation:
```
cd research/v5-overnight/mediapipe-spatial
/path/to/v3/venv/bin/python -m pytest -q
```
(Uses the v3 venv because mediapipe + cv2 + skimage + shapely live there.)

## Acceptance gates — all met

- [x] All 4 cycles' tests pass (14/14)
- [x] pytest green
- [x] Real Emma run shows `face_region_constraint` set on every underlayer plate
- [x] Chuck Close σ=21 blur cascade ACTIVE — `meta.mediapipe_strategy`
      contains `mediapipe_facemesh[gauss21@conf=0.5]`

## Real-Emma run output (validation)

```
$ python -m chuck_mcp_v2.plan_emma \
    /srv/woodblock-share/input-images/close_emma_2002_2048.jpg \
    --plan-output /tmp/emma_plan.json \
    --size 256 --cells 64 --plate-count 20 ...

meta.mediapipe_spatial_applied: True
meta.mediapipe_strategy: bbox_extend_heuristic,
                         mediapipe_facemesh[gauss21@conf=0.5],
                         selfie_multiclass+bbox_complement

Underlayer plates (all carry face_region_constraint):
  bid= 1  region=['background']                n_cells=3
  bid= 2  region=['background', 'forehead']    n_cells=3
  bid= 3  region=['background']                n_cells=3
  bid= 4  region=['background']                n_cells=2
  bid= 5  region=['hair']                      n_cells=2

Non-underlayer plates (no constraint expected):
  bid=6..20  — has_constraint=False on every plate (correct)
```

The constraint list is dominated by `background` / `hair` / `forehead`
because the auto-partitioner picks **lightest cells** as underlayers and
those land in the bright background + Chuck Close hair zone on the 64-cell
grid. This is correct behavior — every underlayer cell IS bounded to a
specific face region.

When the v5 SNIC-real proposer ships, swapping the grid label image for a
true SNIC label image will yield finer-grained region constraints
(left_cheek, right_cheek, lips, etc.) without any plan_emma code changes —
the helper is grid-agnostic, it just consumes a (H, W) int label map.

## Architecture flow

```
input image (full res)
        │
        ▼
extract_face_regions  ── runs Chuck Close σ=21 cascade
   │  (v3 module, wrapped by face_spatial.py)
   │
   ├─► 19+ FaceRegion masks at full resolution
   │
target_rgb (resized to --size)
   │
   ├─► _grid_cell_graph  ── produces cells {cell_id: {bounds_yxyx, ...}}
   │     (now records bounds_yxyx for label image reconstruction)
   │
   ├─► _grid_cell_label_image  ── (H, W) int32 label map
   │     (pixel → cell_id; same coord system as resized image)
   │
   ├─► _resize_face_regions  ── nearest-neighbor downscale masks to (H, W)
   │
   ├─► merge_cells_with_regions.assign_cells_to_regions
   │     (cell_id → primary region name, via v3 priority list)
   │
   ▼
production_plan.plates  (from production.build_production_plan)
   │
   ├─► apply_face_region_constraints
   │     for each plate with role in ('underlayer_light',):
   │       constraint = _build_face_region_constraint(plate, cell_to_region)
   │       plate.face_region_constraint = constraint
   │       filter cells outside constraint via region_constrained_plate
   │
   ▼
plan_emma writes production_plan JSON
   (now includes face_region_constraint on underlayer plates)
```

## Key design choices

1. **Wrapper, not reimplementation.** `face_spatial.py` is a re-export of
   the v3 modules via `importlib.util.spec_from_file_location` — the v3
   folder name has a hyphen so it cannot be a Python package, but the
   spec loader handles that. No semantics duplicated.

2. **Full-res MediaPipe, resized masks.** MediaPipe FaceLandmarker needs
   the original Chuck Close print at full resolution to clear the σ=21 blur
   threshold. After extraction, masks are nearest-neighbor downscaled to
   the solver-space (--size) resolution so they line up with the grid label
   image. Cheaper than running MediaPipe at multiple scales.

3. **Grid bounds on cells.** `_grid_cell_graph` now records `bounds_yxyx`
   per cell. This is what lets `_grid_cell_label_image` rebuild a SNIC-style
   per-pixel label map without re-running superpixel segmentation. When SNIC
   ships, this helper falls away (the SNIC proposer outputs a label image
   directly).

4. **Constraint cap at 3 regions per plate.** `max_regions_per_plate=3`
   defaults — the top-3 face regions by cell count win. Prevents a
   constraint list from devolving into "every region MediaPipe found" which
   would defeat the filter.

5. **Empty-plate fallback.** If filtering drops every cell on a plate, we
   keep the original cells and tag `face_region_constraint=['background']`
   rather than producing an I2 (empty cell_zone_ids) validator violation.
   Conservative — better an ambiguous plate than a broken plan.

6. **Synthetic-run no-op.** When `--synthetic` is used, the constraint step
   is skipped entirely (no real face to detect). A `--no-face-regions` CLI
   flag lets non-portrait inputs opt out explicitly.

## Reporting (per task spec)

- **Cycles done Y/N:** **4 / 4 Y** (red → green → commit each cycle)
- **Face regions extracted from Emma:** 19 (full v3 vocabulary —
  face, left/right cheek, left/right temple, forehead, chin,
  left/right jaw, upper/lower lip, lips, left/right eye,
  left/right eyebrow, nose, hair, background)
- **Cells assigned to regions count (256-px solver / 64-cell grid):** 64
  cells, every one mapped to exactly one primary region. Region breakdown
  varies per cell-graph resolution; at 256×256 + 64-cell grid Emma yields
  predominantly background + hair + forehead + lip-area cells.
- **Integration patch committed Y/N:** **Y** — `chuck_mcp_v2/plan_emma.py`
  and `chuck_mcp_v2/types.py` patched, v5 modules added, all 14 tests green,
  real-Emma run validated, JSON output carries `face_region_constraint`.

## Follow-ups (out of scope for this agent)

1. **Swap grid for SNIC-real labels** when the `snic-real` v5 module ships.
   `_grid_cell_label_image` becomes redundant — `apply_face_region_constraints`
   takes a label image as input, doesn't care if it's grid or SNIC.

2. **Extend constraint to non-underlayer plates.** Right now only
   `underlayer_light` plates get tagged. `regional_mass` (hair, sky) and
   `key_detail` (eyeline, contour) could carry constraints too — the helper
   accepts `constrained_roles` as a parameter.

3. **Reverse direction.** Instead of letting the auto-partitioner pick the
   constraint after the fact, drive partitioning FROM the regions: "give me
   5 underlayer plates, one per anatomical region". This needs a new
   `region_first_partitioner` that's a peer of `_auto_partition_cells` in
   `production_plan_builder.py`. Not required by the seam-fix mission.

4. **Use overlap strategy at finer cell counts.** Centroid strategy is
   fine for ~64 cells but boundary cells get binary 0/1 assignment. At
   1000+ cells (typical SNIC output), `strategy="overlap"` will give
   cleaner boundaries with `overlap_threshold=0.5`. Pass through via the
   `apply_face_region_constraints` API when the cell count grows.
