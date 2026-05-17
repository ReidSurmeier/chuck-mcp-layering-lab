# SNIC-Real Cell Proposer — TDD Build Notes

Agent: **SNIC-REAL** | Swarm: swarm-1778989256284-xvs2l5 | Date: 2026-05-17

Mission: Replace the fixed-grid cell-proposal placeholder in
`research/v4-build/production-solver/production_plan_builder.py` (via the
canonical `chuck_mcp_v2.plan_emma` entry point) with a real SNIC superpixel
proposer driven by actual image pixels — eliminating the empty-plate
kraft-paper failure on the v4 acceptance sheet (row 3 of
`/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v4-seam-fix-real-emma/acceptance_sheet.png`).

---

## TDD Cycle Summary

All four cycles completed RED → GREEN → COMMIT. Tests live at
`research/v5-overnight/snic-real/test_snic_real.py`. Full pytest run:
**4 passed in 205.80s**.

| Cycle | Test | Commit | Notes |
|-------|------|--------|-------|
| 1 | `test_snic_on_real_emma_produces_at_least_1500_cells` | `24fcc10` | RED for ModuleNotFoundError; GREEN at 2072 cells |
| 2 | `test_snic_emma_cells_have_at_least_5_hue_clusters` | `86e2be0` | 10/10 hue bins populated (>=30 chromatic cells each) |
| 3 | `test_cells_have_lab_chroma_role_hint_adjacency` | `140af33` | All 6 role_hints active; adjacency reciprocal |
| 4 | `test_plan_emma_uses_snic_not_grid` | `d10c867` | E2E subprocess against `chuck_mcp_v2.plan_emma`; production_plan.meta.cell_proposal_source == "snic" |

---

## Real Emma Run (`close_emma_2002_2048.jpg`)

### snic_proposer.propose_cells (standalone, full resolution)

```json
{
  "wall_seconds": 51.40,
  "source": "snic",
  "image_shape": [2048, 1658],
  "n_cells": 2072,
  "hue_clusters": 10,
  "role_distribution": {
    "skin": 679,
    "background": 614,
    "other": 502,
    "hair": 221,
    "lip": 50,
    "eyes": 6
  },
  "adjacency_mean": 5.33,
  "adjacency_median": 5.0,
  "adjacency_min": 1,
  "adjacency_max": 16
}
```

### chuck_mcp_v2.plan_emma E2E (solver-space, --size 128 --cells 2000)

```json
{
  "plan_id": "plan_1778990623246_0dc978f8",
  "plate_count": 26,
  "total_pulls": 118,
  "cell_count": 2039,
  "image_shape": [128, 104],
  "meta": {
    "plate_count_target": 26,
    "role_distribution": {
      "underlayer_light": 6,
      "local_chroma": 9,
      "regional_mass": 8,
      "key_detail": 3
    },
    "validate_ok": true,
    "cell_proposal_source": "snic",
    "cell_proposal_target_cells": 2000,
    "hue_cluster_count": 8
  }
}
```

The solver-space `cell_count` drops slightly (2072 → 2039) because the
1658×2048 SNIC label map is nearest-neighbour downscaled into the 104×128
solver grid; a small handful of border cells are absorbed by neighbours but
all 26 plates carry real cells (no empty plates).

### Performance

| Stage | Wall (s) | Notes |
|-------|----------|-------|
| pysnic on 1658×2048 Emma (target_cells=2000) | 51.4 | priority queue, single pass |
| `_snic_cell_graph` adapter (NN-downscale + per-cell stats) | <1 | numpy reductions |
| Full `plan_emma --size 128 ... --no-face-regions --no-mokuhanga-pigments` | 52 | dominated by SNIC + JAX warm-up |

Cycle 1+2+3 each take ~50–54s because `propose_cells` is called fresh per
test (the SNIC step itself). Cycle 4 adds ~60s for the subprocess. Cache or
fixture sharing not added — keeping per-test isolation is the safer default
during TDD.

---

## Before / After: production_plan.json (grid placeholder vs SNIC)

### Before (v4 grid placeholder, abstracted)

```json
"meta": {
  "plate_count_target": 26,
  "role_distribution": { ... },
  "validate_ok": true
}
```
No provenance. Cells are 80×80px tiles indexed (row, col) with averaged RGB;
many plates end up with cells of nearly identical mean colour because the
grid never aligned with skin/hair/eye boundaries. On the
`2026-05-17_v4-seam-fix-real-emma` acceptance sheet, row 3 (plate previews)
rendered as kraft-paper-empty rectangles — the plates carried geometry but
no actual ink coverage on the underlying image structure.

### After (v5 SNIC, captured above)

```json
"meta": {
  ...
  "cell_proposal_source": "snic",
  "cell_proposal_target_cells": 2000,
  "hue_cluster_count": 8
}
```
Cells now follow Lab-edge boundaries: skin tones, hair strands, lip regions
land in their own superpixels. The auto-partitioner inside
`production_plan_builder._auto_partition_cells` finally has chromatic
diversity to cluster on (chroma + hue partition spans >5 distinct buckets).
Pixel-bearing plates means non-empty plate previews on the next acceptance
sheet render.

---

## Implementation Map

```
research/v5-overnight/snic-real/
├── __init__.py
├── conftest.py            # repo root + here on sys.path
├── snic_proposer.py       # propose_cells, hue_cluster_count, _run_snic, _rgb_to_lab
├── test_snic_real.py      # 4 TDD cycles
└── NOTES.md               # this file

chuck_mcp_v2/plan_emma.py  # patched:
  - V5_SNIC_REAL_DIR added to sys.path
  - _snic_cell_graph(image_path, solver_shape, requested_cells) adapter
  - run() routes to SNIC for image inputs, grid for synthetic / --use-grid-cells
  - production_plan.meta.cell_proposal_source + hue_cluster_count stamped
  - new flags: --use-grid-cells, --no-mokuhanga-pigments
```

### Backend choice

`_run_snic` tries `pysnic.algorithms.snic` first (priority-queue SNIC, MIT
licensed, `pip install pysnic`). On import failure or runtime exception it
falls back to `skimage.segmentation.slic` on the same Lab image (still real,
still image-driven — no grid). On the Emma run pysnic delivered cleanly,
so backend = "snic" in both standalone and E2E runs.

### Role-hint heuristic

Six buckets: skin / hair / eyes / lip / background / other. The heuristic
combines Lab lightness, chroma, hue angle (atan2(b*, a*)) and a spatial
prior keyed on the face quadrants of a typical Chuck Close portrait
(eyes 0.30–0.55 vertical, lip 0.55–0.85 vertical, margins for background).
It is intentionally coarse — `production_plan_builder._auto_partition_cells`
re-derives final roles via chroma/lightness percentiles. The hint is for
warm-starting and acceptance-sheet diagnostics.

---

## Acceptance Gates — Verification

- [x] All 4 TDD cycles' tests pass (verified `pytest -v` → 4 passed)
- [x] Real Emma run completes end-to-end via `python -m chuck_mcp_v2.plan_emma`
- [x] Output production_plan.json has `cell_count >= 1500` (2039 actual)
- [x] Output production_plan.json has `hue_cluster_count >= 5` (8 actual)
- [x] `cell_proposal_source: "snic"` stamped on the production_plan.meta
- [x] snic_proposer.py + tests live at `research/v5-overnight/snic-real/`
- [x] Integration patch in canonical `chuck_mcp_v2/plan_emma.py` path

### Commit lineage

```
24fcc10  TDD cycle 1: SNIC-real cell proposer green on full-res Emma
86e2be0  TDD cycle 2: hue cluster diversity green (Emma yields 10 chromatic bins)
140af33  TDD cycle 3: per-cell properties green (lab/chroma/role/adjacency)
d10c867  TDD cycle 4: plan_emma routes through SNIC, not grid (integration patch)
```

---

## Follow-ups (out of scope for this mission)

- Speed: pysnic Python loop dominates wall time. The
  `research/papers/segmentation-cellgraph` notes recommend (b) C++ ref via
  pybind11 or (c) a JAX port for 2-4× speedup. Not blocking — SNIC at 51s on
  full-resolution Emma is acceptable for current iteration cadence.
- Diagnostics: add per-cell CO / EV / ICV stats to enable principled
  parameter tuning (Tier 2 from the cell-graph research notes).
- The `_snic_cell_graph` adapter loads pysnic twice per call (once for
  labels, once for per-cell aggregation in plan_emma). A future refactor
  could share state, but the cost is dominated by SNIC itself, not the
  reaggregation.
- The mediapipe-spatial path (`apply_face_region_constraints`) was kept
  gated behind `--no-face-regions` for the cycle 4 test because cv2 and
  mediapipe are not installed in the validators-reconstruction venv.
  When deps land, the v5 spatial-constraint step will benefit from the
  real SNIC label map (now plumbed through `solver_label_image` from
  `_snic_cell_graph`).
