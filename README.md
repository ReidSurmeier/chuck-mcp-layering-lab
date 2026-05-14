# Chuck MCP Layering Lab

Chuck MCP Layering Lab is a separate experiment for testing mokuhanga-style
layer planning, jigsaw color organization, pigment guidance, and MCP validation
around the Chuck Close / Emma printmaking problem.

This repo is intentionally separate from `emma-mokuhanga-mcp`. Do not mix these
repos, outputs, or MCP tool surfaces.

## Status

Current branch: `main`

Fork lineage:

- upstream base: `ReidSurmeier/woodblock-reidsurmeier-wtf`
- fork point: `6db6f11 Rewrite Chuck MCP README for role-based solver`
- saved baseline branch: `checkpoint/pre-cell-graph-jigsaw-tints`
- current repo remote: `ReidSurmeier/chuck-mcp-layering-lab`

The current build runs end-to-end through the MCP registry, the JAX solver,
cell-graph analysis, printability repair, production-batch planning, and visual
carousel export.

It is not yet a final carving plan. The best older 9-pull solver still matches
the input image more closely by Delta E. This fork is improving organization,
jigsaw grouping, review tooling, and printability constraints so the next solver
iteration can optimize the right structure instead of producing pixel-level
reconstruction plates.

## Goal

The project is testing a production-oriented abstraction:

```text
input image
  -> compressed differentiable study stack
  -> cell graph / jigsaw regions
  -> production batches and repeated pulls
  -> CNC-safe vector plates
```

The solver should build an image through overlapping translucent or semi-opaque
impressions, not through isolated pixel masks. Early plates may contain detailed
carved geometry, but their role should still be supportive: light, transparent,
and useful underneath later color. Later plates should carry stronger chroma,
regional hue changes, shadows, and key/detail work.

For the Emma reference, the useful scale reference is still Yasu Shibata's
production: 27 woodblocks, 113 colors, and 132 pulls. A 9-12 impression JAX
stack is a study, not an adequate production count.

## Current Pipeline

The active v23 pipeline is:

1. `S1` ingest and cache target image.
2. `S3` hue-family analysis.
3. `S3.b` SLIC/cell-graph construction for regional jigsaw reasoning.
4. `S4` role-aware warm start.
5. `S5` JAX/JAXopt inverse solver.
6. `S6.b` jigsaw organization using the persisted cell graph.
7. `S6.c` Delta-E-guarded printability repair before vector export.
8. `S6.d` read-only production batch proposal.
9. `S7+` state masks, block packing, SVG/export surfaces.

The render model is still RGB/JAX pigment blending. It is useful for iteration,
but physical mokuhanga color needs swatches and calibration before editioning.

## Implemented Changes

- Persisted cell graph:
  - `cell_labels.npy`
  - `cell_graph.json`
  - per-cell color, tone, role hints, adjacency, and area diagnostics

- Jigsaw organization:
  - uses the persisted cell graph instead of re-segmenting
  - avoids recovering tint in cells already active enough
  - separates adaptive support roles from local wash/detail roles

- Printability repair:
  - removes tiny islands before SVG export
  - guards repair with Delta E so topology cleanup cannot silently wreck color
  - reports component, island, partial-cell, overlap, and low-alpha pressure

- Flexible pigment and wash library:
  - expanded from the parent catalog to 36 entries
  - includes natural/synthetic pigment anchors and adaptive wash colors
  - treats the list as expandable guidance, not a fixed enforced palette
  - `suggest_pigment_mix` returns premix starting ratios when no single pigment
    is a good match

- Production batch planner:
  - proposes a `4 + 4 + detail` review structure
  - first batch: light pink, blue, orange, and green support roles
  - second batch: regional color/depth roles
  - final batch: regional hue shifts, shadows, contours, and key/detail regions
  - writes `production_batch_plan.json`

- Carousel review export:
  - writes one slide per plate and one slide after each cumulative pull
  - includes both the production-batch proposal and the flat solver sequence
  - writes a contact sheet and manifest for quick review

## MCP Tool Surface

The MCP registry is built from [backend/mcp/registry.py](/home/reidsurmeier/src/chuck-mcp-layering-lab/backend/mcp/registry.py).

Important tools:

- `propose_stack`
- `inspect_plan`
- `forward_render`
- `score_stack_delta_e`
- `score_candidate_stack`
- `solver_telemetry`
- `get_pigments`
- `suggest_pigment_mix`
- `cell_at`
- `inspect_cell`
- `score_printability`
- `propose_plate_reorganization`
- `plan_production_batches`
- `export_print_plan`
- `export_svg`
- `export_block_svgs`
- `generate_carve_order`

`propose_plate_reorganization` and `plan_production_batches` are read-only
planning tools. They propose organization changes; they do not mutate solver
masks yet.

## Setup

Use Python 3.11+ with an NVIDIA GPU.

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab

python3 -m venv .venv-v23
. .venv-v23/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[solver,mcp,io,dev]"
```

Verify JAX sees the GPU:

```bash
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-v23/bin/python - <<'PY'
import jax
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
PY
```

Expected backend:

```text
gpu
```

## Run Through MCP

```bash
WOODBLOCK_DISABLE_SAM=1 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-v23/bin/python - <<'PY'
from backend.mcp.registry import call_mcp_tool

image = "/srv/woodblock-share/input-images/close_emma_2002_2048.jpg"
r = call_mcp_tool("propose_stack", {
    "path": image,
    "solve_profile": "thorough",
    "m_prior": 10,
})
print(r.model_dump(mode="json"))
PY
```

Run as an MCP server:

```bash
.venv-v23/bin/chuck-layering-mcp
```

## Validation Runner

The local validation script generates the full review package:

```bash
WOODBLOCK_DISABLE_SAM=1 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=. \
.venv-v23/bin/python scripts/validate_v23_run.py \
  --input /srv/woodblock-share/input-images/close_emma_2002_2048.jpg \
  --solve-profile fast \
  --m-prior 10 \
  --run-name chuck-batch-production-carousel-v2-fast-m10-main-20260514-1735
```

Default artifact roots:

```text
/srv/woodblock-share/output-images/
/srv/woodblock-share/chuck-clean-outputs/
/srv/woodblock-share/chuck-carousel-slides/
```

Latest reviewed carousel:

```text
/srv/woodblock-share/chuck-carousel-slides/chuck-batch-production-carousel-v2-fast-m10-main-20260514-1735
```

That run produced 75 carousel slides, including 48 production-proposal slides
and the flat solver pull sequence.

## Current Validation Result

Latest run:

```text
chuck-batch-production-carousel-v2-fast-m10-main-20260514-1735
```

Summary:

| Metric | Value |
|---|---:|
| impressions | 12 |
| mean DeltaE76 | 13.992 |
| p95 DeltaE76 | 36.155 |
| printability score | 69.66 |
| production proposal slides | 48 |
| total carousel slides | 75 |

Interpretation:

- The current build is working mechanically.
- The new batch proposal is easier to inspect than the old contact sheet.
- The plate organization is closer to a production-planning abstraction.
- The image match is still worse than the old 9-pull benchmark.
- The next solver change should optimize hierarchical batches directly, not
  merely draw a production proposal after a flat solver has already run.

## Hierarchical Solver Direction

The next serious change should be staged and bounded:

1. Solve the first light-support batch against low-frequency color and luminance.
2. Freeze or softly constrain that batch.
3. Solve the second color/depth batch against residual color and regional mass.
4. Solve detail/key plates against high-frequency residuals.
5. Run a bounded feedback pass that can adjust early plates only for
   low-frequency errors inside a trust region.

That keeps the useful hierarchy without allowing every later error to push the
first support plates into noisy detail masks.

## Tests

Current regression command:

```bash
PYTHONPATH=. /home/reidsurmeier/src/woodblock-reidsurmeier-wtf/.venv-v23/bin/python -m pytest \
  backend/tests/v23/stages/test_s3b_cell_graph.py \
  backend/tests/v23/stages/test_s4_warmstart.py \
  backend/tests/v23/stages/test_s5_solver.py \
  backend/tests/v23/stages/test_s6b_jigsaw_organize.py \
  backend/tests/v23/stages/test_s6c_printability_repair.py \
  backend/tests/v23/stages/test_s7_block_pack.py \
  backend/tests/v23/stages/test_orchestrator.py \
  backend/tests/v23/direct/test_cell_graph_tools.py \
  backend/tests/v23/direct/test_batch_planning_tools.py \
  backend/tests/v23/direct/test_d9b_tools.py \
  backend/tests/v23/unit/test_forward_render_km.py \
  backend/tests/v23/unit/test_topology_repair.py \
  -q
```

Last local result:

```text
109 passed in 135.22s
```

## Known Gaps

- Production batch planning is still a proposal, not a solver-mutating stage.
- The carousel proposal does not yet optimize batch composites against the
  target.
- Early support plates are still too sparse in some runs.
- Dark/key information can enter the review sequence too early.
- Actual JAXopt iteration counts are not persisted; telemetry reports the
  configured profile budget.
- Calibration still needs real paper/pigment swatch data before color recipes
  should be trusted for editioning.
- SVG export should remain gated by printability/topology review.

## License

MIT. Copyright 2026 Reid Surmeier.
