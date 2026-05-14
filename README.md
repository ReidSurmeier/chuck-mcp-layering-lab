# Chuck MCP Layering Lab

Chuck MCP Layering Lab is a separate experimental repo for testing broader
underlayers, jigsawed regional color plates, image-driven accent passes, and
pigment mix guidance on top of the current Chuck MCP tool stack.

This repo starts from `ReidSurmeier/woodblock-reidsurmeier-wtf` at
`6db6f11`. It is intentionally separate from `emma-mokuhanga-mcp`; do not mix
those repos or tool surfaces.

## Current Status

The current `main` branch contains the layering-lab fork:

- saved baseline: `chuck-mcp-speckle-m10-20260512`
- parent role-based tag: `chuck-mcp-rolebased-joint-20260512`
- parent fork point: `6db6f11 Rewrite Chuck MCP README for role-based solver`

The parent solver works on GPU and produces a real overlapping pull stack. This
fork changes the next hypothesis: early base plates must be broad by
construction, middle color plates should prefer separated jigsaw regions, and
missing colors should be expressible as pigment-mix recipes instead of forcing
the optimizer to invent noisy masks.

## What It Does

Chuck MCP Layering Lab takes one image and generates:

- an ordered stack of translucent/semi-opaque impressions
- cumulative print previews after every pull
- per-impression alpha masks
- a final composite preview
- plan metadata for MCP tools
- optional SVG/carving exports
- pigment mix suggestions for colors outside the catalog

The system does **not** recover historical or true underlayers from an image.
It designs plausible underprint candidates under the current pigment/rendering
model.

## Layering Lab Direction

The reference images in `/srv/woodblock-share/Examples` show a different
construction logic from the earlier pixel-reconstruction runs:

- the first block is light yellow because the reference needs a high-luminance
  warm support field; in another image this role should be the lightest broad
  support pigment selected from that image
- base colors carry large diffuse areas before final detail appears
- red is a separate color role here because it is a high-chroma regional accent
  with crisp boundaries, not because the solver should always add red
- later hue shifts are jigsawed as separated regional blocks with clear borders
- optical mixing comes from stack order, local opacity, and premixed pigment
  choices, not from every plate fading into every neighbor

That makes the most important failure clear: broad base roles must not start as
skinny pixel-level detail. In this fork, the first support role is inferred from
low-frequency target structure, and high-chroma regional colors are seeded as
their own accent plates when warranted.

## Current Solver Changes

The current S5 solver uses:

- JAX + JAXopt L-BFGS on CUDA
- light-to-dark print ordering, with black/key detail last
- bounded internal solve grids for 2K images on 12 GB GPUs
- a role layout after print ordering:
  - early pulls: broad underlayer controls
  - middle pulls: regional color/shadow controls
  - final pulls: detail/key controls
- different parameterization by role:
  - underlayers use a 12x coarser control grid
  - middle impressions use a 4x coarser control grid
  - detail impressions remain full solve-grid
- a role-aware warm start:
  - broad underlayer seed from low-frequency color/tonal structure
  - underlayer pigment inferred from broad-support color, preferring lighter
    pigments early
  - separate high-chroma accent seed when a regional hue shift exists
  - base-hue Tan seeds are blurred before reaching S5
- jigsaw pressure for middle plates:
  - pairwise overlap penalty between middle color plates
  - stronger high-frequency penalty on underlayers
- edge-weighted RGB loss
- low-pass target loss
- layer-weighted TV
- local-support/speckle penalty
- dark-on-bright penalty

The pigment catalog is expanded from the parent 13 entries to 24 entries:
cadmium/hansa yellows, orange, cadmium red, quinacridone magenta, violets,
ultramarine/cobalt/prussian/phthalo/cerulean blues, viridian/phthalo/sap greens,
burnt/raw sienna and umber, yellow ochre, alizarin crimson, vermilion, naphthol
red, forest green, and ivory black.

The staged warm-up solver is available but not enabled by default:

```bash
WOODBLOCK_ROLE_WARMUP=1
```

Validation in the parent repo showed staged warm-ups over-shaped early pulls on
the Emma test image. The lab fork keeps the default as joint optimization over
role-parameterized groups while changing the warm-start and role geometry.

## Solve Profiles

`propose_stack` accepts `solve_profile` and optional `m_prior`.

| Profile | Max L-BFGS iterations | Default `m_prior` | Internal pixel budget |
|---|---:|---:|---:|
| `fast` | 60 | 6 | 256k |
| `default` | 180 | 8 | 512k |
| `thorough` | 400 | 10 | 768k |

`m_prior` is validated from 4 through 12.

For large images, the solver optimizes on a bounded internal grid and returns
full-resolution masks/previews. Override the grid cap with:

```bash
WOODBLOCK_SOLVER_MAX_PIXELS=1200000
```

## Latest Reference Run

Input:

```text
/srv/woodblock-share/input-images/close_emma_2002_2048.jpg
```

Parent role-based output:

```text
/srv/woodblock-share/output-images/chuck-rolebased-joint-main-20260512-160644
```

Clean image-only output folder:

```text
/srv/woodblock-share/chuck-clean-outputs/chuck-rolebased-joint-main-20260512-160644
```

Metrics:

| Metric | Value |
|---|---:|
| mean DeltaE76 | 7.277 |
| p95 DeltaE76 | 18.258 |
| solver wall time | 42.86s |
| optimized shape | 974 x 789 |
| downsample scale | 0.47558 |
| impressions | 9 |
| first 3 pull components at alpha >= 0.30 | 1364 |

Parent print order:

1. cadmium yellow
2. hansa yellow
3. cadmium orange
4. quinacridone magenta
5. burnt sienna
6. cobalt violet
7. viridian green
8. cobalt blue
9. ivory black

Parent comparison to the saved M10 build:

| Metric | Saved M10 | Role-Based Current |
|---|---:|---:|
| mean DeltaE76 | 6.941 | 7.277 |
| p95 DeltaE76 | 17.772 | 18.258 |
| first 3 pull components | 2847 | 1364 |
| solver wall time | 27.7s | 42.9s |

Interpretation: the parent role-based solver made early pulls less fragmented
but still leaked final-image detail into color plates. This fork is testing the
next structural constraint set before accepting more SVG/carving output.

## Setup

Use Python 3.11+ on Linux/WSL2 with an NVIDIA GPU. The current tested host uses
JAX 0.10.0 with CUDA 13 packages.

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab

python3 -m venv .venv-v23
. .venv-v23/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[solver,mcp,io,dev]"
```

Verify GPU JAX:

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

## Run Through MCP Registry

This runs the same in-process tool path the MCP server exposes:

```bash
WOODBLOCK_DISABLE_SAM=1 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-v23/bin/python - <<'PY'
from backend.mcp.registry import call_mcp_tool

image = "/srv/woodblock-share/input-images/close_emma_2002_2048.jpg"

result = call_mcp_tool("propose_stack", {
    "path": image,
    "solve_profile": "thorough",
})
print(result.model_dump(mode="json"))
PY
```

Useful follow-up tools:

```python
call_mcp_tool("inspect_plan", {"plan_id": plan_id, "focus": "composite"})
call_mcp_tool("inspect_plan", {"plan_id": plan_id, "focus": "per_impression"})
call_mcp_tool("forward_render", {"plan_id": plan_id})
call_mcp_tool("score_stack_delta_e", {"plan_id": plan_id})
call_mcp_tool("score_candidate_stack", {"plan_id": plan_id})
call_mcp_tool("solver_telemetry", {"plan_id": plan_id})
call_mcp_tool("suggest_pigment_mix", {"target_hex": "#c65a40"})
call_mcp_tool("export_svg", {"plan_id": plan_id})
call_mcp_tool("export_block_svgs", {"plan_id": plan_id})
```

## Run As MCP Server

Console entry point:

```bash
.venv-v23/bin/woodblock-mcp
```

Claude Code registration example:

```bash
claude mcp add chuck_layering_lab --scope user -- \
  ssh reidsurmeier2@100.67.23.102 \
  "wsl -d Ubuntu -- /home/reidsurmeier/src/chuck-mcp-layering-lab/.venv-v23/bin/chuck-layering-mcp"
```

Then verify:

```bash
claude mcp list | grep chuck_layering_lab
```

## Tool Surface

The registered tool surface is generated from:

```text
backend/mcp/registry.py
```

Primary tools:

- `ingest_reference_image`
- `analyze_image`
- `build_hue_family_map`
- `propose_stack`
- `inspect_plan`
- `forward_render`
- `simulate_overprint`
- `score_stack_delta_e`
- `score_candidate_stack`
- `solver_telemetry`
- `suggest_pigment_mix`
- `export_print_plan`
- `export_svg`
- `export_block_svgs`
- `generate_carve_order`

Additional modules provide HITL edits, calibration, session, carve, and overlay
tools.

## Output Locations

Default plan/session artifacts:

```text
~/.woodblock/v23/
```

Shared full outputs from validation runs:

```text
/srv/woodblock-share/output-images/
```

Clean image-only outputs:

```text
/srv/woodblock-share/chuck-clean-outputs/
```

Use the clean folder when reviewing visuals. It contains only:

- final composite
- incremental pull contact sheet
- impressions + cumulative pull contact sheet
- individual cumulative pull images

## Architecture

```text
MCP client
  -> backend/mcp/v23_server.py
  -> backend/mcp/registry.py
  -> backend/mcp/tools/*
  -> backend/services/v23/orchestrator.py
  -> S1 ingest
  -> S2 optional SAM gateway
  -> S3 hue family map
  -> S4 layering-lab warm start
  -> S5 JAX role/jigsaw inverse solver
  -> S6 three-state mask classifier
  -> S7 block packing
  -> S8 topology diagnostics/repair hooks
  -> S9 SVG vectorization
  -> S10 manifest and ZIP emit
```

## Render Model

Current default rendering is Tier 1:

- RGB/JAX forward stack
- pigment anchors from the 24-pigment layering-lab catalog
- rendered as if pigments were pre-mixed in a well

This is directionally useful but not physically final mokuhanga overprint
simulation. `suggest_pigment_mix` gives practical premix starting ratios for
unavailable colors, but final color decisions still need swatches on the target
paper. T2 empirical swatch correction exists as a local path after calibration.
Spectral/two-flux rendering remains future work.

## What Is Still Not Fixed

The layering-lab solver still needs:

- SLIC/superpixel brushed-zone parameters for middle impressions
- hard jigsaw region assignment before vectorization
- explicit acceptance gates for base-role topology
- hard pre-vectorization machinability scoring
- component-count acceptance thresholds in normal tool output
- better persistence of actual JAXopt `iter_num`, not just max iteration budget
- broader test coverage across `/Volumes/woodblock/Examples`
- more careful visual gates before accepting SVGs as carving geometry

The current lab build plan lives here:

```text
docs/layering-lab-build-plan.md
```

## Tests

Focused checks for this fork:

```bash
PYTHONPATH=. .venv-v23/bin/python -m pytest \
  backend/tests/v23/unit/test_forward_render_km.py \
  backend/tests/v23/stages/test_s4_warmstart.py \
  backend/tests/v23/stages/test_s5_solver.py \
  backend/tests/v23/stages/test_s10_emit.py \
  backend/tests/v23/direct/test_d9b_tools.py \
  backend/tests/v23/unit/test_pydantic_block.py \
  -q
```

Last focused result:

```text
76 passed in 79.61s
```

## License

MIT. Copyright 2026 Reid Surmeier.
