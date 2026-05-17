# Chuck MCP Layering Lab

Chuck MCP Layering Lab is a separate experiment for building and testing a
mokuhanga-style Block / Impression / Mask planner around the Chuck Close /
Emma printmaking problem.

This repository is intentionally separate from `emma-mokuhanga-mcp`. Do not mix
the repos, outputs, MCP tool surfaces, or issue trackers.

## V1 Acceptance

V1 is a visually plausible mokuhanga block/proof planner, not a claim that the
input image has been quantitatively solved or that the artist's real process has
been recovered.

Primary acceptance gates:

- validators score authoritative Masks and proof states, not contact-sheet
  pixels;
- the Order and proof progression should read like an incremental woodblock
  print;
- Mask geometry should be connected and separable enough for jigsaw carving;
- Underprints are designed support structures, not inferred physical evidence;
- final-match dE is reported as telemetry and an improvement target.

See `CONTEXT.md`, `docs/adr/0006-validator-truth-over-previews.md`, and
`docs/adr/0007-v1-accepts-plausible-print-plan.md`.

## Current Status

Current branch: `main`

GitHub repo: `ReidSurmeier/chuck-mcp-layering-lab`

Fork lineage:

- upstream base: `ReidSurmeier/woodblock-reidsurmeier-wtf`
- current repo remote: `ReidSurmeier/chuck-mcp-layering-lab`

Latest verified Emma run:

```text
/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/
/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-13/sheet_iter_13.png
```

Iter 13 fixed two important measurement/plumbing issues:

- plate-not-composite now scores Mask truth and passes `28/28`;
- the outer loop now warm-starts from previous solved/repaired state instead of
  replaying the same solve.

Latest metrics:

```text
gates:        3 / 5
dE_mean:      18.778
dE_p95:       40.3015
underprint:   34.35% legacy overlap metric
GPU:          JAX 0.10.0, backend gpu, CudaDevice(id=0)
```

This is not good enough as a final print plan yet. The remaining hard failures
are carved-region topology, jigsaw separation, Underprint methodology scoring,
and color/render fidelity.

## Active Pipeline

The research path currently used for Emma validation is:

```text
input image
  -> SNIC/cell graph
  -> production plan candidates
  -> hybrid optimizer
  -> morphology repair
  -> alpha/proof artifact dump
  -> validator plan
  -> review sheet + metrics
```

Important local entry points:

- `chuck_mcp_v2/plan_emma.py`
- `research/v4-build/hybrid-optimizer/alternating_loop.py`
- `research/v5-overnight/loop-runner/run_iter.sh`
- `research/v5-overnight/loop-runner/build_validator_plan.py`
- `research/v3-construction/validators-reconstruction/run_all_validators.py`
- `research/v5-overnight/alpha-proof-dumper/dumper.py`

## Setup

Use the renderer environment for the current research path:

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab
. .venv-renderer/bin/activate
python -m pip install -e ".[solver,mcp,io,viz,dev]"
```

Verify JAX sees the GPU:

```bash
.venv-renderer/bin/python - <<'PY'
import jax
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
PY
```

Expected backend on this machine:

```text
gpu
```

The CUDA runtime currently emits a kernel-driver-version warning in logs, but
JAX still reports `CudaDevice(id=0)` and runs the Emma solver on GPU.

## Run Latest Emma Loop

```bash
bash research/v5-overnight/loop-runner/run_iter.sh 13 extreme-cells 26
```

That writes:

```text
/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/
/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-13/
```

The row is appended to:

```text
research/v5-overnight/loop-runner/iterations.csv
```

## Focused Tests

```bash
.venv-renderer/bin/python -m pytest -q research/v3-construction/validators-reconstruction
.venv-renderer/bin/python -m pytest -q research/v4-build/hybrid-optimizer
.venv-renderer/bin/python -m pytest -q research/v5-overnight/loop-runner
```

Latest observed results:

```text
validators:       17 passed
hybrid optimizer: 13 passed
loop runner:       1 passed
```

## MCP Tool Surface

The MCP registry is built from `backend/mcp/registry.py`.

Important tool families:

- stack proposal and inspection: `propose_stack`, `inspect_plan`,
  `forward_render`, `score_candidate_stack`, `score_stack_delta_e`;
- pigment and render introspection: `get_pigments`, `suggest_pigment_mix`,
  `solver_telemetry`;
- cell and printability review: `cell_at`, `inspect_cell`,
  `score_printability`, `propose_plate_reorganization`;
- production planning/export: `plan_production_batches`,
  `plan_adaptive_ink_stack`, `export_print_plan`, `export_svg`,
  `export_block_svgs`, `generate_carve_order`.

Use public-facing docs and issues with the glossary terms Block, Impression,
Mask, Pigment, Order, Underprint, Review preview, and Validator truth. Legacy
research modules may still use `Plate` internally.

## Product / Issue Tracker

GitHub issues for this repo live at:

```text
https://github.com/ReidSurmeier/chuck-mcp-layering-lab/issues
```

Current PRD and vertical slices:

- `#1` PRD: Chuck MCP V1 plausible mokuhanga plan
- `#2` typed validator-plan module
- `#3` split Review preview / Validator truth / archive outputs
- `#4` carved-region Mask topology
- `#5` semantic Underprint scoring
- `#6` README and MCP tool-surface alignment
- `#7` render-tier adapter

Because this clone also has an `upstream` remote, pass
`--repo ReidSurmeier/chuck-mcp-layering-lab` to `gh issue` commands.

## Research Notes

Useful docs:

- `docs/diagnosis/2026-05-17-v5-validator-and-outer-loop.md`
- `docs/architecture/deepening-opportunities-2026-05-17.md`
- `docs/audit-response-and-reconstruction-plan-2026-05-17.md`
- `research/v5-overnight/loop-runner/FINAL_REPORT.md`

The best next algorithmic work is not more random restarts. It is improving the
Mask topology before continuous color solving: connected carved regions,
physical spacing, semantic Underprint scoring, and then better Overprint render
tiers.
