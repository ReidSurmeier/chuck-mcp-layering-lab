# AGENTS.md — Chuck MCP Layering Lab

This repo is the Chuck MCP experiment. It is separate from
`emma-mokuhanga-mcp`; do not touch that repo from this workspace unless the user
explicitly changes scope.

## Mission

Build and test a mokuhanga-style Block / Impression / Mask planner for the
Chuck Close / Emma printmaking problem. The output is a visually plausible
block/proof plan designed by printmaking rules, not a recovered artist process.

V1 acceptance is physical-plan plausibility:

- validators score authoritative Masks and proof states;
- Review previews are human-facing only;
- final-match dE is telemetry, not the sole hard gate;
- Blocks, Impressions, Masks, Pigments, Order, and Underprints should be
  readable to a printmaker.

## Agent skills

### Issue tracker

Issues and PRDs for this repo are tracked in GitHub Issues for
`ReidSurmeier/chuck-mcp-layering-lab` via the `gh` CLI. Because this clone also
has an `upstream` remote, pass `--repo ReidSurmeier/chuck-mcp-layering-lab`
explicitly. See `docs/agents/issue-tracker.md`.

### Triage labels

The repo uses the default Matt Pocock triage vocabulary: `needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See
`docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo: read root `CONTEXT.md` and relevant `docs/adr/`
entries before changing domain behavior. See `docs/agents/domain.md`.

## Current Verified State

Latest Emma run:

```text
/home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/
/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-13/sheet_iter_13.png
```

Current metrics:

```text
gates:        3 / 5
dE_mean:      18.778
dE_p95:       40.3015
PNC:          28 / 28 pass
jigsaw:       14 / 28 pass
underprint:   34.35% legacy overlap metric
```

JAX reports GPU on this machine:

```text
JAX 0.10.0
backend gpu
CudaDevice(id=0)
```

## Commands

Activate the current environment:

```bash
cd /home/reidsurmeier/src/chuck-mcp-layering-lab
. .venv-renderer/bin/activate
```

Run focused tests:

```bash
.venv-renderer/bin/python -m pytest -q research/v3-construction/validators-reconstruction
.venv-renderer/bin/python -m pytest -q research/v4-build/hybrid-optimizer
.venv-renderer/bin/python -m pytest -q research/v5-overnight/loop-runner
```

Run a single Emma iteration:

```bash
bash research/v5-overnight/loop-runner/run_iter.sh 13 extreme-cells 26
```

Inspect latest metrics:

```bash
jq '{outer_iter_count,n_gates_passed,delta_e_mean,delta_e_p95,history,notes}' \
  /home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/hybrid_result.json
```

## Key Decisions

- ADR-0006: validators score truth objects, not Review previews.
- ADR-0007: V1 accepts a plausible print plan, not a solved reconstruction.
- ADR-0005: topology rules belong in scoring/repair or upstream proposal, not
  as soft differentiable loss terms.
- ADR-0002: Mixing and Overprint are different render events.

## Active Issues

- `#1` PRD: Chuck MCP V1 plausible mokuhanga plan
- `#2` typed validator-plan module
- `#3` split Review preview / Validator truth / archive outputs
- `#4` carved-region Mask topology
- `#5` semantic Underprint scoring
- `#6` README and MCP tool-surface alignment
- `#7` render-tier adapter

Use:

```bash
gh issue list --repo ReidSurmeier/chuck-mcp-layering-lab --state open
```

## Next Engineering Priority

Do not spend the next pass on random restarts. Iter 13 proves the outer loop now
carries state forward, but the result still fails physically. The next useful
build is carved-region Mask topology before continuous color solving, followed
by semantic Underprint scoring and clean output packaging.
