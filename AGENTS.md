# AGENTS.md — v23-MCP migration handoff (Claude Code → Codex 5.5)

You are picking up the **v23-MCP** build mid-stream from Claude Code. Same machine, same files. Read this top-to-bottom before touching anything.

## 0. Mission

`v23-MCP` is a mokuhanga (Japanese woodblock) print-planning tool. Ingest one PNG/JPG → produce a CNC-ready multi-impression carving plan styled after Chuck Close's *Emma* (2002) Pace-Editions process. Architecture is a **10-stage pipeline (S1-S10)** exposed as an **MCP server** with **40 tools across 7 tiers**.

Output is **plausible underprint candidates** — designed by printmaking rules, not inferred from physical evidence. WB-LANG-01/-02 lint enforces this in tests.

## 1. Repo state (as of handoff)

- **Repo**: `/home/reidsurmeier/src/woodblock-reidsurmeier-wtf`
- **Branch**: `main`
- **HEAD**: `691e86a` (D14.e `split_impression` by mask_island real) + uncommitted D14.f `pin_region` real on top
- **Tree**: dirty — `hitl.py`, `test_d14d_hitl_real.py`, `test_d9b_tools.py`, `AGENTS.md`. Commit before any other work.
- **venv**: `.venv-v23/` (Python 3.12)
- **Test count**: 231 passed, 1 skipped, 22 xfailed, 1 xpassed
- **JAX**: CPU-only on this box. CUDA install abandoned (network too slow for cuDNN wheels). Solver works fine on tests ≤ 64² with `WOODBLOCK_DISABLE_SOLVER=1` autouse bypass.

## 2. Quickstart

```bash
cd /home/reidsurmeier/src/woodblock-reidsurmeier-wtf

# Activate venv
source .venv-v23/bin/activate

# Run full v23 suite
WOODBLOCK_HOME=/tmp/wb-codex timeout 300 .venv-v23/bin/python -m pytest backend/tests/v23/ -q

# Run a single test file
.venv-v23/bin/python -m pytest backend/tests/v23/direct/test_d14d_hitl_real.py -x -q

# Run the MCP server (manual smoke)
.venv-v23/bin/python -m backend.mcp.server
```

**Expected**: `226 passed, 22 xfailed, 1 xpassed`. The 22 xfailed are sentinel placeholders for unwired tools — they flip to passed as you wire each one.

## 3. Architecture map (S1-S10)

| Stage | What it does | File |
|-------|--------------|------|
| S1 | Ingest + canonicalise PNG, SHA-256 + EXIF strip, register session | `backend/services/v23/stages/s1_ingest.py` |
| S2 | SAM2.1 region prior via HTTP gateway (`WOODBLOCK_DISABLE_SAM=1` bypass) | `backend/services/v23/stages/s2_sam.py` |
| S3 | Hue family classification (7 families) + per-family label map PNG | `backend/services/v23/stages/s3_hue_family.py` |
| S4 | Tan RGB-geometry warm-start (convex-hull + Delaunay barycentric) | `backend/services/v23/stages/s4_warmstart.py` |
| S5 | JAX + jaxopt L-BFGS inverse solver, sigmoid box reparam | `backend/services/v23/stages/s5_solver.py` |
| S6 | Three-state mask classification (visible / covered / support / none) | `backend/services/v23/stages/s6_three_state_mask.py` |
| S7 | DSATUR-style greedy block packing + pull-group assignment | `backend/services/v23/stages/s7_block_pack.py` |
| S8 | Topology repair (morph open + close, ΔE-regression-guarded) | `backend/services/v23/core/topology_repair.py` |
| S9 | SVG vectorize per impression (uses `backend/svg_postprocess.py` + potrace) | **NOT YET WIRED** — see § 8 |
| S10 | ZIP emit (manifest.json + recipe.md + composite PNG + per-impression PNGs) | `backend/services/v23/stages/s10_emit.py` |

Chained by: `backend/services/v23/orchestrator.py::run_pipeline_partial`.

## 4. Persistence layout

When the solver runs, the orchestrator writes per-plan artifacts under:

```
~/.woodblock/v23/sessions/<sid>/plans/<plan_id>/
├── plan.json          # PartialPlan as JSON
├── state_stack.npy    # (M, H, W) uint8 three-state classification
├── alpha_stack.npy    # (M, H, W) float32 per-pixel coverage from S5
├── pigment_idx.npy    # (M,) int32 pigment ids
└── target.npy         # (H, W, 3) float32 canonical sRGB target (added D14.c)
```

`PartialPlan` lives in `backend/services/v23/orchestrator.py`. All downstream tools key off `plan.alpha_stack_path` etc.

**Plan dirs `_tmp` are scratch from SAM stage — do not assume the plan_id matches `_tmp`.**

## 5. MCP tool surface (40 tools, 7 tiers)

| Tier | File | Status |
|------|------|--------|
| Tier 0 — Core (6) | `backend/mcp/tools/core.py` | REAL for `propose_stack`, `inspect_plan`, `forward_render`, `score_candidate_stack`, `score_stack_delta_e`, `export_print_plan`. **9 IMPL_PENDING** for sub-renderings (heatmap, quad, pixel, etc) |
| Tier 1 — HITL (10) | `backend/mcp/tools/hitl.py` | REAL: `compare_plans`, `merge_impressions`, `split_impression`(mask_island), `simplify_masks_for_carving`, `pin_region`(all 3 actions). **5 IMPL_PENDING**: `alternative_stacks`, `generate_stack_candidates`, `merge_impressions_by_hue_family`, `split_impression`(hue_subcluster), `adjust_pull_groups`, `compare_alternate_recipes` |
| Tier 2 — Overlay (3) | `backend/mcp/tools/overlay.py` | REAL: `simulate_overprint` (t1_mixbox). **3 IMPL_PENDING** for t2/t3 tiers |
| Tier 3 — Introspect (6) | `backend/mcp/tools/introspection.py` | REAL: all 6 (`get_pigments`, `get_emma_priors`, `get_defaults`, `solver_telemetry`, `dE_at`, `pigment_at`). 1 mock-fallback for unknown plan_id |
| Tier 4 — Session (4) | `backend/mcp/tools/session.py` | REAL: all 4 |
| Tier 5 — Carve (3) | `backend/mcp/tools/carve.py` | **3 IMPL_PENDING**: `svg_per_impression`, `svg_per_block`, `carve_order` — S9 not wired |
| Tier 6 — Calibration (2) | `backend/mcp/tools/calibration.py` | **2 IMPL_PENDING**: `capture_swatch`, `fit_pigments` — ColorChecker detection not wired |

**Total real today: 23/40. Remaining to wire: 17.**

Grep `IMPL_PENDING` to find every remaining stub.

## 6. TDD discipline (RIGID — do not break)

Every wire-up: **RED → GREEN → COMMIT**.

1. Write a failing test in `backend/tests/v23/direct/test_d14X_<name>.py` (D14.f, D14.g, ...).
2. Run it, confirm it fails for the right reason (KeyError, ImportError, etc).
3. Implement the real backing.
4. Run that file alone → green.
5. Run full suite → no regressions.
6. Commit with message: `v23-MCP D14.<letter> wire <tool> real` + body explaining inputs/outputs/refusals.

**Don't refactor production code unless the test forced it.** Surgical changes only.

## 7. Banned terms (WB-LANG-01 / WB-LANG-02)

Lint enforced in tests:

- **NEVER** say: "recovered underlayer(s)", "true underlayer(s)", "physical evidence", "recovers the original"
- "Mixbox predicts the print" requires the qualifier "as if pre-mixed in a well"
- Output framing: "**plausible underprint candidates**" + "**designed by printmaking rules, not inferred from physical evidence**"

Even in negated form ("NOT recovered underlayers"), the regex matches. Reword to avoid the substring entirely.

## 8. Next deliverables (sequenced)

In execution order. Each is a single TDD cycle ending in one commit.

| ID | Tool / feature | Notes |
|----|----------------|-------|
| D14.f | `pin_region(plan_id, region, action, pigment_id)` | **DONE (uncommitted)**. Direct alpha-stack edit: force/forbid/merge with bbox + clamping. No solver re-run. |
| D14.g | `adjust_pull_groups(plan_id, hints)` | Plan metadata mutation — re-pack DSATUR with constraints `{"merge_pull_groups": [...], "split_block": N, "force_block": {impression_id: block}}`. New `plan_id` persisted. |
| D14.h | `merge_impressions_by_hue_family(plan_id, family_name)` | Auto-select impressions where pigment ∈ family, delegate to `merge_impressions`. Family map: see `backend/services/v23/core/hue_families.py`. |
| D14.i | `alternative_stacks(plan_id, n)` | Re-run solver `n` times with seed perturbation OR M-prior perturbation. Parallel calls to `run_pipeline_partial`, persist n new plans, return ranked list by `score_candidate_stack`. |
| D14.j | `split_impression(by="hue_subcluster")` | Load target.npy at the impression's mask region, k-means in Oklab on masked pixels, k=2 default. Split alpha plane by Oklab assignment. |
| D14.k | Tier 0 sub-rendering tools real | `inspect_plan` returns extra artifact paths (heatmap, quad, pixel) — render on-demand from persisted alpha_stack + target.npy. |
| D14.l | **S9 SVG per impression** | For each impression: `(alpha_stack[i] > 0.5).astype(uint8)` → potrace → SVG with kento reg marks. Wire `svg_per_impression`, `svg_per_block`, `carve_order`. |
| D14.m | Calibration: `capture_swatch` real | OpenCV ColorChecker detection, extract 24 patch sRGB, persist as calibration_set.json under session. |
| D14.n | Calibration: `fit_pigments` real | Solve per-pigment t1→t2 LUT entry from swatch overprints. Persists `mixbox_lut.npz` at `~/.woodblock/v23/luts/`. |
| D14.o | Tier 2 t2/t3 overlay tools | `simulate_overprint(tier="t2")` loads the fitted LUT; `compare_render_tiers` diffs t1 vs t2 composites. |
| Ring 5 | Corpus regression framework | `backend/tests/v23/corpus/` — golden plan.json per fixture, ΔE budget gate (mean ≤ 1.5, p95 ≤ 3.0) on `corpus/close_emma_2002.png`. |
| Tier-1 gate | Run `propose_stack` on `corpus/close_emma_2002.png` | Expected to FAIL ΔE budget on CPU JAX at full res — that's OK, document the gap. Real gate runs on Windows WSL2 GPU host. |

## 9. Conventions

- **Test rings** under `backend/tests/v23/`:
  - `direct/` — pure-python tool/orchestrator calls
  - `transport/` — MCP transport layer
  - `conversation/` — mock-Opus conversation flows
  - `stages/` — per-stage unit tests
  - `unit/` — algorithmic units
  - `corpus/` — Ring 5 regression (mostly empty, build this out)
  - `solver_smoke/` — JAX solver convergence
  - `scaffold/` — skeleton-level smoke
- **Autouse fixtures** in `conftest.py` set `WOODBLOCK_DISABLE_SAM=1` + `WOODBLOCK_DISABLE_SOLVER=1`. Solver-exercising tests `monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER")`.
- **400 LOC cap** per file. Split before approaching the limit.
- **`_isolate(monkeypatch, tmp_path)`** helper sets `WOODBLOCK_HOME=tmp_path` + reloads `backend.mcp.paths` + `backend.services.v23.session` + `backend.services.v23.orchestrator` — copy this pattern in new test files.
- **Plan-mutating tools** persist a NEW `plan_id` under the SAME session — never overwrite parent.
- **`ToolResult`** dataclass: `ok: bool, data: dict | None, errors: list[WoodblockError]`. Real tools never raise — they return refusal-tier `WoodblockError`s in `errors`.
- **`WoodblockError` tiers**: `refusal` (4xx-class, ok=False), `degraded` (partial, ok=True with errors), `internal` (5xx-class, panic).

## 10. Gotchas

- `.gitignore` actively excludes `.claude/` and `CLAUDE.md` — those are Claude Code scratch and **should not** be committed. Codex's own scratch (if any) should be added similarly.
- `npm` worktree pollution: `.claude/worktrees/agent-*` ARE git submodules from prior swarm runs — leave alone, don't `git add` them.
- `WOODBLOCK_DISABLE_SAM=1` is autouse default — opt out per-test with `monkeypatch.delenv`.
- SAM gateway is `http://localhost:8001` (v20 sidecar). Not running in this dev environment. All SAM-touching code degrades gracefully.
- JAX `forward_render_jax.forward_render` expects `(H, W, M)` alpha, NOT `(M, H, W)`. `alpha_stack` is stored as `(M, H, W)` — transpose when calling.
- `gitnexus` FTS index hooks complain about read-only DB — ignore, they're stale-cache warnings, not real failures.
- `jaxopt` `DeprecationWarning` — known, no maintainer, not actionable.
- `scikit-image` `morphology` warnings (`min_size` deprecation, `binary_closing` deprecation) — track but don't fix mid-build.

## 11. References

- **Build chain doc**: `docs/v23-mcp/build-chain.md` (D-numbered stages)
- **Addenda**: `docs/v23-mcp/addendum-v3.md`, `addendum-v4.md`, `addendum-v5.md` (fixes 1-N + WB-LANG specs)
- **Research**: `docs/v23-mcp/research-v23-mcp-defaults.md` (locked technical defaults: ΔE2000 target, M_prior range, block count target, etc.)
- **Tan algorithm**: see comments in `backend/services/v23/stages/s4_warmstart.py`
- **Mixbox**: see `backend/services/v23/core/forward_render_jax.py` (vendored 13-pigment table)
- **Corpus**: `corpus/close_emma_2002.png` (the Pace-Editions Tier-1 reference image — do not edit)
- **Pigment list**: 13 entries, indexed 0-12, names in `backend/mcp/tools/introspection.py::_PIGMENT_NAMES`

## 12. Cost / runtime budget

- Tests: < 1 min for full v23 suite on CPU
- Solver `fast` profile: ~5 s for 32² test image, ~60 s budget for production
- Solver `default`: 180 s budget
- Solver `thorough`: 600 s budget
- Real production-size image (≥ 1 Mpx) requires the GPU host — defer

## 13. Hand-off check

When Codex picks this up, the first action should be:

```bash
cd /home/reidsurmeier/src/woodblock-reidsurmeier-wtf
git log --oneline -10
WOODBLOCK_HOME=/tmp/wb-codex-smoke timeout 300 .venv-v23/bin/python -m pytest backend/tests/v23/ -q --tb=line
```

If you see `226 passed, 22 xfailed, 1 xpassed` you're synced. Begin with **D14.f** (`adjust_pull_groups`).

If the count differs, run `git status` + `git log` and reconcile before writing any code.
