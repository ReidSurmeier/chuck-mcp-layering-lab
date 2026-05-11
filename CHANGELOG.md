# Changelog

All notable changes to woodblock_stack (v23-MCP) are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning will follow [SemVer](https://semver.org/spec/v2.0.0.html) once a release ships. Until then, dated unreleased scaffold milestones are tracked under `[Unreleased]` and rolled into a `0.0.1` scaffold tag at the end of D1–D3.

## [Unreleased]

### Added

- Repository hygiene scaffolding: `README.md` (full), `LICENSE` (MIT), `CONTRIBUTING.md` (TDD discipline + 400 LOC cap + branch/commit format + banned-terms list), `.editorconfig`, `.github/CODEOWNERS`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/blocker.yml`, `.github/ISSUE_TEMPLATE/build-step.yml`, `docs/architecture/README.md`, `docs/adr/0001`..`0005`, this `CHANGELOG.md`.
- ADR-0001 — MCP-first architecture (v22 web-app → v23 MCP server pivot).
- ADR-0002 — 3-tier forward-render (overlay-not-mixing lock per addendum-v4).
- ADR-0003 — stdio-over-SSH transport (synthesizer reconciliation).
- ADR-0004 — Day-1 MCP tool surface capped at 11 (addendum-v3 fix 2).
- ADR-0005 — Topology rules 6+7 move out of the optimizer (addendum-v3 fix 1).

## [0.0.1] — v23-MCP scaffold (2026-05-10 → 2026-05-11)

Pre-ship scaffold milestone. MCP server boots, types validate, filesystem state layer works. No solver yet, no real tools beyond import stubs.

### Added (chronological, oldest → newest)

- `97a9de6` — **Wave A baseline + CONTEXT.md + v23 plan reference.** Imported salvaged helpers from v20 separation pipeline (`separate_v20.py` lifted utilities, `svg_postprocess.py` selected functions). Locked the v23 glossary into `CONTEXT.md`. Pinned the master plan path.
- `ea94635` — **CONTEXT.md: v23-MCP drift fixes per validator P-8.** Walked back ambiguous language drift around "detected underlayers" / "recovered hidden block" claims. Validator-flagged corrections folded into the glossary's banned-terms list.
- `94f571a` — **v23-MCP D1 scaffold — 8/8 tests green (TDD).** Repo shape: `backend/mcp/__init__.py`, `backend/services/v23/__init__.py`, `backend/tests/v23/conftest.py`, CI workflow stub, banned-terms grep test. First green pytest count.
- `9aacde1` — **v23-MCP D2 Pydantic types — 25/25 v23 tests green (TDD).** `backend/services/v23/types.py` with Block / Impression / Mask / Pigment / PullGroup / Plan + `mcp/errors.py` ToolResult envelope with 4-tier WoodblockError severity. `Plan` rejects `mode` field (addendum-v3 fix 3 — `solve_profile` only).
- `94ba8a0` — **v23-MCP D3.1+D3.2 filesystem-state — 9/9 green.** ULID generator (26-char lexsortable), `backend/mcp/paths.py` with `WB_DATA_DIR` resolution + `..` rejection, session_dir + plan_dir computations.
- `e251d71` — **CONTEXT.md v23 — add Overprint/Mixing/Glazing/Render tier (addendum-v4).** Locked the overlay-not-mixing physics framing into the glossary; added flagged ambiguity entry for the "Mixbox predicts the print" claim. Sets up ADR-0002 + WB-LANG-02 lint.

### Locked

- Glossary terms: Block, Impression, Mask, Pigment, Order, Underprint, Stack, Plan, Pull group, Strategy template, Solve profile, Overprint, Mixing, Glazing, Render tier.
- Posture: "plausible underprint candidates that reduce reconstruction error under this pigment/printing model. Never recovered true underlayers."
- 11-tool day-1 surface (ADR-0004).
- 3-tier render hierarchy with T1 ship gate (ADR-0002).
- Topology rules 6+7 post-solve only (ADR-0005).
- stdio-over-SSH transport (ADR-0003).
- MCP-first product framing (ADR-0001).

### Deferred

- v23.1: HITL tools (pin/merge/split/adjust), calibration upload (`capture_swatch`, `fit_pigments`, `apply_calibration`), T2 empirical 2-layer LUT render, session juggling, minimal viewer at `/v23/review/{plan_id}`.
- v24: T3 K-M two-flux recursion with 8λ (K, S) per-pigment fit, CHROMA validation oracle.
- v23.x: MCP-over-SSE/HTTP transport upgrade for multi-client (Mac + Linux on one GPU).
