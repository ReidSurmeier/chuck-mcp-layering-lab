# Contributing to woodblock_stack (v23-MCP)

## TDD discipline (non-negotiable)

Every code change goes through `red → green → refactor`. The build sequence at `/tmp/research-v23-mcp-build-sequence.md` enumerates every step as `D<N>.<n>`; each step pairs a test file with the code file it justifies.

1. **Red.** Write the test first under `backend/tests/v23/<ring>/`. Run `pytest -k <expr> -x` and confirm it fails for the right reason.
2. **Green.** Write the minimum code under `backend/{mcp,services/v23}/` to make the test pass. Do not write a second test until the first is green.
3. **Refactor.** Tighten the implementation. Tests still pass. LOC count tracked.

If you cannot articulate the test, you do not yet understand the change. Stop and re-read the plan section in `/mnt/c/Users/reidsurmeier2/Books/printmaking/v23/research-v23-mcp-plan-v2.1.md`.

## File size limit — 400 LOC cap

Per CLAUDE.md anti-patterns. Specialist agents hit context exhaustion above 400 LOC. If a stage target exceeds 400 LOC (e.g. S5 solver at 380), split before writing. The plan's per-stage budgets in §6 are upper bounds, not goals.

When a file approaches 350 LOC, open an issue with the `build-step` template proposing the split.

## Branch naming

```
pipeline/v23-D<N>-<slug>
```

Examples:
- `pipeline/v23-D1-scaffold`
- `pipeline/v23-D5-sam-gateway`
- `pipeline/v23-D10-topology-repair`
- `pipeline/v23-repo-hygiene` (non-numbered hygiene branches use a descriptive slug)

One branch per `D<N>` day. Merge to `main` once Senior Review passes (Stage 3 of the 7-stage pipeline).

## Commit message format

```
v23-MCP D<N>.<n> <slug> — <green>/<total> green
```

Examples:
- `v23-MCP D1.1 scaffold — 8/8 green`
- `v23-MCP D2.5 plan-rejects-mode — 25/25 v23 tests green (TDD)`
- `v23-MCP D5.3 real-sam-roundtrip — 4/4 green`

The trailing `<count>/<count> green` is a **load-bearing claim** — the test count must match `pytest backend/tests/v23/ -q | tail -1` at HEAD. Senior Review verifies.

For non-step hygiene work, drop the `D<N>.<n>` segment:
- `v23-MCP repo hygiene — GitHub templates + 5 ADRs`
- `CONTEXT.md v23 — add Overprint/Mixing/Glazing/Render tier (addendum-v4)`

## Banned terms (CI-enforced)

`WB-LANG-01` — domain language hygiene (CONTEXT.md ban list):

```
\bplate\b
\bseparator\b
\blayer\b                # use Impression for the pass, Mask for the region
\bunderbase\b
\bunderlayer\b
detect.{0,5}underlayer
recover.{0,5}underprint
true.{0,5}hidden.{0,5}block
\bpass\b                 # ambiguous; use Impression
```

`WB-LANG-02` — overlay-not-mixing lock (addendum-v4):

```
mixbox.*predicts.*the.{0,5}print
blend.*used.*for.*overprint
pre-mix.*synonym.*for.*overprint
```

Plus a **positive lint**: any string presented to the artist that mentions Mixbox must contain `mixing` or `pre-mixed` in the same paragraph so the qualifier rides along.

Both rules run in `.github/workflows/v23.yml` and `backend/tests/v23/scaffold/test_language_lint.py`. Grep must return zero matches before push.

## Pre-commit hook requirements

Local pre-commit must:

1. `ruff check backend/` — zero warnings on `E F I UP B`
2. `mypy backend/mcp backend/services/v23` — zero errors
3. `pytest backend/tests/v23/ -q` — all green (or test was just added; document the red in the PR)
4. `grep -E -i -f scripts/banned_terms.txt backend/` — zero matches
5. Confirm branch name matches `^pipeline/v23-(D\d+-|repo-)[a-z0-9-]+$`

Install with `pre-commit install` once `scripts/banned_terms.txt` lands (D1.4).

## Build sequence reference

Every step ID (`D1.1`, `D6.5`, `D10.3`, ...) is defined in `/tmp/research-v23-mcp-build-sequence.md`. Open that file, find your row, copy the `TDD test` path verbatim into your PR description so reviewers can re-run it.

The realistic milestone framing is in addendum-v3 fix 6: D23 = MCP up + 3 corpus golden, D30-35 = Tier-1 5/5, D45 = buffer, D60 = full 33-tool surface.

## Schema changes

The manifest schema (`schema_version: "v23.0"`) is **additive-only** until v23 ships. If your change requires a breaking field rename or removal, the PR must include an ADR under `docs/adr/` justifying the bump to `v23.1`.

## ADR requirement

Add an ADR when **all three** are true (per `~/.claude/skills/grill-with-docs/ADR-FORMAT.md`):

1. Hard to reverse
2. Surprising without context
3. The result of a real trade-off (alternatives existed)

Number sequentially: scan `docs/adr/` for the highest existing index and add one. Keep ADRs 30–60 lines. Name the rejected alternative, name the trade-off, cite the user quote when applicable.

## PR template

Every PR must fill the template at `.github/PULL_REQUEST_TEMPLATE.md`. The TDD evidence, build-sequence step ID, CONTEXT.md compliance check, banned-terms grep result, manifest schema-unchanged confirmation, and rollback tag are all required.
