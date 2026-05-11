# Architecture index

The binding plan + addendums for woodblock_stack (v23-MCP) live on the SMB share at:

```
/mnt/c/Users/reidsurmeier2/Books/printmaking/v23/
```

This directory exists outside the git repo intentionally — the plan is research notes, not source. Hashes + dates below pin the version this scaffold was built against. Re-hash with `sha256sum <file>` after any edit and update this index.

## Master plan + binding addendums

| File | Role |
|---|---|
| `research-v23-mcp-plan-v2.1.md` | **Master plan v2.1** — supersedes v1/v2 on conflict. §7 math, §10 33-tool surface, §11 transport, §6 stage S1–S10, §9 Emma priors, §9.5 underprint rules, §10.6 print recipe format. |
| `research-v23-mcp-user-addendum-v4.md` | **Overlay-not-mixing lock.** 3-tier render hierarchy, WB-LANG-02 lint, `get_render_tier` tool. See ADR-0002. |
| `research-v23-mcp-user-addendum-v3.md` | **5 concrete fixes to v2.** Topology out of optimizer (fix 1, ADR-0005), 11-tool day-1 cap (fix 2, ADR-0004), `mode` → `solve_profile` (fix 3), 5-component score (fix 4), measurable backend boundary (fix 5), timeline framing (fix 6). |
| `research-v23-mcp-user-addendum-v2.md` | v2 architectural refinement (mostly subsumed by v3+v4). 7 underprint rules. |
| `research-v23-mcp-user-addendum.md` | v1 addendum — posture lock, verifier-not-generator framing. |

## Supporting specialist briefs

| File | Role |
|---|---|
| `research-v23-mcp-protocol.md` | Transport rationale (stdio over SSH, ADR-0003). FastMCP 2.x, GPU semaphore, lifespan. |
| `research-v23-mcp-reuse.md` | Wave A salvage map — what carries from v20/v22 into v23 (~500 LOC SVG postprocess, ~80 LOC SAM, full corpus). |
| `research-v23-mcp-state.md` | Filesystem state layer: ULID, `~/.woodblock/v23/`, session lifecycle, 30 d / 50-session LRU. |
| `research-v23-mcp-tools.md` | Full 33-tool catalog with 7-section UX docstrings. |
| `research-v23-mcp-interfaces.md` | Pydantic v2 contracts (Block / Impression / Mask / Pigment / Plan / ToolResult / WoodblockError). |
| `research-v23-mcp-testing.md` | 4-ring test taxonomy (scaffold / unit / stages / transport / conversation). |
| `research-v23-mcp-edges.md` | Failure-mode catalog: timeouts, GPU contention, graceful cascade. |
| `research-v23-mcp-build-sequence.md` | D1–D23 aspirational TDD step list (lives at `/tmp/`, not SMB). Source of truth for `D<N>.<n>` IDs. |
| `research-v23-mcp-calibration.md` | Optional swatch upload path (T2 unlock). |
| `research-v23-mcp-corrections.md` | Validator B-N blocker catalog. |
| `research-v23-mcp-repo-layout.md` | Module tree, package layout, import boundaries. |
| `research-v23-mcp-ux.md` | Opus system prompt + tool docstring style guide. |
| `research-v23-overlap-math.md` | Forward-render math options (superseded on tier-1 choice by addendum-v4). |
| `research-v23-pipeline.md` | S1–S10 stage rationale. |
| `research-v23-defaults.md` | Defaults catalog: ΔE 1.5/3.0, solve_profile budgets, M_prior. |
| `research-v23-emma-priors.md` | 6-family OKLab anchors, accent rule, keyblock rule. |
| `research-v23-data-model.md` | Pydantic schemas in full. |
| `research-v23-history.md` | v0–v22 retrospective. |
| `research-v23-mask-topology.md` | 3-state mask classifier rationale (visible / covered / support / none). |
| `CONTEXT.md` | (copy on SMB; canonical in repo root) glossary. |

## Pin

| Field | Value |
|---|---|
| Plan version | v2.1 |
| Addendum chain | v1 → v2 → v3 → v4 (v4 latest binding) |
| Scaffold built against | 2026-05-11 |
| Scaffold commit | `pipeline/v23-repo-hygiene` HEAD |

To verify the local SMB copy matches what this scaffold was built against, run:

```bash
ls -la /mnt/c/Users/reidsurmeier2/Books/printmaking/v23/research-v23-mcp-plan-v2.1.md
ls -la /mnt/c/Users/reidsurmeier2/Books/printmaking/v23/research-v23-mcp-user-addendum-v4.md
```

`mtime` should be 2026-05-11 or earlier. If newer, re-read both files and check whether any decision in `docs/adr/` is invalidated before continuing the build sequence.

## Quick navigation

- New to the repo? Read `CONTEXT.md` first (glossary), then this file (architecture pointer), then `docs/adr/0001..0005` in order.
- Starting a build step? Find the row in `/tmp/research-v23-mcp-build-sequence.md`, then read the matching plan section.
- Debugging a tool? Read `research-v23-mcp-tools.md` for the 7-section docstring template + `research-v23-mcp-edges.md` for the failure mode.
