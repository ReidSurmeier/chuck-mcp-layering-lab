# 0001 — MCP-first architecture (v22 web-app → v23 MCP server pivot)

Status: accepted (2026-05-11)

v22 was framed as a public-facing web app: user uploads an image at `tools.reidsurmeier.wtf`, watches a progress bar, downloads a ZIP. v22's plan-of-record (`/mnt/c/Users/reidsurmeier2/Books/printmaking/v22/`) committed to a FastAPI surface, an SSE progress stream, a Next.js wizard, and a registered-user state model. v22 stalled because the *judgment* in mokuhanga plan synthesis — pick subject, pick template, read a 5-component score breakdown, decide whether to re-solve — is not a UI concern. It is a conversation. A web wizard reified the conversation into buttons and immediately leaked back into ambiguity (the validator's P-8 drift fixes were almost entirely UI copy walking back into "we detect underlayers" language).

v23 pivots to MCP-first. The user's verbatim ask:

> "build an MCP-first research/production assistant"

Opus 4.7 owns the conversation, the semantic judgments (subject classification, template choice, recipe paraphrasing per addendum-v3 fix 5), and the artist-facing voice. The MCP server owns the measurable computation (SAM regions, OKLab clusters, JAX inverse solver, mask topology, SVG export). The boundary is enforced: `analyze_image` returns `Mpx | dpi | hue_clusters | dominant_colors | region_geometry | est_complexity` and refuses to return `subject_label`. The model reads the image preview and decides.

## Alternative considered

**Keep v22's FastAPI + Next.js shape, fix the language drift with stricter copy review.** Rejected because the drift was structural, not editorial: a wizard with a "Detect underlayers" button cannot be honest no matter how the copy reads. The button itself is the lie. v22's `routes/mokuhanga.py` (149 LOC) is dropped wholesale per `research-v23-mcp-reuse.md` §1.

**Hybrid: ship MCP server AND a thin v22-compatible web wizard.** Rejected on bandwidth + boundary clarity. v20's public web demo at `color.reidsurmeier.wtf` already serves the "no underprint solver, just color sep" use case. v23 the assistant ≠ v20 the demo.

## Trade-off accepted

- **Gained:** honest language by construction, single-user single-GPU semaphore is trivially correct, no FastAPI / no SSE plumbing / no auth surface, faster ship (no Next.js wizard to build), the conversation IS the UI.
- **Lost:** no public web demo for v23 (v20 fills that gap), no anonymous user path, requires Claude Code as the client. Multi-client (Linux + Mac simultaneous on one GPU) deferred to v23.x per `research-v23-mcp-protocol.md` §2.

## Consequence — opt-in viewer is read-only

`/v23/review/{plan_id}` ships as a Next.js viewer ≤ 800 LOC, read-only, no upload form, no detection button. It reads `~/.woodblock/v23/plans/<plan_id>/manifest.json` and renders it. The viewer never initiates a solve. The MCP server is the only thing that creates plans.
