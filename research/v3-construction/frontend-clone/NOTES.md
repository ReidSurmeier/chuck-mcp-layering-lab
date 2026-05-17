# NOTES — Frontend Clone Research Synthesis

Date: 2026-05-16
Researcher: FRONTEND-CLONE
Swarm: chuck-mcp v3 construction (swarm-1778978903817-bqgh16)

## Source repo identified

`/home/reidsurmeier/src/chuck-mcp-layering-lab/` itself is the active color-separator codebase. There's no separate repo. The Next.js root at the chuck-mcp-layering-lab folder ships `color.reidsurmeier.wtf` today via the `src/app/color-separator/` route, host-routed in `src/middleware.ts`. A stale, older copy lives at `chuck-mcp-layering-lab/color-separator/` (sibling folder, not connected to the running app) — ignore it.

## Recommended clone strategy: in-place fork

Add `src/app/colorv2/` parallel to `src/app/color-separator/`, share the same Next deploy and host-based middleware, point the new pages at a separate chuck-mcp Python backend over a second proxy `/api/chuck/*` → `${CHUCK_BACKEND_URL}`. Leave the color-separator app running in parallel for the entire v2 build period; flip Cloudflare DNS to colorv2 only after sign-off.

## Top-3 component reuses (verbatim)

1. **The plate loading animation stack** — 5 coordinated pieces in `globals.css` + `PlatesGrid.tsx`: `@keyframes skeleton-pulse`, `@keyframes plateAppear`, the inline `animationDelay: i * 60ms` stagger on each card, the indeterminate progress bar, the `.main-canvas.is-loading` dim. All five are CSS-only (no JS lib), all five must move together. Used in v2 for the 27-block grid streaming reveal.
2. **`ProgressBar.tsx`** — 163 lines, already handles stage labels, ETA, stall detection (3s pct-no-change → indeterminate fallback), elapsed time. Drop-in for solver progress; only the `STAGE_LABELS` map needs to change.
3. **The SSE streaming infrastructure** — `src/lib/api.ts` `fetchPreviewStream` (lines 624–745) + `src/app/api/preview-stream/route.ts` (78 lines, the manual stream-pipe trick that defeats undici's "other side closed" error). This is battle-hardened against Cloudflare + standalone Next + 30s heartbeats. Rebuild the chuck-mcp progress endpoint on the same pattern; both the proxy route and the client parser are mostly reusable.

Honorable mentions: `PlateZoom.tsx` (45 lines), `colors.ts` helpers, the entire `globals.css` design system, the `ApiError` + `fetchWithRetry` skeleton.

## Page count

| Page | Status |
|---|---|
| `/colorv2/upload` | NEW — image picker + intent prompt |
| `/colorv2/preview/[plan_id]` | NEW — composite + 27-block grid + scrubber + sidebar |
| `/colorv2/proof/[plan_id]` | NEW — 7-checkpoint vs Hokusai/Close + heatmap + 6 validators |
| `/colorv2/sign-off/[plan_id]` | NEW — export-carving-files trigger + folder tree |
| `/colorv2/layout.tsx` | NEW (small) — copy of color-separator layout |
| `/colorv2/components/IntentSidebar.tsx` | NEW (medium) — rewrite of NavPanel |
| `/colorv2/components/BlocksGrid.tsx` | NEW (small) — adaptation of PlatesGrid |
| `/colorv2/components/BlockZoom.tsx` | NEW (small) — adaptation of PlateZoom |
| `/colorv2/components/SolverProgress.tsx` | NEW (small) — adaptation of ProgressBar |
| `/colorv2/components/PullScrubber.tsx` | NEW (medium) — no antecedent in color-separator |
| `/colorv2/components/ProofGrid.tsx` | NEW (medium) — 7-checkpoint contact sheet renderer |
| `/colorv2/components/HeatmapOverlay.tsx` | NEW (small) — d3-scale on PNG canvas |
| `/colorv2/components/ValidatorBadges.tsx` | NEW (small) — 6-row pass/fail |
| `/colorv2/hooks/useChuckPlan.ts` | NEW (large) — replaces useColorSeparator |

Plus backend wiring:
- `/colorv2/api/chuck/plan/route.ts` (POST)
- `/colorv2/api/chuck/plan/[plan_id]/route.ts` (GET)
- `/colorv2/api/chuck/plan/[plan_id]/progress/route.ts` (SSE)
- `/colorv2/api/chuck/plan/[plan_id]/pull/[pull_id]/render/route.ts` (GET)
- `/colorv2/api/chuck/plan/[plan_id]/pull/[pull_id]/heatmap/route.ts` (GET)
- `/colorv2/api/chuck/plan/[plan_id]/block/[block_id]/svg/route.ts` (GET)
- `/colorv2/api/chuck/plan/[plan_id]/proofs/route.ts` (GET)
- `/colorv2/api/chuck/plan/[plan_id]/image/[kind]/route.ts` (GET, kind ∈ {composite, target})
- `/colorv2/api/chuck/regenerate-blocks/route.ts` (POST)
- `/colorv2/api/chuck/export-carving/route.ts` (POST)
- `/colorv2/api/chuck/interpret/route.ts` (POST)

Total new pages: 4 routes. Total new files: ~25. Total new lines (estimate, not counting copied CSS or copied skeletons): ~1800.

## Estimated frontend dev hours

Based on the day-by-day plan in `CLONE_STRATEGY.md` and Reid's vibe-coding velocity (1 specialist agent at Opus 4.6, parallelized where possible per disjoint-files rule):

| Phase | Work | Hours |
|---|---|---|
| Day 1 | Scaffold subtree + middleware + env vars + hello-world deploy | 2 |
| Day 2 | `/upload` page + form + LLM interpret preview + redirect | 3 |
| Day 3 | `useChuckPlan` hook + SSE wiring + `/preview` shell | 5 |
| Day 4 | `BlocksGrid` + `BlockZoom` + `IntentSidebar` + composite display + pull scrubber | 6 |
| Day 5 | `/proof` page (proof grid + heatmap + validator badges) + `/sign-off` page (export trigger + folder tree) | 5 |
| Day 6 | Error states + mobile responsive + Sentry + analytics events | 3 |
| Day 7 | End-to-end against real chuck-mcp backend + bug fixes | 4 |
| Buffer | Cloudflare cache surprises, standalone-build copy bugs, deployment friction | 4 |

**Total: ~32 hours of frontend specialist work.** Sequence-bound at 7 days even if parallel agents help, because each day depends on the previous (hook → page → component → integration). Compresses to 5 days if backend stubs land early.

This estimate assumes:
- chuck-mcp backend is feature-complete enough to serve test data by Day 3
- No design changes mid-build
- The plate loading animation works on the new SSE event cadence without re-engineering
- The 7 reference proof images are sourced and placed in `public/seo/reference/` before Day 5

## Open questions for orchestrator

1. **Subdomain name conflict**: design doc says `chuck.reidsurmeier.wtf`; task spec says `colorv2.reidsurmeier.wtf`. Pick one and update both docs. Suggest `colorv2.` to match Reid's existing naming pattern, with `chuck.` as a CNAME alias.
2. **Should the color-separator app continue serving in parallel during v2 build?** Default yes — same Next.js deploy, same `tools.reidsurmeier.wtf` umbrella, no breakage risk for existing users.
3. **Photoshop integration escape hatch** (per design doc deferred-to-V2 list, also referenced in `regenerate_blocks` tool signature). V1 frontend should leave the upload slot for `photoshop_mask` as a hidden field — wire later without re-shipping.
4. **Iteration anchor management**: `previous_plan.json` symlink lives server-side. Does frontend need to display "iter 3 of 5" with a chain of past plan_ids? Yes per design doc — wire a small "iterations" widget in the sidebar from Day 4 onward.
5. **The two indeterminate progress bar implementations** (`@keyframes indeterminate` lines 1114 and `@keyframes plates-loading-slide` lines 481) — both are live in globals.css. Audit which one the new pages actually use to avoid CSS bloat in the dedicated colorv2 build.

## Constraint compliance check

- All paths absolute  ✓
- Output stayed inside `research/v3-construction/frontend-clone/`  ✓
- No code cloned or modified — spec only  ✓
- Top-3 reuses reported above  ✓
- Estimated hours reported above (~32h)  ✓
- Source repo path: `/home/reidsurmeier/src/chuck-mcp-layering-lab/`  ✓

## Files produced

- `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/frontend-clone/INVENTORY.md` — full inventory of existing pages, components, API surface, design tokens
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/frontend-clone/CLONE_STRATEGY.md` — what to keep, what to rewire, file-by-file mapping, next.config.ts changes, SSE/WebSocket decision, day-by-day phasing, risk register
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/frontend-clone/package_diff.md` — 5 adds, 4 cleanup candidates (deferred), bundle impact
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/frontend-clone/new_pages_spec.md` — wireframe + API contract + user actions for all 4 new pages, plus shared layout and route summary
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/research/v3-construction/frontend-clone/NOTES.md` — this file
