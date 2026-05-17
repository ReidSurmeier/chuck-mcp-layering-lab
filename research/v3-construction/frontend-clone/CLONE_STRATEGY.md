# Clone Strategy — colorv2.reidsurmeier.wtf

Date: 2026-05-16
Researcher: FRONTEND-CLONE
Target build window: Week 4 of v3 construction (per `docs/v2-design-locked-2026-05-16.md`)
Posture: research/spec only. NO code is created or modified by this document.

## TL;DR

Fork the existing color-separator frontend **in place** inside `chuck-mcp-layering-lab` by adding a new app subtree `src/app/colorv2/` (and supporting `src/lib/chuck-api.ts` + new API proxy routes under `src/app/api/chuck/`). Keep `src/app/color-separator/` intact and live until v2 is signed off. Add a host-based middleware branch routing `colorv2.*` to `/colorv2`. Reuse the same component library, plate loading animation, design tokens, and SSE streaming patterns verbatim — only the API client and the page composition change.

## Where to fork from

**Repo**: `/home/reidsurmeier/src/chuck-mcp-layering-lab/` (do NOT create a separate repo).
**Source subtree**: `src/app/color-separator/` + `src/lib/api.ts` + `src/lib/types.ts` + `src/lib/colors.ts` + `src/app/globals.css` + `src/middleware.ts` + `next.config.ts` + `src/app/api/*`.
**Stale folder to ignore**: `/home/reidsurmeier/src/chuck-mcp-layering-lab/color-separator/` (the top-level sibling) — older snapshot, not connected to the live deploy.

Rationale: a sibling app inside the same Next.js deployment shares fonts, base CSS, build cache, deploy pipeline, and the existing Cloudflare tunnel. A new repo would force duplicating all five and add ~6 hours of setup that buys nothing.

## What to keep verbatim (zero edits)

| Asset | Path | Why |
|---|---|---|
| Font files | `public/fonts/*.woff*` | Same visual language requirement |
| Global stylesheet | `src/app/globals.css` | Plate loading animation, sliders, buttons, layout grammar, swatches — Reid wants these identical |
| `colors.ts` helpers | `src/lib/colors.ts` | `rgbToHex`, `hexToRgb`, etc. — used by every plate card |
| `PlateZoom.tsx` | `src/app/color-separator/components/PlateZoom.tsx` | Modal pattern, no chuck-specific deps |
| `PlatesGrid.tsx` skeleton render block | `src/app/color-separator/components/PlatesGrid.tsx` lines 51–123 | THE plate loading animation — staggered appear + skeletons |
| `ProgressBar.tsx` | `src/app/color-separator/components/ProgressBar.tsx` | Stage labels need updating but structure stays |
| Sentry config | `next.config.ts` | Already silent, no-op for v2 |
| CSP headers | `next.config.ts` `headers()` | Carry forward |
| Cache-busting headers | `next.config.ts` | Same Cloudflare-purge issue |
| middleware base | `src/middleware.ts` | Extend, don't replace |

## What to rewire

| Concern | Today | After v2 |
|---|---|---|
| Backend protocol | HTTP FormData → FastAPI on `localhost:8001` | HTTP + SSE → chuck-mcp Python service (port TBD; design doc says systemd user unit `chuck-mcp.service`) |
| Endpoint surface | `/api/preview*`, `/api/plates*`, `/api/merge`, `/api/upscale`, `/api/separate`, `/api/auto-optimize`, `/api/result/{id}` | `/api/chuck/plan`, `/api/chuck/regenerate-blocks`, `/api/chuck/export-carving`, `/api/chuck/plan/{plan_id}`, `/api/chuck/plan/{plan_id}/proof/{checkpoint_id}`, `/api/chuck/plan/{plan_id}/pull/{pull_id}`, `/api/chuck/plan/{plan_id}/block/{block_id}`, `/api/chuck/plan/{plan_id}/progress` (SSE) |
| Data unit | "plate" = one color separation | "block" = physical woodblock (1–27) + "pull" = atomic print application (1–132) — different mental model |
| State container | `useColorSeparator` hook, ALL local React state | `useChuckPlan(plan_id)` hook + SWR/react-query for server state — plans are persisted by chuck-mcp |
| File flow | upload → process → display in same page | multi-step: `/upload` → POST plan → redirect `/preview/<plan_id>` |
| Output | client-side JSZip | server-side `export_carving_files()` MCP tool writes to `~/cnc-carving-jobs/` |
| Auth | `BACKEND_API_KEY` env, X-API-Key header | same pattern, env name `CHUCK_API_KEY` |
| Upscale dependency | local FastAPI `/api/upscale` | external — calls `color.reidsurmeier.wtf` over Tailscale per design doc Week 4 |

## What new pages to add

Per the locked design (workflow steps 1, 8, 9, 11) and the task spec, four new top-level routes:

1. `/colorv2/upload` — image picker + free-form intent prompt + LLM interpretation panel preview + "Generate plan" submit
2. `/colorv2/preview/[plan_id]` — composite vs target + 27-block grid + pull scrubber + sidebar (prompt, interpretation, provenance) + action bar (Adjust / Re-prompt / Single-block patch / Sign off)
3. `/colorv2/proof/[plan_id]` — 7-checkpoint proof series side-by-side vs Hokusai/Chuck Close reference + per-pull load-bearing heatmap
4. `/colorv2/sign-off/[plan_id]` — final review of 6 validator scores + "Export carving files" trigger + folder path display after export

Plus a root rewrite: `/colorv2/` → middleware sends `colorv2.*` host to one of the above (likely `/upload` if no `plan_id` cookie, else last-viewed `plan_id`).

Detailed wireframes for each in `new_pages_spec.md`.

## next.config.ts changes

```ts
// Add to existing next.config.ts (do not replace)
const chuckBackendUrl = process.env.CHUCK_BACKEND_URL ?? "http://localhost:8005";  // new port for chuck-mcp service

async rewrites() {
  return {
    afterFiles: [
      // existing
      { source: "/api/:path*", destination: `${backendUrl}/api/:path*` },
      // new
      { source: "/api/chuck/:path*", destination: `${chuckBackendUrl}/api/:path*` },
    ],
  };
}
```

And introduce env vars:
- `CHUCK_BACKEND_URL` (server-side proxy target, defaults to `http://localhost:8005`)
- `CHUCK_API_KEY` (forwarded as `X-API-Key`)
- `NEXT_PUBLIC_CHUCK_API_BASE` (set to `""` so client requests hit the Next proxy, mirroring how `NEXT_PUBLIC_BACKEND_URL` is left empty today)

## middleware.ts changes

```ts
// Extend the existing host-based router
if (hostname.startsWith("colorv2.") && request.nextUrl.pathname === "/") {
  return NextResponse.rewrite(new URL("/colorv2/upload", request.url));
}
// Existing color.* branch stays untouched
```

Matcher should expand to `["/", "/upload", "/preview/:plan_id", "/proof/:plan_id", "/sign-off/:plan_id"]` so the subdomain mapping covers deep links.

## SSE / WebSocket considerations for live solver progress

The chuck-mcp solver workflow per `docs/v2-design-locked-2026-05-16.md` is:

- Upload → upscale → LLM interpret → algorithmic baseline → solver fills blocks 5–27 → render preview
- Each stage takes seconds to minutes; the solver loop in particular runs gradient×mask + counterfactual ablation per pull (up to 132 pulls)
- User needs continuous feedback so they don't abandon the tab

**Recommendation: keep SSE, reject WebSockets.**

Reasons:
1. The existing `fetchPreviewStream` + `fetchPlatesStream` infrastructure is battle-tested through `src/app/api/preview-stream/route.ts` (handles undici `other side closed` errors via manual stream piping) and through the streaming-friendly Cloudflare config. Reusing the pattern saves debugging time.
2. The Cloudflare-proxy `X-Accel-Buffering: no` header trick is already configured.
3. The progress flow is purely one-way (server → browser). WebSockets buy nothing.
4. Per-pull events fit the existing `data: {...}\n\n` event format cleanly: emit `{"type": "pull", "pull_id": 47, "block_id": 12, "preview_b64": "..."}` per pull, terminate with `{"type": "done", "plan_id": "..."}`.

**New SSE endpoints** (chuck backend → Next proxy → browser):
- `GET /api/chuck/plan/{plan_id}/progress` (SSE) — emits stage transitions, validator partial scores, per-pull progress
- The 4 sync endpoints (`plan`, `regenerate`, `export`, `result-image`) stay request/response

**Service worker**: not needed for V1. Plans take 30s–10min on a single image; users will keep the tab focused. If we want offline preview viewing post-plan, that's V2 work. (Service workers also conflict with the `Cache-Control: no-store` strategy already in `next.config.ts`.)

**Heartbeats**: copy the 30s `still_processing` pattern from `fetchPreviewStream` lines 650–654 — Cloudflare drops idle SSE after 100s, so backend must emit at least every 30s.

## Files to copy → rename mapping

| Source | Destination | Treatment |
|---|---|---|
| `src/app/color-separator/page.tsx` | `src/app/colorv2/preview/[plan_id]/page.tsx` | Heavy edit — replace `useColorSeparator` with `useChuckPlan`, swap plate grid for block+pull surfaces |
| `src/app/color-separator/layout.tsx` | `src/app/colorv2/layout.tsx` | Copy verbatim, retitle |
| `src/app/color-separator/components/NavPanel.tsx` | `src/app/colorv2/components/IntentSidebar.tsx` | Major rewrite — params slider grid replaced by prompt textarea + interpretation panel; reuses color picker, action buttons |
| `src/app/color-separator/components/PlatesGrid.tsx` | `src/app/colorv2/components/BlocksGrid.tsx` | Light edit — 27-block grid uses same animation classes, but each card carries block_id + pull list |
| `src/app/color-separator/components/PlateZoom.tsx` | `src/app/colorv2/components/BlockZoom.tsx` | Light edit — show plate SVG (mirrored) + pull metadata |
| `src/app/color-separator/components/ProgressBar.tsx` | `src/app/colorv2/components/SolverProgress.tsx` | Edit STAGE_LABELS map for new stages: "Upscaling", "Interpreting prompt", "Cell graph (SNIC)", "Block partition (DSATUR)", "Solver pull 47/132", "Validators", "Rendering proofs", "Complete" |
| `src/app/color-separator/hooks/useColorSeparator.ts` | `src/app/colorv2/hooks/useChuckPlan.ts` | Major rewrite — see HOOK CONTRACT below |
| `src/app/color-separator/constants.ts` | `src/app/colorv2/constants.ts` | Replace `VersionId` with `PlanStatus`, add `Block`, `Pull`, `ProofCheckpoint` types |
| `src/lib/api.ts` | `src/lib/chuck-api.ts` (new) | New file — reuse `ApiError` + `fetchWithRetry` skeleton, swap endpoints |
| `src/lib/types.ts` | `src/lib/chuck-types.ts` (new) | Domain types: `Plate`, `Pull`, `ProofState` mirroring the Python dataclasses |
| `src/app/api/preview-stream/route.ts` | `src/app/api/chuck/plan/[plan_id]/progress/route.ts` | Adapt the manual-stream-pipe trick |
| `src/app/api/preview/route.ts` | `src/app/api/chuck/plan/route.ts` | Same validation skeleton; POST that returns `{plan_id, preview_url}` |
| `src/app/api/result/[resultId]/route.ts` | `src/app/api/chuck/plan/[plan_id]/image/[kind]/route.ts` | Adapt for serving composite + block previews + pull renders |

## Hook contract (`useChuckPlan`)

```ts
function useChuckPlan(plan_id?: string) {
  return {
    // identity
    planId: string | null,
    status: 'idle' | 'uploading' | 'interpreting' | 'planning' | 'solving' | 'rendering' | 'ready' | 'error',

    // user inputs
    file: File | null,
    intentPrompt: string,
    setIntentPrompt: (s: string) => void,

    // server state
    plan: ChuckPlan | null,  // { plan_id, blocks: Block[27], pulls: Pull[132], proof_states: ProofState[7], validator_scores, continuity_score, interpretation, defaults_applied, iteration_count, nag }
    interpretation: PromptInterpretation | null,  // { forbidden_pigments, region_overrides, ... } from LLM
    proofCheckpoints: ProofCheckpoint[],

    // streaming progress
    progressStage: string | null,
    progressPct: number,
    currentPull: { id: number; block_id: number; preview_url: string } | null,

    // actions
    submitPlan: () => Promise<{ plan_id: string }>,
    rePrompt: (newPrompt: string) => Promise<void>,
    regenerateBlocks: (block_ids: number[]) => Promise<void>,
    signOff: () => Promise<{ carving_job_path: string }>,

    // navigation helpers
    scrubToPull: (pull_id: number) => void,
    selectBlock: (block_id: number | null) => void,

    // errors
    error: AppError | null,
    clearError: () => void,
  };
}
```

## Migration phasing within Week 4

Per the task spec we DO NOT clone now. When Week 4 of the v3 build starts, run the work in this order:

1. **Day 1**: scaffold `src/app/colorv2/` empty pages + `src/lib/chuck-api.ts` stub + extend middleware + add env vars → deploy and verify the subdomain loads "hello world"
2. **Day 2**: build `/colorv2/upload` page (single file picker + textarea, posts to `/api/chuck/plan` and redirects to `/preview/<plan_id>`) → verify roundtrip with mocked backend
3. **Day 3**: build `useChuckPlan` hook + `/colorv2/preview/[plan_id]` shell (composite display + skeleton 27-block grid) → wire SSE progress
4. **Day 4**: build full `BlocksGrid` + `BlockZoom` + sidebar `IntentSidebar` (prompt + interpretation + action bar)
5. **Day 5**: build `/colorv2/proof/[plan_id]` (7-checkpoint comparison + heatmap) + `/colorv2/sign-off/[plan_id]` (validator scores + export trigger)
6. **Day 6**: error states, mobile responsive, Sentry wiring, analytics events
7. **Day 7**: end-to-end against real chuck-mcp backend with Emma 2002 test image, fix integration bugs

## Risk register

| Risk | Mitigation |
|---|---|
| Plate loading animation breaks because the "block" mental model fires events differently | Test SSE event cadence early; emit `block` events with the same `index`/`total` fields the existing skeleton math expects, even if we have to fake stagger for synchronous renders |
| Cloudflare timeouts on 5–10min solver runs | Same `X-Accel-Buffering: no` + 30s heartbeat pattern as preview-stream; if the solver runs 600s, may need DNS-only mode + VPS proxy (per Reid's `project_known_issues.md`) — already solved for color-separator |
| `BACKEND_API_KEY` name collision in `.env.local` | Use `CHUCK_API_KEY` namespace strictly; verify env loads correctly per the standalone build pattern (Reid's pinned gotcha) |
| Standalone build copy step missed for new pages | Reid's documented deploy pattern: `cp -r .next/static .next/standalone/.next/static && cp -r public .next/standalone/public` — no change needed, same script |
| Stale `color-separator/` sibling folder getting picked up | Add a delete-on-fork-day TODO; verify by `find` that no Next route references it |
