# Frontend Inventory — color-separator (source for colorv2 fork)

Date: 2026-05-16
Researcher: FRONTEND-CLONE
Source repo: `/home/reidsurmeier/src/chuck-mcp-layering-lab/` (Next.js root + colocated FastAPI backend)
Live URL of source: `color.reidsurmeier.wtf`
Target URL after fork: `colorv2.reidsurmeier.wtf` (also written as `chuck.reidsurmeier.wtf` in design doc — locked spec uses `chuck.` for the MCP-level URL but task spec uses `colorv2.`; aliasing decision pending — see NOTES.md)

## Source repo identification

There is **no separate** `color-separator` repo on Reid's machine. The frontend lives directly inside `chuck-mcp-layering-lab` and was lifted from an earlier color-separator project. Evidence:

- `/home/reidsurmeier/src/chuck-mcp-layering-lab/package.json` — `"name": "color-separator"`, `"homepage": "https://tools.reidsurmeier.wtf/color-separator"`, repo URL `ReidSurmeier/color-separator`
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/src/middleware.ts` — host-based rewrite from `color.*` → `/color-separator`
- `/home/reidsurmeier/src/chuck-mcp-layering-lab/color-separator/` — a leftover sibling folder with duplicate-but-older `components/`, `hooks/`, `layout.tsx`, `page.tsx`. DO NOT use as the fork source — it's stale (`page.tsx` is 10613 bytes, the active one is at `src/app/color-separator/page.tsx` and has been migrated into the `src/` layout).

**Canonical frontend root**: `/home/reidsurmeier/src/chuck-mcp-layering-lab/src/`

## Top-level repo layout

| Path | Purpose |
|---|---|
| `src/app/` | Next.js 16 App Router pages |
| `src/lib/` | Frontend shared modules (api client, types, colors, cnc-engine) |
| `src/middleware.ts` | host-based subdomain routing |
| `public/` | Static assets — fonts, images, screenshots, videos, seo |
| `backend/` | FastAPI service (port 8001) — to be REPLACED by chuck-mcp in v2 |
| `next.config.ts` | proxy rewrite `/api/*` → `${BACKEND_URL}` (default `http://localhost:8001`), CSP, Sentry |
| `package.json` | Next 16.2.1, React 19.2.4, Tailwind 4, base-ui, framer-motion, jszip, makerjs, clipper2-ts |

## Pages (app router)

| Route | File | Lines | Purpose |
|---|---|---|---|
| `/` | `src/app/page.tsx` | 5 | `redirect("/color-separator")` |
| `/color-separator` | `src/app/color-separator/page.tsx` | 273 | Main UI — the entire color-sep app |
| `/cnc` | `src/app/cnc/page.tsx` | 98 | Post-processing for CNC-ready SVGs (kento, sheet layout, EPS/DXF export) |
| `/about` | `src/app/about/page.tsx` | 172 | Static info page |
| `/color-separator/layout.tsx` | layout | 20 | Sub-layout, `force-dynamic` + `force-no-store` |
| `/cnc/cnc.css` | route-scoped css | n/a | CNC-specific styles |
| `src/app/globals.css` | global css | 1238 | Design system — typography, sliders, plate cards, **plate loading animation**, progress bar, layout grammar |
| `src/app/garden.css` | global css | 560 | Garden / decorative styles |

## API routes (Next.js proxy layer in front of FastAPI backend)

| Route | File | Backend target | Notes |
|---|---|---|---|
| `POST /api/preview` | `src/app/api/preview/route.ts` | `${BACKEND_URL}/api/preview` | Validates body (10MB cap, mime allow-list), forwards with `X-API-Key` |
| `POST /api/preview-stream` | `src/app/api/preview-stream/route.ts` | `${BACKEND_URL}/api/preview-stream` | SSE passthrough — manual stream pipe to avoid undici close errors |
| `POST /api/separate` | `src/app/api/separate/route.ts` | `${BACKEND_URL}/api/separate` | Returns ZIP blob |
| `GET /api/result/[resultId]` | `src/app/api/result/[resultId]/route.ts` | `${BACKEND_URL}/api/result/{id}` | Fetches stored composite PNG, sanitizes ID |

Note: the rest of the API surface (`/api/plates-stream`, `/api/plates-svg`, `/api/merge`, `/api/upscale`, `/api/health`, `/api/auto-optimize`, `/api/job/*`, `/api/analytics/*`) is proxied at the Next config level via `rewrites()` in `next.config.ts` — no Next route handler, direct passthrough to backend. The four routes that DO have handlers are the ones needing extra logic (auth, validation, blob shaping, streaming workaround).

## Components (color-separator scope)

| File | Lines | Role |
|---|---|---|
| `src/app/color-separator/components/NavPanel.tsx` | 649 | Left sidebar — version select, file picker, palette swatches, all parameter sliders, process/reset/download buttons, merge UI, error toast, image info display |
| `src/app/color-separator/components/PlatesGrid.tsx` | 129 | Plate card grid with merge selection, **skeleton placeholders during streaming load** |
| `src/app/color-separator/components/PlateZoom.tsx` | 45 | Modal overlay zooming a single plate image |
| `src/app/color-separator/components/ProgressBar.tsx` | 163 | Top-of-screen progress bar — stage label, %, ETA, elapsed, plate counter, stall detection |
| `src/app/color-separator/constants.ts` | 26 | Types (`VersionId`, `PaletteColor`, `PlateImage`, `AppError`) + version list |
| `src/app/color-separator/hooks/useColorSeparator.ts` | 1129 | Single mega-hook owning ALL state, all API calls, file handling, debounced re-runs, merge logic, client-side ZIP build, diagram PNG generation |

The CNC page has its own parallel set (`src/app/cnc/components/{CncNavPanel,PlatePreview}.tsx` + `src/app/cnc/hooks/useCncProcessor.ts`) — out of scope for colorv2 fork.

## Plate loading animation — full mapping

The "plate loading animation" Reid wants preserved verbatim is composed of FIVE coordinated pieces:

1. **Skeleton cards in the grid** (`PlatesGrid.tsx` lines 104–123) render greyed placeholder `<div class="plate-skeleton">` cards for every plate still loading, calculated from `platesTotalCount - plateImages.length`.

2. **Skeleton pulse animation** (`globals.css` lines 1127–1188) — `@keyframes skeleton-pulse` and `@keyframes skeletonPulse` (two slight variants), 1.2–1.5s ease-in-out infinite, opacity 0.3↔0.9. Targets `.plate-skeleton-img`, `.plate-skeleton-swatch`, `.plate-skeleton-text`.

3. **Per-plate appear animation** (`globals.css` lines 1163–1175) — `@keyframes plateAppear` slides each new card in from `translateY(8px)` with a staggered `animationDelay: ${i * 60}ms` set inline in `PlatesGrid.tsx` line 72. This produces the cascading reveal as plates stream in.

4. **Top progress bar** (`ProgressBar.tsx` + `globals.css` lines 1114–1125) — `@keyframes indeterminate` sliding 20%-width fill, plus `plates-loading-slide` for the small indeterminate strip variant (lines 481–488). Two indeterminate-bar implementations coexist.

5. **Main canvas dim** (`globals.css` line 1026 `.main-canvas.is-loading`) — `is-loading` class set in `page.tsx` line 229 fades the entire canvas while processing.

**Dependencies for the animation**:
- CSS only — no JS animation lib for the plate loading itself
- `framer-motion@12.38.0` is installed but used elsewhere (not in the plate grid load path)
- Inline style on each plate card supplies the `animationDelay: i * 60ms` stagger — see `PlatesGrid.tsx` line 72

All five pieces must be ported verbatim to preserve the look. They depend on:
- `globals.css` (full file)
- `PlatesGrid.tsx` (the skeleton render block + animationDelay inline style)
- `ProgressBar.tsx`
- the SSE stream driving incremental plate arrival (`fetchPlatesStream` in `src/lib/api.ts`)

If colorv2 drops SSE streaming, the cascading-appear effect disappears with it. **Recommendation: keep streaming as the v2 mental model**, even though chuck-mcp's primary lifecycle is plan-then-render rather than plate-by-plate.

## API surface called by the existing frontend

From `src/lib/api.ts` (745 lines), all routed through `${BACKEND_URL}` (defaults to `""` so they hit the Next.js proxy → FastAPI):

| Function | Endpoint | Method | Purpose |
|---|---|---|---|
| `checkHealth()` | `/api/health` | GET | Backend health, RAM/SAM cache status |
| `fetchPreview()` | `/api/preview` | POST FormData | Sync preview (composite blob + manifest header) |
| `fetchPreviewStream()` | `/api/preview-stream` | POST FormData → SSE | Streaming preview with per-stage progress |
| `fetchSeparation()` | `/api/separate` | POST FormData | Bulk separation, returns ZIP |
| `fetchAutoOptimize()` | `/api/auto-optimize` | POST → SSE | Iterative LLM-driven param tuning |
| `fetchUpscale()` | `/api/upscale` | POST FormData | Pre-upscale image, cache hash |
| `fetchMerge()` | `/api/merge` | POST FormData | Merge plate pairs by centroid index |
| `fetchPlatesStream()` | `/api/plates-stream` | POST FormData → SSE | Stream individual plates as base64 PNG events |
| `fetchPlatesSvg()` | `/api/plates-svg` + `/api/job/{job_id}` | POST + poll | Async high-res SVG/PNG generation |
| `trackEvent()` | `/api/analytics/event` | POST JSON | Fire-and-forget analytics |

Plus backend MCP surface (`backend/mcp/registry.py` exposes tools `core`, `planning`, `calibration`, `hitl`, `introspection`, `overlay`, `session`, `carve`) — not yet consumed by frontend.

## Design tokens / styles

- **Fonts** (in `public/fonts/`): `AUTHENTICSans-90.woff`, `AUTHENTICSans-Condensed-90.woff`, `DepartureMono-Regular.woff/.woff2`
- **Color palette**: black text on white, `#14ff00` highlight-green (active button, plate-selected, plates-loading-fill), reds `#e05c5c` (merge-selected outline). All defined inline in CSS — no token system.
- **Typography**: AUTHENTICSans-90 body, monospace for hex values + data rows
- **Tailwind**: v4 with `@tailwindcss/postcss`, but the actual styles are 1800 lines of plain CSS in `globals.css` + `garden.css`. Tailwind is set up but barely used.
- **Sliders**: hand-styled `input[type="range"]` with 6px square thumbs (globals.css lines 55–90)
- **Buttons**: flat grey (#ddd), green when `data-active="true"`, no border-radius (lines 92–119)
- **Layout grammar**: fixed-left nav-panel ~280px, full-screen main-canvas with composite image + plates grid below

## What lives in `src/lib/`

| File | Purpose |
|---|---|
| `api.ts` | All API calls, retry/timeout policies, request dedup, error class |
| `types.ts` | `PlateInfo`, `Manifest`, `SeparationParams`, `PreviewResult`, `OptimizeIteration` |
| `colors.ts` | `rgbToHex`, `hexToRgb`, `plateFilenameStem`, `safeZipFolderName` |
| `cnc-types.ts`, `cnc-engine.ts`, `cnc-export.ts` | CNC-page logic (out of scope for colorv2 fork) |
| `utils.ts` | Tailwind `cn()` helper |

## Routes that depend on host-based routing

`src/middleware.ts` rewrites `host=color.*` + `path=/` to `/color-separator`. This means **the same Next.js deployment serves multiple subdomains** via host header. When colorv2 lands, we add a parallel rewrite: `host=colorv2.*` → `/colorv2` (or whatever the new page path is).

## Bundled but underused dependencies

`makerjs@0.19.2`, `clipper2-ts@2.0.1-15`, `dompurify@3.3.3` are used only on the `/cnc` page. `@sentry/nextjs` is wired in `next.config.ts` but is silent (`silent: true`). `shadcn@4.1.0` is installed but I see no `components/ui/*` shadcn-generated files in `src/`. `class-variance-authority` and `tw-animate-css` are similarly unused in the color-separator subtree.
