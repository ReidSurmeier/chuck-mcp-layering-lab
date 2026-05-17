# Package Diff — color-separator → colorv2

Source: `/home/reidsurmeier/src/chuck-mcp-layering-lab/package.json` (existing root) as the basis. The fork ships in the SAME `package.json` (no separate repo), so this diff is what to ADD or REMOVE from the existing dependency list.

## Baseline (current)

```json
"dependencies": {
  "@base-ui/react": "^1.3.0",
  "@sentry/nextjs": "^10.47.0",
  "@types/dompurify": "^3.0.5",
  "class-variance-authority": "^0.7.1",
  "clipper2-ts": "^2.0.1-15",
  "clsx": "^2.1.1",
  "dompurify": "^3.3.3",
  "framer-motion": "^12.38.0",
  "jszip": "^3.10.1",
  "lucide-react": "^1.6.0",
  "makerjs": "^0.19.2",
  "next": "16.2.1",
  "react": "19.2.4",
  "react-dom": "19.2.4",
  "shadcn": "^4.1.0",
  "tailwind-merge": "^3.5.0",
  "tw-animate-css": "^1.4.0"
}
```

## Adds (justified)

| Package | Version | Why |
|---|---|---|
| `swr` | `^2.3.0` | Server state for plan queries — replaces ad-hoc `useEffect` polling. The hook `useChuckPlan(plan_id)` benefits from SWR's stale-while-revalidate and request dedup (chuck plans are immutable per `plan_id`, perfect SWR fit). Alternative: `@tanstack/react-query` (larger, more powerful, overkill here). |
| `zod` | `^3.24.0` | Validate the chuck-mcp JSON responses on the client. The MCP backend wraps `claude -p` and the design doc explicitly mentions "3 retries on malformed output" — defending the frontend with zod parsing catches drift before it crashes the UI. |
| `react-zoom-pan-pinch` | `^3.7.0` | The 27-block grid + composite + per-pull renders all need pan/zoom on large images (Emma is 2048px upscaled to 8192px). Existing `PlateZoom.tsx` is a simple modal; v2 needs proper inspection of dense block detail. |
| `d3-scale` + `d3-array` | `^4.0.2`, `^3.2.4` | The load-bearing heatmap on `/proof/[plan_id]` overlays per-pixel ΔE on the composite. d3-scale handles the color ramp cleanly. |

Total bundle impact estimate: ~60KB gzipped (swr 8KB, zod 13KB, react-zoom-pan-pinch 9KB, d3 30KB). Acceptable on this app (already serves font files plus 1.2MB of CSS).

## Removes (cleanup candidates — defer the actual remove to a separate refactor PR)

| Package | Why it's safe to drop |
|---|---|
| `shadcn` | Installed `^4.1.0` but no `components/ui/*` files exist in `src/` to consume it. Dead dep. |
| `tw-animate-css` | Not referenced in any `*.tsx`/`*.css` under `src/` (grep confirmed). |
| `class-variance-authority` | Only useful if you're building a design-system component library; the app uses hand-written CSS. Not imported anywhere in `src/`. |
| `@types/dompurify` | Types for `dompurify@^3.3.3`, but as of dompurify v3.x the package ships its own types. Likely vestigial. Verify before remove. |

**Recommendation: do not remove during the fork.** Touch only what the fork strictly needs. Run a separate dependency audit during the Week 6 cleanup.

## Keep but rescope

| Package | Current use | Future use |
|---|---|---|
| `jszip` | Client-side ZIP build for plate PNG/SVG bundles | KEEP — but ZIP build moves to the backend (chuck-mcp's `export_carving_files` writes server-side). Used only as fallback "Download bundle" in `/sign-off`. |
| `makerjs`, `clipper2-ts` | CNC SVG geometry on `/cnc` page | KEEP — `/cnc` page stays live. Some block-render code in v2 might reuse `makerjs` for kento-jig SVG composition; defer until Week 4 reveals the need. |
| `framer-motion` | Loaded but not used in color-separator grid (which uses CSS keyframes) | KEEP — likely useful for the pull scrubber on `/preview/[plan_id]` |
| `lucide-react` | Icons | KEEP — sidebar action bar will use them |
| `dompurify` | Sanitizing user-rendered HTML | KEEP — the LLM interpretation panel may display formatted text |

## devDependencies

No changes. The existing Playwright + ESLint + Tailwind + TypeScript setup covers v2 testing needs.

## Engines / runtime

No change. `engines: {"node": ">=24"}` and React 19 / Next 16 stay.

## Sentry

Already wired in `next.config.ts` via `withSentryConfig(nextConfig, { silent: true })`. Set up a new Sentry project for v2 events (separate DSN, environment tag `colorv2`). No package change.

## Net delta

- Adds: 5 packages (`swr`, `zod`, `react-zoom-pan-pinch`, `d3-scale`, `d3-array`)
- Removes: 0 packages (cleanup deferred)
- Bundle size: +~60KB gzipped
