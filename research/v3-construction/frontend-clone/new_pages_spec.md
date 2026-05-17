# New Pages Spec — colorv2 routes

Source design contract: `/home/reidsurmeier/src/chuck-mcp-layering-lab/docs/v2-design-locked-2026-05-16.md`

All four pages live under `src/app/colorv2/` and share the existing `globals.css` design system. Each section below is a wireframe-in-text plus the explicit API contract and user actions.

---

## 1. `/colorv2/upload`

**File**: `src/app/colorv2/upload/page.tsx`
**Entry point**: middleware rewrites `colorv2.reidsurmeier.wtf/` to here when there's no `last_plan_id` cookie. Direct visit OK.

### Layout

```
+----------------------------------------------------------------+
| colorv2 . chuck-mcp planner                  [about] [archive] |  <- top bar (back-to-tools style)
+----------------------------------------------------------------+
|                                                                |
|   STEP 1   choose source image                                 |
|   ┌────────────────────────────────────────────┐               |
|   │  drop image or click to browse              │               |
|   │  ─────────────────────────────────          │               |
|   │  preview: 1843×2456 JPG, 2.1MB              │               |
|   └────────────────────────────────────────────┘               |
|                                                                |
|   STEP 2   intent prompt (optional)                            |
|   ┌────────────────────────────────────────────┐               |
|   │ "warm undertones, avoid phthalo blue,       │               |
|   │  3 underlayers max, prioritize skin"        │               |
|   └────────────────────────────────────────────┘               |
|   leave blank to use Emma defaults                             |
|                                                                |
|   ┌─────────────────────┐    ┌───────────────┐                |
|   │  preview interp.    │    │  generate →   │                |
|   └─────────────────────┘    └───────────────┘                |
|                                                                |
|   [collapsible: previous plans] →                              |
|     - 2026-05-14 emma  (plan_id: emma-c4f2)                    |
|     - 2026-05-12 reid_self  (plan_id: self-a17b)               |
+----------------------------------------------------------------+
```

### Data shown

- Image preview (width × height × MIME × file size) — exactly the same `imageInfo` object pattern as the existing `handleFileSelect` in `useColorSeparator.ts` lines 430–479
- Live LLM interpretation preview (optional, behind `preview interp.` button — costs one cheap `claude -p` call that does NOT yet solve)
- List of recent plans from `localStorage.recentPlanIds` (no backend call)

### API calls

| Trigger | Endpoint | Method | Payload | Response |
|---|---|---|---|---|
| `preview interp.` click | `/api/chuck/interpret` | POST | `{image_filename, intent_prompt}` | `{interpretation: {forbidden_pigments, region_overrides, ...}, cost_estimate_usd}` |
| `generate →` click | `/api/chuck/plan` | POST FormData | `image: File, intent_prompt: string, iteration_anchor_path?: string` | `{plan_id, preview_url, status: "planning"}` then redirect to `/colorv2/preview/<plan_id>` |
| recent plans list | none — localStorage | — | — | — |

### User actions

- Drag/drop or click to upload image (mime allow-list `image/jpeg, image/png, image/webp, image/tiff`, max 50MB per `next.config.ts` middlewareClientMaxBodySize)
- Type intent prompt
- Click `preview interp.` → expands interpretation panel inline, does not navigate away
- Click `generate →` → posts plan request, shows blocking loader ("interpreting → planning..."), then redirects on first SSE event with `plan_id`
- Click an old plan in the list → navigates to `/colorv2/preview/<plan_id>` (server returns cached plan)

### Edge cases

- Image < 2048px on long edge: backend triggers auto-upscale over Tailscale per design Week 4. Show "upscaling..." toast while it runs.
- Backend rejects intent_prompt (zod schema mismatch from LLM): show backend's `interpretation_error.hint` inline, keep form state, do not redirect.
- Generate clicked twice rapidly: dedupe via the same in-flight map pattern as `src/lib/api.ts` lines 65–73.

---

## 2. `/colorv2/preview/[plan_id]`

**File**: `src/app/colorv2/preview/[plan_id]/page.tsx`
**Entry point**: redirected from `/upload` after plan submission, or direct deep link.

### Layout (per design doc "Preview surfaces" section)

```
+----------------------------------------------------------------------+
| ← upload   plan: emma-c4f2   iter 1   continuity 0.84   [proof →]   |
+--------------------------------------+-------------------------------+
|  composite vs target (top)           |  SIDEBAR                      |
|  ┌──────────┐  ┌──────────┐         |                               |
|  │  TARGET  │  │ COMPOSITE│         |  intent prompt:               |
|  │          │  │          │         |  "warm undertones, avoid      |
|  └──────────┘  └──────────┘         |   phthalo blue..."            |
|  (paper texture overlay when         |                               |
|   showing composite, ORIGINAL label  |  interpretation:              |
|   when toggled via spacebar)         |  - forbidden: phthalo_blue    |
|                                      |  - underlayers cap: 3         |
+──────────────────────────────────────+  - region overrides: none     |
|  27 BLOCKS (primary review grid)     |                               |
|  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐     |  defaults applied:            |
|  │1 │ │2 │ │3 │ │4 │ │5 │ │6 │     |  - role priors                 |
|  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘     |  - underlayer hue rules        |
|  ...                                 |                               |
|  ┌──┐                                |  provenance per underlayer:   |
|  │27│                                |  block 04: ALGO (cool-under-  |
|  └──┘                                |    shadow rule)               |
|  each card: mirrored plate SVG       |  block 06: TEXT ("warm under- |
|  on wood-grain ground + pull list    |    tones" prompt phrase)      |
|  + block_id label                    |                               |
+──────────────────────────────────────+  validators (live):           |
|  PULL SCRUBBER (1 → 132)             |  ☑ plate_not_composite  0.31  |
|  ━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━  |  ☑ role_purity          0.92  |
|  pull 47 / 132  (block 12, pass 2)   |  ☑ jigsaw_separation    PASS  |
|  preview at this pull: ...           |  ☐ proof_progression    0.71  |
|                                      |  ☑ underlayer_reversal  PASS  |
|                                      |  ─ final_match  ΔE 4.8 (info) |
|                                      |                               |
|                                      |  [adjust] [re-prompt]         |
|                                      |  [single-block patch]         |
|                                      |  [sign off → carve]           |
+──────────────────────────────────────+-------------------------------+
```

### Data shown

- Target image (the upload)
- Current composite render
- 27 block cards (each: mirrored plate SVG preview, block_id, list of pull_ids that use this block, pigment swatch row)
- Pull scrubber (1–132) with current-pull preview image showing cumulative state through that pull
- Sidebar: prompt, interpretation panel (LLM output), defaults applied, per-underlayer provenance (ALGO vs TEXT + which prompt phrase triggered it), live validator scores
- Iteration count + diminishing-returns nag after iter 5

### API calls

| Trigger | Endpoint | Method | Response |
|---|---|---|---|
| page mount | `/api/chuck/plan/[plan_id]` | GET | full plan JSON (Plate[27], Pull[132], ProofState[7], validator_scores, interpretation, defaults_applied, iteration_count, nag) |
| page mount (in parallel) | `/api/chuck/plan/[plan_id]/progress` | SSE | live progress while status != "ready" |
| pull scrubber drag | `/api/chuck/plan/[plan_id]/pull/[pull_id]/render` | GET | PNG of cumulative state through that pull |
| block card click | `/api/chuck/plan/[plan_id]/block/[block_id]/svg` | GET | mirrored plate SVG (full res) |
| composite/target toggle (spacebar) | none — DOM toggle | — | — |
| `re-prompt` action | `/api/chuck/plan` | POST | `{intent_prompt, iteration_anchor_path: /tmp/plans/[plan_id]/plan_vN.json}` → returns new `plan_id`, redirect |
| `single-block patch` action | `/api/chuck/regenerate-blocks` | POST | `{plan_id, blocks: number[], photoshop_mask?: string}` → returns updated plan, refresh in place |
| `sign off → carve` action | navigate to `/colorv2/sign-off/[plan_id]` | — | — |
| `proof →` link | navigate to `/colorv2/proof/[plan_id]` | — | — |

### SSE event vocabulary

The progress stream emits one of these per `data: {...}\n\n` frame (mirrors existing `fetchPreviewStream` parser in `src/lib/api.ts` lines 624–745):

| stage | extra fields | meaning |
|---|---|---|
| `"upscaling"` | `pct` | RealESRGAN running over Tailscale |
| `"interpreting"` | `pct` | LLM translating intent_prompt |
| `"snic"` | `pct` | cell-graph build via SNIC |
| `"baseline_underlayers"` | `pct`, `block_count` | algorithmic 4–9 underlayer proposal |
| `"text_overrides"` | `pct` | applying intent_prompt overrides |
| `"block_partition"` | `pct` | DSATUR partition into 27 blocks |
| `"solving"` | `pct`, `current_pull`, `total_pulls`, `current_block_id` | per-pull progress |
| `"validators"` | `validator_id`, `score` | live validator scores |
| `"rendering_proofs"` | `pct`, `checkpoint_id` | computing 7 proof states |
| `"heartbeat"` | — | keepalive, ignore |
| `"complete"` | `result_id`, `plan_summary` | done |
| `"error"` | `error`, `recoverable: bool` | terminal or recoverable failure |

### User actions

- Spacebar toggles target ↔ composite (same UX as existing color-separator)
- Click block card → opens `BlockZoom` modal with full-res mirrored plate SVG + pull metadata
- Drag pull scrubber → loads cached pull-N render PNG (debounced 200ms)
- Click `adjust` → expands inline param tweaker (deferred to V2)
- Click `re-prompt` → opens textarea inline, submits as new iteration anchored to current plan
- Click `single-block patch` → opens multi-select of block_ids + (deferred) Photoshop mask upload, posts to `regenerate-blocks`
- Click `sign off → carve` → navigates to `/colorv2/sign-off/[plan_id]`

### Reuses from color-separator (verbatim CSS classes)

- `.plate-card`, `.plate-card-image`, `.plate-card-info`, `.plate-card-swatch`, `.plate-card-hex` → become `.block-card`, same animation timing
- `.plates-grid` → `.blocks-grid` (same flex layout, larger gap for 27 cards)
- `.plate-skeleton*` → `.block-skeleton*` for streaming-block placeholders (same `@keyframes skeleton-pulse`)
- `.progress-bar-*` → all reused for the top SSE progress bar
- `.about-overlay`, `.about-box` → patterns for the sidebar info panels
- `.merge-progress-overlay`, `.merge-spinner` → reused for `single-block patch` regeneration state

---

## 3. `/colorv2/proof/[plan_id]`

**File**: `src/app/colorv2/proof/[plan_id]/page.tsx`
**Entry point**: `proof →` button on `/preview/[plan_id]` page, or direct deep link.

### Layout (per design doc "Verification surfaces")

```
+----------------------------------------------------------------------+
| ← preview   plan: emma-c4f2   proof verification                    |
+----------------------------------------------------------------------+
|                                                                      |
|  YOUR PROOF SERIES (7 checkpoints)        REFERENCE PROOF SERIES     |
|  ┌────┐┌────┐┌────┐┌────┐                ┌────┐┌────┐┌────┐┌────┐  |
|  │ p4 ││ p8 ││p12 ││p16 │ ...           │ h4 ││ h8 ││h12 ││h16 │  |
|  └────┘└────┘└────┘└────┘                └────┘└────┘└────┘└────┘  |
|  ┌────┐┌────┐┌────┐                      ┌────┐┌────┐┌────┐         |
|  │p20 ││p24 ││p132│                      │h20 ││h24 ││h132│         |
|  └────┘└────┘└────┘                      └────┘└────┘└────┘         |
|                                          (toggle: Hokusai / Close)  |
|                                                                      |
|  proof_progression_score  0.71                                       |
|                                                                      |
+----------------------------------------------------------------------+
|                                                                      |
|  PER-PULL LOAD-BEARING HEATMAP                                       |
|  ┌──────────────────────────────────────┐  pull selector             |
|  │  composite with red overlay where    │  ◀  47 / 132  ▶            |
|  │  ΔE on removal > threshold           │                            |
|  │                                      │  removal_delta_e: 14.2     |
|  │                                      │  load_bearing: yes         |
|  │                                      │  (would be rejected if no) |
|  └──────────────────────────────────────┘                            |
|                                                                      |
|  CONTINUITY SCORE   0.84                                             |
|                                                                      |
|  6 VALIDATORS    (full results)                                      |
|  ☑ plate_not_composite  0.31  (threshold ≤ 0.6 — pass)               |
|  ☑ role_purity          0.92                                         |
|  ☑ jigsaw_separation    min 5.4mm (≥ 5.0 required)                  |
|  ☐ proof_progression    0.71  (advisory — flag if < 0.75)            |
|  ☑ underlayer_reversal  all 27 blocks flipped, no pulls flipped     |
|  ─ final_match          ΔE_2000 mean 4.8 (info only)                |
|                                                                      |
|  [back to preview]   [sign off → carve →]                            |
+----------------------------------------------------------------------+
```

### Data shown

- 7 checkpoint proof images (after pulls 4, 8, 12, 16, 20, 24, 132 per design)
- Reference proof series (toggle: Hokusai Great Wave or Chuck Close Emma) — static images served from `public/seo/reference/` (to be populated)
- Per-pull heatmap overlay (composite + red-tint where removing-this-pull would shift > threshold ΔE)
- Continuity score (numeric)
- All 6 validators with PASS/FAIL + score + threshold

### API calls

| Trigger | Endpoint | Method | Response |
|---|---|---|---|
| page mount | `/api/chuck/plan/[plan_id]/proofs` | GET | `{checkpoints: [{checkpoint_id, image_url, pull_ids_so_far}], continuity_score, validators}` |
| checkpoint card click | opens zoomed view, no API call | — | — |
| pull selector ◀ ▶ | `/api/chuck/plan/[plan_id]/pull/[pull_id]/heatmap` | GET | PNG with red-tint overlay where load-bearing |
| reference series toggle | none — local state, static images | — | — |

### User actions

- Click any of the 7 proof cards → full-screen modal
- Use pull selector to step through 1–132, watching the heatmap
- Toggle reference between Hokusai / Chuck Close
- Click `back to preview` → returns to `/preview/[plan_id]`
- Click `sign off → carve →` → forwards to `/sign-off/[plan_id]`

### Edge cases

- A validator fails: the row gets red background, the `sign off` button is disabled with tooltip "validator X failed, address before signing off"
- Heatmap not yet computed for the selected pull: show spinner, request lazy

---

## 4. `/colorv2/sign-off/[plan_id]`

**File**: `src/app/colorv2/sign-off/[plan_id]/page.tsx`
**Entry point**: `sign off → carve` button on `/preview/` or `/proof/`.

### Layout

```
+----------------------------------------------------------------------+
| ← proof   plan: emma-c4f2   sign-off                                |
+----------------------------------------------------------------------+
|                                                                      |
|  READY TO CARVE?                                                     |
|                                                                      |
|  this will export 27 mirrored plate SVGs + soto-kento jig SVG +     |
|  pigment recipes + 132 pull renders + proof contact sheets to:      |
|                                                                      |
|  ~/cnc-carving-jobs/emma-2026-05-16/                                 |
|                                                                      |
|  ────────────────────────────────────────────────                    |
|                                                                      |
|  validator summary (all 6 must pass)                                 |
|  ☑ plate_not_composite                                               |
|  ☑ role_purity                                                       |
|  ☑ jigsaw_separation                                                 |
|  ☐ proof_progression  ← advisory, sign-off allowed                  |
|  ☑ underlayer_reversal                                               |
|  ─ final_match                                                       |
|                                                                      |
|  plan metadata                                                       |
|    iterations: 3                                                     |
|    total pulls: 132                                                  |
|    unique pigments: 11                                               |
|    edition target: 10 prints                                         |
|    estimated total ink (full edition): 480ml                         |
|                                                                      |
|  ┌────────────────────────────┐  ┌────────────────────────────┐    |
|  │  download zip (fallback)   │  │  export to ~/cnc-carving-  │    |
|  │                            │  │  jobs/  →                  │    |
|  └────────────────────────────┘  └────────────────────────────┘    |
|                                                                      |
|  after export — what's next?                                         |
|  1. load block_01.svg on RISD ShopBot                                |
|  2. cut soto-kento jig once, reuse across editions                   |
|  3. mix pigments per recipes/pigment_mix_recipes.md                  |
|  4. print master proof, pin to wall                                  |
|  5. print edition of 10 following recipes/pull_order_schedule.md    |
|                                                                      |
+----------------------------------------------------------------------+
```

### After export-clicked state

```
+----------------------------------------------------------------------+
|  ✓ exported                                                          |
|                                                                      |
|  ~/cnc-carving-jobs/emma-2026-05-16/                                 |
|                                                                      |
|  carving job folder:                                                 |
|    plates/        27 SVG + 27 preview PNG                            |
|    jig/           1 SVG (soto-kento)                                 |
|    proofs/        5 contact-sheet PNGs                               |
|    pulls/         132 cumulative-state PNGs                          |
|    recipes/       3 docs (mix recipes + ink estimate + pull schedule)|
|    meta/          5 JSON/CSV/YAML (plan, interpretation, iterations, |
|                   validator scores, CNC tool config)                 |
|    previous_plan.json   symlink → plan_v01.json                      |
|                                                                      |
|  total files: 169                                                    |
|  total size: 142 MB                                                  |
|                                                                      |
|  [open folder]  [back to preview]  [new plan →]                      |
+----------------------------------------------------------------------+
```

### Data shown

- Validator summary (re-displayed from `/proof` page, condensed)
- Plan metadata: iteration count, total pulls, unique pigment count, edition target, ink estimate
- Export target path
- Post-export: actual file count, total size, folder structure tree

### API calls

| Trigger | Endpoint | Method | Response |
|---|---|---|---|
| page mount | `/api/chuck/plan/[plan_id]` | GET | full plan (already cached by SWR from `/preview` page) |
| `export to ~/cnc-carving-jobs/` click | `/api/chuck/export-carving` | POST | `{plan_id, output_dir: "~/cnc-carving-jobs"}` → `{carving_job_path, files: string[]}` (may take 30s–2min) |
| `download zip (fallback)` click | `/api/chuck/plan/[plan_id]/zip` | GET | streamed ZIP blob (built server-side) |
| `open folder` click (post-export) | system file-manager URL `file://...` | — | works on Linux desktop only |
| `new plan →` click | navigate to `/upload` | — | — |

### User actions

- Confirm validators pass (visual only — backend will hard-gate)
- Click `export to ~/cnc-carving-jobs/` (primary) or `download zip` (fallback for remote use)
- Wait for backend to write files (show progress bar with file count)
- Post-export: click `open folder` (Linux desktop) or copy path
- Click `new plan →` to start fresh

### Edge cases

- Backend reports validator failure on final check: show red banner, link back to `/proof/` page
- Disk full: backend returns 507, frontend shows clear error + suggests trash old jobs
- User navigates away mid-export: export is server-side, completes regardless; show toast on return if `localStorage.pending_exports[plan_id]` matches
- Re-export the same plan: backend appends `_v02` suffix to the folder name

---

## Shared layout: `src/app/colorv2/layout.tsx`

Copy of `src/app/color-separator/layout.tsx` with new `metadata`:

```ts
export const metadata = {
  title: "colorv2 — chuck-mcp mokuhanga planner",
  description: "Plan a Chuck Close-style multi-block woodblock print from one image",
};
```

Keeps `noStore()`, `force-dynamic`, `force-no-store` for the same Cloudflare-cache-busting reason.

---

## Route summary

| Path | Loads | Persists | SSE |
|---|---|---|---|
| `/colorv2/upload` | nothing server-side; localStorage for recent plans | sets `recentPlanIds[]` in localStorage | none |
| `/colorv2/preview/[plan_id]` | full plan + live progress | none (server owns plan state) | yes — solver progress |
| `/colorv2/proof/[plan_id]` | proofs + validators + heatmaps on-demand | none | none (proofs are computed-once) |
| `/colorv2/sign-off/[plan_id]` | plan summary | sets `pending_exports[plan_id]` while export runs | none (export is fast-ish: <2min) |
