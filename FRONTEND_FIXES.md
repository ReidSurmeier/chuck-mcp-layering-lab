# Frontend Fixes — colorsep-v20-optimize

## Changes

### 1. Progress bar moved to viewport top (`globals.css`)
- Changed `.progress-bar-root` from `top: 28px` to `top: 0`
- z-index 250 already above header (200)
- ARIA attributes already present: `role="progressbar"`, `aria-valuenow`, `aria-valuemin`, `aria-valuemax`, `aria-label`, `aria-live="polite"`

### 2. upscale_scale added to fetchPlatesStream and fetchMerge (`api.ts`)
- `fetchPlatesStream` was missing `upscale_scale` in FormData — backend ignored 4x setting for plate images
- `fetchMerge` was also missing it
- `buildFormData` (used by preview-stream, preview, separate) already had it in all version blocks

### 3. Download progress bar tracks actual percentage (`NavPanel.tsx`)
- `.download-progress-fill` was using CSS animation with fixed 30% width
- Now parses percentage from `downloadProgress` string (e.g. "downloading 45%")
- Sets explicit `width` style and disables animation when percentage available
- Added ARIA attributes: `role="progressbar"`, `aria-valuenow`, `aria-label`

## Test Results

### New tests (e2e/ui-fixes.spec.ts) — 5/5 PASS
- Progress bar visible at top within 1s
- Progress percentage increases during processing
- Download progress bar tracks percentage
- 4x upscale_scale sent in plates-stream request
- Zero console errors during full flow

### Existing tests (e2e/color-separator.spec.ts) — 6/7 PASS
- Test 3 (plates timing ≤2s after composite) fails — **pre-existing backend issue**
  - Frontend dispatches both requests concurrently (verified)
  - Backend serializes GPU access: preview-stream completes first, then plates-stream
  - Not fixable from frontend — needs backend request queuing or parallel GPU access

## Files Changed
- `src/app/globals.css` — progress bar `top: 0`
- `src/lib/api.ts` — `upscale_scale` in `fetchPlatesStream` + `fetchMerge`
- `src/app/color-separator/components/NavPanel.tsx` — download progress bar width tracking
- `e2e/ui-fixes.spec.ts` — new test file
