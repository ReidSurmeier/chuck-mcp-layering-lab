# Mask Validation Rules

Locked contract between user-painted PSD output and the chuck-mcp v2 solver input.

`ingest_masks.py:ingest_painted_psd()` enforces every rule below.

## Rule 1: Canvas dimension exact match

**Check**: `psd.width == manifest.canvas.width_px AND psd.height == manifest.canvas.height_px`

**Severity**: Hard error.

**User-facing message**:
> Canvas was resized. Expected {W}×{H}, got {w}×{h}. Photoshop may have asked
> you to resize on save — undo and re-save without resampling.

**Why**: If canvas resized, every pixel coordinate is off. Kento marks drift, slot masks no longer align to the source image. Unrecoverable.

## Rule 2: Kento reference layer present

**Check**: Some layer has name containing `"kento_marks"`. By default the template uses `"[REF] kento_marks_DO_NOT_PAINT"`.

**Severity**: Hard error.

**User-facing message**:
> The kento reference layer is missing. You may have deleted it. Re-open the
> original template, copy the kento layer over, save, and try again.

**Why**: Even though we don't always bitwise-compare kento (Rule 7), losing it means we can't verify registration at all.

## Rule 3: Binary mask threshold

**Check**: For each `slot_NN_*` layer with painted content, the grayscale composite must have
≥99% of pixels clustered at 0±2 or 255±2.

```python
flat = grayscale.flatten()
clustered_pct = ((flat <= 2) | (flat >= 253)).mean() * 100.0
assert clustered_pct >= 99.0
```

**Severity**: Hard error.

**User-facing message**:
> {layer_name}: not binary ({clustered:.2f}% clustered at extremes, need ≥99%).
> You probably used a soft brush — switch to a hard-edge brush (hardness=100%)
> or apply Image → Adjustments → Threshold (127) to the layer.

**Why**: Mokuhanga underlayer plates are binary (carved or not). Anti-aliased grey edges are not physically realizable. Solver expects boolean masks.

**Tolerance** of 2 (not 0) accounts for:
- JPEG-style compression artifacts if user saved as JPG then re-imported (don't do this!)
- Photoshop's internal anti-alias from "no anti-alias" tool settings (still leaks ~1px of grey)
- Resampling slop if user accidentally resized then undid

## Rule 4: Coverage range

**Check**: Each slot's coverage percentage must be `min_coverage_pct ≤ cov ≤ max_coverage_pct`.

Default: `min=0.1%`, `max=80.0%`.

**Severity**: Hard error.

**User-facing messages**:
> {layer_name}: coverage {pct:.3f}% below minimum 0.1%. Probably an
> accidental dot — please re-mask.

> {layer_name}: coverage {pct:.3f}% above maximum 80%. Probably forgot
> to mask off — please clip.

**Why**:
- Below 0.1% (~6 pixels on a 1240×1748 canvas) is almost always an accidental tool slip.
- Above 80% means the user inverted the mask (painted background instead of foreground).

## Rule 5: Kento no-paint zones

**Check**: For each slot mask, count painted pixels inside any kento bbox (see `kento_spec.md` for bbox derivation). Total must be 0.

**Severity**: Hard error.

**User-facing message**:
> {layer_name}: {N}px painted over kento marks ({list_of_zones}).
> Registration will fail — please clear corners.

**Why**: If a slot mask covers the kento, the CNC plate generator will carve the kento away. Loss of registration = whole print fails. Hard fail at validation, not at carve time.

## Rule 6: Layer offset / bounds within canvas

**Check**: For each painted slot, layer bounding box (offset + size) must be entirely within canvas bounds.

```python
assert layer.offset[0] >= 0
assert layer.offset[1] >= 0
assert layer.offset[0] + layer.width <= psd.width
assert layer.offset[1] + layer.height <= psd.height
```

`_layer_to_canvas_array()` clips to canvas defensively, so this is a warning, not a hard error.

**Severity**: Warning.

**User-facing message**:
> {layer_name}: layer extends outside canvas bounds. Clipped to canvas.

**Why**: Photoshop allows layer pixel data to extend beyond the canvas (e.g. if you moved a large layer). It's never useful here — solver only cares about in-canvas pixels.

## Rule 7: Bitwise kento integrity (OPTIONAL, off by default)

**Check**: If user has not interacted with kento layer, its rendered content == template's rendered content.

```python
template_kento = render_kento_layer(spec)
user_kento = kento_layer.composite()
assert np.array_equal(np.array(template_kento), np.array(user_kento))
```

**Severity**: Warning by default. Hard error if `strict_kento=True`.

**User-facing message**:
> Kento layer has been modified from the template. This is fine if you
> just changed visibility/opacity, but if you painted on or transformed
> it, registration may have drifted.

**Why**: Defensive net for cases where the user accidentally edited the kento layer. Skipped by default because PSD opacity changes shouldn't fail validation.

## Rule 8: Unknown slot names (manifest mismatch)

**Check**: Every `slot_*` layer name in the PSD must appear in the manifest's slot list.

**Severity**: Warning.

**User-facing message**:
> Unknown slot {layer_name} (not in manifest). Ignored.

**Why**: User may have added their own layer for experimentation. Don't error — just ignore.

## Rule 9: Missing slots (warning only)

**Check**: Every slot named in the manifest should have a valid mask after all other rules.

**Severity**: Warning.

**User-facing message**:
> Slots without valid masks: {list}. Solver will skip these unless they
> were declared essential.

**Why**: Users iterate. They paint slot_01 first, save, run the solver, see the result, then come back and paint slot_02. Don't force them to paint all 9 at once.

## Aggregation logic

`ingest_painted_psd(..., strict=True)`:
1. Run all checks.
2. If ANY hard error → raise `ValidationError` listing every error.
3. If only warnings → return `(masks, report)` with `report.ok == True`.

`ingest_painted_psd(..., strict=False)`:
1. Run all checks.
2. Always return `(masks, report)`. Caller inspects `report.errors`.
3. Slots that failed validation are NOT in `masks` (so solver only sees valid masks).

## Test matrix (covered by `gen_template.py` / `ingest_masks.py` end-to-end test)

| Scenario | Expected outcome |
|---|---|
| All slots painted binary, kento intact | All masks returned. `report.ok=True`. |
| One slot empty | Empty slot omitted from masks. Warning only. `report.ok=True`. |
| One slot non-binary (soft brush) | Hard error. Strict mode raises. |
| One slot painted over kento | Hard error. Strict mode raises. |
| Canvas resized | Hard error. Strict mode raises. |
| Kento layer deleted | Hard error. Strict mode raises. |
| Layer named `slot_999_lol` | Warning only. Ignored. |
| Slot with coverage 0.05% | Hard error (below min). |
| Slot with coverage 95% | Hard error (above max). |

## Tuning parameters (manifest defaults)

Every threshold lives in `manifest.validation`:

```json
{
  "validation": {
    "binary_threshold_pct": 99.0,
    "binary_tolerance": 2,
    "min_coverage_pct": 0.1,
    "max_coverage_pct": 80.0
  }
}
```

To loosen for prototype iterations: drop binary_threshold_pct to 95.0. To tighten for final production: raise binary_tolerance to 0 (= exactly 0 or 255 only).
