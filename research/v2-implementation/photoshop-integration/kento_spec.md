# Kento Mark Coordinate Convention

Locked spec for chuck-mcp v2. Reused by:
- `photoshop-integration/gen_template.py` (template generation)
- `cnc-mokuhanga-carve/*` (CNC plate generation — kento must match downstream)
- `binary-mask-jax/*` (solver hard-constraint validation)

## Background

Kento (見当) is the Japanese woodblock registration system. Traditionally:
- **Kagi-ita** (鉤板) — L-shaped notch at one corner (typically lower right). Locks two edges.
- **Hikitsuke** (引付) — straight notch on adjacent edge (typically bottom). Locks one edge.

Two marks = full 2D registration. Sub-mm precision when carved correctly.

In chuck-mcp v2 we use a richer convention because we're generating both the painting template (Photoshop) and the CNC carving template (Onshape/Carbide Create). We need *redundant* marks so the user can visually verify alignment in either tool, and so the solver can detect any rotation/scale drift in the PSD round-trip.

## Mark types

| Mark | Symbol | Position | Function |
|---|---|---|---|
| Corner L (×4) | `⌐` `¬` `⌐` `¬` (rotated) | each corner, inset from edge | 2D position lock |
| Center crosshair (×1) | `+` | canvas center | rotation + scale verification |

All marks: pure black (#000000) on transparent / white background. Single channel.

## Coordinate convention

Origin: **top-left** pixel of the canvas (Photoshop / PIL / numpy convention).
Y axis: increases downward.
X axis: increases rightward.

## Geometric parameters (locked for v2)

| Parameter | Value | Notes |
|---|---|---|
| `kento_inset_mm` | 10.0 | Distance from canvas edge to L-corner vertex |
| `kento_arm_mm` | 15.0 | Length of each L arm |
| `kento_stroke_mm` | 0.5 | Line width (visible but not bulky) |
| `crosshair_arm_mm` | 5.0 | Half-length of each crosshair arm |

These convert to pixels via the canvas DPI:

| DPI | inset_px | arm_px | stroke_px | crosshair_px |
|---|---|---|---|---|
| 150 | 59 | 89 | 3 | 30 |
| 300 | 118 | 177 | 6 | 59 |
| 600 | 236 | 354 | 12 | 118 |

Conversion: `px = round(mm * dpi / 25.4)`.

## Corner positions

For canvas of width `W` and height `H` pixels with `pad = kento_inset_px`:

| Corner | Vertex (x, y) | L opens toward |
|---|---|---|
| NW (top-left) | `(pad, pad)` | SE (canvas interior) |
| NE (top-right) | `(W - pad, pad)` | SW (canvas interior) |
| SW (bot-left) | `(pad, H - pad)` | NE (canvas interior) |
| SE (bot-right) | `(W - pad, H - pad)` | NW (canvas interior) |

**Definition of "L opens toward"**: the two arms extend in those directions from the vertex.

For NW (flip_x=False, flip_y=False):
- Horizontal arm: from `(pad, pad)` to `(pad + arm, pad)`
- Vertical arm: from `(pad, pad)` to `(pad, pad + arm)`

The vertex of the L is at the named corner pixel coordinate, **and that pixel itself is part of both arms**. Sub-pixel accuracy of the vertex IS the registration anchor.

## Per-paper-size pixel coordinates (300 DPI)

| Paper | Canvas (px) | NW | NE | SW | SE | Center |
|---|---|---|---|---|---|---|
| A3 (297×420 mm) | 3508×4961 | (118, 118) | (3390, 118) | (118, 4843) | (3390, 4843) | (1754, 2480) |
| A4 (210×297 mm) | 2480×3508 | (118, 118) | (2362, 118) | (118, 3390) | (2362, 3390) | (1240, 1754) |
| A5 (148×210 mm) | 1748×2480 | (118, 118) | (1630, 118) | (118, 2362) | (1630, 2362) | (874, 1240) |
| A6 (105×148 mm) | 1240×1748 | (118, 118) | (1122, 118) | (118, 1630) | (1122, 1630) | (620, 874) |

(`gen_template.make_template_spec()` returns these programmatically.)

## No-paint zones (validation hooks)

Each kento mark has a bounding box that the user MUST NOT paint inside. Bounding boxes have a 2×stroke margin around the visible mark, to account for anti-aliasing slop.

For each L at vertex `(vx, vy)` with arm direction `(dx, dy)` where `dx, dy ∈ {-1, +1}`:

```
margin = stroke_px * 2
bbox = (
    min(vx, vx + dx * arm_px) - margin,  # x0
    min(vy, vy + dy * arm_px) - margin,  # y0
    max(vx, vx + dx * arm_px) + margin,  # x1
    max(vy, vy + dy * arm_px) + margin,  # y1
)
```

For the center crosshair at `(cx, cy)` with half-arm `ch`:

```
bbox = (cx - ch - margin, cy - ch - margin, cx + ch + margin, cy + ch + margin)
```

`ingest_masks.py:_overlap_with_zones()` enforces this rule.

## Validation: detecting drift in round-trip

When a user-painted PSD comes back, we can verify the kento layer is bitwise-identical to the template's kento layer:

```python
template_kento = render_kento_layer(spec)
user_kento_layer = next(L for L in psd if 'kento_marks' in L.name)
user_kento = user_kento_layer.composite()
assert np.array_equal(np.array(template_kento), np.array(user_kento))
```

If equal → no resampling / cropping / rotation happened during the round-trip.
If not equal → reject. Photoshop transformed the kento layer (probably by user error).

This check is OPTIONAL because most users won't manipulate the locked reference layer, but it's a cheap safety net.

## Why not Photoshop guides?

Photoshop has "guides" (View → New Guide) which are non-printable horizontal/vertical lines. We don't use them because:

1. **Guides aren't serialized to PSD pixel data** — they live in the document's resource block. psd-tools doesn't have native API for them (writeable). Adobe's spec calls them "Grid and guides resource" (resource ID 0x0408).
2. **Guides don't survive flattening / export** — if user flattens to PNG, guides vanish.
3. **Guides aren't physically printable** — they only exist in-app. We need printable marks for the CNC plate downstream.

Pixel-based kento marks are the only honest solution that round-trips through PSD, PNG, and the CNC carving stage.

## Future: vector kento via Photoshop shape layers

Down the road, kento marks could be vector shape layers (Photoshop "shape layer" / vector mask). psd-tools 1.17.0 reads but does not yet write vector mask paths. PhotoshopAPI (EmilDohne) may close this gap. For v2 we stick to rasterized marks at canvas DPI — they're physically faithful at print resolution.

Sources:
- Salter, "Japanese Woodblock Printing" (UH Press, 2002) — pages 67-72 on kagi-ita / hikitsuke practice
- Vollmer, "Japanese Woodblock Print Workshop" (Watson-Guptill, 2015) — chapter 4
- Adobe Photoshop File Formats Specification, "Grid and guides resource" (0x0408)
- psd-tools `psd_tools.psd.image_resources.PrintFlags` — registration_marks boolean (irrelevant — that's a print-output flag)
