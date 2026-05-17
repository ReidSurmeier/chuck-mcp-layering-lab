---
title: "SVG Mirror Flip — Implementation for chuck-mcp v2 Block Export"
sources:
  - "https://www.w3.org/TR/SVG11/coords.html"
  - "https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform"
  - "https://gist.github.com/topics/svg-transform"
  - "https://github.com/cnc-club/gcodetools"
  - "https://www.fablewood.com/cutting-a-kento-registration/"
relevance: "MUST-READ — block prints mirror-image of what's carved, so SVG export must apply horizontal flip BEFORE tool compensation, BEFORE kento placement, BEFORE G-code emission. Spells out the exact transform: scale(-1, 1) translate(-width, 0). Includes the matched flip on the metadata coordinates so the printer-side preview can show the source-image orientation."
tags: [svg, mirror, transform, horizontal-flip, export, coordinate-system]
---

# SVG Mirror Flip Implementation

## Why mirror-flip?

A woodblock prints **mirror-image** of what's carved on it. If you carve
the letter `R` on the block, you print `Я`. Therefore the SVG sent to
CNC must be the **mirror of the desired final printed image**.

This is a one-line transform that, if forgotten, ruins all 27 blocks.

## The transform

In SVG coordinate space (origin top-left, +X right, +Y down), horizontal
mirror about the vertical axis at x = width:

```svg
<g transform="scale(-1, 1) translate(-1000, 0)">
  <!-- all carve geometry goes here -->
  <!-- assume content was drawn in 1000mm-wide coordinate system -->
</g>
```

The two-step interpretation:
1. `scale(-1, 1)` flips X about the origin (X=0 axis)
2. `translate(-1000, 0)` moves the flipped content back into positive-X
   space (a 1000mm-wide canvas)

**Order matters:** SVG transforms compose right-to-left. The translate
happens **after** the scale visually, but is written first in the
attribute. The transform matrix:

```
| -1   0   1000 |
|  0   1     0  |
|  0   0     1  |
```

For chuck-mcp v2 with block stock 280 × 400 mm:

```svg
<svg width="280mm" height="400mm" viewBox="0 0 280 400">
  <g transform="scale(-1, 1) translate(-280, 0)">
    <!-- carve geometry, drawn in the "as-it-will-print" orientation -->
  </g>
</svg>
```

This produces a file where the geometry as stored matches the carved
mirror, and a CAM tool reading the SVG without applying the transform
itself will see the correct toolpath input.

## Where the flip fits in the pipeline

```
SOURCE IMAGE
  ↓ (S6 vectorization → polygons in source coordinate frame)
POLYGONS_IN_PRINT_FRAME
  ↓ (S7 export: horizontal_flip)            ← THIS STEP
POLYGONS_IN_CARVE_FRAME
  ↓ (tool radius compensation, Minkowski offset)
TOOLPATH_POLYGONS
  ↓ (SVG export with metadata)
SVG_FOR_CAM
  ↓ (VCarve → ShopBot SBP)
PHYSICAL CARVE
```

**Critical**: tool compensation runs on **flipped** geometry, not on
original. Because the bit cuts on the right side of the path for an
outside contour by convention, and "right side" reverses under flip,
the flip-before-compensate order avoids sign errors.

## Python implementation (chuck-mcp side)

```python
# chuck-mcp/exporters/mirror_flip.py
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform

def mirror_polygons(polys: list[Polygon], block_width_mm: float) -> list[Polygon]:
    """Horizontally flip polygons about the block's vertical centerline.

    The block coordinate system is:
      X: 0 to block_width_mm, increasing to the right (image-space)
      Y: 0 to block_height_mm, increasing downward
    After flip, an image feature at X=10 ends up at X=block_width-10.

    Args:
        polys: list of shapely Polygon objects in image-space
        block_width_mm: outer block dimension along X axis

    Returns:
        list of Polygon objects in carve-space (mirrored)
    """
    # affine_transform takes (a, b, d, e, xoff, yoff) for matrix
    #   [a b xoff]
    #   [d e yoff]
    # We want X' = -X + block_width, Y' = Y
    # => a=-1, b=0, xoff=block_width, d=0, e=1, yoff=0
    matrix = (-1.0, 0.0, 0.0, 1.0, block_width_mm, 0.0)
    return [affine_transform(p, matrix) for p in polys]


def emit_svg_with_flip(polys: list[Polygon],
                       block_width_mm: float,
                       block_height_mm: float,
                       block_id: int) -> str:
    """Emit chuck-mcp v2 block SVG with mirror flip and metadata."""

    flipped = mirror_polygons(polys, block_width_mm)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{block_width_mm}mm" height="{block_height_mm}mm" '
        f'viewBox="0 0 {block_width_mm} {block_height_mm}">',
        f'  <metadata>',
        f'    <chuck-mcp:block xmlns:chuck-mcp="https://chuck-mcp/v2"',
        f'      id="{block_id}"',
        f'      orientation="carve-frame"  ',
        f'      flip-applied="true"',
        f'      print-frame-transform="scale(-1, 1) translate(-{block_width_mm}, 0)"/>',
        f'  </metadata>',
    ]

    for poly in flipped:
        d = _polygon_to_svg_path(poly)
        lines.append(f'  <path d="{d}" fill="black" stroke="none"/>')

    lines.append('</svg>')
    return "\n".join(lines)


def _polygon_to_svg_path(poly: Polygon) -> str:
    """Convert a shapely Polygon to an SVG path 'd' attribute."""
    parts = []
    # Exterior
    coords = list(poly.exterior.coords)
    parts.append(f'M {coords[0][0]:.3f} {coords[0][1]:.3f}')
    for x, y in coords[1:]:
        parts.append(f'L {x:.3f} {y:.3f}')
    parts.append('Z')
    # Interior rings (holes)
    for ring in poly.interiors:
        coords = list(ring.coords)
        parts.append(f'M {coords[0][0]:.3f} {coords[0][1]:.3f}')
        for x, y in coords[1:]:
            parts.append(f'L {x:.3f} {y:.3f}')
        parts.append('Z')
    return ' '.join(parts)
```

## Test cases (regression suite)

The flip is a one-liner, but it's load-bearing. Tests:

```python
def test_flip_preserves_y():
    """Y coordinates unchanged under horizontal flip."""
    poly = Polygon([(10, 20), (30, 20), (30, 40), (10, 40)])
    flipped = mirror_polygons([poly], block_width_mm=100)[0]
    assert flipped.bounds[1] == 20.0  # ymin
    assert flipped.bounds[3] == 40.0  # ymax

def test_flip_negates_x():
    """X coordinates reflect about block centerline."""
    poly = Polygon([(10, 0), (30, 0), (30, 10), (10, 10)])
    flipped = mirror_polygons([poly], block_width_mm=100)[0]
    # X=10 → X=90, X=30 → X=70
    assert flipped.bounds[0] == 70.0  # xmin
    assert flipped.bounds[2] == 90.0  # xmax

def test_flip_is_involutive():
    """Applying flip twice = identity."""
    poly = Polygon([(10, 20), (30, 40), (50, 60)])
    flipped_once = mirror_polygons([poly], block_width_mm=100)[0]
    flipped_twice = mirror_polygons([flipped_once], block_width_mm=100)[0]
    assert flipped_twice.equals(poly)

def test_flip_preserves_topology():
    """Inner holes preserved under flip."""
    outer = [(10, 10), (90, 10), (90, 90), (10, 90)]
    inner = [(40, 40), (60, 40), (60, 60), (40, 60)]
    poly = Polygon(outer, [inner])
    flipped = mirror_polygons([poly], block_width_mm=100)[0]
    assert len(flipped.interiors) == 1
```

## A common bug to avoid: double-flip

If both chuck-mcp and a downstream CAM tool apply mirror, the final
carve is back-to-original orientation. This produces a print that's
mirror of the source image — visually similar to a "wrong" carve but
caught only when comparing to source-image reference.

**Mitigation**: bake the flip into the SVG geometry **and** write
`flip-applied="true"` into the SVG metadata. Downstream CAM tools that
respect chuck-mcp metadata will skip a second flip. Tools that don't
will require user vigilance.

Also: include a **fiducial** in the SVG metadata showing what "up" means
in the carve-frame:

```svg
<metadata>
  <chuck-mcp:fiducial>
    <chuck-mcp:point x="280" y="0" label="print-frame-(0,0)"/>
    <chuck-mcp:point x="0" y="0" label="carve-frame-(0,0)"/>
    <chuck-mcp:note>If you carve this file as-is and print on paper,
      the result will match the source image. If you see mirrored text
      on the print, the SVG was flipped twice.</chuck-mcp:note>
  </chuck-mcp:fiducial>
</metadata>
```

## Why not just flip the source image?

In principle, flipping the source image before vectorization gives the
same result. Two reasons not to:

1. **Solver convenience**: chuck-mcp's S5 solver wants to compare its
   forward render directly against the source image. Both are in
   "print-frame". Flipping the source means flipping the comparison
   constantly.

2. **Metadata clarity**: keeping the carve-frame flip explicit at S7
   means the SVG file's metadata can declare it. A flipped source image
   loses this information.

3. **Vector vs raster**: flipping a Polygon is exact; flipping a raster
   image introduces resampling artifacts unless done at the pixel
   boundary.

## Citations

- W3C SVG 1.1 coordinates spec: https://www.w3.org/TR/SVG11/coords.html
- MDN SVG transform: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
- gcodetools (Inkscape Python extension): https://github.com/cnc-club/gcodetools
- Andy Farkas kento (print-frame discussion): https://www.fablewood.com/cutting-a-kento-registration/
