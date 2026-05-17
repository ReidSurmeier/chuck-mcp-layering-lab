# Swatch Plate SVG Assets — Concrete Implementation Spec

This is the SVG that chuck-mcp v2 ships in `assets/calibration_plate_v1.svg`. Reid can CNC-carve this directly (cnc-woodblock-tools pipeline) or hand-carve following the printed template.

## Final dimensions (settled)

- Plate: **240mm × 360mm** (fits Iwano kozo A3 with margins)
- Swatches: **18mm × 18mm** with **3mm gutter**
- Grid: **15 rows × 7 columns = 105 swatches** per plate
- ColorChecker reservation area: **90mm × 60mm** top center
- ArUco fiducials: **20mm × 20mm** at all 4 corners (10mm from edge)
- Kento marks: bottom-left L-shape + bottom-right straight bar (standard mokuhanga)
- Concentration labels: **3mm** font, top of each column
- Pigment ID labels: **3mm** font, left of each row

## SVG file (production-ready)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="240mm" height="360mm"
     viewBox="0 0 240 360"
     style="font-family: sans-serif">

  <!-- =================================================================== -->
  <!-- Background: white plate                                              -->
  <!-- =================================================================== -->
  <rect x="0" y="0" width="240" height="360"
        fill="white" stroke="black" stroke-width="0.2"/>

  <!-- =================================================================== -->
  <!-- ArUco fiducial markers (rendered as PNG references at carve time)   -->
  <!-- IDs 0-3 are corner anchors. Generated from cv2.aruco.DICT_4X4_50.   -->
  <!-- =================================================================== -->
  <g id="aruco-markers">
    <image x="10"  y="10"  width="20" height="20"
           xlink:href="aruco_4x4_50_id0.png"/>
    <image x="210" y="10"  width="20" height="20"
           xlink:href="aruco_4x4_50_id1.png"/>
    <image x="10"  y="330" width="20" height="20"
           xlink:href="aruco_4x4_50_id2.png"/>
    <image x="210" y="330" width="20" height="20"
           xlink:href="aruco_4x4_50_id3.png"/>
  </g>

  <!-- =================================================================== -->
  <!-- ColorChecker reservation (carved as blank, ColorChecker laid on top) -->
  <!-- =================================================================== -->
  <g id="cc-reservation">
    <rect x="75" y="15" width="90" height="60"
          fill="none" stroke="black" stroke-width="0.3"
          stroke-dasharray="2,1"/>
    <text x="120" y="48" text-anchor="middle"
          font-size="4" fill="#555">PLACE COLORCHECKER HERE</text>
    <text x="120" y="55" text-anchor="middle"
          font-size="3" fill="#888">(not carved)</text>
  </g>

  <!-- =================================================================== -->
  <!-- Column labels (concentration ladder)                                 -->
  <!-- Origin x of leftmost column = 48mm. Column pitch = 21mm.            -->
  <!-- =================================================================== -->
  <g id="col-labels" font-size="3" text-anchor="middle">
    <text x="57"  y="86">3%</text>
    <text x="78"  y="86">6%</text>
    <text x="99"  y="86">12%</text>
    <text x="120" y="86">25%</text>
    <text x="141" y="86">50%</text>
    <text x="162" y="86">75%</text>
    <text x="183" y="86">100%</text>
  </g>

  <!-- =================================================================== -->
  <!-- Row labels (pigment IDs - placeholders, customized per plate)       -->
  <!-- Origin y of top row swatch = 90mm. Row pitch = 21mm.                -->
  <!-- =================================================================== -->
  <g id="row-labels" font-size="3" text-anchor="end">
    <text x="45" y="100">P01</text>
    <text x="45" y="121">P02</text>
    <text x="45" y="142">P03</text>
    <text x="45" y="163">P04</text>
    <text x="45" y="184">P05</text>
    <text x="45" y="205">P06</text>
    <text x="45" y="226">P07</text>
    <text x="45" y="247">P08</text>
    <text x="45" y="268">P09</text>
    <text x="45" y="289">P10</text>
    <text x="45" y="310">P11</text>
    <text x="45" y="331">P12</text>
    <!-- Three more rows on a wider plate (see below) -->
  </g>

  <!-- =================================================================== -->
  <!-- Swatch grid: 15 rows × 7 columns                                     -->
  <!-- Each swatch represents a stripe that gets carved into the block.    -->
  <!-- At carve time, these are solid relief (raised) areas the brush inks. -->
  <!-- The renderer generates this programmatically; example for row 0:    -->
  <!-- =================================================================== -->
  <g id="swatch-grid" fill="black">
    <!-- Row 0 (P01) - shown explicitly for first row -->
    <rect x="48"  y="90"  width="18" height="18"/>
    <rect x="69"  y="90"  width="18" height="18"/>
    <rect x="90"  y="90"  width="18" height="18"/>
    <rect x="111" y="90"  width="18" height="18"/>
    <rect x="132" y="90"  width="18" height="18"/>
    <rect x="153" y="90"  width="18" height="18"/>
    <rect x="174" y="90"  width="18" height="18"/>

    <!-- Remaining 14 rows generated by template substitution at build time -->
    <!-- For y = 111, 132, 153, ..., 384 (overflows; see fit note below) -->
  </g>

  <!-- =================================================================== -->
  <!-- Kento registration marks (traditional mokuhanga)                     -->
  <!-- =================================================================== -->
  <g id="kento" fill="black">
    <!-- L-shape kento, bottom-left corner -->
    <path d="M 5,355 L 5,335 L 25,335 L 25,357 L 7,357 Z"
          fill="none" stroke="black" stroke-width="0.5"/>
    <!-- Straight bar kento, bottom-right -->
    <rect x="200" y="354" width="35" height="3" fill="black"/>
  </g>

  <!-- =================================================================== -->
  <!-- Version stamp (machine-readable text, carved into block)             -->
  <!-- =================================================================== -->
  <text x="120" y="355" text-anchor="middle" font-size="3" fill="#333">
    CHUCK-MCP CALIBRATION PLATE v1.0
  </text>
</svg>
```

## Fit problem and resolution

15 rows × 21mm pitch = 315mm needed for grid, but only 360 - 90 (top reserve) - 25 (bottom kento) = 245mm available. **The grid doesn't fit at 15 rows.**

Two production options:

### Option A: TWO plates (RECOMMENDED)

Plate 1: pigments P01-P10 (10 rows × 21mm = 210mm fits)
Plate 2: pigments P11-P20 (or more if Reid expands)

Each plate is 240mm × 360mm. Same fiducials, same code. The `pigment_row_on_plate` argument plus a `plate_id` argument distinguish them.

Total carving cost: 2 blocks instead of 1. But each block is half the carving time of trying to cram 15 rows into one.

### Option B: Compact layout (single plate)

Reduce row pitch to **15mm** (12mm swatch + 3mm gutter), keep 7 columns at 21mm pitch. Grid height = 15 × 15 = 225mm, fits in 245mm available.

12mm × 12mm swatch = 144 sq mm. At 10 px/mm in the warped image, that's ~120 × 120 = 14,400 pixels per swatch. Still plenty for median sampling.

Downside: tighter spacing makes hand-carving harder. And printing requires more precise inking — small swatches show pigment irregularity more.

### Option C: Wide plate (single plate, tall format)

Make plate 240mm × 600mm (uncommon paper size). Most washi sheets max at A3 (297×420). Rejected.

**Recommendation: Option A.** Two plates, 10 pigments per plate. Maximum Reid capacity = 20 pigments without re-carving. He has ~15-25, so this is right-sized.

## Programmatic generation of the SVG

```python
# chuck_mcp_v2/calibration/svg_generation.py

from pathlib import Path
from typing import Sequence
import cv2
import numpy as np

PLATE_W = 240
PLATE_H = 360
SWATCH = 18
GUTTER = 3
PITCH = SWATCH + GUTTER  # 21
GRID_X0 = 48
GRID_Y0 = 90
N_COLS = 7
N_ROWS = 10  # per Option A
ARUCO_SZ = 20


def generate_aruco_pngs(out_dir: Path, ids: Sequence[int] = (0, 1, 2, 3)):
    """Generate PNG files for ArUco markers, suitable for SVG embedding."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for aid in ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, aid, 400)
        cv2.imwrite(str(out_dir / f"aruco_4x4_50_id{aid}.png"), img)


def render_plate_svg(
    pigment_labels: Sequence[str],   # length N_ROWS
    concentration_labels: Sequence[str] = ("3%", "6%", "12%", "25%", "50%", "75%", "100%"),
    plate_id: str = "v1_plate_1",
    output_path: Path = Path("assets/calibration_plate_v1.svg"),
):
    assert len(pigment_labels) == N_ROWS

    svgs = ['<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{PLATE_W}mm" height="{PLATE_H}mm" '
            f'viewBox="0 0 {PLATE_W} {PLATE_H}" style="font-family: sans-serif">']

    # Background
    svgs.append(f'<rect x="0" y="0" width="{PLATE_W}" height="{PLATE_H}" '
                f'fill="white" stroke="black" stroke-width="0.2"/>')

    # ArUco markers
    for aid, (x, y) in [(0, (10, 10)), (1, (210, 10)), (2, (10, 330)), (3, (210, 330))]:
        svgs.append(f'<image x="{x}" y="{y}" width="{ARUCO_SZ}" height="{ARUCO_SZ}" '
                    f'xlink:href="aruco_4x4_50_id{aid}.png"/>')

    # ColorChecker reservation
    svgs.append('<rect x="75" y="15" width="90" height="60" fill="none" '
                'stroke="black" stroke-width="0.3" stroke-dasharray="2,1"/>')
    svgs.append('<text x="120" y="48" text-anchor="middle" font-size="4" fill="#555">'
                'PLACE COLORCHECKER HERE</text>')

    # Column labels
    for ci, label in enumerate(concentration_labels):
        cx = GRID_X0 + ci * PITCH + SWATCH / 2
        svgs.append(f'<text x="{cx}" y="86" text-anchor="middle" font-size="3">{label}</text>')

    # Row labels
    for ri, label in enumerate(pigment_labels):
        ry = GRID_Y0 + ri * PITCH + SWATCH / 2 + 1.5
        svgs.append(f'<text x="45" y="{ry}" text-anchor="end" font-size="3">{label}</text>')

    # Swatch grid
    for ri in range(N_ROWS):
        for ci in range(N_COLS):
            x = GRID_X0 + ci * PITCH
            y = GRID_Y0 + ri * PITCH
            svgs.append(f'<rect x="{x}" y="{y}" width="{SWATCH}" height="{SWATCH}" fill="black"/>')

    # Kento marks
    svgs.append('<path d="M 5,355 L 5,335 L 25,335" fill="none" stroke="black" stroke-width="0.5"/>')
    svgs.append('<rect x="200" y="354" width="35" height="3" fill="black"/>')

    # Version stamp
    svgs.append(f'<text x="120" y="355" text-anchor="middle" font-size="3" fill="#333">'
                f'CHUCK-MCP CAL {plate_id}</text>')

    svgs.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(svgs))
    return output_path
```

## CNC carving notes

This plate is ideal for the `cnc-woodblock-tools` pipeline. Recommendations:
- **End-mill: 1.5mm flat for swatches**, 0.5mm for fine ArUco features.
- **Cherry or shina ply**, ~20mm thick. Standard mokuhanga choice.
- **Carve depth: 1.5mm**. Deep enough for nori paste to not bridge, shallow enough for fine detail.
- **No "outline" carve**; just relief (carve AWAY everything outside the marked regions).

## Hand-carving notes

If Reid hand-carves:
- Print this SVG at 100% scale on paper, glue to the block face with rice paste.
- Use a fine V-gouge for ArUco edges (precision matters for detection).
- Swatches are easier — flat chisel or U-gouge.
- 1.5-2mm depth is fine.

## Asset checklist for chuck-mcp v2 repo

```
chuck-mcp-layering-lab/
  chuck_mcp_v2/
    calibration/
      svg_generation.py       # render_plate_svg function
      pipeline.py             # full pipeline
      mcp_tools.py            # MCP tool definitions
  assets/
    calibration_plate_v1.svg  # rendered for default 10 pigments
    aruco/
      aruco_4x4_50_id0.png   # 400×400 px
      aruco_4x4_50_id1.png
      aruco_4x4_50_id2.png
      aruco_4x4_50_id3.png
    plate_print_preview.png   # full-resolution preview for verification
    plate_carve_guide.pdf     # print-ready, scaled for hand-carving
  pigments/                   # populated by user via MCP tool
    _inventory.yaml
    gunjo_handmade_2026_05.yaml
    ...
```
