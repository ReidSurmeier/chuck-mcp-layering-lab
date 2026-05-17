---
title: "ShopBot SBP Post-Processor Reference for chuck-mcp v2"
sources:
  - "https://shopbottools.com/wp-content/uploads/2024/01/SBG00253150707CommandRefV3.pdf"
  - "https://www.opensbp.com/files/QuickReference_bothpages.pdf"
  - "https://shopbottools.com/wp-content/uploads/2024/01/SBG00314150707ProgHandWin.pdf"
  - "https://forum.vectric.com/viewtopic.php?f=4&t=14783"
  - "https://docs.vectric.com/docs/V10.0/VCarvePro/ENU/Help/form/post-processor-editing/"
  - "https://shopbottools.com/2008/01/09/customizing-vcarve-pro-and-partworks/"
  - "https://www.opensbp.com/"
relevance: "MUST-READ — ShopBot uses .sbp files (OpenSBP language), NOT G-code. Any chuck-mcp 'gcode export' is misnamed. The correct V1 implementation is: SVG → VCarve Pro/Aspire → ShopBot post → .sbp file. Documents the SBP command syntax for the day chuck-mcp wants to skip VCarve and emit SBP directly."
tags: [shopbot, sbp, post-processor, vcarve, opensbp, gcode, file-format]
---

# ShopBot SBP Post-Processor Reference

## The format question — answered

**ShopBot's PRSalpha (RISD's CNC) reads `.sbp` files, NOT `.nc` / `.gcode` /
`.tap` / G-code.**

This is the most consequential implementation detail in this domain. Two
people researching chuck-mcp v2 will independently make the wrong assumption
that the CNC eats G-code; it doesn't. The output format is **OpenSBP**, a
ShopBot-specific text language. The conversion from G-code-style toolpath
intent to OpenSBP must happen somewhere.

The three viable pipelines:

### Pipeline A (production-ready, recommended for V1)
```
chuck-mcp → SVG (with metadata) → Vectric VCarve Pro → ShopBot post → .sbp
                                          ↓
                              user assigns tools + DOC manually in VCarve
```
- **Pros**: VCarve handles tool compensation, multi-pass strategy,
  G-code-style optimization, post-processor selection. Industry standard
  at RISD-tier shops. ShopBot ships with VCarve in many bundles.
- **Cons**: requires per-block manual VCarve session. 27 blocks = 27 setups.
  Partially automatable via VCarve's batch tools and gadget scripting.

### Pipeline B (semi-automated, V1.5)
```
chuck-mcp → SVG → automated VCarve project generator (Python or VCarve
                  gadgets in JavaScript) → batch-export .sbp per block
```
- **Pros**: scales to 27 blocks. Same tool-comp engine as Pipeline A.
- **Cons**: VCarve's automation API is limited; VCarve Pro Gadgets
  (JS-based) work but are clunky. Likely 1-2 weeks of integration work.

### Pipeline C (direct SBP emit, V2 stretch)
```
chuck-mcp → SVG → custom Python emitter → .sbp directly
```
- **Pros**: full automation, fully versionable, no VCarve dependency.
- **Cons**: must implement tool compensation, multi-pass roughing, climb-
  vs-conventional logic, lead-in / lead-out from scratch. 4-6 weeks of
  work plus QA against physical carves.

**V1 recommendation: Pipeline A.** Document the workflow. If 27 manual
VCarve sessions turn out to be the bottleneck, escalate to Pipeline B.

## SBP language essentials

OpenSBP is a text format. One command per line, comma-separated parameters.
Lines starting with `'` are comments. Selected core commands:

### Movement commands

| Cmd | Meaning | Example |
|---|---|---|
| `M2,x,y` | Move 2-axis (XY) to absolute position at cut feedrate | `M2, 5.0, 3.2` |
| `M3,x,y,z` | Move 3-axis to absolute XYZ at cut feedrate | `M3, 5.0, 3.2, -0.062` |
| `J2,x,y` | Jog 2-axis (rapid) | `J2, 0, 0` |
| `J3,x,y,z` | Jog 3-axis (rapid) | `J3, 0, 0, 0.25` |
| `MX,d` `MY,d` `MZ,d` | Move single axis to absolute position | `MZ, -0.062` |
| `JZ,d` | Jog single axis (rapid) | `JZ, 0.25` |

### Speed commands

| Cmd | Meaning | Example |
|---|---|---|
| `MS,xy,z` | Set cut speeds for XY and Z axes (in/sec for PRSalpha) | `MS, 2.0, 0.5` |
| `JS,xy,z` | Set jog speeds (rapid travel) | `JS, 6.0, 1.5` |

PRSalpha default cut MS = 2.0 in/sec (= 120 IPM), Z = 0.5 in/sec (= 30 IPM).

### Spindle / tool

| Cmd | Meaning |
|---|---|
| `SO,1,1` | Spindle on (output 1) |
| `SO,1,0` | Spindle off |
| `C9` | "Toggle spindle" macro (depends on shop config) |
| `&Tool = 1` | Set tool number variable |
| `C6` | Pause for manual tool change (macro) |

### File structure

```
'  chuck-mcp v2 — block 03 of 27 — relief carve
'  Date: 2026-05-16
'  Tool: 1/8" up-spiral, bit #2

SA          ' Set absolute mode
CN, 90      ' Use VCarve default file naming
JZ, 0.5     ' Lift Z to safe height (0.5")
MS, 2.0, 0.5  ' Cut speed: XY = 2 in/sec, Z = 0.5 in/sec
JS, 6.0, 1.5  ' Jog speed: XY = 6 in/sec, Z = 1.5 in/sec

C6          ' Pause for tool change to bit #2
SO, 1, 1    ' Spindle on
PAUSE 4     ' Spindle spin-up

'  Pass 1 — pocket roughing
J3, 0.0, 0.0, 0.25
M3, 0.0, 0.0, 0.0
M3, 0.0, 0.0, -0.049    ' First pass depth = -1.25 mm
M2, 2.5, 0.0
M2, 2.5, 1.0
M2, 0.0, 1.0
M2, 0.0, 0.0
'  ... (additional passes)

JZ, 0.5     ' Lift to safe height
SO, 1, 0    ' Spindle off
END
```

### Critical operator commands (always at top)

```
SA       ' Absolute positioning mode
&Tool=1  ' Initialize tool counter
JZ, 0.5  ' Always lift to safe Z first
JS, 6, 1.5    ' Jog speeds
MS, 2, 0.5    ' Move (cut) speeds
```

### Units

PRSalpha typically runs in **inches** at RISD/US shops, but supports
**mm** via `VU, mm` directive. **chuck-mcp v2 should standardize on mm**
(everything else in the project is metric). Add at top:

```
VU, mm
```

Note: depths in the example above are in inches (`-0.049` = -1.25mm). For
mm-mode files, write `-1.25` directly.

## VCarve Pro ShopBot post-processor names

When opening VCarve and saving the toolpath, select one of:

- **`ShopBot TC (mm) (*.sbp)`** — Tool Change variant, mm units — *RECOMMENDED*
- `ShopBot (mm) (*.sbp)` — single-tool variant, mm units
- `ShopBot Arcs (mm) (*.sbp)` — preserves arcs as native circular moves
  (smaller files, sometimes smoother on circles)
- `ShopBot TC (inch)` — inch version of TC

The `TC` (Tool Change) variants emit a `C6` pause command at every tool
boundary, which is what you want for the 4-bit pipeline.

The post-processors live at:
```
C:\sbparts\VCarvePro_forShopBotPosts\        (default install)
C:\ProgramData\Vectric\VCarve Pro\PostP\     (alternate)
```

If the ShopBot post isn't in VCarve's post-processor list, copy the `.pp`
files from `sbparts` to the VCarve `PostP` directory.

## Pipeline C — custom SBP emitter sketch

For when (not if) chuck-mcp wants to skip VCarve:

```python
# chuck-mcp/exporters/sbp.py
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon

@dataclass
class ToolParams:
    diameter_mm: float
    cut_feed_mm_s: float       # MS xy
    plunge_feed_mm_s: float    # MS z
    jog_feed_mm_s: float       # JS xy
    spindle_rpm: int
    doc_mm: float              # depth-of-cut per pass
    stepover_frac: float       # 0.0 - 1.0
    tool_number: int

def emit_sbp(block_id: int,
             relief_polys: list[Polygon],
             jig_polys: list[Polygon],
             tools: dict[str, ToolParams],
             carve_depth_mm: float = 2.5,
             safe_z_mm: float = 6.0) -> str:
    """Emit a complete .sbp file for one chuck-mcp block."""

    lines = []
    lines.append(f"'  chuck-mcp v2 block {block_id:03d}")
    lines.append("VU, mm")
    lines.append("SA")
    lines.append(f"JZ, {safe_z_mm}")

    # Pass 1: roughing
    rough = tools['1/4_compression']
    lines.append(f"&Tool = {rough.tool_number}")
    lines.append("C6")  # tool change pause
    lines.append(f"MS, {rough.cut_feed_mm_s}, {rough.plunge_feed_mm_s}")
    lines.append(f"JS, {rough.jog_feed_mm_s}, {rough.plunge_feed_mm_s * 3}")
    lines.append("SO, 1, 1")
    lines.append("PAUSE 4")
    for poly in offset_polygons(relief_polys, -rough.diameter_mm/2):
        lines.extend(emit_pocket_passes(poly, rough, carve_depth_mm))
    lines.append(f"JZ, {safe_z_mm}")
    lines.append("SO, 1, 0")

    # Passes 2-4 — analogous, with progressively smaller tools

    lines.append("END")
    return "\n".join(lines)

def emit_pocket_passes(poly: Polygon, tool: ToolParams, total_depth: float) -> list[str]:
    """Multi-pass spiral pocket fill of one polygon."""
    lines = []
    n_passes = ceil(total_depth / tool.doc_mm)
    depth_per_pass = total_depth / n_passes
    for i in range(n_passes):
        z = -depth_per_pass * (i + 1)
        # Spiral inward from polygon boundary
        for ring in spiral_offset_rings(poly, tool.diameter_mm * tool.stepover_frac):
            x0, y0 = ring.coords[0]
            lines.append(f"J3, {x0:.3f}, {y0:.3f}, {z + 1.0:.3f}")
            lines.append(f"M3, {x0:.3f}, {y0:.3f}, {z:.3f}")
            for x, y in ring.coords[1:]:
                lines.append(f"M2, {x:.3f}, {y:.3f}")
    return lines

def offset_polygons(polys, distance):
    """Minkowski offset for tool compensation."""
    return [p.buffer(distance, cap_style=2, join_style=2) for p in polys]

def spiral_offset_rings(poly, stepover):
    """Generate inward-spiral rings for pocket toolpath."""
    rings = []
    current = poly
    while current.area > 0:
        rings.append(current.exterior)
        current = current.buffer(-stepover, cap_style=2, join_style=2)
        if current.is_empty:
            break
    return rings
```

This is ~150 LOC of skeleton; full Pipeline C implementation runs ~600-800
LOC depending on lead-in/lead-out sophistication.

## Practical risks with direct SBP emission

1. **No lead-in / lead-out**: VCarve plunges with ramping or helical lead-
   in to protect the bit. Custom emitter needs the same.
2. **No collision avoidance**: VCarve checks tool against fixture clamps
   and stock surface. Custom needs explicit safe-Z management.
3. **No arc-fitting**: VCarve emits arcs as `CG` (curve, G-code-like) for
   smoother motion. Plain `M2` line segments work but produce visible
   facets on curved features.
4. **No feedrate override**: VCarve adjusts feed at corners (corner
   slowdown). Custom emitter sees fast feed → corner overshoot.

For V1: **use Pipeline A.** Custom SBP emit is interesting but premature.

## Citations

- ShopBot OpenSBP command reference v3:
  https://shopbottools.com/wp-content/uploads/2024/01/SBG00253150707CommandRefV3.pdf
- OpenSBP project: https://www.opensbp.com/
- OpenSBP quick reference card:
  https://www.opensbp.com/files/QuickReference_bothpages.pdf
- ShopBot Part File Programming Handbook:
  https://shopbottools.com/wp-content/uploads/2024/01/SBG00314150707ProgHandWin.pdf
- VCarve Pro ShopBot post-processor forum:
  https://forum.vectric.com/viewtopic.php?f=4&t=14783
- Vectric post-processor editing docs:
  https://docs.vectric.com/docs/V10.0/VCarvePro/ENU/Help/form/post-processor-editing/
