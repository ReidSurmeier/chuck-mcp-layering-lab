---
title: "SVG to G-code / SBP Pipelines for chuck-mcp v2"
sources:
  - "https://github.com/cnc-club/gcodetools"
  - "https://github.com/drandrewthomas/gcodercnc2d5"
  - "https://gcodercnc.com/"
  - "https://docs.vectric.com/docs/V10.0/VCarvePro/ENU/Help/form/post-processor-editing/"
  - "https://www.scan2cad.com/blog/cad/convert-svg-gcode/"
  - "https://cncphilosophy.com/inkscape-g-code-tutorial/"
  - "https://forums.maslowcnc.com/t/inkscape-for-gcode-generation-quick-instructions-to-get-started/12049"
  - "https://shopbottools.com/training/tutorials/"
relevance: "MUST-READ — surveys the 5 candidate pipelines from chuck-mcp's SVG to ShopBot SBP, and picks the V1 winner. Vectric VCarve Pro wins for V1 production; gcodetools (Inkscape) is the open-source fallback; Carbide Create is non-starter (Carbide 3D-only); Easel is non-starter (Inventables-only)."
tags: [svg, gcode, sbp, vcarve, inkscape, gcodetools, pipeline, cam]
---

# SVG → G-code / SBP Pipelines

## The five candidates

| Pipeline | Output | Cost | Automation | Verdict |
|---|---|---|---|---|
| **Vectric VCarve Pro** | .sbp + .nc + .tap | $700 (lifetime) | Limited gadgets | **V1 winner** |
| **Vectric Aspire** | .sbp + .nc + .tap | $1995 (lifetime) | Same as VCarve + 3D modeling | Overkill for v2 |
| **Inkscape gcodetools** | G-code only | Free / OSS | Full (Python) | **OSS fallback** |
| **GCoderCNC web app** | G-code only | Free | Limited (web UI) | Hobbyist tier |
| **Carbide Create** | .nc only | Free | Limited | Wrong CAM target (Carbide 3D) |
| **Easel** | proprietary | $19/mo | Cloud-locked | Wrong CAM target (Inventables) |
| **Custom Python emitter** | .sbp directly | dev time | Full | **V2 stretch goal** |

## Pipeline A — Vectric VCarve Pro (V1)

**Recommended for production. Already known at RISD.**

### Workflow

```
1. chuck-mcp exports SVG (per-block, with metadata)
2. User opens VCarve Pro, "Import Vector File" → select SVG
3. VCarve imports the SVG paths as vectors at their stored coordinates
4. User selects "Material Setup" → maple plywood, 12 mm thick
5. User assigns toolpath strategy per vector:
   - Outer roughing region → "Pocket Toolpath" with 1/4" compression
   - Bulk relief → "Pocket Toolpath" with 1/8" up-spiral
   - Detail boundary → "Profile Toolpath" with 1/16" up-spiral
   - Kento bevel (jig only) → "V-Carve Toolpath" with 30° V-bit
6. User saves toolpath, post-processes with "ShopBot TC (mm) (*.sbp)"
7. .sbp file copied to ShopBot via USB or network drive
8. Repeat for blocks 2-27
```

### Pros
- Industry-standard CAM. Well-supported by ShopBot.
- VCarve handles tool compensation, lead-in/out, multi-pass strategy
- Tool library is editable — chuck-mcp can ship a `chuck-mcp-v2.tdb`
  tool database file that defines bits #1-4 with correct feeds/speeds
- Manual review at every block catches solver bugs before they ruin
  wood (worth ~$2 of material per saved error)
- Multiple format outputs (.sbp, .nc, .tap) — chuck-mcp v2 outputs
  generic SVG that works for any future CAM

### Cons
- 27 blocks × 5 min/setup = 2+ hours of click-work per edition
- Manual step → human error vector
- License is per-machine; RISD shop has one, home shop needs separate

### Automation hook: VCarve Gadgets

VCarve has a JavaScript-based "Gadget" API. chuck-mcp can ship a gadget
that:
1. Takes a directory of SVG files (27 files)
2. For each: opens, applies a saved toolpath template, saves .sbp
3. Outputs all 27 .sbp files into a target directory

Effort: 1-2 weeks of development. Worth it if the project does multiple
27-block editions.

Reference: https://shopbottools.com/2008/01/09/customizing-vcarve-pro-and-partworks/

## Pipeline B — Inkscape gcodetools (open-source fallback)

**Recommended as a fully-OSS pipeline if VCarve cost is a blocker.**

Repository: `cnc-club/gcodetools` (296 stars, Python, active)

### Workflow

```
1. chuck-mcp exports SVG per block (with chuck-mcp metadata)
2. Open SVG in Inkscape
3. Extensions → Gcodetools → Tools Library
   - Define tools: 1/4" cylindrical, 1/8" cylindrical, 1/16" cylindrical,
     30° V-bit
4. Extensions → Gcodetools → Orientation Points
   - Define machine origin (0, 0) and Z-zero
5. Extensions → Gcodetools → Path to Gcode
   - Output G-code file
6. SBP-convert step (chuck-mcp-side, see below): G-code → .sbp
```

### Pros
- Free, open-source, Python-scriptable
- Can run headless via Inkscape command-line (`inkscape --extension`)
- gcodetools is mature (2014+), supports tool radius offset, profile,
  pocket, lathe modes
- Output is **G-code**, not SBP — so a final G-code→SBP conversion is
  needed (trivial: see `web_shopbot_sbp_postprocessor.md`)

### Cons
- gcodetools documentation is sparse (per multiple forum complaints)
- Less robust than VCarve for complex multi-pass strategies
- No native ShopBot .sbp output — chuck-mcp must convert G-code to SBP

### G-code → SBP conversion sketch

```python
# chuck-mcp/exporters/gcode_to_sbp.py
import re

def convert_gcode_to_sbp(gcode_text: str) -> str:
    """Convert standard G-code to OpenSBP."""
    out = ["VU, mm", "SA"]
    current_z = None
    for line in gcode_text.splitlines():
        line = line.strip().upper()
        if line.startswith(";") or line.startswith("("):
            out.append(f"'  {line}")
            continue
        # G0 = rapid (jog), G1 = cut feed
        # G0 X Y Z → J3 or M3
        # G1 X Y Z → M3 or M2
        m = re.match(r"G0?(\d+)\s+(.*)", line)
        if not m:
            continue
        g_num = int(m.group(1))
        params = dict(re.findall(r"([XYZ])([-+]?[0-9.]+)", m.group(2)))
        x = params.get('X')
        y = params.get('Y')
        z = params.get('Z')
        cmd_prefix = "J" if g_num == 0 else "M"
        if x is not None and y is not None and z is not None:
            out.append(f"{cmd_prefix}3, {x}, {y}, {z}")
        elif x is not None and y is not None:
            out.append(f"{cmd_prefix}2, {x}, {y}")
        elif z is not None:
            out.append(f"{cmd_prefix}Z, {z}")
    out.append("END")
    return "\n".join(out)
```

This is the **minimum viable** converter. Real production needs:
- Spindle on/off (M3, M5) → SO commands
- Tool change (M6) → C6 macro
- Feedrate (F) → MS commands
- Arc commands (G2, G3) → CG commands

Estimated full converter: 200-300 LOC.

## Pipeline C — Custom Python SBP Emitter (V2 stretch)

See `web_shopbot_sbp_postprocessor.md` for full sketch.

**Not recommended for V1.** Reasons:
1. Tool compensation done in chuck-mcp code = bug risk
2. Lead-in / lead-out / corner slowdown = nontrivial CAM
3. VCarve's 30 years of CAM polish is hard to replicate

Defer until: (a) doing 100+ blocks/year, (b) VCarve workflow proven
inadequate, (c) team has dedicated CAM dev cycles.

## Pipeline D — GCoderCNC web app (out of scope)

Free web tool at https://gcodercnc.com/. Browser-based, takes SVG, emits
G-code. Lower fidelity than gcodetools. Worth knowing exists for
prototyping a single block, not for chuck-mcp v2 production.

## Pipeline E — Carbide Create / Easel (vendor-locked, out of scope)

Both are CAM tools tied to specific CNC hardware brands. Won't drive a
ShopBot. Useful as references for "what a polished SVG-to-CAM UX looks
like" but not production options.

## SVG dialect for chuck-mcp v2

For Pipeline A (VCarve) and Pipeline B (gcodetools) compatibility, the
chuck-mcp v2 SVG export should follow these conventions:

### Color-coded layer semantics
Borrow from Draw2Cut (per existing chuck-mcp vectorization research):

| Stroke color | Meaning | CAM action |
|---|---|---|
| `#000000` (black) | Carve area (relief) | Pocket toolpath |
| `#FF0000` (red) | Outer boundary | Profile toolpath |
| `#0000FF` (blue) | Kento mark | Profile + V-carve |
| `#00FF00` (green) | Block edge / jig pocket | Profile outside, no offset |
| `#FFFF00` (yellow) | Optional V-carve detail | V-carve toolpath |

VCarve respects vector colors at import — user can right-click a color
and assign a saved toolpath template.

### Layer naming
- `layer-0-relief` — main relief carve geometry
- `layer-1-boundary` — outer boundary of relief area
- `layer-2-kento` — registration marks (on jig SVG; absent on block SVGs)
- `layer-3-edge` — block outer edge for jig fit (on jig SVG only)
- `layer-4-engraving` — block ID label (engrave 0.5 mm deep)

### Metadata

```xml
<metadata>
  <chuck-mcp:block xmlns:chuck-mcp="https://chuck-mcp/v2"
    schema-version="1.0"
    block-id="03"
    edition-size="27"
    paper-size="B4"
    paper-width-mm="257"
    paper-height-mm="364"
    block-width-mm="280"
    block-height-mm="400"
    block-thickness-mm="12"
    carve-depth-mm="2.5"
    end-mill-min-diameter-mm="1.59"
    flip-applied="true"
    grain-direction-degrees="0"
    pass-strategy="rough-bulk-detail"
    chuck-mcp-version="2.0.0"/>
</metadata>
```

VCarve will ignore the metadata but it's preserved in the file for the
chuck-mcp side preview / verification tooling.

## Practical V1 decision

**Pipeline A (Vectric VCarve Pro)** is the V1 choice because:
1. RISD already has VCarve licenses on the ShopBot computer
2. Maple plywood + ShopBot is VCarve's home turf
3. Manual per-block review is **good** for an edition of 10 — catches
   solver bugs before $5 of maple is destroyed
4. ~2 hours of click-work per 27-block edition is acceptable for the
   target use case (Reid's RISD thesis-scale project, not a print farm)

If/when this changes, **Pipeline B (gcodetools)** becomes the OSS path
of least resistance, **then** Pipeline C (custom emitter) when volume
justifies the dev cost.

## Citations

- gcodetools: https://github.com/cnc-club/gcodetools
- gcodercnc2d5: https://github.com/drandrewthomas/gcodercnc2d5
- GCoderCNC web app: https://gcodercnc.com/
- Vectric VCarve docs: https://docs.vectric.com/docs/V10.0/VCarvePro/ENU/Help/form/post-processor-editing/
- Scan2CAD svg-to-gcode guide: https://www.scan2cad.com/blog/cad/convert-svg-gcode/
- Inkscape G-code tutorial: https://cncphilosophy.com/inkscape-g-code-tutorial/
- Maslow CNC Inkscape guide: https://forums.maslowcnc.com/t/inkscape-for-gcode-generation-quick-instructions-to-get-started/12049
- ShopBot Tutorials: https://shopbottools.com/training/tutorials/
