---
title: "End-Mill Selection for chuck-mcp v2 Relief Carving in Maple Plywood"
sources:
  - "https://www.cnccookbook.com/feeds-speeds-cnc-wood-cutting/"
  - "https://www.toolgrit.com/guides/cnc-wood-speeds-feeds-guide"
  - "https://twotrees3d.com/blogs/twotrees-blog/3d-carving-cnc-projects-how-to-carve-relief-art-with-your-cnc-router"
  - "https://idcwoodcraft.com/products/30-degree-extreme-detail-v-bit-bit-for-cnc-routers-1-4-shank"
  - "https://www.matterhackers.com/articles/end-mill-comparison-guide"
  - "https://www.harveyperformance.com/in-the-loupe/understanding-wood-properties-for-cnc-woodworking-projects/"
  - "https://woodweb.com/knowledge_base/CNC_Bit_Life_in_Baltic_Birch.html"
relevance: "MUST-READ — locks the V1 end-mill set: 1/4\" flat (roughing) + 1/8\" flat (bulk carve) + 1/16\" flat (detail) + 30° V-bit (inside corners). Specifies feeds and speeds for maple plywood. Defines depth-of-cut per pass and stepover for each tool. This is the bit set chuck-mcp v2 should target."
tags: [end-mill, feeds-speeds, maple, plywood, v-bit, ball-nose, multi-pass]
---

# End-Mill Selection for chuck-mcp v2 Relief Carving

## V1 recommended bit set

**Four-bit set, ordered roughing → finishing:**

| # | Bit | Diameter | Role | Min feature it produces |
|---|---|---|---|---|
| 1 | 1/4" flat compression spiral | 6.35 mm | Roughing — clear bulk waste outside the relief area | n/a (clears air) |
| 2 | 1/8" flat 2-flute up-spiral | 3.18 mm | Bulk relief carving + jig pocket | 3.2 mm |
| 3 | 1/16" flat 2-flute up-spiral | 1.59 mm | Fine relief detail + kento walls | 1.6 mm |
| 4 | 30° V-bit | 0 mm tip | Inside corners + kento bevel | depth-dependent |

All bits must be **solid carbide** — plywood glue lines destroy HSS
quickly (per WoodWeb knowledge base). 1/4" shank is the ShopBot standard.

## Why each bit

### #1 — 1/4" flat compression spiral (roughing)

- **Purpose**: clear large negative-space regions efficiently. On a chuck-
  mcp plate, this is "anything outside the carved relief area" — typically
  60-90% of the block.
- **Compression spiral** = upcut on the bottom, downcut on the top. Vital
  for plywood: prevents tear-out on **both** top veneer and bottom veneer.
- **Step-down**: 3-4 mm per pass in maple plywood (0.5 × diameter rule)
- **Stepover**: 60% (3.8 mm) — gets bulk waste cleared fast
- **Feed**: 100-140 IPM
- **Spindle**: 16,000-18,000 RPM
- **Chip load**: 0.003-0.005 IPT (2-flute → feed = 18000 × 2 × 0.004 = 144 IPM)

### #2 — 1/8" flat 2-flute up-spiral (bulk relief)

- **Purpose**: carve the actual relief area to depth (2.5 mm carved) at
  reasonable speed. This is the workhorse bit for chuck-mcp v2.
- **Up-spiral** is preferred over compression at this size because:
  - Compression spirals < 1/4" have weak cutting edges
  - Maple plywood top veneer is thinner than bit's compression flute
    geometry would help with
  - Up-spiral evacuates chips upward, important for fine features
- **Step-down**: 1.25 mm per pass (0.4 × diameter, conservative for hardness)
- **Stepover**: 40% (1.27 mm) for the body, 20% (0.64 mm) at boundaries
- **Feed**: 80-120 IPM
- **Spindle**: 16,000-18,000 RPM
- **Tool radius compensation**: offset polygon by **1.59 mm OUTWARD** for
  contours (cut on the outside of the line so the relief feature stays
  full size)

### #3 — 1/16" flat 2-flute up-spiral (detail)

- **Purpose**: clean up the boundary of the relief area to its true shape,
  produce sharp 1.6-mm-radius internal corners, carve kento walls.
- This is the bit that defines chuck-mcp's **minimum feature size = 1.6 mm**.
- **Step-down**: 0.5-0.8 mm per pass (0.5 × diameter, but cherry-conservative)
- **Stepover**: 20% for finishing pass (0.32 mm)
- **Feed**: 60-90 IPM (slower — small bit, lower deflection budget)
- **Spindle**: 18,000 RPM
- **Watch for**: tool deflection (1/16" bit will deflect 0.05-0.1 mm under
  full side-load). Mitigate by running a **second light spring pass** at
  0 depth-of-cut increment after the nominal final pass.

### #4 — 30° V-bit (inside corners + kento bevel)

- **Purpose**: produce internal corners sharper than the 0.8 mm radius
  that 1/16" flat leaves. Carve the 15-30° bevel on kento marks.
- **30°** chosen over 60° because:
  - Mokuhanga features rarely need extreme depth from a V-bit
  - 30° is the **detail** V-bit (per IDC Woodcraft: "extreme detail
    carving") — narrower kerf for tighter inside corners
  - 60° is sign-making territory; chuck-mcp is finer
- **Feed**: 40-60 IPM (V-bits are slower because cut width varies)
- **Depth**: dictated by feature width — `depth = width / (2·tan(15°)) = 1.866·width`
- **Spindle**: 18,000-22,000 RPM

### What about ball-nose bits?

**Not recommended for V1.** Ball-nose is for continuous 3D surfaces with
varying depth. chuck-mcp v2 produces **binary masks** (carved vs raised,
flat top + flat bottom). Flat bits + V-bits handle this 2.5D geometry
completely. Adding ball-nose introduces scallop-pattern surface artifacts
on the raised relief, which capture pigment unpredictably.

If a future v3 wants graduated relief depth (for, e.g., bokashi gradient
effects), ball-nose comes back. For v2: skip.

## Feeds and speeds table (maple plywood, ShopBot PRSalpha)

| Bit | DOC/pass | Stepover | Feed (IPM) | Spindle (RPM) | Chip load (IPT) |
|---|---|---|---|---|---|
| 1/4" compression | 3.0 mm | 60% (3.8 mm) | 120 | 18,000 | 0.004 |
| 1/8" up-spiral | 1.25 mm | 40% (1.27 mm) | 100 | 18,000 | 0.0028 |
| 1/16" up-spiral | 0.65 mm | 20% (0.32 mm) | 70 | 18,000 | 0.002 |
| 30° V-bit | depth-by-width | n/a | 50 | 20,000 | varies |

**IPM = inches per minute**, **IPT = inches per tooth**. ShopBot PRSalpha
top feed is ~10 IPM/sec acceleration; these feeds are well within its
operating envelope.

Notes:
- Maple plywood is harder than solid maple (the cross-grain plies stiffen
  the assembly). Add 10-15% margin to feeds and reduce DOC.
- **Burn marks = feed too slow** for the spindle speed. If you see brown
  on maple, INCREASE feed first (not decrease).
- **Chip-out on edges = climb-cut required** for finishing pass.
- **Glue-line dulling**: rebuild bit life expectation to 25-40% of solid-
  wood numbers. Plan tool changes between full editions, not within a
  single block.

## Multi-pass strategy (for one block)

```
PASS 1 — ROUGH OUTSIDE          | bit: 1/4" compression
  - Clears all material outside the "carved relief area" to full depth
  - 60% stepover, 3 mm step-down per pass
  - Leave 0.5 mm stock on inside boundaries for #2
  - Ends with: bulk waste gone, relief outline rough-defined

PASS 2 — BULK INSIDE THE RELIEF | bit: 1/8" up-spiral
  - Cuts the actual relief shape to within 0.2 mm of final
  - Climb direction on perimeter, conventional on pocketing
  - 0.5 mm stock left on boundary for #3
  - Ends with: relief shape fully roughed, boundary still oversize

PASS 3 — FINISH BOUNDARY        | bit: 1/16" up-spiral
  - Profile-cut along true boundary at full depth
  - Climb cut, no stock-left
  - Optional second pass at 0 depth increment as spring pass
  - Ends with: relief outline at exact size, 0.8mm radius inside corners

PASS 4 — INSIDE CORNERS         | bit: 30° V-bit  (OPTIONAL)
  - Only where the SVG marks "sharp_corner" metadata
  - V-carve into corners to relieve the 0.8mm radius
  - 0.5-1.0 mm depth typical

PASS 5 — KENTO BEVEL            | bit: 30° V-bit  (jig only, not block)
  - Bevel the baren-clearance side of kento marks
  - 0.5 mm depth, 15-30° from vertical
```

**Total time estimate** (per 257×364mm block, typical relief coverage):
- Pass 1: 4-6 min
- Pass 2: 8-12 min
- Pass 3: 6-10 min
- Pass 4: 1-2 min if used
- **Per-block total: ~20-25 minutes**
- **27 blocks: 9-11 hours of pure carve time** (plus tool changes, stock
  changes, setup)

## Tool radius compensation summary

For chuck-mcp's SVG export, every polygon contour needs the right offset:

| Geometry type | Offset | Tool used | Direction |
|---|---|---|---|
| Relief feature **outer boundary** | + tool_radius | 1/16" → +0.80 mm | OUTWARD |
| Relief feature **internal hole** | - tool_radius | 1/16" → -0.80 mm | INWARD |
| Roughing pocket interior | - tool_radius | 1/8" → -1.60 mm | INWARD |
| Kento paper-engaging wall | 0 | 1/8" → 0 | direct (climb cut) |
| Block outer edge for jig fit | - 0.075 mm | 1/8" | INWARD (allow 0.15mm clearance) |

These offsets are applied to the **mirror-flipped SVG**, NOT the original
image. Flip first, offset second.

## Cherry vs maple (since v2 chose maple)

| Property | Maple (selected) | Cherry (Lyon's choice) |
|---|---|---|
| Janka hardness | 1,450 (hard maple) | 950 |
| Top-veneer tear-out | Moderate | Low |
| Min feature stiffness | Excellent | Very good |
| Tool wear | Higher | Lower |
| Ink absorption | Lower | Higher |
| Print fidelity | Sharper edges | Slightly softer edges |
| Cost | Lower (more common) | Higher |
| Plywood availability | Common (cabinet-grade) | Limited (specialty) |

**Maple is the modern CNC-friendly choice.** Cherry is the traditional
mokuhanga choice. The Janka hardness gap means **maple holds fine
features better** (Lyon's 15-block "Anthony" worked in cherry; a 27-block
chuck-mcp print can be more demanding on feature size and edge crispness,
which maple handles).

Caveat: maple's harder surface means **higher baren pressure required**
for ink transfer. Practitioners report water-based pigments take 30-50%
more strokes vs cherry. Acceptable for an edition of 10; not for an
edition of 200.

## Citations

- CNCCookbook — feeds/speeds for wood: https://www.cnccookbook.com/feeds-speeds-cnc-wood-cutting/
- ToolGrit speeds and feeds guide: https://www.toolgrit.com/guides/cnc-wood-speeds-feeds-guide
- TwoTrees 3D relief carving: https://twotrees3d.com/blogs/twotrees-blog/3d-carving-cnc-projects-how-to-carve-relief-art-with-your-cnc-router
- IDC Woodcraft 30° V-bit: https://idcwoodcraft.com/products/30-degree-extreme-detail-v-bit-bit-for-cnc-routers-1-4-shank
- MatterHackers end mill guide: https://www.matterhackers.com/articles/end-mill-comparison-guide
- Harvey Performance wood properties: https://www.harveyperformance.com/in-the-loupe/understanding-wood-properties-for-cnc-woodworking-projects/
- WoodWeb — bit life in plywood: https://woodweb.com/knowledge_base/CNC_Bit_Life_in_Baltic_Birch.html
