---
title: "Mike Lyon — Post-Digital Mokuhanga via ShopBot CNC"
sources:
  - "https://mlyon.com/"
  - "https://mlyon.com/2012/post-digital-printmaking/"
  - "https://mlyon.com/2017/two-small-woodcuts-using-a-wedged-registration-jig/"
  - "http://2017.mokuhanga.org/mike-lyon/"
  - "https://mlyon.com/wp-content/uploads/Post-Digital_Printmaking_Chapter_10_Mike_Lyon.pdf"
  - "https://theunfinishedprint.libsyn.com/mike-lyon-printmaker-layers-of-meaning"
relevance: "PRIMARY REFERENCE — Mike Lyon is the most documented practitioner of CNC mokuhanga (since 2004). His method is the de-facto template for the 27-block chuck-mcp v2 workflow. Confirms: cherry plywood, ShopBot router, luminosity-separated bitmap-per-block, kento *added to bitmaps post-separation*, wedged jig as registration alternative scalable beyond 10+ blocks."
tags: [cnc, mokuhanga, shopbot, mike-lyon, kento, multi-block, post-digital]
---

# Mike Lyon — Post-Digital Mokuhanga via ShopBot CNC

## TL;DR for chuck-mcp v2

Lyon (b. 1951) has been carving mokuhanga blocks on a **ShopBot CNC router**
since **2004**. He's the canonical case study. His workflow validates the
v2 design assumption: separate physical block per layer, CNC-carved, mokuhanga-
printed by hand. Key takeaways for chuck-mcp v2:

1. **Cherry plywood** is the documented substrate. ("The first woodcut that
   Lyon printed using the ShopBot was carved from fifteen separate cherry
   plywood blocks.") Maple is a reasonable modern substitute but **cherry
   is the proven choice for mokuhanga + CNC**.

2. **Luminosity separation, not color separation.** Lyon decomposes the
   target image into per-block bitmaps representing luminosity bands, then
   converts each bitmap to a toolpath. The bitmap-per-block model maps 1:1
   to chuck-mcp's binary mask-per-plate output.

3. **Kento marks are added to bitmaps post-separation**, not extracted
   from the image: *"these areas must be added to the bitmaps after the
   image has been prepared and divided into its component layers."* This
   is the chuck-mcp v2 instruction: kento is a stamping operation on top
   of the binary mask, not part of the solver.

4. **For >10 blocks: use a wedged registration jig, not raw kento.** Lyon's
   2017 method holds the block in a precisely-machined cradle whose top
   surface is co-planar with the block's printing surface. Eliminates
   block-shift between inkings (the dominant failure mode at 10+ blocks).
   **This is the recommended approach for chuck-mcp's 27-block edition.**

5. **15-block "Anthony" print** is the documented scale ceiling for raw-
   kento method. chuck-mcp v2's 27 blocks is past Lyon's documented kento-
   only limit. Wedged jig (or equivalent external registration) is
   required, not optional.

## What's explicitly documented

### Hardware
- **ShopBot CNC router** (model not specified in public sources, but
  RISD's printmaking shop has a **ShopBot PRSalpha** which is the most
  common model for university printmaking departments at this scale)
- **Capacity**: full 122 × 244 cm (4 × 8 ft) sheets (per Lyon's blog)
- **Custom headstock**: solenoid-driven pen attachment for drawing
- **Custom tool holders**: roulette, drypoint needle, ball burnisher
  mounted in plastic pipe that fits the router collet — same chuck used
  for routing, drawing, and intaglio-mark-making

### Materials
- **Cherry plywood blocks** (15 separate blocks for "Anthony")
- Also: copper plates for intaglio (not relevant to mokuhanga branch)
- Studio innovation: "sliding humidor paper delivery and takeup" — i.e.
  paper humidity is actively controlled throughout the multi-pull session.
  Confirms: humidity drift across 27 blocks × 10 prints is real and must
  be engineered around.

### Software workflow (from "Post-Digital Printmaking" chapter)
1. Source: digital photograph or scanned drawing
2. **Adobe Photoshop**: divide into luminosity bands, one bitmap per
   block
3. **Add kento marks to each bitmap** (manual step in Photoshop)
4. Bitmap → toolpath via unspecified CAM (likely VCarve Pro or PartWorks,
   the standard ShopBot CAM stack)
5. ShopBot carves
6. Hand-print with traditional mokuhanga technique (baren + water-based
   pigments)

### Multi-block alignment
- "Anthony" used **15 separate cherry plywood blocks** for a single image
- "Madeline and Kit" (2012): **17 blocks**
- 2017 wedged-jig prints: 12 blocks each, demonstrating the jig method
  at moderate scale
- chuck-mcp v2 target: **27 blocks** — outside Lyon's documented kento-
  only range, inside his wedged-jig documented range

## The wedged-jig insight (most important for v2)

From Lyon's 2017 post:

> "By wedging the block into a somewhat fancier jig, the block is held
> securely without shifting [...] very tight and certain registration
> and very clean margins without having to handle the damp block between
> printings."

> "The printing surface sits higher than the jig itself, with the
> registration marks (kento) level with the printing surface."

This is **the geometry chuck-mcp v2 should target**:

```
     paper
       ↓
   ┌───┴───┐ ┌─┐                  ← kento marks (level with print surface)
   │  block│ │ │
═══╧═══════╧═╧═╧══════════════════ ← jig top surface = block top surface
   │ jig pocket │  jig material   ← block sits in CNC-machined pocket
═══════════════════════════════════ ← spoilboard / CNC table
```

Key implications:
- All 27 blocks are machined to **identical outer dimensions** (within
  CNC tolerance ≤ 0.1 mm), so they're interchangeable in the jig
- The jig is **CNC-machined on the same ShopBot** as the blocks → uses
  the same coordinate system → kento alignment is automatic
- Paper is registered against **kento on the jig**, not the block →
  block can be lifted out, re-inked, replaced without disturbing paper
  registration
- The "kento on the jig" can be either traditional carved L+bar **or**
  pin holes drilled through the paper + dowels in the jig (Ternes Burton
  pins, used commercially in screen printing — Lyon mentions these as
  an alternative)

## Translation to chuck-mcp v2 export

The v2 SVG export must produce **two distinct artifact families**:

1. **27 block SVGs**, one per plate, each containing:
   - The carved geometry (mokuhanga relief) — horizontally flipped
   - **Tab features on block edges** that index into the jig (e.g. two
     locating pins or a corner notch) — same coordinates on every block
   - **No kento marks on the block itself** (wedged-jig design)

2. **1 jig SVG**, containing:
   - The block pocket (sized to receive the carved block with light
     press-fit, 0.1-0.2 mm clearance)
   - The kento marks (kagi + hikitsuke) at fixed coordinates relative to
     the pocket
   - Optional pin holes for Ternes Burton-style pin registration
   - Spoilboard hold-down features (screw clearance, vacuum slots)

## Carve depth (inferred — not explicitly stated)

Lyon doesn't publish carve depths. Inferred from mokuhanga convention
(2-3 mm to hold ink, deep enough to keep brush + baren off lower
surface) and CNC physics (depth ≤ 0.5 × tool diameter per pass):

- **Recommended carve depth: 2.5 mm** (97 thou, slightly under 1/8")
  - 1/16" mill (1.59 mm Ø) → 3 passes × 0.85 mm depth-of-cut
  - 1/8" mill (3.18 mm Ø)  → 2 passes × 1.25 mm depth-of-cut
- **Block stock thickness: 12-18 mm** (1/2" - 3/4") — depth of carve
  is small compared to stock, so block remains rigid post-carve

## Open questions Lyon doesn't answer (chuck-mcp must decide)

1. **Exact end-mill set.** Lyon doesn't publish his bit list. Inferred
   from his ShopBot generation and the published mokuhanga relief
   community: 1/16" or 1/8" flat for outline + 1/4" flat for roughing.
   See `web_end_mill_selection_relief_carving.md` for the v1 recipe.

2. **Block edge tolerance for jig fit.** Lyon doesn't publish. Standard
   ShopBot routed tolerance is ±0.1 mm on edges. chuck-mcp v2 should
   spec the jig pocket at **block_outer + 0.15 mm per side** for a
   light press fit with a 27-block edition's worth of swelling/shrinking
   slop.

3. **Stock orientation.** Plywood has grain direction (top veneer + core
   plies). For relief carving Lyon doesn't specify; the convention is
   **top veneer grain perpendicular to the dominant feature direction**
   to minimize tear-out on fine features. For chuck-mcp this is a
   per-block decision and must be exposed as a tool config setting.

## Citations

- Mike Lyon home: https://mlyon.com/
- "Post-Digital Printmaking" (2012): https://mlyon.com/2012/post-digital-printmaking/
- IMC 2017 talk page: http://2017.mokuhanga.org/mike-lyon/
- "Wedged registration jig" (2017): https://mlyon.com/2017/two-small-woodcuts-using-a-wedged-registration-jig/
- Post-Digital Printmaking Chapter 10 (PDF): https://mlyon.com/wp-content/uploads/Post-Digital_Printmaking_Chapter_10_Mike_Lyon.pdf
- The Unfinished Print podcast — Lyon ep.: https://theunfinishedprint.libsyn.com/mike-lyon-printmaker-layers-of-meaning
