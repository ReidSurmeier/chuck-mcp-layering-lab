---
title: "Block Stock Dimensions and Materials for chuck-mcp v2 (B4 mokuhanga)"
sources:
  - "https://imcclains.com/catalog/blocks/shina.html"
  - "https://imcclains.com/catalog/blocks/14Shina.html"
  - "https://www.jacksonsart.com/en-us/japanese-shina-plywood-4mm-woodblock-300x450mm"
  - "https://www.schoolartsupplies.com.au/japanese-shina-wood-sheets-262-x-360-x-4mm"
  - "https://mokuhanga-school.jp/tools-and-materials/product/shina-plywood-large/category_pathway-42"
  - "https://www.cnccookbook.com/feeds-speeds-cnc-wood-cutting/"
  - "https://kynetoncraftery.com/japanese-shina-plywood-woodblock-4mm/"
relevance: "MUST-READ — locks the block-stock dimensions for chuck-mcp v2: 280×400mm maple plywood, 12mm thick. Paper is B4 (257×364mm) plus 23mm margin per side. Standardizes the rectangle for all 27 blocks and the jig pocket. Sets up CNC nesting on standard 4'×8' or 2'×4' plywood sheets."
tags: [block, stock, dimensions, plywood, b4, paper, jig, nesting]
---

# Block Stock Dimensions for chuck-mcp v2

## The dimensional chain

B4 paper → kento clearance → block image area → block stock outer dims →
jig pocket → CNC sheet nesting. Each step compounds.

### Paper

**JIS B4: 257 × 364 mm** — the canonical mokuhanga paper size for medium-
format prints. Sized for kozo/gampi sheets and most commercial mokuhanga
washi papers (Iwano, Awagami). This is the right target for chuck-mcp v2.

Real mokuhanga papers are sold slightly oversized to accommodate trimming;
expect ±2-3 mm on each dimension. Design the kento marks for **the nominal
257×364 mm**, accept ±2 mm slop in the paper, hand-trim post-print.

### Image area

Mokuhanga conventionally keeps an unprinted border of **15-30 mm** on all
four sides. For chuck-mcp v2 (multi-block precision target):

| Side | Margin | Reasoning |
|---|---|---|
| Bottom | **20 mm** | Where kento marks live; must fit them + clearance |
| Right | **15 mm** | Kagi-kento corner; tighter than bottom |
| Top | **15 mm** | Pure border, baren start zone |
| Left | **15 mm** | Pure border |

**Image area: 227 × 329 mm** (257-30=227 wide, 364-35=329 tall — note
asymmetric vertical because bottom margin is 5mm bigger for kento).

For the v2 target file format, the image-area rectangle should be exposed
in the SVG metadata so the printer can align the source image manually if
auto-alignment fails.

### Block stock dimensions

Block must cover the **image area + kento marks + safety margin**:

```
   ┌────────────────────────────────────────┐
   │                                        │  ↑
   │   ┌─────────────────────────────────┐  │  │ 15 mm top border
   │   │                                 │  │  │
   │   │                                 │  │  │
   │   │   image area                    │  │  │
   │   │   227 × 329 mm                  │  │  │  block stock
   │   │                                 │  │  │  280 × 400 mm
   │   │                                 │  │  │
   │   │                                 │  │  │
   │   └─────────────────────────────────┘  │  │
   │                                ┌─┐     │  │
   │                                │ │     │  │
   │                                └─┘     │  │
   │                                  ┌┐    │  ↓ kento + jig clearance
   │                                  ││    │     
   │                                  ╶┘    │     
   └────────────────────────────────────────┘
   ←──────────────  280 mm  ──────────────→
```

**Block outer dimensions: 280 × 400 mm × 12 mm thick.**

Rationale:
- **280 wide** = 257 (paper) + 15 (right margin & kagi clearance) + 8 (jig
  pocket fit clearance)
- **400 tall** = 364 (paper) + 20 (bottom kento + jig clearance) + 16 (top
  border + jig fit)
- **12 mm thick** = standard 1/2" plywood (≈12.7 mm) — common stock,
  cheaper than 18mm, rigid enough at this footprint, gives enough Z
  capacity for the 2.5 mm carve depth + jig pocket inset

### Material: maple plywood

Per the V2 design lock, the substrate is **maple plywood**. Practical
specifications:

| Spec | Value | Notes |
|---|---|---|
| Species | Hard maple (face veneer) | Janka 1,450 |
| Construction | Standard 5-7 ply | Birch core acceptable |
| Thickness | 12 mm (1/2" nominal) | Actual: 12.0-12.7 mm |
| Sheet size | 1220 × 2440 mm (4'×8') | Cabinet-grade |
| Veneer thickness | ≥ 0.6 mm | Below 0.6 mm: tear-out risk |
| Glue line | Type II (water-resistant) | Stands up to water-based pigment |
| Grade | A1 or A2 face | No knots, no patches |

Estimated cost: $80-120/sheet at US cabinet suppliers. One 4'×8' sheet
yields 27 blocks at 280×400 mm with ~25% waste — sufficient for one
chuck-mcp v2 print. Buy 2 sheets to have a backup.

### Comparison: maple vs shina (the traditional choice)

Standard mokuhanga uses **shina plywood** (Japanese basswood). Commercial
sizes:

| Source | Dimensions | Thickness |
|---|---|---|
| McClain's | various, common 4"×6" to 9"×12" | 0.25" (6 mm) and 0.5" |
| Jackson's Art | 300 × 450 mm | 4 mm |
| Heidelberg Schoolart | 262 × 360 mm | 4 mm |
| Mokuhanga-school.jp | 300 × 450 mm (large) | 9 mm |

**Why chuck-mcp v2 chose maple over shina** (presumed rationale):
- Shina is hard to source in the US in large enough sheets and consistent
  thickness for CNC
- Shina Janka = ~410 lbf vs maple 1,450 lbf; maple holds fine relief
  features 3.5× better
- Shina is more prone to CNC tear-out at small features (per
  `web_cnc_min_feature_size_endmill_relief.md`: needs 25-50% feature-size
  margin in shina vs cherry/maple)
- Shina's water-absorption is much higher → swells with water-based ink,
  causing block-dimension drift across an edition
- Maple's 12mm-thick sheets are commodity items at every hardwood supplier

**Shina trade-offs preserved**:
- Maple needs more baren pressure for ink transfer (harder surface)
- Maple top veneer can mirror-grain — designs that cross grain will print
  with slight directional artifacts

### Stock nesting on a 4'×8' sheet

```
1220 × 2440 mm sheet, 280 × 400 mm blocks:

Rows: floor(2440 / 400) = 6 rows
Cols: floor(1220 / 280) = 4 cols
Total: 24 blocks / sheet

With ~5 mm kerf allowance between blocks and ~10 mm edge margins:
Effective area per block: 290 × 410 mm
Rows: floor(2420 / 410) = 5
Cols: floor(1200 / 290) = 4
Total: 20 blocks / sheet

→ chuck-mcp v2's 27 blocks need 2 sheets of 4'×8' maple plywood
  (~$200 in materials, $250 with kerf/scrap allowance)
```

### Jig dimensions

The soto-kento jig is one larger piece carved on the same ShopBot:

```
Jig outer: 350 × 470 × 12 mm (matches block thickness, 35mm border)
Block pocket: 280.15 × 400.15 mm (0.075mm clearance per side, light
              press-fit), 12 mm deep (block sits flush)
Kento marks: positioned outside the block pocket, in the jig border
Hold-down: 4× #10 screw holes at corners for fixturing to spoilboard
```

Jig material: same maple plywood for thermal stability matching the
blocks. (If blocks and jig expand differently with humidity, kento
alignment drifts. Same material = same expansion.)

### Per-block engraved label

Each block should carry a CNC-engraved label on its **edge** (not the
face) indicating block number, edition, and orientation. Sample:

```
chuck-mcp / 03 of 27 / R-up
```

Cut with the 1/16" bit at 0.5 mm depth on the bottom edge of the block.
Total engraving toolpath: ~30 seconds per block. Worth every second when
sorting 27 nearly-identical rectangles months later.

### Validation checklist for stock

- [ ] Maple plywood, A1 face, 12 mm nominal thickness
- [ ] Outer dimensions 280 × 400 mm, ±0.2 mm
- [ ] Edges square within 0.3 mm over 400 mm (out-of-square will misregister)
- [ ] No surface defects in the image-area zone
- [ ] Grain direction recorded (perpendicular to dominant features preferred)
- [ ] Block number engraved on bottom edge before relief carve
- [ ] Jig pocket fit verified with one block before carving the other 26

## Citations

- McClain's shina catalog: https://imcclains.com/catalog/blocks/shina.html
- Jackson's Art shina: https://www.jacksonsart.com/en-us/japanese-shina-plywood-4mm-woodblock-300x450mm
- School Art Supplies shina 262×360: https://www.schoolartsupplies.com.au/japanese-shina-wood-sheets-262-x-360-x-4mm
- Mokuhanga School shina large: https://mokuhanga-school.jp/tools-and-materials/product/shina-plywood-large/category_pathway-42
- McClain's 1/4" shina: https://imcclains.com/catalog/blocks/14Shina.html
- Kyneton Craftery shina 4mm: https://kynetoncraftery.com/japanese-shina-plywood-woodblock-4mm/
