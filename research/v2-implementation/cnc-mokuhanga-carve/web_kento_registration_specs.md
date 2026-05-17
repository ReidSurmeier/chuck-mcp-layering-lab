---
title: "Kento Registration — Authoritative Specifications for CNC Implementation"
sources:
  - "https://art-design-glossary.musabi.ac.jp/kagi-kento-and-hikitsuke-kento/"
  - "https://art-design-glossary.musabi.ac.jp/uchi-kento-soto-kento/"
  - "https://art-design-glossary.musabi.ac.jp/registration-marks/"
  - "http://woodblock.com/encyclopedia/entries/018_36/018_36.html"
  - "https://imcclains.com/productinfo/documents/KentoJapaneseRegistration.pdf"
  - "https://www.japanesetools.com.au/products/kento-nomi-mokuhanga-registration-mark-tool"
  - "https://www.goodsjapan.com/michihamono-woodcarving-kento-nomi-15mm-woodblock-print-chisel-wood-carving-tool/a-21177"
relevance: "MUST-READ — defines the physical geometry of kento marks. Critical correction from my pre-research mental model: kento marks are CARVED-BELOW block surface (depression for paper to drop into), NOT raised proud of the surface. The L-corner is a notched-out region that catches the corner of the paper, and the straight bar is a notched-out groove. Both are 1.5mm deep + slight bevel. This is the only spec that produces sub-mm registration across an edition of 10."
tags: [kento, mokuhanga, registration, kagi, hikitsuke, uchi-kento, soto-kento, depth-spec]
---

# Kento Registration — Authoritative Specifications

## The two marks

Traditional mokuhanga uses **two kento marks per block**, always in the
same relative position:

| Mark | Japanese | Shape | Position |
|---|---|---|---|
| Kagi-kento | 鉤見当 (lit. "hook indicator") | L-shaped corner | Bottom-right corner of block |
| Hikitsuke-kento | 引付見当 (lit. "pull-to indicator") | Straight bar | Along bottom edge, ~2/3 of width from right |

(Position is "bottom" in printing orientation — i.e. the edge of the
block closest to the printmaker. Paper is placed top-down onto the block.)

## Geometry — corrected understanding

**The kento is a depression carved INTO the block, not a feature raised
OUT of it.**

From Musabi Art University glossary:
> "The wood of the registration marks has been shaved down below the
> original level of the block, and in addition, has been beveled off
> away from the brushing area."

From McClain's "How to Carve Kento":
> "The depth should be two to three times the thickness of the paper.
> More specifically, the initial cut is made about one-eighteenth of an
> inch or 1.5 mm deep."

From Woodblock.com encyclopedia:
> "[The registration marks] allow the printing tool [baren] to pass over
> them without making contact" — confirming the depression geometry.

The physical purpose:
1. Paper sits on top of the block.
2. The corner of the paper drops into the L-notch of the kagi-kento.
3. The bottom edge of the paper rests against the wall of the hikitsuke-
   kento.
4. The baren passes over the kento marks without smearing pigment into
   them (because they're recessed, not proud).

## Canonical dimensions for chuck-mcp v2

Synthesizing all sources:

| Parameter | Value | Source |
|---|---|---|
| Depth of kento cut | **1.5 mm** | McClain's, "about 1/18 inch" |
| Cut wall angle | Vertical wall on paper-engaging side, 15-30° bevel away on baren-side | Musabi, "beveled off away from the brushing area" |
| Kagi (L-corner) leg length | **15-25 mm** per leg | inferred from kento-nomi chisel width 15mm + Andy Farkas guide |
| Hikitsuke (straight bar) length | **20-40 mm** | inferred from kento-nomi reach |
| Kagi position | Inside the paper margin, bottom-right | Musabi |
| Hikitsuke position | Bottom edge, distance from kagi ≈ 2/3 × paper width | Musabi |
| Distance from paper edge to inner wall of kento | **0 mm** — kento walls define paper position | by definition |

The 15 mm kento-nomi chisel width is the practical scale: traditional
hand-carved kento marks are *about as wide as the chisel*, ~15 mm. CNC
can go smaller, but mokuhanga registration practice expects ≥15 mm
contact length per mark for stable paper holding.

## Uchi-kento vs Soto-kento — choose Soto for chuck-mcp v2

Two implementation styles:

### Uchi-kento (内見当, "inner registration")
- Marks carved directly **into each block**
- Requires re-carving on every block (27 times for v2)
- More precise per-block, but accumulates registration error across
  blocks if any kento mark is misplaced
- Block size must be larger than (image + margin) to fit the kento
  marks within the block footprint
- **Standard for ≤10 blocks. Not recommended for chuck-mcp v2's 27.**

### Soto-kento (外見当, "outer registration")
- Marks carved into a **separate registration board (kentoban)** of
  the same thickness as the block
- The block slots into the kentoban; paper registers against the
  kentoban's kento marks
- The block itself can be raw rectangular stock — no kento on the block
- One set of kento marks shared across all 27 blocks → no accumulation
  of per-block error
- **Recommended for chuck-mcp v2.** Same principle as Mike Lyon's
  wedged-jig method.

The wedged jig is just a structurally robust soto-kento with a pocket
that holds the block in fixed XY position.

## CNC implementation of kento marks

### Geometry
For a Soto-kento jig in chuck-mcp v2:

```
Top view of jig:
  ┌─────────────────────────────────────┐
  │ jig                                 │
  │  ┌───────────────────────────────┐  │
  │  │                               │  │
  │  │   block pocket                │  │
  │  │   (sized to take 27 blocks)   │  │
  │  │                               │  │
  │  │                               │  │
  │  │                               │  │
  │  └───────────────────────────────┘  │
  │                              ┌──┐   │  ← hikitsuke (~25mm long)
  │                              │  │   │
  │                              └──┘   │
  │                                  ┌┐ │  ← kagi (L-shape, 20mm legs)
  │                                  ││ │
  │                                  ╶┘ │
  └─────────────────────────────────────┘

Side view of kento cut:
                ┌───────────────────────
                │  jig top surface
            ╲   │
             ╲  │   ← 15-30° bevel (baren clearance)
              ╲ │
               ╲│
                ├──── ← 1.5 mm depth
                │
              vertical wall
              (paper edge sits here)
```

### G-code / SBP toolpath for kento

Use a **flat 1/8" (3.18 mm) end mill** with profile + pocket strategy:

1. **Profile-mode** the inner (paper-engaging) wall of the L and the bar:
   - Tool follows the inner line, full-depth cut (1.5 mm), single pass
   - Climb-cut for clean edge
   - Tool radius (1.59 mm) compensated OUTWARD on this wall (paper edge
     must drop precisely against the wall)

2. **Pocket-mode** the area outboard of the L/bar to the block edge:
   - Standard pocketing toolpath, 0.5-0.8 mm depth-of-cut per pass
   - 60% stepover for efficiency (this region isn't critical-precision)

3. **Bevel the outer edge** with a V-bit (optional but recommended):
   - 30° or 45° V-bit
   - 0.5-1.0 mm depth, angled away from the paper-engaging wall
   - Provides baren clearance per Musabi's "beveled off" instruction

### Cross-block alignment of kento

**This is the load-bearing constraint for the 27-block edition.**

For soto-kento (recommended):
- Kento marks are carved into the jig **once**.
- All 27 blocks are rectangles cut from the same stock, with the same
  outer dimensions (within ±0.1 mm CNC tolerance).
- Each block fits the jig identically. Registration is automatic.
- Failure mode: blocks shrink/swell with humidity, jig pocket fit
  degrades. Mitigation: cut blocks slightly oversize (0.2 mm) and sand
  to exact fit, OR maintain studio humidity at 50% RH ± 3% (industry
  standard for printing rooms).

For uchi-kento (NOT recommended for v2):
- Kento marks must be carved at the exact same coordinates on every
  block. Requires consistent stock placement on CNC table.
- Use **at least 2 dowel pins** or a **corner stop** to fix every block
  to the same machine origin before carving.
- Across 27 blocks the per-block placement error compounds; expect
  ±0.3-0.5 mm registration drift end-to-end of the edition. Above the
  user's "sub-mm precision" target.

## Validation checklist before sending blocks to CNC

For each block:

- [ ] Image content is **horizontally flipped** (mirror), because the
      block prints mirror onto the paper
- [ ] Block outer dimensions match jig pocket spec (within ±0.1 mm)
- [ ] No image features smaller than the end-mill minimum-feature size
      (see `web_end_mill_selection_relief_carving.md`)
- [ ] Image positioned within the jig's "active print area" (i.e. inside
      the paper margin minus the kento clearance zone)
- [ ] Top veneer of plywood: grain direction noted (perpendicular to
      dominant features preferred)

For the jig (one-time):

- [ ] Kento marks at correct position relative to paper size (B4 =
      257 × 364 mm for chuck-mcp v2)
- [ ] Kento depth = 1.5 mm
- [ ] Bevel cut on baren-clearance side of kento (15-30°)
- [ ] Block pocket sized for press-fit (block_outer + 0.15 mm per side)
- [ ] Pocket depth = block thickness ± 0 mm (block top must be flush
      with jig top, or very slightly above)

## Citations

- Musabi Art University glossary (kagi+hikitsuke):
  https://art-design-glossary.musabi.ac.jp/kagi-kento-and-hikitsuke-kento/
- Musabi (uchi vs soto):
  https://art-design-glossary.musabi.ac.jp/uchi-kento-soto-kento/
- Encyclopedia of Woodblock Printmaking (Woodblock.com):
  http://woodblock.com/encyclopedia/entries/018_36/018_36.html
- McClain's "How to Carve Kento":
  https://imcclains.com/productinfo/documents/KentoJapaneseRegistration.pdf
- Kento-nomi tool specs (15mm flat blade):
  https://www.japanesetools.com.au/products/kento-nomi-mokuhanga-registration-mark-tool
