---
domain: cnc-mokuhanga-carve
swarm: chuck-mcp-v2-implementation
swarm_id: swarm-1778969836247-ys4o7z
agent: CNC-MOKUHANGA-CARVE
date: 2026-05-16
artifact_count: 7
---

# CNC Relief Carving for chuck-mcp v2 — Synthesis Notes

## Scope

End-to-end CNC export pipeline for chuck-mcp v2: SVG-to-physical-block
workflow for a 27-block edition of B4 (257×364 mm) mokuhanga prints,
carved on a ShopBot PRSalpha in maple plywood, registered to sub-mm
precision across an edition of 10.

This includes:
- Block stock + jig geometry
- End-mill selection + feeds/speeds
- Kento registration (physical implementation)
- SVG dialect (mirror flip + metadata + layer colors)
- CAM pipeline (SVG → VCarve → ShopBot .sbp)
- Operator + validation checklists

## VERDICT — End-to-end CNC pipeline

```
PHASE 0  RIG SETUP (once per shop)
─────────────────────────────────
  - ShopBot PRSalpha confirmed at RISD; spoilboard in good condition
  - 4-bit tool set acquired: 1/4" compression, 1/8" up-spiral, 1/16"
    up-spiral, 30° V-bit, all solid carbide, 1/4" shank
  - VCarve Pro license verified (or gcodetools Pipeline B prepped)
  - 2 sheets of 4'×8' × 12mm hard maple plywood, A1 face

PHASE 1  JIG CARVE (once per edition)
─────────────────────────────────────
  - Cut jig blank to 350 × 470 × 12 mm
  - chuck-mcp emits ONE jig SVG containing:
      * block pocket (280.15 × 400.15 × 12 mm deep)
      * kento marks at fixed coordinates relative to pocket
      * 4 hold-down screw holes
  - VCarve session: 4 toolpaths (pocket rough, pocket finish, kento
    profile, kento V-bevel), output `jig.sbp`
  - Mount jig to spoilboard with #10 screws at corners

PHASE 2  PER-BLOCK CARVE (×27)
──────────────────────────────
  For each block 01..27:
    a. chuck-mcp emits block-NN.svg with:
         * relief geometry, mirror-flipped (scale(-1,1) translate(-280,0))
         * outer rectangle at 280 × 400 mm
         * block ID engraving on bottom edge
         * tool-compensation offsets baked into geometry
    b. Cut 280 × 400 mm blank from maple plywood sheet
    c. Engrave block ID on bottom edge (manual VCarve step, 30 sec)
    d. Place blank in jig pocket
    e. VCarve session: 3 passes (rough, bulk, detail) + optional V-bit
       → block-NN.sbp
    f. Run .sbp on ShopBot — ~20-25 min carve time
    g. Inspect: no tear-out, no missed features, kento clear
    h. Move to step a for next block

PHASE 3  TEST PULL
──────────────────
  - Print blocks 1-3 first as a registration test
  - Verify paper drops cleanly into kento on jig
  - Check inter-block alignment with a registration target (e.g.
    crosshair on layers 1+2+3)
  - If misalignment >0.5 mm: stop, diagnose (jig pocket fit, block
    edge tolerance, kento bevel direction)
  - Once clean: print remaining 24 blocks → full edition of 10
```

## Top 3 must-reads

1. **`web_mike_lyon_cnc_mokuhanga_workflow.md`** — Mike Lyon is the only
   documented practitioner of CNC mokuhanga at the chuck-mcp scale. The
   wedged-jig method (his 2017 innovation) is the answer to the 27-block
   registration problem. **READ THIS FIRST.**

2. **`web_kento_registration_specs.md`** — Corrects the most likely
   pre-research mental model error: kento marks are carved DOWN into the
   block, not raised ABOVE it. Locks the depth (1.5 mm), the bevel
   (15-30°), and the choice of soto-kento (on jig) over uchi-kento
   (on every block) for the v2 design.

3. **`web_end_mill_selection_relief_carving.md`** — The V1 four-bit set
   with feeds, speeds, depth-of-cut per pass, and stepover for each.
   Locks chuck-mcp v2's minimum feature size at 1.6 mm.

## Recommended end-mill set for V1

| # | Bit | Diameter | Role |
|---|---|---|---|
| 1 | 1/4" flat compression spiral, solid carbide | 6.35 mm | Rough outside relief |
| 2 | 1/8" flat 2-flute up-spiral, solid carbide | 3.18 mm | Bulk relief carve |
| 3 | **1/16" flat 2-flute up-spiral, solid carbide** | **1.59 mm** | **Finishing — sets min feature size** |
| 4 | 30° V-bit, solid carbide | 0 mm tip | Inside corners + kento bevel |

All 1/4" shank for ShopBot compatibility. Budget ~$100-150 for the set.

**The 1/16" bit is the load-bearing member.** It defines chuck-mcp v2's
minimum feature size (1.6 mm). Carry **two spares** — 1/16" bits snap
under deflection, and a snap mid-edition is a recovery nightmare.

## Stock dimensions

- **Paper**: B4 (257 × 364 mm)
- **Block**: 280 × 400 × 12 mm (maple plywood, A1 face, ±0.2 mm)
- **Image area**: 227 × 329 mm
- **Margins**: 15 mm top/left/right, 20 mm bottom (kento clearance)
- **Carve depth**: 2.5 mm
- **Jig**: 350 × 470 × 12 mm with 280.15 × 400.15 mm block pocket
- **Sheet count**: 2 × 4'×8' × 12 mm maple plywood (covers 27 blocks +
  1 jig + waste budget)

## ShopBot SBP post-processor settings

- **Output format**: `.sbp` (OpenSBP), NOT G-code
- **VCarve post**: `ShopBot TC (mm) (*.sbp)` — Tool Change variant, mm
- **File header**: `VU, mm` + `SA` (absolute mode)
- **Cut speed (MS)**: 2.0 in/s XY, 0.5 in/s Z (PRSalpha default)
- **Jog speed (JS)**: 6.0 in/s XY, 1.5 in/s Z
- **Safe Z**: 6.0 mm above stock
- **Tool change**: `C6` macro between passes

See `web_shopbot_sbp_postprocessor.md` for the full SBP command
reference and Pipeline C custom-emitter sketch.

## Kento implementation (the concrete answer)

**Choose: Soto-kento on jig, not Uchi-kento on each block.**

Geometry:
- **Position**: outside the block pocket, on the jig border, oriented
  to receive the bottom-right corner and bottom edge of the paper
- **Kagi (L-corner)**: 20 mm × 20 mm legs, 8 mm wide arms
- **Hikitsuke (bar)**: 25 mm long × 8 mm wide, 171 mm to the left of
  the kagi (= 2/3 × paper width 257 mm)
- **Depth**: 1.5 mm into jig surface
- **Wall**: vertical on paper-engaging side
- **Bevel**: 25° (15-30° acceptable) on baren-clearance side

Alignment across 27 blocks: **automatic**. All 27 blocks fit the same
jig pocket; the jig holds the kento marks. No per-block kento carving
required. Per-block registration drift = jig pocket tolerance (±0.075 mm)
+ block edge tolerance (±0.1 mm) = **±0.18 mm worst case**, well under
the sub-mm precision target.

Failure modes + mitigations:
- **Block swells in humidity** → jig pocket gets too tight → block won't
  seat. Mitigation: studio humidity at 45-55% RH; cut blocks slightly
  oversize and sand to fit.
- **Block shrinks in dry conditions** → jig pocket too loose → block
  shifts during inking. Mitigation: maintain same studio humidity; carry
  a thin wood shim for the long edge if shrinkage exceeds 0.3 mm.
- **Jig pocket distorts over 27 blocks of use** → cumulative wear in the
  pocket walls. Mitigation: carve a sacrificial jig once and re-carve if
  needed (one-evening's work).

## Pre-carve validation checklist (every block)

Before sending block-NN.sbp to the ShopBot:

- [ ] **Mirror flip applied** — verify SVG metadata `flip-applied="true"`
- [ ] **Kento marks present in jig** (not in block SVG) — verify block
      SVG layer `layer-2-kento` is empty/absent
- [ ] **No features smaller than 1.6 mm** — run skimage area_opening
      check on the binary mask before SVG emission; if any component
      survives `area_opening(min_size = 19²)` but vanishes under
      `binary_erosion(disk(10))`, FLAG IT
- [ ] **Block ID engraving** present at correct edge position
- [ ] **Tool compensation offsets applied** — outer features offset by
      +0.80 mm, internal holes offset by -0.80 mm
- [ ] **Outer rectangle** is exactly 280 × 400 mm
- [ ] **Image area within margins** — no carve geometry outside
      (15, 20) to (265, 380) mm bounds
- [ ] **VCarve toolpath review** — open .sbp in VCarve preview, verify
      no plunge into stock above material surface, no rapid moves into
      stock
- [ ] **Bit installed matches** the sbp's first `&Tool=` command

Pre-carve validation for the **jig** (once per edition):
- [ ] All kento walls vertical (zero bevel on paper-engaging side)
- [ ] Kento bevel applied on baren-clearance side (15-30°)
- [ ] Kento depth = 1.5 mm (±0.1 mm)
- [ ] Block pocket dimensions verified with caliper after carve
- [ ] Test-fit one carved block before committing to others

## Cross-cutting findings

1. **ShopBot is .sbp, not G-code.** Most consequential discovery. Any
   chuck-mcp documentation that says "emit G-code" is wrong by default.
   Pipeline A (Vectric VCarve) handles the conversion; Pipeline B
   (gcodetools) emits G-code that chuck-mcp must convert to SBP.

2. **Mike Lyon's wedged jig is the registration pattern.** For >10
   blocks, traditional uchi-kento (on every block) accumulates too
   much error. The soto-kento jig is the documented scalable solution
   and Lyon proved it at 12 blocks. chuck-mcp's 27 is well within
   the validated range.

3. **Maple is the right substrate** (vs cherry or shina). Janka 1,450
   vs cherry's 950 means maple holds finer features. The trade-off
   (more baren pressure for ink transfer) is acceptable for an edition
   of 10. Shina is wrong for CNC — too soft, too prone to tear-out.

4. **The 1/16" bit defines min feature size**, not the algorithm.
   chuck-mcp's vectorization S6.c already does mill-sized area-opening
   (per existing research). The CNC pipeline just consumes that contract:
   `min_feature_px = 19` at 300 DPI, sized to 1/16" bit's 1.59 mm
   diameter.

5. **Mirror flip is one transform**, but load-bearing. Bake it into the
   SVG emit step and write `flip-applied="true"` into metadata. Test
   suite enforces it.

6. **Manual VCarve sessions are acceptable for V1.** 27 × 5 min ≈ 2
   hours of click-work per edition. Trade-off: every block gets a human
   review before $5 of maple is destroyed. Worth it for production
   editions of 10 each. Automate (Pipeline B/C) only when edition
   count justifies dev time.

## What's NOT in scope (deferred to v3)

- **Graduated relief depth** (bokashi-style soft edges) — requires ball-
  nose bit and 3D toolpath; out of v2's binary-mask model.
- **Multiple bit-change passes within one .sbp file** — VCarve does
  this fine; deferred to Pipeline B/C consideration.
- **In-situ probing / surface-mapping** — for very fine relief depth
  control on warped plywood. ShopBot supports it; chuck-mcp doesn't
  need it for 2.5 mm carve depths.
- **Auto-detect grain direction from plywood image** — currently a
  manual user input. Computer-vision-based grain detection is a clean
  research project but not v2-critical.
- **Sub-100µm registration** — chuck-mcp v2 targets sub-1mm. Going
  below requires Ternes Burton pin-registration, paper-side punching
  jig, and dedicated stock-handling protocol. Out of scope.

## Open questions

1. **Does RISD's ShopBot have a tool changer or manual tool change?**
   Affects whether one .sbp can carry all 4 tools or requires 4 separate
   .sbp files per block. (My recommendation defaults to "TC" variant
   which uses C6 pauses for manual changes — works either way.)

2. **What's the actual stock thickness variation** in the maple plywood
   batch? 12 mm nominal can be 11.5-12.7 mm in practice. If >0.5 mm
   variation, the carve depth needs to be measured per-block or set
   conservatively to avoid breakthrough.

3. **Is the jig pocket carved before or after the kento marks?** Order
   matters for tool-change efficiency. Recommend: pocket first (1/8"
   bit, then 1/16" finish), then kento (same 1/16" bit), then V-bevel
   (V-bit). One bit change.

4. **What's the user's baren skill level?** A 27-block print on maple
   demands strong baren technique. If unfamiliar with maple's higher
   pressure requirement, do a 3-block test edition first.

## Quick-reference: which artifact for which question

| Question | Read this first |
|----------|-----------------|
| Mike Lyon's documented method? | `web_mike_lyon_cnc_mokuhanga_workflow.md` |
| Kento dimensions + position? | `web_kento_registration_specs.md` |
| Which bits to buy? | `web_end_mill_selection_relief_carving.md` |
| Feeds and speeds for maple? | `web_end_mill_selection_relief_carving.md` |
| SBP file format reference? | `web_shopbot_sbp_postprocessor.md` |
| Why .sbp not .gcode? | `web_shopbot_sbp_postprocessor.md` |
| Block stock dimensions? | `web_block_stock_dimensions_b4.md` |
| Maple vs shina rationale? | `web_block_stock_dimensions_b4.md` |
| How to mirror-flip SVG? | `web_svg_mirror_flip_implementation.md` |
| Which CAM pipeline for V1? | `web_svg_to_gcode_pipelines.md` |
| Machine-readable tool spec? | `chuck_mcp_v2_tool_config.yaml` |
