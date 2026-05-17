---
title: "Minimum feature size and end-mill geometry for relief carving — practical reference"
sources:
  - "https://jiga.io/cnc-machining/cnc-design-guide/"
  - "https://www.matterhackers.com/articles/end-mill-comparison-guide"
  - "https://leweiprecision.com/design-rules-for-cnc-milling-parts/"
  - "https://www.onshape.com/en/blog/cnc-machining-101-endmills-other-considerations"
  - "https://docs.vectric.com/docs/V10.0/Aspire/ENU/Help/form/VCarve%20Toolpath%20Creator/"
relevance: "MUST-READ — establishes the minimum-feature-size budget chuck-mcp needs to enforce upstream of the SVG exporter. Cherry plywood with 1/16\" (0.0625\"≈1.59mm) end-mill = minimum internal radius ~0.8mm; minimum slot width ~1.6mm; minimum positive feature ~1.6mm (else snaps under tool side-load). Translates directly to S6.c's area_threshold and endmill_radius_px values."
tags: [cnc, end-mill, minimum-feature, relief-carving, woodblock, manufacturability]
---

# Minimum Feature Size for Relief Carving on Cherry / Shina Plywood

## TL;DR pixel/mm budget for chuck-mcp

For a target ShopBot setup with 1/16" or 1/8" end mill on cherry or
shina plywood (relief carving for woodblock printing), the printability
budget at the SVG-output boundary is:

| Tool         | Diameter | Min slot width | Min positive feature width | Min internal radius |
|--------------|----------|----------------|---------------------------|---------------------|
| 1/16" flat   | 1.59 mm  | 1.59 mm        | ≥ 1.6 mm                  | 0.8 mm              |
| 1/8" flat    | 3.18 mm  | 3.18 mm        | ≥ 3.2 mm                  | 1.6 mm              |
| 1/16" ball   | 1.59 mm  | 1.59 mm        | ≥ 2.0 mm (less stiff)     | 0.8 mm              |
| 60° V-bit    | 0 mm tip | depends on depth| n/a (varying width)       | 0 mm (point)        |

**Convert to pixels** at the chuck-mcp render resolution. If the source
image is 300 DPI = 11.81 px/mm:

- 1/16" mill: `min_feature_px = ceil(1.59 * 11.81) = 19`
- 1/8" mill: `min_feature_px = ceil(3.18 * 11.81) = 38`

These are the **direct settings for S6.c's area_threshold** (= `min_feature_px²`)
and `endmill_radius_px` (= `min_feature_px // 2`).

## Why these limits exist

### Tool deflection / snap

Per MatterHackers: "the smaller your end mill, the more fragile it gets,
so if you try to cut through material too fast with a 1/16" end mill
you're snapping it off into the workpiece." For relief carving, the
side-load on the tool when cutting along a thin positive feature can
cause:

- Tool deflection → drift from planned path (>0.1mm common)
- Tool snap → ruined workpiece + lost tool

Rule of thumb: depth-of-cut per pass ≤ 0.5 × tool diameter for hardwood.
1/16" end mill → ≤ 0.8 mm depth per pass.

### Minimum internal radius

Any internal corner is rounded to (tool radius). A 1/16" end mill cannot
cut a corner sharper than 0.8mm radius. If the SVG calls for a sharper
corner, the toolpath either:

- Rounds the corner (loses geometric fidelity but safe)
- Overshoots the corner (gouges adjacent material)
- Requires a smaller tool change (cost: tool-change time + setup)

chuck-mcp's S6.c should pre-round all internal corners to ≥ tool radius
before SVG export. This is a **morphological closing with disk(r)** of
the *negative* space (or equivalently, opening-by-reconstruction of the
positive mask).

### Cherry vs shina

Cherry is harder (Janka ~950 lbf) — less tool deflection but more
stress on the bit. Allows tighter features at the cost of slower feed.
Shina (Japanese basswood/lime) is softer (Janka ~410 lbf) — bigger
tolerances needed because fibers tear out at small features.

Practical for our purposes:

- Cherry: trust the published min feature sizes above
- Shina: add 25-50% safety margin (i.e. min positive feature ≥ 2.0mm
  for 1/16" mill in shina)

### V-carve special case

V-bits can carve features narrower than their tip diameter by reducing
depth. A 60° V-bit at depth d produces a cut of width `2·d·tan(30°)
= 1.155·d`. This means V-carve **doesn't have a hard minimum feature
width**; instead, it has a maximum-depth budget for fine features.

For chuck-mcp, the V-carve path is a separate downstream consideration —
the SVG just needs to identify which curves are V-carvable (medial-axis
+ width field).

## Stepover / scallop for relief surface finish

For relief printing, the ink-bearing top surface needs ≤ 25-50 µm
scallop height between adjacent passes (depends on paper). For a ball
nose end-mill of radius r at stepover s, scallop height ≈ s²/(8r).

To get ≤ 25 µm scallop with 1/16" ball nose (r=0.8mm):
- s² ≤ 8 × 0.8 × 0.025 = 0.16 → s ≤ 0.4 mm

So 0.4mm stepover for the finishing pass — that's ~80 raster lines per
inch. Roughing pass can use 0.6 × tool diameter = 1mm stepover.

## Recommended chuck-mcp S6.c → SVG export config

For a default cherry-plywood + 1/16" mill carve at 300 DPI source:

```yaml
s6c:
  area_threshold_px: 361        # 19² = (1.6mm @ 300DPI)²
  endmill_radius_px: 10         # 1/2 × 19 ≈ 0.8mm
  delta_e_budget_per_plate: 2.0 # CIE76 ΔE
  positive_feature_close: true  # morphological closing on positive
  negative_feature_open: true   # morphological opening on negative
  edge_smooth_lambda: 0.5       # Visvalingam-Whyatt area tolerance

vectorize:
  engine: potrace
  turdsize: 361                 # = area_threshold_px (consistency)
  alphamax: 0.6                 # preserve corners
  opttolerance: 0.1             # tight curve fit
  turnpolicy: minority

postprocess:
  simplify_engine: shapely_vw   # Visvalingam-Whyatt
  simplify_area_tol_px2: 25     # (5px)² — sub-feature smoothing
  preserve_topology: true
  ensure_min_width: true        # opening-by-reconstruction guard
```

## Citations

- Jiga CNC Design Guide. "Features & Size Limits of CNC Machining."
  https://jiga.io/cnc-machining/cnc-design-guide/
- MatterHackers. "CNC End Mill Comparison Guide."
  https://www.matterhackers.com/articles/end-mill-comparison-guide
- Lewei Precision. "Design Rules for CNC Milling Parts."
  https://leweiprecision.com/design-rules-for-cnc-milling-parts/
- Onshape. "CNC Machining 101 — Endmills and Other Considerations."
- Vectric Aspire / VCarve docs (V-carve toolpath specifics).
