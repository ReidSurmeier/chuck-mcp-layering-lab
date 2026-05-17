---
domain: vectorization-cnc
swarm: chuck-mcp-layering-lab
swarm_id: swarm-1778962124344-s4cm4l
agent: VECTORIZATION-CNC
date: 2026-05-16
artifact_count: 11
---

# Vectorization & CNC — Synthesis Notes

## Scope

Raster-to-vector conversion at chuck-mcp's S6.c → SVG boundary, plus
the downstream concerns the cnc-woodblock-tools project will pick up:
topology-preserving simplification, minimum-feature enforcement,
tool-radius compensation, and relief carving on cherry/shina plywood
with 1/16" and 1/8" end-mills.

## Top 5 must-reads (ranked for S6.c relevance)

1. **Potrace — Selinger 2003** [`web_potrace_selinger_2003.md`]
   The right inline exporter for chuck-mcp's per-plate binary masks.
   `turdsize` is literally chuck-mcp's island-pressure threshold;
   `alphamax`/`opttolerance` are the only two knobs needed at the SVG
   boundary. Closed paths by construction. Drop-in.

2. **Vincent 1993 — Morphological area openings** [`web_morphological_area_opening_vincent_1993.md`]
   Formalizes the cleanup that *should* run inside S6.c before
   Potrace. Area opening alone is insufficient: pair it with
   opening-by-reconstruction at radius = end-mill_radius_px to kill
   thin necks. Idempotent, topology-preserving, near-linear time.

3. **Min feature size for relief carving** [`web_cnc_min_feature_size_endmill_relief.md`]
   Translates physical end-mill geometry into the px/mm budget S6.c
   needs. 1/16" mill ≈ 1.6mm minimum feature ≈ 19px @ 300 DPI. Cherry
   trusts published limits; shina needs 25-50% safety margin.

4. **Polygon simplification — Douglas-Peucker / Visvalingam-Whyatt**
   [`web_visvalingam_whyatt_polygon_simplification.md`]
   The path between Potrace output and CNC toolpath has too many
   vertices. Visvalingam-Whyatt with topology guard (Shapely
   `preserve_topology=True`) is the right default.

5. **VTracer** [`web_vtracer_visioncortex.md`]
   Alternative to Potrace; supports stacked/cutout layered output that
   maps cleanly to chuck-mcp's plate model. O(n) vs Potrace's O(n²)
   fitting. Not strictly better — corner detection is less robust on
   engineered geometry — but worth A/B testing.

## Other artifacts (high → low S6.c relevance)

- `arxiv_2206.04655_live_layerwise_image_vectorization.md` — LIVE
  (CVPR 2022). Conceptually maps to chuck-mcp's layered model but too
  slow for the inline path. Possible offline hero-image refiner.
- `web_diffvg_li_2020.md` — DiffVG. Foundational; only relevant if we
  add a differentiable post-pass on top of Potrace output.
- `arxiv_2503.16424_bezier_splatting.md` — Faster DiffVG drop-in.
  30x/150x speedup for open curves. Only matters if (a) we adopt
  differentiable post-processing and (b) we have open curves to
  optimize.
- `arxiv_2102.02798_im2vec.md` — Im2Vec. Survey-completeness; not
  applicable.
- `arxiv_2110.04830_marvel_manga_drl_vectorization.md` — MARVEL. Same.
- `arxiv_2501.18951_draw2cut.md` — Draw2Cut (CHI 2025). Relevant for
  the SVG dialect handoff to cnc-woodblock-tools: color = toolpath
  type, kento metadata, real-time alignment for QA.
- `arxiv_2212.07941_cnc_toolpath_survey.md` — Toolpath generation
  survey. Establishes vocabulary (iso-parametric, iso-scallop,
  contour-parallel, V-carve). Useful for cnc-woodblock-tools, less so
  for chuck-mcp itself.

## The single biggest pipeline recommendation

**Move area-opening + opening-by-reconstruction INTO S6.c, BEFORE
Potrace, with parameters derived from the end-mill diameter.**

Currently chuck-mcp's S6.c reports "island pressure" then hands the
*uncleaned* mask to the SVG exporter. Potrace then independently
deletes spurs via `turdsize`. This double-bookkeeping is:

- **Inconsistent**: S6.c's pressure metric and Potrace's `turdsize`
  use different filters (S6.c is area-based; `turdsize` is contour-
  enclosed-area). Results can diverge for ring-shaped components.
- **ΔE-blind on the Potrace side**: Potrace deletes spurs without
  consulting the ΔE budget. So a "low-pressure" mask can blow its
  budget at vectorization time without S6.c knowing.
- **Tool-radius-blind**: neither filter enforces minimum *width* on
  retained features. A 1-px-wide spur survives both filters; it then
  vanishes under tool-radius offset, creating a silent geometry loss.

The fix is a single pre-Potrace cleanup step inside S6.c:

```python
# chuck-mcp/s6c/printability_repair.py
import skimage.morphology as morph

def cleanup_for_cnc(mask: np.ndarray,
                    min_feature_px: int,
                    endmill_radius_px: int) -> np.ndarray:
    """Cleanup mask in-place to be CNC-printable, ΔE-aware.

    Combines area-opening (kill small components) with
    opening-by-reconstruction (kill thin necks). Symmetric
    closing+reconstruction for negative-space speckle.
    """
    # POSITIVE space: area opening + width opening
    m = morph.area_opening(mask, area_threshold=min_feature_px**2,
                           connectivity=2)
    eroded = morph.binary_erosion(m, morph.disk(endmill_radius_px))
    m = morph.reconstruction(eroded, m)

    # NEGATIVE space: area closing + width closing
    m = morph.area_closing(m, area_threshold=min_feature_px**2,
                           connectivity=2)
    dilated = morph.binary_dilation(m, morph.disk(endmill_radius_px))
    m = morph.reconstruction(dilated, m, method='erosion')

    return m.astype(np.uint8)
```

Then Potrace runs with `turdsize = min_feature_px²` for redundancy and
`alphamax = 0.6`, `opttolerance = 0.1`.

**Why this is the biggest win**:

- Eliminates a class of silent failures (spurs that survive ΔE check
  but die at tool-radius offset)
- Unifies S6.c pressure metrics with vectorizer behavior
- Makes the printability budget physically grounded (end-mill geometry)
  rather than a pixel-count heuristic
- Costs ~20 lines of skimage. No new dependencies, no GPU.

## Recommended pipeline additions (lower-priority)

1. **Post-Potrace Visvalingam-Whyatt** with `preserve_topology=True`
   at area tolerance `(min_feature_px/4)²`. Halves SVG control-point
   count without breaking topology.

2. **Tool-radius offset preview** in S6.c. Compute the
   Minkowski-erosion of the mask by `disk(endmill_radius_px)`; if any
   component vanishes, that component will not survive the carve.
   Currently this only fails at cnc-woodblock-tools time, far from the
   solver that could fix it.

3. **Kento + toolpath-type SVG metadata**. Borrow Draw2Cut's
   convention: stroke color = toolpath type (contour / pocket /
   V-carve). Embed `<metadata>` with registration mark coordinates.
   Lets cnc-woodblock-tools generate G-code without re-inferring
   intent.

4. **Optional differentiable post-pass** (LIVE/Bezier-Splatting
   architecture) for hero prints only. Constrain to "nudge control
   points within ±1 mm of Potrace output, reject any update that
   creates a self-intersection". Off by default; user-toggleable in
   the cnc-woodblock-tools UI.

5. **Stepover / scallop calculator** in cnc-woodblock-tools. For
   relief printmaking the ink-bearing top surface needs ≤ 25-50 µm
   scallop. With a 1/16" ball nose at scallop_target = 25 µm:
   stepover ≤ √(8·r·s) = √(8·0.8·0.025) = 0.4 mm. Bake this into the
   downstream toolpath-generation defaults.

## Tool-compensation references for cherry / shina plywood

Compensation = post-vectorization Minkowski offset by tool radius,
applied to:

- **Outer contours**: offset **outward** by `endmill_radius` for cut
  along the boundary (the tool cuts on the outside of the line).
- **Inner contours (holes)**: offset **inward** by `endmill_radius`.
- **V-carve regions**: compute medial-axis + width field; depth at
  each point = `width / (2 · tan(V_angle/2))`.

Practical references:

- **Clipper2** (Angus Johnson) — open-source polygon offsetting C++
  with bindings. Robust against degenerate cases that crash JTS. The
  cnc-woodblock-tools EPS/DXF export already depends on this — keep
  it as the tool-offset primitive.
- **Shapely `.buffer(distance, cap_style=2, join_style=2)`** — wraps
  GEOS; cap_style=flat, join_style=mitre for engineered geometry.
  Equivalent to Clipper2 for our purposes, simpler dep tree.
- **VCarve / Aspire toolpath docs** — practical CAM reference for
  multi-pass relief carving (roughing pass at 60% stepover with
  larger tool, finishing pass at 20% stepover with smaller tool).

Specific compensation values per tool / material from
`web_cnc_min_feature_size_endmill_relief.md`:

| Tool         | Wood   | Buffer offset | Notes |
|--------------|--------|---------------|-------|
| 1/16" flat   | Cherry | 0.80 mm       | Standard offset |
| 1/16" flat   | Shina  | 0.95 mm       | +20% for tear-out |
| 1/8" flat    | Cherry | 1.60 mm       | Standard offset |
| 1/8" flat    | Shina  | 1.90 mm       | +20% for tear-out |
| 1/16" ball   | Cherry | 0.80 mm       | For finishing pass only |
| 60° V-bit    | Either | varies w/ depth | width = 1.155·depth |

## What's NOT in scope (deferred)

- **5-axis toolpath generation** — survey (arxiv_2212.07941) covers it
  but chuck-mcp is firmly 2.5D.
- **Curvature-aware adaptive stepover** — for ultra-fine reliefs;
  irrelevant for ink-bearing surfaces which are flat by design.
- **RL-based toolpath optimization** (arxiv_2009.14365,
  arxiv_2404.07209) — additive-manufacturing-focused; relief carving
  is subtractive and the optimization basin is much smaller.
- **3D Bas-relief generation from images** — out of scope; chuck-mcp's
  output is layered planar plates, not continuous 3D depth.

## Open questions for the swarm

1. What is chuck-mcp's source-image resolution range? The 300 DPI
   numbers above assume scan resolution; if the solver works at lower
   resolution the min_feature_px shrinks proportionally.
2. Does S6.c currently expose a tool-diameter parameter, or is it
   hardcoded to a pixel value? The morphology fix above only works if
   we plumb `endmill_diameter_mm` + `dpi` through to S6.c.
3. Is the SVG output consumed by cnc-woodblock-tools as-is, or is
   there a CAM-aware converter in between? If the latter, the
   metadata-based toolpath-type hints (Draw2Cut style) need to be
   defined there, not in chuck-mcp.

## Quick-reference: which paper for which question

| Question | Read this first |
|----------|-----------------|
| How does Potrace work?  | `web_potrace_selinger_2003.md` |
| Color vector tracing?   | `web_vtracer_visioncortex.md` |
| Mask cleanup math?      | `web_morphological_area_opening_vincent_1993.md` |
| Min feature for 1/16"?  | `web_cnc_min_feature_size_endmill_relief.md` |
| Path simplification?    | `web_visvalingam_whyatt_polygon_simplification.md` |
| Deep-learning options?  | `arxiv_2206.04655_live_*.md` (best) + others |
| Diff. vector graphics?  | `web_diffvg_li_2020.md` |
| Fast diff. rendering?   | `arxiv_2503.16424_bezier_splatting.md` |
| User-facing CNC HCI?    | `arxiv_2501.18951_draw2cut.md` |
| Toolpath taxonomy?      | `arxiv_2212.07941_cnc_toolpath_survey.md` |
