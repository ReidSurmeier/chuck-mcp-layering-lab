---
title: "Differentiable Vector Graphics Rasterization for Editing and Learning"
authors: ["Tzu-Mao Li", "Michal Lukáč", "Michaël Gharbi", "Jonathan Ragan-Kelley"]
year: 2020
venue: "SIGGRAPH Asia 2020 (ACM ToG)"
url: "https://people.csail.mit.edu/tzumao/diffvg/"
pdf: "https://people.csail.mit.edu/tzumao/diffvg/diffvg.pdf"
code: "https://github.com/BachiLi/diffvg"
relevance: "HIGH (background) — DiffVG is the foundational primitive that LIVE, Im2Vec, VectorFusion, Bezier Splatting, etc. all build on. Not directly useful for chuck-mcp's deterministic exporter, but if we ever differentiable-render the chuck-mcp output to fine-tune plate boundaries against the source photograph, this is the engine."
tags: [diffvg, differentiable-rasterization, vector-graphics, antialiasing, gradient]
---

# DiffVG — Differentiable Vector Graphics Rasterization

## What it is

DiffVG is a 2D rasterizer for SVG primitives (paths, polygons, ellipses,
gradients) whose output is **differentiable** w.r.t. all primitive
parameters: control-point positions, colors, stroke widths, gradient
stops. This means you can compute a gradient of any raster-domain loss
back to the SVG parameters and optimize.

## Key insight

Naive rasterization is non-differentiable at primitive boundaries (the
indicator function jumps). DiffVG fixes this by **pre-filtering**:
analytically computing the area of each pixel covered by each primitive
under a small anti-aliasing kernel. The covered area is differentiable in
the primitive's control points → gradients flow.

Two anti-aliasing schemes:

1. **Analytical pre-filtering**: integrate the primitive's coverage over
   a triangular or box kernel per pixel. Exact gradients.
2. **Multi-sample anti-aliasing (MSAA)** with edge sampling: cheaper,
   stochastic gradients via Monte-Carlo edge sampling.

## API

```python
import pydiffvg

# Define primitives
path = pydiffvg.Path(num_control_points, points, is_closed)
shape_group = pydiffvg.ShapeGroup(shape_ids, fill_color, stroke_color)

# Render
scene_args = pydiffvg.RenderFunction.serialize_scene(
    canvas_w, canvas_h, shapes, shape_groups)
img = pydiffvg.RenderFunction.apply(canvas_w, canvas_h,
                                     num_samples_x, num_samples_y,
                                     seed, None, *scene_args)

# Backprop a raster-domain loss
loss = (img - target).pow(2).mean()
loss.backward()
# points.grad, fill_color.grad now populated
```

## Applications enabled

- **Image-guided SVG editing** (move control points to minimize a
  perceptual loss vs. a target image)
- **Painterly rendering** (fit N Béziers to a photo)
- **Seam-carving for vector graphics** (vector content-aware resize)
- **Generative SVG models** trained with raster supervision
  (Im2Vec, LIVE, VectorFusion all use DiffVG)

## What DiffVG does NOT do

1. **No topology preservation**. A path can self-intersect, develop
   negative-area lobes, or invert orientation during optimization.
2. **No minimum feature size**. Stroke width can drift to zero.
3. **No occlusion-order awareness**. Layer ordering is a hyperparameter
   set by the caller, not optimized.
4. **No fabrication-aware metrics**. The renderer is a visual fidelity
   engine; geometry validity is the caller's responsibility.

## Why this matters for chuck-mcp

DiffVG is the right engine for an **optional post-pass**: after the
deterministic S6.c→Potrace exporter produces a valid SVG, you could
run a short DiffVG optimization to nudge control points to better match
the original photograph (subject to constraints fixing path count and
preventing self-intersection). This is exactly the trick LIVE uses, but
the determinism of the input SVG means the optimization basin is small
and the topology is unlikely to break.

For the main loop, DiffVG is too heavy (GPU required) and too permissive
(no topology guarantees). Stay with Potrace inline; consider DiffVG only
for hand-curated hero prints.

## Performance

- C++ core with Python bindings
- CUDA backend; CPU fallback works but slow
- ~10-100ms per forward pass for moderate complexity SVGs on V100
- Bezier Splatting (arXiv:2503.16424) achieves 30x forward / 150x
  backward speedup over DiffVG for open curves — see separate note.

## Citation

Li, T-M., Lukáč, M., Gharbi, M., Ragan-Kelley, J. "Differentiable Vector
Graphics Rasterization for Editing and Learning." ACM Trans. Graphics
(SIGGRAPH Asia 2020). https://people.csail.mit.edu/tzumao/diffvg/
