---
title: "Bezier Splatting for Fast and Differentiable Vector Graphics Rendering"
authors: ["Xi Liu", "Chaoyi Zhou", "Nanxuan Zhao", "Siyu Huang"]
arxiv_id: "2503.16424"
year: 2025
venue: "arXiv preprint (v4 2025)"
url: "https://arxiv.org/abs/2503.16424"
relevance: "MEDIUM — Drop-in faster replacement for DiffVG. 30x forward / 150x backward for open curves. Relevant if chuck-mcp ever adds a differentiable post-pass to fine-tune control points. Also exports to standard SVG XML."
tags: [differentiable-rasterization, gaussian-splatting, bezier, performance]
---

# Bezier Splatting — Fast Differentiable Vector Graphics

## Idea

Replace DiffVG's analytical pre-filtering with **2D Gaussian splats along
the Bezier curve**. Each curve is sampled at fixed parameter intervals
and a small Gaussian footprint is rendered at each sample. The Gaussian
mean and covariance are differentiable in the underlying control points,
so gradient flow is preserved.

This is the Gaussian Splatting trick (originally for 3D radiance fields)
applied to 2D vector primitives.

## Why it's faster than DiffVG

- DiffVG: analytical coverage requires solving cubic equations per pixel
  per primitive; expensive especially for backward.
- Bezier Splatting: forward = render N Gaussians (cheap GPU op),
  backward = compute Gaussian gradients (standard GS code).

Reported speedups (paper Table 2):
- 30x forward rasterization for open curves vs DiffVG
- 150x backward rasterization for open curves vs DiffVG

## Adaptive densification

Like 3DGS, this paper does pruning + densification: low-contribution
Gaussians are pruned, and high-error regions get more samples. This
escapes local minima better than DiffVG's fixed parameterization.

## SVG interoperability

Final output is converted to standard XML SVG (Bezier paths with
control points), so it plugs into the same downstream tools as DiffVG.

## Limitations for chuck-mcp

Same as DiffVG, plus:

1. **Open-curve-optimized** — the speedup is most pronounced for open
   curves. chuck-mcp produces closed-region masks; closed-curve speedup
   is smaller (paper reports ~10x).
2. **Gaussian smear at boundary** — edges are softer than DiffVG.
   Possibly a problem if downstream CAM needs sharp edges; need to
   verify with our pipeline.
3. **No topology guarantees** — same caveats.

## Recommended posture

If chuck-mcp adds a differentiable post-pass (LIVE-style nudging on top
of Potrace output), use Bezier Splatting instead of DiffVG. Otherwise
not relevant.

## Citation

Liu, X. et al. "Bezier Splatting for Fast and Differentiable Vector
Graphics Rendering." arXiv:2503.16424, 2025.
