# Differentiable Vector Graphics Rasterization for Editing and Learning (DiffVG)

authors: Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, Jonathan Ragan-Kelley
venue: ACM TOG 39(6), SIGGRAPH Asia 2020
paper: https://people.csail.mit.edu/tzumao/diffvg/diffvg.pdf
code: https://github.com/BachiLi/diffvg
relevance: Foundational paper for the entire LIVE / vectorization line of work. Defines the differentiable rasterizer (analytical prefilter + multisample anti-aliasing variants) that makes gradients of rendered raster pixels w.r.t. vector path control points / fill colors well-defined. Everything below (LIVE, Segmentation-guided Vectorization, Layered Semantic Simplification, Clair Obscur, Optimize&Reduce, VectorFusion, SVGDreamer) sits on top of DiffVG.

Not on arXiv (TOG-only). Listed here as a reference, not a separate arxiv artifact.

Key technical contributions:
- Observation: vector graphics rasterization is differentiable after pixel prefiltering. They build the gradient by differentiating through the prefilter convolution rather than the discontinuous coverage indicator.
- Two prefiltering options:
  1. Analytical prefilter — fast, can produce artifacts at sharp corners.
  2. Multisample anti-aliasing — slightly slower, unbiased high-quality gradients.
- Supports polygons, quadratic & cubic Bezier curves, circles, ellipses, gradient + solid fills, strokes.
- PyTorch + TensorFlow bindings; this is the workhorse renderer most subsequent layer-wise vectorization papers use.

Why chuck-mcp should care:
- chuck-mcp currently has a JAX forward render. If chuck-mcp wants to support vector path output directly, DiffVG (or a JAX port — see diffvg-jax forks) gives the gradient backbone for plate boundary geometry, not just plate alpha. That unlocks geometry refinement during the inverse solve, instead of only doing it post-hoc in the CNC pipeline.
- The analytical prefilter is fast enough for interactive use; the multisample variant is the one to use when chuck-mcp wants gradient signal through sharp jigsaw boundaries.

Reference for further reading: project page https://people.csail.mit.edu/tzumao/diffvg/ has slides + supplementary that walk through the prefilter derivation.
