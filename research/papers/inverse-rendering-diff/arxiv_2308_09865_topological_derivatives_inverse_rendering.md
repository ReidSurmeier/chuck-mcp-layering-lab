# A Theory of Topological Derivatives for Inverse Rendering of Geometry

authors: Ishit Mehta, Manmohan Chandraker, Ravi Ramamoorthi
arxiv: 2308.09865
venue: ICCV 2023
project: https://ishit.github.io/td/
relevance: ICCV 2023 paper that introduces TOPOLOGICAL DERIVATIVES — a way to take a gradient step that NUCLEATES a new hole or new phase (new region/plate) directly from image residuals, instead of relying on silhouette gradients which are sparse and fail when no region exists yet to be deformed. This is the most cited solution to chuck-mcp's "new plate creation" problem: how do you let the solver SPAWN a new chroma-accent plate when the current stack can't explain a region?

Key idea (from abstract + project page):
- Standard differentiable rasterizers only give gradients on existing boundaries. If a chroma-accent (e.g., red plate) doesn't yet exist in the active stack, there's no gradient pushing the solver to create one.
- Topological derivatives compute the change in rendered image intensity from INTRODUCING a vanishing hole or phase at every pixel. That residual map is itself a gradient telling the optimizer WHERE to nucleate a new region.
- Validated on 2D closed-curve optimization and 3D surface optimization, with applications including image vectorization, vector-graphics generation from text, single-image reconstruction of shape ambigrams, and multi-view 3D reconstruction.

Algorithmic moves to import for chuck-mcp:
1. Replace the manual "should we add a red plate?" heuristic with a topological-derivative residual map computed AFTER batch 1 (light supports) is fit. Wherever the residual sign indicates "adding a new opaque/translucent layer of color X here would lower image loss," nucleate a plate there.
2. This is provably better than re-running the same flat solver with m_prior+1: a flat solver has no incentive to use the extra plate at a high-residual location specifically. Topological derivatives are pixel-local "spawn here" gradients.
3. Pair with a regularization that suppresses spurious tiny nucleations (matches chuck-mcp's existing tiny-island pressure score).

Abstract (full):
"We introduce a theoretical framework for differentiable surface evolution that allows discrete topology changes through the use of topological derivatives for variational optimization of image functionals. While prior methods for inverse rendering of geometry rely on silhouette gradients for topology changes, such signals are sparse. In contrast, our theory derives topological derivatives that relate the introduction of vanishing holes and phases to changes in image intensity. As a result, we enable differentiable shape perturbations in the form of hole or phase nucleation. We validate the proposed theory with optimization of closed curves in 2D and surfaces in 3D to lend insights into limitations of current methods and enable improved applications such as image vectorization, vector-graphics generation from text prompts, single-image reconstruction of shape ambigrams and multi-view 3D reconstruction."

Notes on integration:
- For 2D vector-style problems (chuck-mcp is 2D in image space), the framework is implemented as topological derivative maps over a base region — natural fit for DiffVG-style differentiable rasterizers.
- Hole nucleation = "carve a window in this plate" → useful when chuck-mcp needs an underlayer to be punched-through where a later plate's color should show.
- Phase nucleation = "add a new colored region" → useful when chuck-mcp needs a new chroma/accent plate.

Reference for further reading: project page https://ishit.github.io/td/ links to code, supplementary, and BibTeX.
