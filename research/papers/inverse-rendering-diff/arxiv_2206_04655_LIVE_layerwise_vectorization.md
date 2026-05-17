# Towards Layer-wise Image Vectorization (LIVE)

authors: Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, Humphrey Shi
arxiv: 2206.04655
venue: CVPR 2022 (Oral)
code: https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization (and https://github.com/ma-xu/LIVE)
relevance: THE foundational layer-wise differentiable image-vectorization paper that all the chuck-mcp staged solver ideas inherit from. Builds on DiffVG. Progressively adds Bezier paths one batch at a time, optimizes ALL existing paths each round, uses a UDF (Unsigned Distance Function) contour loss to keep new paths from drifting, and a component-wise initialization that picks the new path's starting location from the largest mis-fit color cluster.

Why this is the closest published proxy for the chuck-mcp staged solver:
- Each LIVE iteration = one chuck-mcp "batch."
- Component-wise initialization = chuck-mcp's role-aware warmstart.
- All-paths joint reoptimization = chuck-mcp's "bounded feedback from later batches into earlier ones."
- UDF loss = a soft trust region around the existing path geometry.

Algorithmic moves to import for chuck-mcp:
1. Use LIVE's progressive add-and-reoptimize loop as the OUTER LOOP of chuck-mcp's staged solver. The inner L-BFGS-B becomes the per-batch refinement.
2. Use LIVE's component-wise initialization (largest unfit connected component in the residual) as the seed for chuck-mcp's next plate, instead of guessing m_prior up front.
3. Use LIVE's UDF loss as the trust-region prior on earlier batches: penalize old plates from moving outside their existing contour while later batches are added. Direct answer to chuck-mcp's "let early batches move only inside a low-frequency trust region."

Abstract:
LIVE proposes a layer-wise image vectorization method that progressively converts a raster image into an SVG of Bezier paths. The method preserves the input image topology in the order of geometric primitives and uses a newly-designed loss + component-wise path initialization technique. Built on DiffVG. Generates compact SVGs that reconstruct the input with only a few paths while remaining semantically consistent with human perception.

Key technical pieces (from the CVPR paper):
- DiffVG differentiable rasterizer for end-to-end gradient flow from rendered raster image back to vector path parameters.
- Component-wise initialization: detect the largest connected mis-fit component in the current residual, place the new path's circular Bezier loop at its centroid.
- UDF (Unsigned Distance Function) loss: weights pixel-wise error inversely with distance to the path contour, so the new path "stays put" near its initialized contour.
- Xing self-intersection loss for control-point geometry validity.
- Joint reoptimization of all paths each round (not just the new one) — gives later batches the ability to nudge earlier batches when their explanation is bad.

References to follow up:
- DiffVG (Li et al. 2020, SIGGRAPH Asia): differentiable rasterizer foundation.
- Segmentation-guided Layer-wise Vectorization (Zhou et al. 2024, arxiv 2408.15741): direct successor — uses Laplacian + Otsu + watershed segmentation instead of color buckets for next-path placement.
- Optimize and Reduce (Hirschorn et al. 2023, arxiv 2312.11334): the top-down counterpart.
- Layered Vectorization via Semantic Simplification (Wang et al. 2024, arxiv 2406.05404): replaces the residual-driven seed with a semantic simplification, more relevant for chuck-mcp's role hierarchy.
- Clair Obscur (Lin et al. 2025, arxiv 2511.20034): adds illumination-aware albedo/shade/light decomposition on top of LIVE-style layering.
