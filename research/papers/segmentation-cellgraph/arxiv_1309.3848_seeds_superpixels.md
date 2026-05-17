---
title: "SEEDS: Superpixels Extracted via Energy-Driven Sampling"
authors: ["Michael Van den Bergh", "Xavier Boix", "Gemma Roig", "Benjamin de Capitani", "Luc Van Gool"]
arxiv_id: "1309.3848"
year: 2012
venue: "ECCV 2012; arXiv preprint posted Sept 2013"
links:
  - "https://arxiv.org/abs/1309.3848"
  - "https://link.springer.com/chapter/10.1007/978-3-642-33786-4_2"
  - "https://link.springer.com/content/pdf/10.1007/978-3-642-33786-4_2.pdf"
  - "https://davidstutz.de/seeds/"
relevance: 7
relevance_reason: "Energy-driven boundary-evolution superpixel method. Stops at any time = produces a valid segmentation at every iteration. Real-time (30 Hz). Useful for chuck-mcp's interactive S3.b preview but less suited to high-fidelity print plate planning."
---

# SEEDS: Superpixels Extracted via Energy-Driven Sampling

## Algorithmic Core

SEEDS belongs to the "boundary evolution" family (per the 2024 Borlido Barcelos survey taxonomy). The pipeline:

1. **Initial partition.** Tile the image with a regular grid of square superpixels at the desired count K. This is the *only* "seed" step — no centroid initialization needed.

2. **Hierarchical block exchange.** SEEDS represents each superpixel as a *quadtree* of nested blocks: the full superpixel, then 1/2 splits, then 1/4 splits, down to single pixels. The algorithm iterates over levels from coarse to fine. At each level, for every block at a superpixel boundary, the algorithm tests whether moving that block from its current superpixel to an adjacent one would *reduce the energy*. If yes, the move is committed.

3. **Energy function.** For each superpixel S, maintain a color histogram H_S over the L-channel and a*b* histograms (in Lab). The energy is:
   ```
   E(partition) = Σ_S  H_S^T * (1 - H_S)  +  λ * boundary_penalty
   ```
   The histogram term is maximized by superpixels whose color distribution is concentrated in a single bin (i.e., homogeneous color); the boundary term penalizes irregular boundaries (typically a count of how many neighbor pairs across the boundary disagree).

4. **Anytime property.** Because every block exchange is a local energy decrease, the partition is *always valid* — connected superpixels, no holes — at every iteration. This means SEEDS can be stopped at any point and the result is still a usable segmentation.

The reported wall-clock is 30 Hz on a 2.8 GHz Intel i7 for 640×480 inputs, putting it firmly in the real-time camp alongside SNIC.

## Behavior and Limitations

SEEDS has two unique properties that distinguish it from the SLIC family:

- **Histograms, not centroids.** The per-superpixel state is a color histogram, not a Lab+xy centroid. This means SEEDS can capture *bimodal* color distributions within a single superpixel — useful when a superpixel straddles a soft edge. SLIC would force such a cell to split or to pick one side of the edge; SEEDS keeps the cell intact and records both modes in its histogram.
- **No compactness parameter.** Regularity is implicit in the initial grid and the per-level block size. SEEDS has fewer knobs to tune (just K and λ for the boundary penalty), which is a virtue for reproducibility.

The flip side: SEEDS' anytime property comes from a hill-climbing schedule that may not reach a global optimum. For chuck-mcp's use case where quality matters more than latency, this is a mild disadvantage versus SLIC/SNIC which iterate to convergence.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

SEEDS is a candidate for chuck-mcp's *preview* path — fast enough to update interactively when the user changes a parameter or repositions the input image. The 30 Hz claim translates to ~33ms for typical preview-resolution images.

However, for the final S3.b output that drives plate planning, SEEDS has one weakness: the histogram-based energy doesn't directly track CIELAB ΔE, which is what mokuhanga visual quality demands. The histogram is binned uniformly in RGB or Lab; bin width determines the smallest color difference SEEDS can detect, and choosing bin width is a hidden hyperparameter. SLIC/SNIC's continuous Lab distance avoids this discretization.

A practical SEEDS use: as a *second opinion* segmentation. Run SEEDS in parallel with SLIC, compute the boundary disagreement map (pixels where SEEDS and SLIC disagree about the cell boundary), and use the disagreement map as input to the QA pipeline. Disagreement regions are the ones most likely to need human review.

Sources:
- [arXiv:1309.3848 abstract](https://arxiv.org/abs/1309.3848)
- [Springer LNCS chapter (ECCV 2012)](https://link.springer.com/chapter/10.1007/978-3-642-33786-4_2)
- [Author full-text PDF](https://link.springer.com/content/pdf/10.1007/978-3-642-33786-4_2.pdf)
- [David Stutz's SEEDS review and benchmark](https://davidstutz.de/seeds/)
