---
title: "Mean Shift: A Robust Approach Toward Feature Space Analysis"
authors: ["Dorin Comaniciu", "Peter Meer"]
venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 24(5), 603-619"
year: 2002
doi: "10.1109/34.1000236"
links:
  - "https://ieeexplore.ieee.org/document/1000236/"
  - "https://comaniciu.net/Papers/MsAnalysis.pdf"
  - "https://www.ipol.im/pub/art/2019/255/article_lr.pdf"
relevance: 7
relevance_reason: "Classical color-space segmentation that converges modes by gradient ascent in density. Naturally groups pixels by perceptual color even across spatial discontinuities — useful as a sanity-check second segmentation for chuck-mcp jigsaw planning."
---

# Mean Shift: A Robust Approach Toward Feature Space Analysis

## Algorithmic Core

Mean shift is a non-parametric mode-seeking algorithm. For an image, embed every pixel as a 5D point `(L, a, b, x, y)` (joint color + spatial). The probability density at any point z is estimated by a kernel density estimator:

```
f(z) = (1/n) * Σ_i  K_h( z - x_i )
```

with kernel K_h (usually Epanechnikov or Gaussian) of bandwidth h. The *mean-shift vector* at z is the gradient of f(z) divided by f(z) — the local "direction of greatest density increase." Iterating z ← z + mean_shift(z) converges every starting point to the nearest local mode of f. Two pixels that converge to the same mode are assigned the same label. The segmentation thus has *exactly as many regions as the density has modes*, with no K parameter; the only knobs are the spatial bandwidth h_s and color bandwidth h_r (two scalars, separating spatial and feature parts of the joint kernel).

The Comaniciu-Meer paper proves convergence of the mean-shift sequence to a stationary point of f and demonstrates two vision applications:

- **Discontinuity-preserving smoothing.** Replace each pixel value by the mode of f reached from that pixel — preserves edges (which are valleys in f) while smoothing within regions.
- **Image segmentation.** Group pixels by shared converged mode, then run a small post-pass merging neighboring regions whose modes are within color-bandwidth h_r.

## Behavior and Limitations

Mean shift is unique among classical segmenters in that it has *no notion of regularity* and *no notion of spatial scale beyond h_s*. Regions are exactly as large as the local density supports — a uniform flat color extends across the whole region it occupies, a textured zone breaks into many small modes. It's also robust to outliers (the gradient ascent in density is dominated by where mass lives, not where points scatter), which means high-frequency JPEG noise rarely corrupts the segmentation. The major weaknesses are (a) wall-clock cost — naive implementation is O(n²) per iteration, though approximate nearest-neighbor structures bring it down to O(n log n), and (b) bandwidth tuning is hard: too small h_r and you get hundreds of regions; too large and you collapse semantic boundaries.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

Mean shift sits in a different niche than SLIC. SLIC produces *regular* superpixels of pre-specified size; mean shift produces *natural* color regions of arbitrary size. The two are complementary:

1. **As a secondary segmentation for sanity-checking the SLIC cells.** Run mean shift with a perceptually-meaningful color bandwidth (h_r = ΔE_JND ≈ 2.3 in CIELAB) and a spatial bandwidth matching the print's smallest meaningful feature (h_s ≈ 8-16 pixels for typical mokuhanga). The resulting "natural" regions can be compared to SLIC's enforced lattice: where they disagree wildly, the SLIC tuning is wrong for that image and S3.b should adapt.

2. **As input to S6.b plate planning when regularity isn't needed.** For prints with very small plate budgets (3-5 plates total), the regularity of SLIC superpixels is not useful — what matters is finding the K = budget natural color modes in the print. Mean shift with low h_r gives exactly this and is the right tool here.

3. **For *post-processing* the cell graph.** After SLIC produces N cells, run mean shift in cell-centroid color space (each cell is now one point in Lab) to cluster *cells* (not pixels) by perceptual color. This is much cheaper than running mean shift at pixel level and produces the natural "plate count" decomposition needed downstream.

Mean shift's discontinuity-preserving smoothing pass is also useful as a *pre*-processing step for SLIC. Many high-resolution prints have paper grain or scan noise that creates phantom Lab gradients at the SLIC resolution; smoothing first with mean shift removes these without blurring across true ink edges. The skimage / OpenCV `pyrMeanShiftFiltering` implements this directly.

Sources:
- [TPAMI 2002 paper at IEEE Xplore](https://ieeexplore.ieee.org/document/1000236/)
- [Comaniciu mirror (PDF)](https://comaniciu.net/Papers/MsAnalysis.pdf)
- [IPOL: An Implementation of the Mean Shift Algorithm](https://www.ipol.im/pub/art/2019/255/article_lr.pdf)
