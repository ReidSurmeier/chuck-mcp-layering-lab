---
title: "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods"
authors: ["Radhakrishna Achanta", "Appu Shaji", "Kevin Smith", "Aurelien Lucchi", "Pascal Fua", "Sabine Süsstrunk"]
venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 34, no. 11, pp. 2274-2282"
year: 2012
doi: "10.1109/TPAMI.2012.120"
links:
  - "https://www.epfl.ch/labs/ivrl/research/slic-superpixels/"
  - "https://www.cs.jhu.edu/~ayuille1/JHUcourses/VisionAsBayesianInference2022/4/Achanta_SLIC_PAMI2012.pdf"
  - "https://infoscience.epfl.ch/entities/publication/605ab010-dcfa-4dcf-bb55-ddf0eda015ba/articledetails"
  - "https://pubmed.ncbi.nlm.nih.gov/22641706/"
relevance: 10
relevance_reason: "Canonical SLIC paper. Already in chuck-mcp's S3.b stage (skimage.segmentation.slic). Defines compactness/sigma/min_size_factor parameters currently tuned by hand."
---

# SLIC Superpixels Compared to State-of-the-Art Superpixel Methods

## Algorithmic Core

SLIC adapts k-means clustering to a five-dimensional feature space `[L, a, b, x, y]`, where `L, a, b` are CIELAB color channels and `x, y` are pixel image coordinates. The algorithm starts by placing K cluster centers on a regular grid with spacing S = √(N/K) where N is the number of image pixels. Each center is then perturbed to the lowest-gradient position within a 3×3 neighborhood to avoid initializing on an edge or noisy pixel. Each pixel is assigned to the nearest cluster center within a 2S × 2S local search region (not the global image), which drops the per-iteration cost from O(NK) for vanilla k-means to O(N), independent of K. The distance metric mixes color and spatial terms with a user-set compactness m:

```
D = sqrt( d_lab^2 + (d_xy / S)^2 * m^2 )
```

where d_lab is Euclidean distance in CIELAB and d_xy is Euclidean distance in image coordinates. The S normalization makes m approximately invariant to the chosen K. After 10 iterations of assign/update (the paper claims convergence is usually reached in 4-10), a post-processing pass enforces 4-connectivity by relabeling small disconnected fragments to the nearest spatially-adjacent label. The TPAMI paper presents this as the natural superpixel analog of Lloyd's algorithm: locality-restricted k-means in a Lab-XY metric, with a single tunable knob (compactness) that buys regularity at the cost of boundary adherence.

## Benchmark Performance

SLIC was benchmarked on the Berkeley Segmentation Dataset (BSDS500) and an MSRA saliency subset against five state-of-the-art baselines at the time: GraphCuts (GCa10/GCb10), Normalized Cuts (NC), QuickShift (QS), TurboPixels (TP), and Watershed (WS). Two metrics dominate: under-segmentation error (UE, fraction of pixels of a ground-truth segment that "bleed" into multiple superpixels) and boundary recall (BR, fraction of ground-truth boundary pixels lying within 2 pixels of a superpixel boundary). SLIC achieves the lowest UE and highest BR across most K values (100-1000 superpixels), while running at roughly 0.5 s for a 481×321 image vs ~150 s for NC and ~10 s for QS. Memory is O(N) — just per-pixel cluster IDs and the K centers — whereas graph methods scale as O(N²). The paper concludes that SLIC's two strengths are (a) Lab-XY clustering with a local 2S × 2S search and (b) connectivity post-processing, and demonstrates these are sufficient for it to dominate the prior SOTA without any graph cut or normalized-cuts machinery.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

The current chuck-mcp S3.b stage already imports `skimage.segmentation.slic` and tunes three parameters: `compactness ∈ {13.0, 19.0}` (emma_lattice runs hotter for more regular blocks; slic mode runs looser to track color), `sigma ∈ {0.65, 1.0}` (pre-blur σ; lower σ in emma_lattice retains fine print structure), and `n_segments = area / 420 or 580`. These are exactly the knobs the Achanta paper defines as primary. The paper validates that this parameterization is reasonable, but two caveats are worth flagging for mokuhanga work: (1) SLIC was tuned and benchmarked on natural photographs where ground-truth segments are objects, not flat color regions, and (2) the connectivity post-processing in skimage uses `min_size_factor / max_size_factor` to merge fragments back, which can cause subtle boundary jitter at high-contrast color edges — exactly the kind of edges a printmaker needs to be crisp for jigsaw carving. For the cell-graph use case, SNIC (2017, same authors) and Achanta's polygon-extraction variant are the natural upgrades: they enforce connectivity by construction (no fragment merging post-hoc) and produce explicit polygon vertices that can drive carving directly.

Sources:
- [SLIC Superpixels — EPFL IVRL project page](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/)
- [TPAMI 2012 paper PDF (JHU mirror)](https://www.cs.jhu.edu/~ayuille1/JHUcourses/VisionAsBayesianInference2022/4/Achanta_SLIC_PAMI2012.pdf)
- [EPFL Infoscience record](https://infoscience.epfl.ch/entities/publication/605ab010-dcfa-4dcf-bb55-ddf0eda015ba/articledetails)
- [PubMed listing](https://pubmed.ncbi.nlm.nih.gov/22641706/)
