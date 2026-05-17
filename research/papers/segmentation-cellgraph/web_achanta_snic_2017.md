---
title: "Superpixels and Polygons Using Simple Non-Iterative Clustering"
authors: ["Radhakrishna Achanta", "Sabine Süsstrunk"]
venue: "CVPR 2017"
year: 2017
links:
  - "https://openaccess.thecvf.com/content_cvpr_2017/papers/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.pdf"
  - "https://www.epfl.ch/labs/ivrl/research/snic-superpixels/"
  - "https://github.com/achanta/SNIC"
  - "https://pypi.org/project/pysnic/"
relevance: 10
relevance_reason: "Direct drop-in upgrade for SLIC in chuck-mcp S3.b. Connectivity guaranteed by construction (no post-hoc fragment merging), non-iterative (lower variance run-to-run), and produces polygons natively — exactly what S6.b jigsaw planning needs."
---

# Superpixels and Polygons Using Simple Non-Iterative Clustering (SNIC)

## Algorithmic Core

SNIC replaces SLIC's iterative local-k-means assignment with a single-pass priority-queue region growing. Seeds are placed on a regular grid (same as SLIC), but instead of repeatedly reassigning every pixel inside a 2S × 2S window, SNIC inserts the 4-neighbors of each seed into a global priority queue keyed by the same 5D CIELAB+xy distance SLIC uses. The min-element is popped, assigned to its nearest centroid, and its unvisited 4-neighbors are pushed. Centroids are updated *online* — each assigned pixel adds its `(L, a, b, x, y)` to its centroid's running average. Because every pixel is assigned exactly once and only via a connected expansion from a seed, the output is provably 4-connected with no post-processing pass. The algorithm runs in O(N log N) (priority queue ops) with O(N) memory — better cache behavior than SLIC because there is no second pass over the image. The distance metric is identical to SLIC:

```
D^2 = ||c_pixel - c_centroid||_Lab^2 + (m/S)^2 * ||xy_pixel - xy_centroid||^2
```

SNIC introduces a second contribution: by tracking, for each superpixel, the *outer* edges (pixels whose 4-neighbors belong to a different label), it can extract a planar polygonal partition directly. Each superpixel becomes a polygon with vertices at junctions of 3 or more labels and edges traced along label boundaries, then Douglas-Peucker simplified.

## Benchmark Performance

On BSDS500, SNIC matches or beats SLIC, SEEDS, ETPS, and LSC on under-segmentation error and boundary recall at every superpixel count from 100 to 1000. Wall-clock is 2-4× faster than SLIC because there's no iteration loop, and memory is lower because the centroid table doesn't need to persist across iterations. Critically for printmaking applications: SNIC's run-to-run variance is zero (no random init given fixed seeds), whereas SLIC's k-means convergence depends on tie-breaking in the local window and produces non-deterministic minor label shifts.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

This is the single highest-leverage upgrade for chuck-mcp S3.b. The current code in `backend/services/v23/stages/s3b_cell_graph.py` calls `segmentation.slic(..., enforce_connectivity=True, min_size_factor=0.22, max_size_factor=3.5)` to chase down fragments. Those fragment-merge parameters are the only thing standing between SLIC output and a clean RAG, and they introduce boundary jitter at exactly the high-contrast Lab edges (e.g. dark line work against paper) that drive jigsaw plate planning. SNIC eliminates this class of bug by construction.

Three concrete chuck-mcp benefits:

1. **Polygon output for free.** The S6.b stage currently has to convert `cell_labels.npy` into vector polygons before sending to a carving plan. SNIC emits polygons natively, with vertices snapped to label junctions — exactly the points a printmaker would mark as kento registration features. This collapses a downstream rasterize→vectorize step.
2. **Deterministic adjacency graph.** Because SNIC is non-iterative, the same input image always produces the same labels and the same RAG. Reproducibility matters for the lab-bench iterations the project is doing on the 36-pigment catalog.
3. **Lower OOM risk on large prints.** SLIC's iteration loop holds two label arrays in memory; SNIC holds one. For the 850k-pixel cap currently in `_MAX_SEGMENT_PIXELS`, SNIC could lift that ceiling without breaking the GPU/CPU memory envelope.

Implementation paths: (a) `pip install pysnic` (MIT, pure-Python, slow), (b) the C++ reference at `github.com/achanta/SNIC` (5-10× faster, needs a Python binding), or (c) port the priority-queue loop to JAX to match the existing forward_render_jax stack. Option (c) is the most aligned with the v23 codebase.

Sources:
- [CVPR 2017 paper PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.pdf)
- [EPFL IVRL SNIC project page](https://www.epfl.ch/labs/ivrl/research/snic-superpixels/)
- [Reference C++ implementation (achanta/SNIC)](https://github.com/achanta/SNIC)
- [pysnic Python binding](https://pypi.org/project/pysnic/)
