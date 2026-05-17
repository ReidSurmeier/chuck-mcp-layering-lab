---
title: "Efficient Graph-Based Image Segmentation"
authors: ["Pedro F. Felzenszwalb", "Daniel P. Huttenlocher"]
venue: "International Journal of Computer Vision (IJCV), 59(2), 167-181"
year: 2004
doi: "10.1023/B:VISI.0000022288.19776.77"
links:
  - "https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf"
  - "https://cs.brown.edu/people/pfelzens/segment/"
  - "http://vision.stanford.edu/teaching/cs231b_spring1415/papers/IJCV2004_FelzenszwalbHuttenlocher.pdf"
  - "https://link.springer.com/article/10.1023/B:VISI.0000022288.19776.77"
relevance: 8
relevance_reason: "Foundational alternative to SLIC: graph-based, edge-driven, naturally produces a region adjacency graph as its primary output. Excellent for hue-shift-aware merging in mokuhanga where regions should be merged by perceptual similarity, not k-means cluster radius."
---

# Efficient Graph-Based Image Segmentation (Felzenszwalb-Huttenlocher 2004)

## Algorithmic Core

The paper builds a weighted graph G = (V, E) where each pixel is a vertex and edges connect 4-connected (or 8-connected) neighbors. Edge weights are the L2 distance between the two pixels in feature space — typically RGB or grayscale intensity, optionally Gaussian-smoothed first with σ=0.8. Segmentation proceeds by a Kruskal-like union-find: edges are sorted in non-decreasing weight order, and for each edge (u, v) with weight w, the two components C_u and C_v containing u and v are merged iff:

```
w  ≤  MInt(C_u, C_v)
    = min( Int(C_u) + τ(C_u),  Int(C_v) + τ(C_v) )
```

where Int(C) is the maximum weight in the minimum spanning tree of component C (the largest edge "internal" to C) and τ(C) = k / |C| is a size-dependent slack term. The constant k is the only user parameter; it controls how aggressively small components get absorbed (a larger k means looser merging and bigger regions). The algorithm runs in O(m log m) where m is the number of edges — essentially the cost of sorting the edge list. The decision rule has a nice property: components are merged precisely when the *external* dissimilarity (the edge weight bridging them) is no greater than the *internal* dissimilarity (the largest spanning-tree edge inside either component), adjusted by τ. This produces segmentations that satisfy a global property (no two adjacent regions could profitably be merged) even though the algorithm makes purely greedy local decisions.

## Behavior and Limitations

Felzenszwalb-Huttenlocher (often called "felzenszwalb" or "fz") is genuinely fast — sub-second on a 481×321 image — and it produces a region adjacency graph as its native output: the union-find structure at termination *is* the RAG. Region sizes vary wildly depending on local image structure: smooth regions become one giant component, textured regions break into many small ones. This is the opposite of SLIC's regularity-by-construction. Pre-smoothing matters a lot; without σ=0.8 Gaussian blur, JPEG artifacts and shot noise create thousands of tiny regions. The min_size parameter is typically used to merge components below ~50-100 pixels back into their neighbors as a post-pass. fz tends to under-segment large flat areas and over-segment textured ones — the exact opposite trade-off from SLIC.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

fz is a complementary primitive to SLIC, not a replacement. For mokuhanga jigsaw planning, the right pipeline likely uses both:

1. **SLIC/SNIC for the lattice.** Get a regular K=N/420 superpixel grid first so that small color details get *some* representation. SLIC's compactness term guarantees no super-tiny or super-elongated cells.
2. **fz-style RAG merging on top.** Treat the SLIC cells as nodes and apply Felzenszwalb's MInt predicate (or skimage's `merge_hierarchical`) to merge adjacent cells whose Lab edge weight is below a paper-Δ-aware threshold. The k parameter becomes the "perceptual ΔE threshold" — set it to JND for the 36-pigment catalog.

This hybrid is what `skimage.future.graph.rag_mean_color` + `merge_hierarchical` already implements. The chuck-mcp pipeline appears to compute adjacency edge strengths in `s3b_cell_graph.py` (`_adjacency` function with edge_strength input), which is the right hook to plug a Felzenszwalb-style merging predicate into. The single technical opportunity: replace the current adjacency thresholding (whatever it currently uses) with the MInt(C_u, C_v) = min(Int(C_u) + k/|C_u|, Int(C_v) + k/|C_v|) rule. This gives perceptually-driven cell merging that respects component-size context: a small cell next to a big uniform cell will merge readily; two big uniform cells with a sharp Lab boundary will resist merging even if the local color delta is moderate.

A second use: fz directly on the un-quantized Lab image (no SLIC pre-step) gives "natural" segmentation that respects mokuhanga hue zones. Useful as a second opinion / sanity check during the planning step.

Sources:
- [IJCV 2004 paper PDF (Brown)](https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)
- [Felzenszwalb project page with code](https://cs.brown.edu/people/pfelzens/segment/)
- [Stanford CS231b mirror](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/IJCV2004_FelzenszwalbHuttenlocher.pdf)
- [Springer official record](https://link.springer.com/article/10.1023/B:VISI.0000022288.19776.77)
