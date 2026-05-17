---
title: "Contour Detection and Hierarchical Image Segmentation"
authors: ["Pablo Arbeláez", "Michael Maire", "Charless Fowlkes", "Jitendra Malik"]
venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 33(5), 898-916"
year: 2011
doi: "10.1109/TPAMI.2010.161"
links:
  - "https://people.eecs.berkeley.edu/~malik/papers/arbelaezMFM-pami2010.pdf"
  - "https://www.semanticscholar.org/paper/Contour-Detection-and-Hierarchical-Image-Arbel%C3%A1ez-Maire/0e5a262bf59b68ba8a7a1103d16fa33a9f5ffc28"
  - "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
relevance: 9
relevance_reason: "gPb-OWT-UCM produces a hierarchy of nested closed contours from any pixel-level contour detector. For mokuhanga, this gives a multi-scale view of where jigsaw cuts could happen — pick the threshold that matches your block budget."
---

# Contour Detection and Hierarchical Image Segmentation (gPb-OWT-UCM)

## Algorithmic Core

The Arbeláez 2011 TPAMI paper has three independently-useful pieces, all of which connect to chuck-mcp.

**Piece 1: gPb (globalized Probability of boundary).** A per-pixel boundary detector that combines (a) local mPb cues from brightness, color (Lab a-channel and b-channel separately), and texture gradients at four orientations and three scales, with (b) a global spectral grouping cue derived from the top-k eigenvectors of an intervening-contour affinity matrix. gPb outputs an oriented boundary probability map at every pixel. State-of-the-art F-measure on BSDS500 at publication time (0.71).

**Piece 2: OWT (Oriented Watershed Transform).** Given the oriented gPb map, OWT runs a watershed seeded at local gPb minima but weighted by the *oriented* gradient at each pixel. The output is a fine-grained over-segmentation where every region boundary is a closed contour. This fixes a long-standing watershed problem (boundaries that don't close into rings) by using the orientation channel of the contour detector.

**Piece 3: UCM (Ultrametric Contour Map).** Starting from OWT's over-segmentation, regions are merged by an order-of-saliency criterion: at each step, the lowest-saliency boundary is dissolved and the two regions on either side merge. The history of these merges defines a hierarchy. Storing the merge level of every boundary pixel in a single image gives the UCM: thresholding the UCM at level t produces, by construction, a valid segmentation with all closed contours of saliency ≥ t. The hierarchy is *ultrametric* — any two regions in the tree have a well-defined merge level — so the UCM single-image representation captures the entire hierarchy with no information loss.

## Benchmark and Significance

On BSDS300 and BSDS500, gPb-OWT-UCM held the SOTA on both contour-detection F-measure and segmentation-covering metrics for several years (until deep contour detectors like HED and structured edge detection arrived). It also defined a benchmark methodology that's still used: precision-recall on boundaries, plus segmentation-covering, plus probabilistic Rand index, evaluated against the multi-annotator BSDS ground truth.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

The UCM idea is the most under-used segmentation concept for printmaking. Mokuhanga planning is inherently a multi-scale question: how many plates can the printer afford, and where should the boundary go at each plate budget? UCM gives the answer in a single representation:

- Threshold the UCM low → many small cells, one plate per cell, finest possible jigsaw detail
- Threshold the UCM high → few large cells, fewer plates, broader color regions
- Pick the threshold that matches the user's plate budget for the print

This is a strict generalization of SLIC's "number of segments" knob. Where SLIC forces you to pick K upfront and rerun if you want a different granularity, UCM lets you scan the full hierarchy from a single computation. For chuck-mcp's S6.b plate-budgeting decision, this is exactly the right abstraction: instead of forcing the planner to commit to a single K for S3.b, hand it the UCM and let it pick its own threshold per-print.

Two integration paths:

1. **Lightweight.** Run SLIC at high K (say K=2000), build a RAG, then apply skimage's `merge_hierarchical` with a saliency function = max Lab gradient across the boundary. The resulting merge history *is* a UCM over the SLIC cells. This is one new function in `s3b_cell_graph.py`.

2. **Full gPb-OWT-UCM.** Use the published OpenCV / Berkeley reference implementation. Higher quality boundaries but order-of-magnitude slower (gPb costs minutes per megapixel without GPU). For high-end mokuhanga commissions, the cost is justifiable. The Berkeley project page (linked) ships MATLAB and C++ reference code; modern reimplementations like Pi Pi Sun's "Holistically-Nested Edge Detection" replace gPb with a CNN to recover the speed.

Note: MCG (Multiscale Combinatorial Grouping, Arbeláez et al. CVPR 2014) extends OWT-UCM by running it at three scales and combining the hierarchies. It produces a multiresolution UCM that's even better suited to the mokuhanga "how many plates?" question.

Sources:
- [PAMI 2011 paper (Berkeley EECS)](https://people.eecs.berkeley.edu/~malik/papers/arbelaezMFM-pami2010.pdf)
- [Berkeley grouping project page (code + data)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/)
- [Semantic Scholar record](https://www.semanticscholar.org/paper/Contour-Detection-and-Hierarchical-Image-Arbel%C3%A1ez-Maire/0e5a262bf59b68ba8a7a1103d16fa33a9f5ffc28)
