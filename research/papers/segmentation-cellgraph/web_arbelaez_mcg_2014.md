---
title: "Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation"
authors: ["Pablo Arbeláez", "Jordi Pont-Tuset", "Jonathan T. Barron", "Ferran Marques", "Jitendra Malik"]
venue: "CVPR 2014 (initial), TPAMI 2017 (extended)"
arxiv_id: "1503.00848"
year: 2014
links:
  - "https://arxiv.org/abs/1503.00848"
  - "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/resources/MCG_CVPR2014.pdf"
  - "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/"
  - "https://github.com/jponttuset/mcg"
relevance: 8
relevance_reason: "Multi-scale extension of OWT-UCM. For chuck-mcp, gives a 3-scale UCM that captures both fine print details (line art) and broad color regions (paper-tinted areas) in one hierarchy."
---

# Multiscale Combinatorial Grouping (MCG)

## Algorithmic Core

MCG is gPb-OWT-UCM run at three image scales (0.5×, 1.0×, 2.0×) and combined into a single hierarchical region tree. The three contributions:

1. **Fast Normalized Cuts.** The bottleneck in gPb is the eigensolver for the affinity matrix. MCG introduces an approximate eigensolver based on the connection between normalized cuts and the *Eigen-CutE* method — gets the top eigenvectors 20× faster with no measurable F-measure loss on BSDS500.

2. **Multiscale Hierarchical Segmenter.** Compute OWT-UCM independently at each of the three scales, then *align* their region trees by intersecting boundaries. The result is a single UCM whose levels combine evidence from all three scales. Fine-scale evidence pushes boundaries to be locally precise; coarse-scale evidence enforces global region coherence. Together they recover both texture-rich detail and uniform-color planes.

3. **Combinatorial Grouping for Object Proposals.** Given the multiscale region tree, enumerate ~2000 object candidates per image by combining adjacent regions and ranking by a learned score. This is the part that made MCG famous for R-CNN-era detection pipelines (replaced by Selective Search in some workflows).

## Benchmark Performance

On BSDS500, MCG-UCM achieves boundary F-measure 0.747 and segmentation-covering 0.61 — both improvements over single-scale OWT-UCM. On PASCAL VOC 2012 object segmentation, MCG candidates achieve >0.90 recall with ~2000 proposals, comparable to selective search at the time. The TPAMI 2017 extension adds *Singlescale Combinatorial Grouping* (SCG), a faster variant that runs in <5s per BSDS image at minor quality cost.

## Relevance to chuck-mcp S3.b → S6.b Pipeline

For chuck-mcp's specific problem — mokuhanga jigsaw planning on prints that mix fine line art with broad flat color regions — multiscale matters more than for natural photography. A single-scale SLIC at fixed K either over-segments the line work (good for detail, but produces hundreds of tiny one-pixel cells that the planner must merge) or under-segments the color blocks (good for plate count, but loses kento-mark structure at line junctions).

MCG's three-scale hierarchy is the right primitive: fine scale resolves line work into segments aligned with the carving plan; coarse scale resolves flat color zones into single cells that map to single plates. The combined UCM hierarchy lets the S6.b planner pick the threshold *per region* if needed — fine threshold inside line-work bounding boxes, coarse threshold inside flat-color regions.

Concrete integration plan if MCG is adopted:

1. Run MCG once per S3.b call (cost: ~10-30s per megapixel on CPU; faster with SCG variant).
2. Use the multiscale UCM as the "merge tree" input to S6.b plate budgeting.
3. The S6.b solver becomes: choose a UCM threshold per region (or globally) such that total plate count ≤ budget and no plate contains incompatible hue regions.

Cost concern: MCG is slower than SLIC by 1-2 orders of magnitude. For interactive iteration during prep this is too slow; for final plate-planning before carving, it's likely fine. A two-tier S3.b ("fast SLIC for preview, MCG for final") matches the workflow.

Sources:
- [arXiv:1503.00848 (TPAMI extended version)](https://arxiv.org/abs/1503.00848)
- [CVPR 2014 paper PDF (Berkeley)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/resources/MCG_CVPR2014.pdf)
- [MCG Berkeley project page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/)
- [MCG reference code on GitHub](https://github.com/jponttuset/mcg)
