---
title: "A Comprehensive Review and New Taxonomy on Superpixel Segmentation"
authors: ["Isabela Borlido Barcelos", "Felipe de C. Belém", "Leonardo de M. João", "Zenilton K. G. do Patrocínio Jr.", "Alexandre Xavier Falcão", "Silvio Jamil Ferzoli Guimarães"]
arxiv_id: "2409.19179"
year: 2024
venue: "ACM Computing Surveys, Vol. 56, Issue 8, Article 200 (August 2024)"
doi: "10.1145/3652509"
links:
  - "https://arxiv.org/abs/2409.19179"
  - "https://arxiv.org/html/2409.19179v1"
  - "https://github.com/IMScience-PPGINF-PucMinas/superpixel-benchmark"
relevance: 9
relevance_reason: "Most authoritative recent (2024) survey of the superpixel landscape. Reviews 59 methods, proposes a useful new taxonomy organized by processing step, and provides a public benchmark. Essential reference for chuck-mcp's algorithm selection."
---

# A Comprehensive Review and New Taxonomy on Superpixel Segmentation

## Summary

ACM CSUR August 2024. Six authors from PUC Minas and Unicamp (Brazil) review 59 superpixel methods spanning 20 years and propose a taxonomy that goes beyond the old "clustering-based vs graph-based vs deep" trichotomy. They evaluate 20 representative algorithms on 9 criteria (connectivity, compactness, delineation, count control, color homogeneity, robustness, running time, stability, visual quality) and publish a benchmark at the linked GitHub repo.

## Proposed Taxonomy

The paper organizes superpixel methods by three processing stages — Initial, Main, Final — and by the level of features each method consumes:

### Main-Processing Categories (Non-CNN)

| Category | Definition | Representative methods |
|---|---|---|
| **Neighborhood-based** | Cluster pixels within a max spatial distance from a reference (seed). | SLIC, LSC, SCALP, Fuzzy SLIC |
| **Dynamic-center-update** | Update centroids online during clustering, often via priority queue. | SNIC, DBSCAN-derivatives |
| **Boundary evolution** | Iteratively update region boundaries (coarse-to-fine block exchange). | SEEDS, ETPS, Van den Bergh |
| **Path-based** | Generate regions as paths in an image graph (IFT, watershed cuts). | ISF, RISF, DISF |
| **Hierarchical** | Build a region tree obeying locality + causality. | SH, gPb-OWT-UCM |
| **Density-based** | Find density peaks in joint feature-spatial space. | QuickShift, Mean Shift |
| **Sparse linear system** | Model segmentation as a sparse matrix problem. | LSC-LR, GMMSP |
| **Data distribution-based** | Assume a parametric pixel distribution and fit. | GMMSP (Gaussian Mixture) |
| **Regional feature extraction** | Extract regional features iteratively. | DISF (Iterative Spanning Forest) |
| **Polygonal decomposition** | Decompose image into non-overlapping polygons. | SNIC-polygon, BASS |
| **Graph-based** | Use graph topology to drive segmentation. | ERS, Felzenszwalb-Huttenlocher |

The CNN-based methods get their own axis: by pixel-to-superpixel assignment strategy (k-means in feature space vs. soft assignment via softmax) and by output type (label map vs. soft membership).

### Feature-Level Categories

- **Pixel-level**: raw color, position (SLIC, SNIC, SEEDS).
- **Mid-level**: patch features, paths, gradients, boundary maps (SCALP, gPb-OWT-UCM).
- **High-level**: saliency, semantic features, object geometry (SSN, LNS-Net, AINet, SPAM).

## Benchmark Results (selected)

The benchmark evaluates 20 methods on BSDS500 and ECSSD. Headline trade-offs reported:

- **Best ICV (color homogeneity):** GMMSP, SCALP, LSC. Worst: pure SLIC at low compactness.
- **Best regularity (CO, GR):** SLIC at high compactness, SEEDS, ETPS. Worst: DISF, deep methods.
- **Best delineation (BR, ASA):** SCALP, deep methods (SSN, AINet). Worst: SLIC at high compactness.
- **Best speed:** SNIC, SLIC. Worst: ERS, MCG.
- **Best robustness to noise:** Fuzzy SLIC, GMMSP. Worst: pure SLIC (no regularization).
- **Best stability (run-to-run determinism):** SNIC, SEEDS. Worst: SLIC at low compactness (k-means depends on init).

## Relevance to chuck-mcp S3.b → S6.b Pipeline

This survey is the right reference document for chuck-mcp to consult when deciding which superpixel algorithm to upgrade to. Mapping the survey's 9 criteria onto mokuhanga jigsaw planning priorities:

| Mokuhanga need | Survey criterion | Best methods |
|---|---|---|
| Carve-able cells (no skinny strands) | Compactness, Regularity | SLIC (high compactness), SNIC, SEEDS |
| Cell ≈ one plate ink color | Color homogeneity (ICV) | GMMSP, SCALP, LSC |
| Respect line-art edges | Delineation (BR, ASA) | SCALP, SPAM/SAM-based, MCG |
| Reproducible runs | Stability | SNIC, SEEDS, Fuzzy SLIC |
| Fast iteration during prep | Running time | SNIC, SLIC |
| Survives scan noise / paper grain | Robustness | Fuzzy SLIC, GMMSP, mean shift pre-filter |
| Predictable plate count | Count control | SLIC, SNIC, GMMSP |

**The clear shortlist for chuck-mcp is SNIC + GMMSP**, with SCALP as a stretch option. SNIC dominates on speed + stability + compactness; GMMSP dominates on ICV (the most important criterion for one-cell-one-plate semantics). A two-pass S3.b that runs SNIC first then merges using GMMSP-like Gaussian-cluster posteriors over Lab would give the best of both — this is essentially the "SLIC + RAG-merge by mean color" pattern but with statistically-grounded merge decisions.

For deep methods: the survey's data confirms what the ill-posed paper argued — deep methods win on delineation but lose on regularity. For mokuhanga where regularity is *carve-ability*, deep is not the right family. SPAM is the exception because it explicitly enforces regularity post-hoc via the SAM mask constraint.

Sources:
- [arXiv:2409.19179 (HTML)](https://arxiv.org/html/2409.19179v1)
- [ACM CSUR DOI 10.1145/3652509](https://doi.org/10.1145/3652509)
- [Survey benchmark on GitHub (PUC Minas)](https://github.com/IMScience-PPGINF-PucMinas/superpixel-benchmark)
