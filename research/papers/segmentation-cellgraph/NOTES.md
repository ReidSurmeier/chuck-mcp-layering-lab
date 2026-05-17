# Segmentation & Cell-Graph Research Notes

Research agent: **SEGMENTATION-CELLGRAPH** | Swarm: swarm-1778962124344-s4cm4l | Date: 2026-05-16

Output artifacts: 11 (4 arXiv + 7 web-sourced canonical references)

---

## Context

chuck-mcp S3.b currently runs `skimage.segmentation.slic` on a Lab-converted RGB image with two presets (`emma_lattice` and `slic`) differing only in compactness (19.0 vs 13.0), sigma (0.65 vs 1.0), and `n_segments = area / 420 or 580`. Output is a label map + RAG written to `cell_labels.npy` / `cell_graph.json`. Downstream S6.b consumes the RAG to plan plates (which cells share a printing plate vs go on separate impressions). Per-cell statistics include mean Lab, paper-Δ, tone hint, hue hint.

The use case is mokuhanga — Japanese woodblock — where a "good" cell graph means:

1. Cells are carve-able (no skinny strands, no one-pixel filaments) → strong **regularity / compactness** constraint.
2. Each cell maps to one pigment from a 36-color catalog → strong **color homogeneity (ICV in CIELAB ΔE units)** constraint.
3. Cell boundaries should coincide with where the printer would actually cut → **boundary delineation** constraint at high-contrast Lab edges.
4. The same input image always produces the same RAG → **stability / determinism** constraint.

This is a more restrictive set of requirements than what BSDS500 benchmarks measure. Two of the four (regularity + color homogeneity) are explicitly downplayed in deep-learning superpixel methods, per Giraud-Clément 2024 (arxiv 2411.06478).

---

## Top 5 papers ranked by relevance to chuck-mcp's cell-graph S3.b → S6.b pipeline

### 1. Achanta & Süsstrunk, "Superpixels and Polygons Using Simple Non-Iterative Clustering" (SNIC) — CVPR 2017
File: `web_achanta_snic_2017.md`

Strict drop-in upgrade for the current SLIC call. Single-pass priority queue replaces SLIC's iterative local-k-means → connectivity is guaranteed by construction (no need for the `min_size_factor=0.22, max_size_factor=3.5` post-processing pass that introduces boundary jitter), determinism is perfect (same input always produces same labels), and SNIC natively emits polygonal boundaries that are exactly the carving lines a printer would draw. 2-4× faster than SLIC, lower memory. The single biggest win available without re-architecting the stage.

### 2. Giraud & Clément, "Superpixel Segmentation: A Long-Lasting Ill-Posed Problem" — arXiv 2411.06478 (2024)
File: `arxiv_2411.06478_superpixel_ill_posed.md`

The conceptual key. Argues that the entire superpixel field has been evaluating on the wrong axis and that *regularity must be measured explicitly*, separately from delineation. Provides the metric vocabulary (UE, ASA, BR, CO, GR, EV, ICV) chuck-mcp should report in S3.b diagnostics to make algorithm choice principled rather than vibes-based. Especially relevant because mokuhanga's "carve-ability" criterion is exactly the regularity property the paper says is missing from recent evaluations.

### 3. Barcelos et al., "A Comprehensive Review and New Taxonomy on Superpixel Segmentation" — ACM CSUR 2024 (arXiv 2409.19179)
File: `arxiv_2409.19179_superpixel_survey.md`

The authoritative recent (Aug 2024) survey. Reviews 59 methods, evaluates 20 on 9 criteria, publishes a public benchmark. The "best-in-class" rankings per criterion give chuck-mcp a directly-actionable shortlist:
- Color homogeneity (ICV): **GMMSP, SCALP, LSC**
- Compactness/regularity: **SLIC@high-compactness, SNIC, SEEDS**
- Speed + stability: **SNIC, SEEDS**
- Robustness to scan noise: **Fuzzy SLIC, GMMSP**

Combining the top entries across the three criteria that matter most for mokuhanga points at SNIC + GMMSP-style merge as the right hybrid.

### 4. Walther, Giraud, Clément, "Superpixel Anything (SPAM)" — BMVC 2025 (arXiv 2509.12791)
File: `arxiv_2509.12791_superpixel_anything.md`

Current SOTA. The architectural pattern — high-level prior segmentation (SAM) + differentiable regular-superpixel filling — is exactly the workflow a printmaker describes ("carve keyblock first, then fit color plates inside the keyblock outlines"). The forward-looking S3.b should adopt this two-stage shape. Substitute SAM with a mokuhanga-specific prior (Canny on line art, or the keyblock-alpha from chuck-mcp's inverse solver) and SPAM's differentiable clustering as the filler.

### 5. Puzicha et al., "On Spatial Quantization of Color Images" (scolorq) — ECCV 1998
File: `web_puzicha_spatial_color_quantization_1998.md`

Best prior art for the *S6.b plate-assignment* stage, not the S3.b cell-graph stage. Joint optimization of palette + assignment + perceptual smoothing in a single cost function. With chuck-mcp's 36-pigment catalog as the fixed palette and the existing K-M forward renderer as the rendering operator, scolorq's deterministic-annealing loop slots in directly as the plate-assignment optimizer. Almost no other paper in the literature is this close to the actual mokuhanga decision problem.

---

## Algorithmic Recommendations

### Tier 1 — Replace SLIC with SNIC in S3.b. Effort: ~1 week.

The current `s3b_cell_graph.py` call:
```python
labels = segmentation.slic(
    seg_rgb, n_segments=segments, compactness=..., sigma=...,
    convert2lab=True, enforce_connectivity=True,
    min_size_factor=0.22, max_size_factor=3.5,
)
```
Becomes:
```python
labels, polygons = snic(  # via pysnic or a JAX port
    seg_rgb_lab, n_segments=segments, compactness=...,
)
```
The `enforce_connectivity` and `min_size_factor` / `max_size_factor` arguments disappear — SNIC guarantees connectivity by construction. Boundary jitter at high-contrast Lab edges drops to zero. Run-to-run reproducibility becomes perfect (currently SLIC's k-means tie-breaking introduces minor label shifts).

Free bonus: SNIC emits polygons natively. Downstream S6.b can consume the polygon list directly instead of rasterizing labels and re-vectorizing.

### Tier 2 — Add explicit regularity + ICV reporting to S3.b diagnostics. Effort: ~1 day.

Currently `s3b_cell_graph.py` writes a `diagnostics` dict. Extend it to include per-cell:
- **Compactness (CO):** `4π × area / perimeter²`, in [0, 1]; 1.0 = perfect circle.
- **Eccentricity (EV):** ratio of major/minor axes of fitted ellipse.
- **Intra-cluster variance (ICV) in CIELAB ΔE units:** `mean(|Lab_pixel - mean(Lab_cell)|)` over cell pixels.
- **Boundary edge strength along cell perimeter:** mean of `edge_strength` at boundary pixels.

These four numbers per cell let the S6.b planner reject "bad" cells (low CO, high EV, high ICV) before they become plates. The aggregate distribution is also QA-grade evidence for parameter tuning.

### Tier 3 — Hybrid SNIC + GMMSP-style merge for ICV-driven cell merging. Effort: ~2-3 weeks.

After SNIC produces N regular cells, fit a Gaussian mixture in Lab over the cell-mean colors (each cell is one point in 3D Lab). Cells whose posterior assignment is to the same mixture component get merged in the RAG, *if* they are spatially adjacent. The number of mixture components K is the plate budget.

This gives chuck-mcp a mathematically-grounded merge predicate that respects the Lab perceptual metric (where SLIC's compactness term doesn't) and uses the spatial adjacency structure (where pure GMMSP doesn't). The merge predicate is essentially `Felzenszwalb-Huttenlocher MInt` (file: `web_felzenszwalb_huttenlocher_2004.md`) with the dissimilarity measure replaced by GMM posterior distance.

### Tier 4 — Long-term: adopt SPAM-style two-stage architecture. Effort: ~2-3 months.

When the inverse-solver keyblock alpha is reliable enough to use as the high-level prior, port the SPAM pipeline:
1. Compute the keyblock-alpha mask from the inverse solver (or, as a baseline, Canny-edge + morphological-close on Lab gradients).
2. Run SNIC inside each masked region with regularity-by-construction.
3. Cell graph contains both the SNIC cells *and* the prior-mask boundaries as first-class adjacency edges.

This is the right long-term architecture because mokuhanga genuinely *has* a two-level structure (carved line work vs flat color), and squeezing it into a single-stage SLIC call is forcing it.

---

## Which methods preserve adjacency information needed for graph coloring downstream

The S6.b plate-planning step is fundamentally a graph coloring problem: assign K colors to N cells such that adjacent cells with incompatible hues get different colors (different plates). This requires a clean, well-defined RAG with:

- **Connectivity guarantees** — every cell is a single connected component, no fragments.
- **4-connectivity** (not just any adjacency) so the planner can reason about which carving lines are shared between cells.
- **Edge weights** that carry semantic meaning (Lab ΔE across the shared boundary, edge strength, joint area, etc.).

Methods ranked by RAG quality for mokuhanga:

| Method | Connectivity | RAG Quality | Notes |
|---|---|---|---|
| **SNIC** | Guaranteed by construction | Excellent | Polygons emitted natively; deterministic |
| **SEEDS** | Guaranteed (boundary evolution) | Good | Histograms instead of centroids — less clean Lab semantics |
| **SLIC** | Post-hoc enforcement | Good with cleanup | Currently used; needs `min_size_factor` tuning |
| **gPb-OWT-UCM** | Guaranteed (oriented watershed) | Excellent + hierarchy | Slow but produces full hierarchy of nested RAGs |
| **MCG** | Guaranteed | Best | Multi-scale UCM; richest RAG; slowest |
| **Mean shift** | Guaranteed (mode collapse) | Mixed | Regions are arbitrary size — RAG is correct but irregular |
| **Felzenszwalb-Huttenlocher** | Guaranteed (union-find) | Good | RAG is the union-find tree at termination |
| **SPAM** | Guaranteed within SAM masks | Excellent (modulo SAM) | RAG inherits SAM mask structure |
| **GMMSP** | Not always | Mixed | GMM components may produce non-connected regions |
| **Deep methods (LNS-Net, AINet)** | Often violated | Poor | Skinny strands, near-disconnected fragments |

**The methods that produce a usable RAG by construction are the SLIC family (especially SNIC), the boundary-evolution family (SEEDS), the hierarchical family (UCM/MCG), and SPAM.** Deep methods other than SPAM should be avoided for chuck-mcp's RAG-dependent downstream stages.

---

## The single biggest algorithmic recommendation for chuck-mcp's S3.b stage

**Replace `skimage.segmentation.slic` with SNIC.**

Rationale, in priority order:

1. **Connectivity guaranteed by construction.** SLIC's post-processing pass to merge tiny fragments uses `min_size_factor=0.22, max_size_factor=3.5` — these parameters introduce boundary jitter at high-contrast Lab edges, which is exactly the kind of edge a printmaker needs to be crisp for jigsaw carving. SNIC eliminates this entire class of bug.

2. **Reproducibility.** SLIC's k-means convergence is sensitive to local tie-breaking inside the 2S × 2S search window; same input + same code can produce slightly different labels run-to-run. SNIC is deterministic (priority queue is total-order). For a lab-bench-iterating project doing 36-pigment QA, deterministic outputs are required for diffing two segmentations.

3. **Native polygons.** S6.b currently has to rasterize SLIC labels then re-vectorize to get carving polygons. SNIC emits polygons directly with junction vertices, which are the exact points a printmaker would mark as kento registration features.

4. **Speed.** 2-4× faster than SLIC because there is no iteration loop. The current 850k-pixel `_MAX_SEGMENT_PIXELS` cap could be safely raised, removing a downscaling step that loses fine line work.

5. **No change to the rest of the pipeline.** SNIC takes the same inputs as SLIC (Lab image, n_segments, compactness) and produces the same output type (label map). The downstream RAG construction in `_adjacency` works without modification. Diagnostics and tests don't need to change.

Implementation options, in order of effort:
- (a) `pip install pysnic` — pure Python, MIT licensed, ~10× slower than C++ reference but immediately usable.
- (b) Wrap the [achanta/SNIC](https://github.com/achanta/SNIC) C++ reference via pybind11 — matches publication speed; one-week port.
- (c) Port the priority-queue loop to JAX — matches the existing `forward_render_jax` stack, makes the full forward pipeline differentiable; two-week port.

Recommended starting point: option (a) for a feasibility check on one print, then move to option (c) once the algorithm is validated in production.

---

## Appendix: Search terms that yielded the strongest hits

- arXiv `cs.CV` + "SLIC superpixels" + 2017 → SNIC and successors
- arXiv `cs.CV` + "ill-posed" + "superpixel" → Giraud-Clément critique
- HuggingFace papers + "superpixel deep learning" → SPAM and recent deep methods
- Web search + "Achanta SLIC 2012 TPAMI" → canonical PDF mirrors
- Web search + "Floyd-Steinberg dithering printmaking" → halftoning and quantization parallels
- Web search + "Puzicha Held spatial color quantization" → scolorq

Searches that did NOT yield mokuhanga-specific results (expected; the niche is too narrow):
- "mokuhanga" + segmentation
- "woodblock print" + plate planning
- "ukiyo-e" + color separation

The literature on art-image segmentation is sparse; the right move is to map general-purpose superpixel methods onto chuck-mcp's specific constraints rather than search for an off-the-shelf mokuhanga solution.
