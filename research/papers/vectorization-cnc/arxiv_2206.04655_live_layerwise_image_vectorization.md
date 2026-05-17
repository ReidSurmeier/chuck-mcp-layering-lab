---
title: "Towards Layer-wise Image Vectorization (LIVE)"
authors: ["Xu Ma", "Yuqian Zhou", "Xingqian Xu", "Bin Sun", "Valerii Filev", "Nikita Orlov", "Yun Fu", "Humphrey Shi"]
arxiv_id: "2206.04655"
year: 2022
venue: "CVPR 2022 (Oral)"
url: "https://arxiv.org/abs/2206.04655"
code: "https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization"
relevance: "HIGH — LIVE is the closest neural method to chuck-mcp's per-plate layered model. Produces compact SVGs with closed Bezier paths in layer order. However, runtime per image is O(minutes) on a GPU and topology of each path is not provably non-self-intersecting; not suitable for the inner loop of S6.c but could be a high-quality offline alternative for hero images."
tags: [vectorization, deep-learning, diffvg, bezier, layer-wise, topology]
---

# LIVE — Layer-wise Image Vectorization

## What it does

LIVE converts a raster image into a layered SVG of closed cubic Bézier
paths. Unlike DiffVG (which optimizes a fixed-cardinality bag of paths)
or Im2Vec (which generates from a learned latent), LIVE adds paths
**progressively**, one layer at a time, until the residual reconstruction
error is below threshold.

## Why this is interesting for chuck-mcp

The "layer-wise" output structure mirrors chuck-mcp's plate model:

- chuck-mcp solves a layered partition; each layer is one ink color
- LIVE outputs a sequence of overlapping Bezier paths in z-order
- The visual semantics are similar: lower layers are partly occluded by
  upper layers
- LIVE's component-wise path initialization uses the **connected
  components of the raster** as seeds — exactly the same primitive
  chuck-mcp's S6.c uses for component-pressure measurement

This means: if you wanted to *replace* the per-plate Potrace step with
something that's aware of the cross-plate layering, LIVE is the closest
research artifact.

## Method

### Component-wise path initialization

Given the input raster, identify connected components of the unexplained
residual. For each new layer, place a closed cubic Bezier ring around the
*largest* component of the residual. The path is then optimized to match
the component boundary using DiffVG.

This is structurally identical to running Potrace on the residual mask of
the most-error pixel, except the optimization is differentiable so the
path can refine sub-pixel.

### UDF (Unsigned Distance guided Focal) loss

The pixel-level loss is weighted by an unsigned distance field computed
from the current SVG render. Pixels far from any vector edge are weighted
down; pixels near the in-progress path edge are weighted up. This
concentrates optimization on the boundary of the current layer.

The published loss form (from the CVPR paper) is:

L_UDF(p) = w(d(p)) · L_focal(I(p), Ĩ(p))

where w is a focal weighting on the unsigned distance d(p) of pixel p to
the closest path edge, and L_focal is Lin et al.'s focal loss.

### Layer-wise optimization

LIVE alternates between:
1. **Add stage**: spawn a new closed Bezier path on the largest residual
   component.
2. **Refine stage**: optimize all current paths jointly via DiffVG. The
   UDF loss prevents earlier layers from drifting.

## Empirical claims

- ~5 paths reproduces a smiley face that DiffVG needs 256 paths for.
- Compact SVG, layer-wise structure consistent with human perception.
- Generalizes to out-of-domain images (unlike learned-prior methods like
  Im2Vec/SVG-VAE).

## Topology guarantees

- **Closed paths only** — each path is a closed cubic Bezier ring.
- **No explicit non-self-intersection constraint** — the optimization can
  in principle produce a path with a self-loop; the paper doesn't claim
  prevention. The component-wise initialization makes it unlikely but not
  impossible.
- **No explicit minimum feature size** — paths can encode features
  arbitrarily small.

## Limitations for chuck-mcp S6.c

1. **Runtime**: minutes per image on a GPU. S6.c needs ms per plate. LIVE
   is not viable as an inline exporter for the printability-repair loop.
2. **Self-intersection** not guaranteed. CAM tools choke on this.
3. **No tool-radius offset / minimum width** awareness.
4. **CUDA / DiffVG dependency** is a heavy ops burden.
5. **Cubic Béziers only** — same downstream conversion need as VTracer.

## How to actually use this

The right deployment is **offline hero-image vectorization**, not the
inline S6.c→SVG path. Concretely:

- For a published edition of a print: run LIVE once, hand-clean the
  output in Illustrator, then feed to cnc-woodblock-tools.
- For batch / preview: stick with Potrace per plate.

## Configuration knobs (from the GitHub repo)

LIVE is configured via `config/base.yaml`. Relevant fields:

- `num_paths` — total number of paths (e.g. 5, 32)
- `num_segments` — Bézier segments per path
- `num_iter` — optimization iterations per stage
- `lr_*` — learning rates for control points and colors
- `udf_weight`, `mse_weight` — loss weights

Default `experiment_5x1`: 5 paths × 1 segment per path.

## Citation

Ma, X. et al. "Towards Layer-wise Image Vectorization." CVPR 2022 (Oral).
arXiv:2206.04655. https://arxiv.org/abs/2206.04655
