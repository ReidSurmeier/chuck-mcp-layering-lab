---
title: "Im2Vec: Synthesizing Vector Graphics without Vector Supervision"
authors: ["Pradyumna Reddy", "Michael Gharbi", "Michal Lukáč", "Niloy J. Mitra"]
arxiv_id: "2102.02798"
year: 2021
venue: "CVPR 2021"
url: "https://arxiv.org/abs/2102.02798"
project: "https://geometry.cs.ucl.ac.uk/projects/2021/Im2Vec/"
relevance: "MEDIUM — Im2Vec is a deep generative VAE+DiffVG approach. Useful for survey context; not directly applicable to chuck-mcp's deterministic pipeline. Confirms that learning-based methods don't enforce topology or feature size, reinforcing the case for keeping Potrace at the S6.c boundary."
tags: [vectorization, vae, deep-learning, diffvg, generative]
---

# Im2Vec — Synthesizing Vector Graphics Without Vector Supervision

## Core idea

Train a neural network to emit SVG paths from a raster image, but
**without** requiring paired vector training data (which is scarce and
non-unique). Instead, train with raster supervision only by rendering
the predicted SVG through DiffVG and comparing the rendered image to
the input image.

## Architecture

- Encoder: CNN over the raster input → latent code
- Decoder: predicts a fixed-cardinality set of closed Bezier paths
  (control points + colors)
- Renderer: DiffVG → predicted raster
- Loss: raster-domain reconstruction loss (perceptual + L1)

Notably, the decoder can emit **variable topology** — paths with holes,
multiple disconnected components — without explicit vector supervision.

## What it provides

- A learned prior over "natural" SVGs in a domain (fonts, emoji, icons)
- A way to vectorize a raster image from that domain
- Differentiable, end-to-end trainable

## What it doesn't provide

1. **Topology guarantees** — same as DiffVG. Paths can self-intersect.
2. **Out-of-domain generalization** — the learned prior is specific to
   the training corpus (fonts, MNIST, emoji in the paper). Not directly
   useful for layered printmaking masks.
3. **Minimum feature size** — none.
4. **Layer / occlusion awareness** — fixed-cardinality bag of paths.

## Why this matters for chuck-mcp

Im2Vec **doesn't directly help** the S6.c → SVG pipeline. It's relevant
mainly as evidence that:

1. Deep-learning vectorization needs a domain-specific corpus to compete
   with hand-written algorithms like Potrace.
2. None of the deep methods enforce machinability constraints.
3. The right place to enforce printability/machinability is **before**
   the vectorizer (in S6.c's mask repair) or **after** (in the CAM tool),
   not in the vectorizer itself.

## Citation

Reddy, P., Gharbi, M., Lukáč, M., Mitra, N. "Im2Vec: Synthesizing Vector
Graphics without Vector Supervision." CVPR 2021. arXiv:2102.02798.
