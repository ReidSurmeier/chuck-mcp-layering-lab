---
title: "On Spatial Quantization of Color Images (scolorq)"
authors: ["Jan Puzicha", "Marcus Held", "Jens Ketterer", "Joachim M. Buhmann", "Dieter W. Fellner"]
venue: "ECCV 1998 / IEEE TIP / Lecture Notes in Computer Science vol. 1406, pp. 563-578"
year: 1998
links:
  - "https://link.springer.com/chapter/10.1007/BFb0055690"
  - "https://link.springer.com/content/pdf/10.1007/BFb0055690.pdf"
  - "https://pubmed.ncbi.nlm.nih.gov/18255438/"
  - "https://github.com/okaneco/rscolorq"
  - "https://www.virtual-boy.com/forums/t/scolorq-an-excellent-image-quantization-algorithm/"
relevance: 8
relevance_reason: "Spatial color quantization unifies palette extraction + dithering + perceptual smoothing in a single cost function. Conceptually parallel to mokuhanga: pick K plates (palette) and decide which pixel goes to which plate (assignment) jointly. Better baseline than running SLIC then median-cut separately."
---

# Spatial Color Quantization (scolorq) — Puzicha et al. 1998

## Algorithmic Core

The Puzicha/Held/Ketterer/Buhmann/Fellner ECCV 1998 paper proposes a *joint* formulation of color palette selection + per-pixel assignment + dithering, all driven by a single perceptual cost function. The standard pipeline at the time was to do these three steps sequentially: (1) extract a K-color palette via median cut or k-means, (2) assign each pixel to its nearest palette color, (3) error-diffuse with Floyd-Steinberg. Each step is locally optimal but globally suboptimal — Floyd-Steinberg's per-pixel propagation can't undo a bad palette choice.

scolorq's cost function:

```
E({α_i}, {c_k}) = Σ_x  ||  I_filtered(x) - (Σ_k α_k(x) * c_k) * h(x)  ||^2
```

where:
- `c_k` are the K palette colors to be optimized
- `α_k(x) ∈ {0, 1}` is the indicator of which palette color is assigned to pixel x
- `h(x)` is a perceptual low-pass filter (modeling that the human eye averages locally)
- `I_filtered` is the original image after the same low-pass filter

The cost is minimized by deterministic annealing over the joint variables — palette colors and assignment indicators are *simultaneously* updated. The α_k(x) become soft probabilities during annealing and harden as temperature drops. The result: palette colors that look good *after* the eye averages them locally, and assignment patterns that act as built-in dithering without explicit error diffusion.

A multiscale Gauss-Seidel optimization handles the O(N×K) cost efficiently.

## Behavior and Results

scolorq is famous for producing remarkably good 4-, 8-, 16-color quantizations of natural photos, often visually preferable to median-cut + Floyd-Steinberg even though it's a single algorithm. The "spatial coherence" in the title is the perceptual filter h(x): adjacent pixels jointly contribute to the cost, so the optimizer is incentivized to choose assignments that look smooth at a few-pixel scale even when the palette is very small.

Open-source implementations exist in C, Python, and Rust (`rscolorq`).

## Relevance to chuck-mcp S3.b → S6.b Pipeline

scolorq is the *most directly relevant prior art* for mokuhanga plate planning, more than SLIC-family superpixels. The mokuhanga problem statement is:

> Given a target image I and a palette of K available pigments (constraint: K ≤ plate budget), produce an assignment of each spatial region to one pigment plate such that the printed result is perceptually closest to I.

This is exactly scolorq with three modifications:

1. **Palette is fixed (the 36 pigments from chuck-mcp's catalog), not optimized.** Skip the c_k update step; only optimize α_k(x).
2. **α_k(x) is region-level, not pixel-level.** Instead of per-pixel indicators, define assignment at the superpixel cell level (one decision per S3.b cell). The cost function still uses pixel-resolution h(x).
3. **The "rendering" operator is mokuhanga overprint K-M recursion, not linear sum.** The `Σ_k α_k(x) * c_k` becomes the layered Kubelka-Munk forward render currently in `forward_render_jax`.

With these three changes, scolorq's deterministic annealing schedule is directly applicable to chuck-mcp's S6.b plate-assignment step. The S3.b cell graph provides the natural set of α-variables; the existing JAX forward renderer provides the rendering operator; only the cost minimization loop needs to be ported.

A second, simpler integration: use scolorq *as-is* to produce an initial guess for the assignment, then use chuck-mcp's existing inverse stack solver to refine. scolorq runs in seconds; the JAX L-BFGS-B refinement runs in minutes. The combined pipeline gets fast initialization that's already perceptually-aware.

Note on the "no Floyd-Steinberg" stance for mokuhanga: actual mokuhanga printers can't dither (one plate produces one solid color region; you cannot interleave pixels of plate A and plate B at scan resolution). But you *can* dither *between adjacent superpixel cells*: have cells A and B alternate in a small region to create a perceptual mid-tone. This is structurally the same as ordered dithering at cell resolution. scolorq's perceptual filter h(x) at cell-scale (say, 20-40 pixels) would naturally find these dithered solutions.

Sources:
- [ECCV 1998 paper at Springer](https://link.springer.com/chapter/10.1007/BFb0055690)
- [LNCS chapter PDF](https://link.springer.com/content/pdf/10.1007/BFb0055690.pdf)
- [Rust implementation: rscolorq](https://github.com/okaneco/rscolorq)
- [scolorq community discussion + examples](https://www.virtual-boy.com/forums/t/scolorq-an-excellent-image-quantization-algorithm/)
