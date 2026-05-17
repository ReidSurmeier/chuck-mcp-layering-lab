# Differentiable Rendering with Perturbed Optimizers

authors: Quentin Le Lidec, Ivan Laptev, Cordelia Schmid, Justin Carpentier
arxiv: 2110.09107
venue: NeurIPS 2021
paper: https://proceedings.neurips.cc/paper/2021/file/ab233b682ec355648e7891e66c54191b-Paper.pdf
relevance: Provides a principled alternative to Gumbel-softmax for differentiating through discrete/non-smooth rendering steps. Crucial for chuck-mcp where plate identity (which pigment, which block) is discrete and the soft-relaxed alpha mattes lose physical meaning if relaxed too far.

Key idea:
- Reframe differentiable rendering as randomized optimization. Instead of replacing the non-differentiable rasterization/argmax operations with smooth approximations (e.g. softmax), inject Gaussian-like noise into the optimizer state and average gradients. This is the "perturbed optimizer" trick from Berthet et al. 2020 applied to rendering.
- The smoothing parameter (noise scale) is the implicit "trust region radius" — large noise = exploring widely, small noise = locked in. Adaptive scheme adjusts smoothing automatically.
- Variance-reduction mechanism keeps the gradient estimator cheap.
- Demonstrated on 6D pose estimation and 3D mesh reconstruction with sharper, less blurred convergence than smoothed-renderer baselines.

Algorithmic moves to import for chuck-mcp:
1. For discrete pigment-slot assignment, swap Gumbel-softmax for a perturbed-optimizer step: sample a few pigment assignments under noise, compute weighted gradient by argmin-of-loss. The noise scale is decayed on a schedule — early batches use high noise (explore), late batches use low noise (commit).
2. The noise scale gives chuck-mcp a NATURAL trust-region knob: shrink it after each batch is "frozen," expand it when starting a new batch.
3. Pairs cleanly with L-BFGS-B: outer loop is perturbed optimization over discrete choices; inner loop is L-BFGS-B over continuous alpha-matte / control-grid params.

Abstract:
"Reasoning about 3D scenes from their 2D image projections is one of the core problems in computer vision. Solutions to this inverse and ill-posed problem typically involve a search for models that best explain observed image data. ... if optimization techniques should be used to explain images, it is crucial to design differentiable functions for the projection of 3D scenes into images, also known as differentiable rendering. Previous approaches to differentiable rendering typically replace non-differentiable operations by smooth approximations, impacting the subsequent 3D estimation. In this paper, we take a more general approach and study differentiable renderers through the prism of randomized optimization and the related notion of perturbed optimizers. ... we propose a variance reduction mechanism to alleviate the computational burden inherent to perturbed optimizers and introduce an adaptive scheme to automatically adjust the smoothing parameters of the rendering process."

References to follow up:
- Berthet et al. (NeurIPS 2020) "Learning with Differentiable Perturbed Optimizers" — the foundational technique.
- "Importance Sampling BRDF Derivatives" (Belhe et al., arXiv:2304.04088) — companion variance-reduction technique for differentiable rendering.
