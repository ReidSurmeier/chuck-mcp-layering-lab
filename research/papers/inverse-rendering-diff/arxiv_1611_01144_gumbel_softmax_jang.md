# Categorical Reparameterization with Gumbel-Softmax

authors: Eric Jang, Shixiang Gu, Ben Poole
arxiv: 1611.01144
venue: ICLR 2017
relevance: Foundational paper for differentiating through categorical / discrete random variables. The default tool for chuck-mcp's discrete pigment-slot selection: each plate has a soft probability distribution over the 36-entry pigment catalog, sampled via Gumbel-softmax with temperature τ. Anneal τ from large (smooth, every pigment partly chosen) to small (one-hot, single pigment per plate) over the solve.

Key idea:
- A categorical variable y ~ Cat(π_1, ..., π_k) is non-differentiable w.r.t. π_i because the sampling step is discrete (argmax of π + Gumbel noise).
- Gumbel-softmax replaces argmax with softmax(τ): y_i ≈ exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ), where g_i ~ Gumbel(0,1).
- As τ → 0, y becomes one-hot (true categorical). As τ → ∞, y becomes uniform.
- Smoothly anneal τ during training: solver explores all pigments early, commits to one late.

Algorithmic moves to import for chuck-mcp:
1. Replace chuck-mcp's hard pigment-slot indexing with a Gumbel-softmax-weighted mixture of pigment K-M coefficients. Forward render uses the soft mixture (continuous); during the solve, anneal τ from ~1.0 down to ~0.1 over the inner L-BFGS-B iterations.
2. At convergence, ROUND to the argmax pigment per plate. The annealing schedule ensures rounding is approximately lossless.
3. For chuck-mcp's plate-batch assignment (which plate goes into batch 1 vs batch 2 vs batch 3), use Gumbel-softmax over batch-index per plate, with τ annealed in the OUTER loop instead.

Caveats specific to chuck-mcp:
- Gumbel-softmax's variance-bias tradeoff is well known. For low-dimensional discrete choices (pigment slot, batch assignment) it works fine. For high-dimensional discrete choices (per-pixel plate assignment), prefer "Perturbed Optimizers" (Le Lidec et al., arXiv:2110.09107) or "Many-Worlds" (Zhang et al., arXiv:2408.16005).

Abstract:
"Categorical variables are a natural choice for representing discrete structure in the world. However, stochastic neural networks rarely use categorical latent variables due to the inability to backpropagate through samples. In this work, we present an efficient gradient estimator that replaces the non-differentiable sample from a categorical distribution with a differentiable sample from a novel Gumbel-Softmax distribution. This distribution has the essential property that it can be smoothly annealed into a categorical distribution. We show that our Gumbel-Softmax estimator outperforms state-of-the-art gradient estimators on structured output prediction and unsupervised generative modeling tasks with categorical latent variables, and enables large speedups on semi-supervised classification."

Companion: Maddison, Mnih, Teh (2017) "The Concrete Distribution" — the same trick discovered independently in the same year.

References to follow up:
- Invertible Gaussian Reparameterization (Potapczynski et al., arXiv:1912.09588) — more flexible alternative.
- Conditional Gumbel-Softmax (Strypsteen, Bertrand, arXiv:2406.01162) — useful when chuck-mcp wants to enforce "if pigment X is chosen for plate i, plate j cannot also choose X" constraints.
