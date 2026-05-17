---
title: Categorical Reparameterization with Gumbel-Softmax
authors: [Eric Jang, Shixiang Gu, Ben Poole]
arxiv_id: 1611.01144
year: 2016
url: https://arxiv.org/abs/1611.01144
applicability_to_chuck_mcp: secondary candidate — Gumbel-Softmax gives a differentiable Bernoulli sample with annealable temperature. The Straight-Through Gumbel-Softmax variant (ST-GS) is a drop-in for STE that adds controlled exploration noise. Useful for chuck-mcp if vanilla STE shows dead-mask collapse or premature region freezing. Same complexity as STE; extra hyperparameter (τ schedule).
---

## What it is

The Gumbel-Max trick: to sample one-hot from a categorical with probabilities π, compute `z = one_hot(argmax_i [g_i + log π_i])` where `g_i ~ Gumbel(0, 1)`. The argmax is non-differentiable, so Gumbel-Softmax replaces it with a temperature-scaled softmax:

```
y_i = exp((log π_i + g_i) / τ) / Σ_j exp((log π_j + g_j) / τ)
```

As `τ → 0`, `y` approaches one-hot. As `τ → ∞`, `y` approaches uniform.

For **Bernoulli** (which is what chuck-mcp binary masks need), this collapses to:

```
g_1, g_0 ~ Gumbel(0, 1)
y = sigmoid((log(p/(1-p)) + g_1 - g_0) / τ)
  = sigmoid((logit(p) + L) / τ)            where L = g_1 - g_0 ~ Logistic(0, 1)
```

i.e. add Logistic noise to the logit, then softmax/sigmoid at temperature τ. Same idea, simpler form.

## Straight-Through Gumbel-Softmax (ST-GS)

The vanilla Gumbel-Softmax sample `y` is **continuous** (soft) — bad if you want a true binary mask in the forward pass. ST-GS fixes this:

```
forward:   y_soft = sigmoid((logit_p + L) / τ)
           y_hard = (y_soft > 0.5).astype(float32)
           y = y_soft + stop_gradient(y_hard - y_soft)        # forward = y_hard, backward = y_soft
backward:  ∂L/∂a flows through y_soft (chain rule through sigmoid + noise)
```

This is **STE with controlled stochastic exploration noise**. The Gumbel/Logistic perturbation forces the optimizer to consider flipping bits near `logit ≈ 0`, breaking premature freezing.

## Temperature annealing

Standard schedule (Jang et al. §3.3): start at `τ ≈ 1.0`, anneal down to `τ ≈ 0.1` or `τ ≈ 0.5` exponentially:

```
τ(t) = max(τ_min, τ_0 * exp(-r * t))     # e.g. τ_0 = 1.0, τ_min = 0.1, r = 1e-5
```

- High τ early: smooth gradients, broad exploration.
- Low τ late: hard sample, gradient matches forward.

For chuck-mcp's L-BFGS-B outer loop, the natural staging is to anneal **between** outer-loop batches, not inside the L-BFGS-B inner step (L-BFGS-B assumes a fixed function — temperature changing during the inner loop will confuse the curvature estimate).

## Properties relevant to chuck-mcp

- **Bias:** Biased like STE but with controllable strength via τ. At τ = 0 it equals STE; at τ = ∞ it's uniform random.
- **Variance:** Higher than STE because of the Gumbel noise. Each forward pass is stochastic — this can confuse L-BFGS-B unless you fix the noise seed for the duration of one outer-loop step.
- **Exploration:** Much better than STE. The Logistic noise has tails — occasionally a small-magnitude logit will flip its mask sample, which lets the optimizer discover better mask configurations.
- **Failure mode:** if τ is too high, the forward `y_hard` is essentially random and the loss is uninformative. If τ is too low, ST-GS degenerates to STE.

## When to prefer ST-GS over plain STE for chuck-mcp

- **STE first.** If vanilla STE works, ship it. ST-GS adds a hyperparameter and noise.
- **Switch to ST-GS if:**
  - Masks freeze early (many pixels at `|logit| >> 1` early, not exploring)
  - Dead-mask collapse: an entire region's mask goes to all-zeros or all-ones in early iterations
  - Final masks are clearly suboptimal — print loss high but binarization gap small (`y_soft ≈ y_hard` means STE worked but found a local min)

## JAX implementation sketch

```python
import jax, jax.numpy as jnp

def st_gumbel_sigmoid(logit_p, key, tau):
    # Logistic noise = Gumbel(0,1) - Gumbel(0,1)
    u = jax.random.uniform(key, logit_p.shape, minval=1e-6, maxval=1 - 1e-6)
    L = jnp.log(u) - jnp.log1p(-u)                     # Logistic(0, 1)
    y_soft = jax.nn.sigmoid((logit_p + L) / tau)
    y_hard = (y_soft > 0.5).astype(y_soft.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)
```

## Trade-off summary vs STE

| Property              | STE             | ST-GS                            |
|-----------------------|-----------------|----------------------------------|
| Forward output        | hard binary     | hard binary                      |
| Backward bias         | high near 0     | tunable via τ                    |
| Backward variance     | zero            | non-zero (Gumbel/Logistic noise) |
| Hyperparameters       | none            | τ schedule (3 numbers)           |
| Exploration           | none            | yes (noise tails)                |
| L-BFGS-B compat       | excellent       | requires fixed noise per step    |
| Implementation LOC    | ~3              | ~6                               |
| Battle-tested in BNNs | yes (since 2013)| yes (since 2016)                 |

For chuck-mcp v2 MVP: **start with STE, keep ST-GS as a 1-line drop-in escalation.**
