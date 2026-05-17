---
arxiv_id: 2505.03201
title: "Enhancing Visual Feature Attribution via Weighted Integrated Gradients"
authors: Kien, Tam, Son, Than, Anh
published: 2025-05-06
url: https://arxiv.org/abs/2505.03201
code: https://github.com/tamnt240904/weighted_ig
relevance_to_chuck_mcp: HIGH — Integrated Gradients is the right tool; baseline weighting matters for our paper-baseline
---

# Weighted Integrated Gradients (WG)

## Core formulation (Sundararajan IG)

```
IG_i(x, x') = (x_i - x'_i) × ∫₀¹ ∂f(x' + α(x-x'))/∂x_i dα
```

For chuck-mcp: `x` = optimised pull mask, `x'` = "no pull" (paper white), feature `i` = a pull's mask pixel. IG tells us how much pulling-or-not-pulling pixel `i` of mask `k` changes the overall composite loss.

## Why baselines matter for us

Path-based attribution depends critically on the baseline `x'`. Their experiments show **24-35% AUC-of-Deletion improvement** over single-baseline IG by weighting multiple baselines by perturbation-based fitness `D_α`.

For chuck-mcp our natural baselines are:
- **Paper white** (no ink anywhere): the canonical "neutral" reference.
- **Final composite minus pull k**: pull-conditioned baseline ("what is the print without this pull?").
- **Yellow-keyblock-only**: warm-start baseline.

Each gives a different IG signal. WG suggests **average them with fitness-based weights** rather than picking one.

## Fitness function `D_α` (binary search, O(log n) per baseline)

```
D_α = min { |S| : f(x ∘ M(S)) ≤ α · f(x) }
```

— the minimum number of features to mask before model output drops to α=0.5 of original. Lower `D_α` = better baseline.

For us: how few pull-pixels must be removed before composite ΔE drops to 50% of the unablated value? Binary search makes this **log-time** per baseline.

## Axioms preserved

- Implementation invariance
- Effectiveness (sensitivity)
- Generalised completeness (total attribution = `f(x) - Σ w_k · f(x'_k)`, weighted baseline output)

So WG **inherits the same theoretical guarantees as Shapley/IG** at a fraction of the cost.

## Recommended chuck-mcp integration

```python
def load_bearing_score_WG(pull_idx, pulls, target, render_fn, deltaE_fn, baselines, weights):
    """Weighted IG attribution for pull k."""
    x = pulls[pull_idx].mask
    score = 0.0
    for x_prime, w in zip(baselines, weights):
        # Straight-line path from baseline x' to x in mask space
        def f(mask_t):
            modified = pulls[:pull_idx] + [pulls[pull_idx].replace(mask=mask_t)] + pulls[pull_idx+1:]
            return deltaE_fn(render_fn(modified), target).sum()
        
        # IG: 30-50 step Riemann sum is standard
        alphas = jnp.linspace(0, 1, 40)
        grads = jax.vmap(lambda a: jax.grad(f)(x_prime + a*(x - x_prime)))(alphas)
        ig = (x - x_prime) * grads.mean(axis=0)
        score += w * jnp.sum(jnp.abs(ig))
    return score
```

40-step IG × 132 pulls × 3 baselines = ~16000 gradient evaluations. At ~5ms each in JAX (small batch on GPU) = 80 seconds. **Over budget.**

## Cheap alternative — single-step gradient (saliency)

The Riemann sum can be approximated by **a single gradient at the input** (= saliency map). This is the "Gradient × Input" approximation of IG, used by SLA paper above. Per chuck-mcp performance budget:

```python
def load_bearing_score_cheap(pull_idx, pulls, target, render_fn, deltaE_fn):
    """1-step IG approximation (Gradient × Input)."""
    composite = render_fn(pulls)
    grad_composite = jax.grad(lambda c: deltaE_fn(c, target).sum())(composite)
    # Project gradient onto pull k's mask footprint
    return jnp.sum(jnp.abs(grad_composite) * pulls[pull_idx].mask)
```

One forward + one backward through the K-M render = **1 gradient call total for all 132 pulls**. Well under 1 second on JAX/GPU.

## Verdict on which method to use

- **Solver inner loop (every iteration):** 1-step Gradient × Mask (SLA-style). Sub-second for all pulls.
- **Pre-finalisation audit pass (once per design):** Full IG with weighted baselines (WG) — Pareto-front quality.
- **Independent verification (offline):** SIM-Shapley with 1000-coalition budget for ground truth on critical designs.

## Reuse: existing baselines for chuck-mcp

Use Sundararajan's "neutral baseline = zero" as the *unweighted* baseline (paper white), and add **the optimised solution minus pull k** as a *fitness-weighted* baseline. The fitness `D_α` naturally drops to zero for non-load-bearing pulls — so the weighting itself flags load-bearing-ness as a side effect.
