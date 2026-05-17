---
title: Learning Sparse Neural Networks through L0 Regularization
authors: [Christos Louizos, Max Welling, Diederik P. Kingma]
arxiv_id: 1712.01312
year: 2017
url: https://arxiv.org/abs/1712.01312
applicability_to_chuck_mcp: strong secondary candidate — Louizos' "hard concrete" distribution stretches a Binary Concrete sample and then **clamps it through a hard sigmoid**, producing samples that have non-zero probability of being **exactly 0 or exactly 1**. This is the only mainstream method that natively produces sometimes-binary, sometimes-continuous samples — and crucially, it admits a **differentiable expected L0 norm**, which lets you regularize for "make this mask use as few pixels as possible". Most useful for chuck-mcp if there's a sparsity prior on mask area (smaller carved regions = cheaper to cut).
---

## The hard concrete trick in one paragraph

Louizos et al. start with a Binary Concrete sample `s ∈ (0, 1)` (= ST-Gumbel-Sigmoid without the hard threshold). They then:

1. **Stretch** the support from `(0, 1)` to `(γ, ζ)` where `γ < 0 < 1 < ζ` (typically `γ = -0.1, ζ = 1.1`):
   ```
   s̄ = s · (ζ - γ) + γ                  s̄ ∈ (γ, ζ)
   ```
2. **Hard-sigmoid clamp** back to [0, 1]:
   ```
   z = min(1, max(0, s̄))                z ∈ [0, 1]
   ```

The result `z` has **point masses at 0 and 1** (whenever the stretched sample fell outside `(0, 1)`) plus a continuous density in between. Sample paths can be exactly 0 or exactly 1 — true binarity for a non-trivial fraction of samples.

## The key equation (expected L0 norm)

For a single gate with log-α and temperature β, the probability that `z ≠ 0` (i.e. the gate is "on") in expectation is:

```
P(z ≠ 0) = sigmoid(log α - β · log(-γ / ζ))
```

This is **differentiable in log α**, so you can add a soft-L0 penalty:

```
L_total = L_recon + λ · Σ_i sigmoid(log α_i - β · log(-γ / ζ))
```

and gradient-descent on `log α_i` directly. The expected sparsity becomes a smooth function of the gate parameters — the central trick of the paper.

## Hyperparameters used in the paper

| Parameter | Symbol | Typical value | Notes                                                |
|-----------|--------|---------------|------------------------------------------------------|
| Temperature | β    | 2/3           | Fixed. Lower → more discrete; too low → high-var grads |
| Stretch left | γ   | -0.1          | Slight overshoot on the 0 side                        |
| Stretch right | ζ  | 1.1           | Slight overshoot on the 1 side                        |
| L0 weight    | λ   | task-dependent | For chuck-mcp = "cost per active pixel"               |

## Forward / backward in JAX

```python
def hard_concrete(log_alpha, key, beta=2/3, gamma=-0.1, zeta=1.1):
    # forward sample
    u = jax.random.uniform(key, log_alpha.shape, 1e-6, 1 - 1e-6)
    s = jax.nn.sigmoid((jnp.log(u) - jnp.log1p(-u) + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    z = jnp.clip(s_bar, 0.0, 1.0)
    return z

def expected_l0(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
    # P(z != 0) — used as a sparsity penalty
    return jax.nn.sigmoid(log_alpha - beta * jnp.log(-gamma / zeta))

def hard_concrete_deterministic(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
    # at inference / final printing — no stochasticity
    s = jax.nn.sigmoid(log_alpha / beta)
    s_bar = s * (zeta - gamma) + gamma
    return jnp.clip(s_bar, 0.0, 1.0)
```

## Why this is genuinely powerful for chuck-mcp

1. **Native partial binarity.** Unlike STE/ST-GS where you have to ad-hoc threshold at the end, hard concrete produces *some* samples that are exactly 0 or 1 during training. You can monitor the fraction-binarized as a quality signal.
2. **L0 penalty.** Chuck-mcp v2 has a natural cost per active mask pixel: more pixels = more carving time + larger ink-laden block surface. λ · expected_l0 is the right regularization shape.
3. **Differentiable through clamp.** `jnp.clip` is sub-differentiable; JAX gives it gradient = 1 inside (0, 1), gradient = 0 outside. This means STE-like saturation behavior is **built in** without manual hard-tanh masking.

## Why it's a secondary choice (not primary)

- **5-10x more LOC** than STE. Stretch, clamp, expected-L0, deterministic-mode-for-inference — 4 helper functions vs STE's 1.
- **Extra hyperparameters** (β, γ, ζ, λ) all interact. Tuning is non-trivial.
- **Still stochastic** during training (Logistic noise). Same L-BFGS-B incompatibility as ST-GS (you need to fix the noise key per outer step).
- **Final inference is not exactly binary.** The clamp produces values in `[0, 1]` — most will be {0, 1} but some will be in between. You'd still need a final argmax/threshold pass.

## Verdict for chuck-mcp

**Reserve hard concrete + L0 for a v3 escalation, not v2 MVP.**

Use it when:
- STE produces masks that are too "blobby" (too much active area, ink waste)
- You want to optimize "minimum-area mask that achieves color target ε" — exactly the L0 form
- Engraving cost becomes a meaningful objective term

For v2 MVP, **STE (with optional ST-GS fallback) is simpler and gets you to a working solver faster**. Hard concrete adds enough complexity that it should only be adopted once you have a baseline measuring stick.
