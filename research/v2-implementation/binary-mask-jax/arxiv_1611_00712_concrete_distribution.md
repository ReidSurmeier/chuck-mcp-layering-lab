---
title: The Concrete Distribution — A Continuous Relaxation of Discrete Random Variables
authors: [Chris J. Maddison, Andriy Mnih, Yee Whye Teh]
arxiv_id: 1611.00712
year: 2016
url: https://arxiv.org/abs/1611.00712
applicability_to_chuck_mcp: foundational — Concrete distribution is the theoretical sibling of Gumbel-Softmax (published the same week from a different lab). For Bernoulli case (which is what chuck-mcp needs), the "Binary Concrete" distribution is exactly: `s = sigmoid((logit_p + L) / τ)` where `L ~ Logistic(0, 1)`. This is the building block Louizos 2018 stretches to get the hard concrete distribution used in L0 regularization.
---

## Same machinery as Gumbel-Softmax, different name

Maddison et al. and Jang et al. published independently in November 2016. They are essentially the same idea, presented from different angles:

- **Jang et al.** — "Gumbel-Softmax" — practical recipe, generative modeling experiments.
- **Maddison et al.** — "Concrete" (CONtinuous relaxation of disCRETE) — measure-theoretic treatment, closed-form density.

For the chuck-mcp **binary** mask use case, the Binary Concrete simplifies to:

```
Let α = p/(1-p) = exp(logit_p), τ > 0 temperature.
Sample L ~ Logistic(0, 1) (equivalently L = log U - log(1-U), U ~ Uniform(0,1)).
Then s = sigmoid((log α + L) / τ) is Binary-Concrete(α, τ).
```

As `τ → 0`, `s` converges to Bernoulli(p). For finite τ, `s ∈ (0, 1)` is continuous — useful for gradient-based optimization, but **not binary**.

## Why chuck-mcp cares about Concrete (vs vanilla STE)

Concrete gives you a **closed-form density** for `s`:

```
log p(s) = log τ + log α - (τ + 1) (log s + log(1-s)) - 2 log(α s^(-τ) + (1-s)^(-τ))
```

This means you can:
1. Write a proper variational lower bound when the binary mask is a latent variable.
2. Compute KL divergence in closed form (KL term in L0-regularization, see Louizos 2018).
3. Anneal `τ` from high (smooth) to low (near-discrete) with a principled story.

For chuck-mcp v2 where region masks are **parameters** (not latent random variables), most of this machinery is overkill — Concrete reduces to "Gumbel-Softmax with a different paper to cite". But it becomes essential if you want to add an **L0 penalty** on mask sparsity (e.g. "I want regions to have as few pixels as possible while still hitting the color target"). For that, see arxiv_1712_01312 (Louizos hard concrete).

## Properties

- **Biased estimator** (low variance, biased toward true categorical only as τ → 0).
- Closed-form log-density, smooth, reparameterizable.
- Does **not** produce true binary forward outputs by itself — must be combined with a threshold + STE to get binary forward.

## In JAX

```python
def binary_concrete(logit_p, key, tau):
    u = jax.random.uniform(key, logit_p.shape, 1e-6, 1 - 1e-6)
    L = jnp.log(u) - jnp.log1p(-u)
    return jax.nn.sigmoid((logit_p + L) / tau)
```

This is identical to the `y_soft` step in ST-Gumbel-Softmax. For binary masks the two methods are computationally identical — the difference is interpretive (Concrete = "I have a continuous random variable on (0,1)"; Gumbel-Softmax = "I have a soft argmax over 2 categories").

## Verdict for chuck-mcp

- **Don't use Concrete directly.** The chuck-mcp binary mask needs to be **hard {0, 1}** at print time. Concrete gives soft `(0, 1)`.
- **Use it as a building block** if going the hard-concrete + L0 route (Louizos), or if a closed-form density is needed for a variational objective.
- For the v2 MVP, **STE is simpler and equally good** for the binary-mask production task.

Mentioned here for completeness because Louizos' hard concrete distribution (the only serious alternative to STE for chuck-mcp) is built directly on top of Binary Concrete.
