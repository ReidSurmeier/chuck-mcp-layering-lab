---
title: Improving Discrete Optimisation Via Decoupled Straight-Through Estimator
authors: [Rushi Shah, Mingyuan Yan, Michael Curtis Mozer, Dianbo Liu]
arxiv_id: 2410.13331
year: 2024
url: https://arxiv.org/abs/2410.13331
applicability_to_chuck_mcp: tuning insight, not a method change — Decoupled-ST shows that conflating forward-pass stochasticity and backward-pass gradient sharpness into a single temperature τ is suboptimal. For chuck-mcp this matters if we use ST-Gumbel-Softmax: pick τ_f independently from τ_b. The empirical takeaway is: optimal configurations lie off the diagonal, often with `τ_f << τ_b`.
---

## The headline claim

Existing STE variants (identity STE, softmax STE, ST-Gumbel-Softmax) all use **one temperature** that simultaneously controls:
- Q1 (forward): how stochastic is the sample drawn?
- Q2 (backward): how sharply are gradients concentrated on the selected category?

These two questions have **different optimal answers** in general. Decoupled-ST uses two temperatures `(τ_f, τ_b)` and shows that the best operating point is far off the `τ_f = τ_b` diagonal.

## The algorithm in two lines

```
forward:   z ~ Categorical( softmax(logits / τ_f) )
backward:  ∂L/∂logits := J(τ_b)^T · ∂L/∂z
           where J(τ_b) = (1/τ_b) · (diag(p_b) - p_b · p_b^T),  p_b = softmax(logits/τ_b)
```

For Bernoulli (the chuck-mcp case):

```
forward:   z = Bernoulli( sigmoid(logit / τ_f) )
backward:  ∂L/∂logit := sigmoid(logit/τ_b) · (1 - sigmoid(logit/τ_b)) / τ_b · ∂L/∂z
```

i.e. compute the forward probability at `τ_f`, sample, but compute the backward sigmoid Jacobian at a *different* temperature `τ_b`.

## Empirical findings

Across three tasks (stochastic binary nets, categorical autoencoders, differentiable logic gates):

| Task                          | Optimal τ_f | Optimal τ_b | Notes                            |
|-------------------------------|-------------|-------------|-----------------------------------|
| Stochastic Binary Net (Fashion-MNIST) | 0.1         | 0.7         | Low fwd stoch, broader gradient   |
| Categorical Autoencoder (MNIST 8-way) | 2.0         | 0.5         | High fwd stoch (exploration), sharp gradient |
| Differentiable Logic Gates (CIFAR-10) | 0.1         | 2.0         | Near-deterministic fwd, broad backward |

The takeaway: **single-temperature methods are fundamentally constrained**. Forward stochasticity and backward gradient sharpness are two different concerns.

## What this means for chuck-mcp

For a binary mask use case most similar to the Stochastic Binary Net row above (binary forward + dead-neuron problem):

- **τ_f ≈ 0.1** — almost deterministic forward (low stochasticity = lower variance = better for L-BFGS-B compatibility).
- **τ_b ≈ 0.7** — broader gradient = avoid dead masks where some pixels never get updated.

For the chuck-mcp case where we *don't even need forward stochasticity* (just hard STE), the natural interpretation is:

- **τ_f → 0** (pure argmax / hard threshold forward, no Gumbel noise)
- **τ_b ≈ 0.7 to 1.0** (sigmoid-style backward, moderate width)

This is **exactly identity-STE with a sigmoid Jacobian backward** — which is actually how most STE implementations work in practice (the backward already uses the sigmoid derivative, not the indicator function).

## When Decoupled-ST is worth adopting for chuck-mcp

- **If using ST-Gumbel-Softmax** (adopted as escalation from plain STE): definitely use the decoupled-temperature variant. Setting `τ_f` low (less forward noise → better L-BFGS-B compat) and `τ_b` moderate (~0.7) is essentially free.
- **If plain STE is working:** no change needed. Plain STE is implicitly `(τ_f = 0, τ_b = 1)` with sigmoid backward Jacobian — already near the empirically optimal region for binary nets.

## JAX implementation

```python
def decoupled_st_bernoulli(logit, key, tau_f=0.1, tau_b=0.7):
    """Decoupled-ST Bernoulli sample with separate fwd / bwd temperatures."""
    # forward: sample at temperature tau_f
    u = jax.random.uniform(key, logit.shape, 1e-6, 1 - 1e-6)
    L = jnp.log(u) - jnp.log1p(-u)                          # Logistic(0, 1)
    p_f = jax.nn.sigmoid((logit + L) / tau_f)               # near-argmax for small tau_f
    z_hard = (p_f > 0.5).astype(logit.dtype)
    
    # backward: replace forward Jacobian with one at temperature tau_b
    p_b = jax.nn.sigmoid(logit / tau_b)
    surrogate = p_b                                          # smooth backward path
    
    return surrogate + jax.lax.stop_gradient(z_hard - surrogate)
```

## Verdict

Decoupled-ST is a **tuning refinement** on ST-Gumbel-Softmax, not a new method. Cite it as the canonical reference if you use ST-GS for chuck-mcp; ignore it if you stick with plain STE.

**Key actionable takeaway:** if you ever switch to ST-Gumbel-Softmax, do not use a single τ — use `τ_f ≈ 0.1, τ_b ≈ 0.7` as starting values per the paper's stochastic-binary-net experiments.
