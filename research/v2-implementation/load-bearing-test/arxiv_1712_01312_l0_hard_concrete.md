---
arxiv_id: 1712.01312
title: "Learning Sparse Neural Networks through L_0 Regularization"
authors: Louizos, Welling, Kingma
venue: ICLR 2018
published: 2017-12-04
url: https://arxiv.org/abs/1712.01312
relevance_to_chuck_mcp: HIGH — the canonical "differentiable L0 / soft-prune via hard concrete gates" recipe
---

# L0 regularisation via Hard Concrete gates — canonical differentiable pruning

## Why this matters for chuck-mcp v2

If we want **continuous soft pruning** (rather than O&R-style hard chop at scheduled checkpoints), this is the canonical recipe. Each pull gets a stochastic gate `z_k ∈ [0, 1]` that the optimiser learns alongside `mask_k`. The gates eventually become exactly 0 or 1 — exact pull selection within an end-to-end differentiable optimisation.

## Core math

L0 norm `||θ||_0 = Σ 1[θ_k ≠ 0]` is non-differentiable. The trick:

1. Replace `θ_k = z_k · w_k` where `z_k` is a stochastic gate.
2. Sample `z_k` from a *hard concrete* distribution: stretched Bernoulli relaxation, then hard-sigmoid.
3. The **expected L0 norm `E[Σ z_k > 0]` is differentiable** w.r.t. the gate parameters `log α_k`.
4. Add to loss: `L_total = L_data + λ · E[||z||_0]`.
5. At inference: clamp `z_k > 0.5` to binary.

For pulls:
```
pull_k_effective_mask = z_k · mask_k
where z_k ~ HardConcrete(log α_k, β, ζ, γ)
```
`λ` is the load-bearing penalty strength. High `λ` → fewer pulls; low `λ` → all pulls remain.

## Soft-vs-hard penalty: this gives both

The hard concrete distribution is the **continuous bridge between soft and hard** that the user is asking about:
- **During optimisation**: `z_k ∈ (0, 1)` is differentiable, gradient flows back via reparameterisation trick.
- **At convergence**: `z_k` concentrates near 0 or 1 (stretching outside [0,1] then clamping).
- **At inference**: threshold to binary → some pulls fully selected, others fully dropped.

## Comparison with O&R

| Aspect | O&R (hard prune at checkpoints) | L0 / Hard Concrete (continuous gates) |
|---|---|---|
| Differentiable through pruning | No (discrete schedule) | Yes (reparameterised gradient) |
| Implementation complexity | Low (one loop) | Medium (stochastic gates, KL terms) |
| Convergence quality | Good (proven by O&R results) | Best in theory, brittle in practice |
| Use with L-BFGS-B | Compatible (between LBFGS calls) | Incompatible (stochastic gradients break LBFGS line search) |
| Speed | Fast | Slow (need stochastic samples) |

**Decision for chuck-mcp**: stick with O&R-style hard prune at checkpoints because we use L-BFGS-B (deterministic, exact line search) for the inner optimisation. Hard-Concrete is suited for SGD-based training, not deterministic Newton-like methods.

## When to revisit Hard Concrete

- If we switch from L-BFGS-B to Adam/SGD for the inner loop (e.g., for very large problems).
- If we want to *jointly learn block assignment*: which physical block each pull goes on. This is a discrete combinatorial selection problem natural to express as stochastic gates over a `(block_id, pass_idx)` lattice.

## What we steal from this paper anyway

Even with hard prune for pulls, we can use **stochastic gates for block assignment** in the graph-coloring / packing phase (S7). Hard-Concrete `z_{pull,block}` over a `N_pull × N_block` matrix gives end-to-end differentiable block assignment.

## Bottom line

For load-bearing-test specifically: **don't use this paper's gate mechanism**. Stick with O&R hard ablation. But keep this in the toolbox for block assignment (S7).
