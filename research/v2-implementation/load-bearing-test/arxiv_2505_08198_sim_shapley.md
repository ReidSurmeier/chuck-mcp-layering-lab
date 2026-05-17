---
arxiv_id: 2505.08198
title: "SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation"
authors: Fan, Li, Zhou, Okada, Hong, Liu, Liu
published: 2025-05-13
url: https://arxiv.org/abs/2505.08198
code: https://github.com/nliulab/SIM-Shapley
relevance_to_chuck_mcp: MEDIUM — relevant only if we *insist* on Shapley. For our use case it is overkill.
---

# SIM-Shapley — Stochastic Iterative Momentum for SV approximation

## What it gives

- Reformulates Shapley as **stochastic optimisation** over coalition samples — drop the `O(d·2^d)` requirement.
- **Linear Q-convergence** with theoretical variance bounds.
- Empirically **85% time reduction** vs SOTA Shapley approximators.
- Game-agnostic (no special structure needed in the value function).

## Method (1-paragraph version)

1. Sample a mini-batch of `m` random coalitions `z_i ∈ {0,1}^d`.
2. Each iteration solves a constrained quadratic over the SV estimate `δ` (with `1^T δ = c` efficiency constraint):
   ```
   L(δ, ν) = (1/m) Σ (v(0) + z_i^T β - v(z_i))^2 + λ||δ||² + ν(1^T δ - c)
   ```
3. Closed-form per-step update via KKT + momentum (Ruppert averaging).
4. Three stability tricks: L2 regularisation, negative-sampling detection, initialisation bias correction.

## Why I'd recommend NOT using full Shapley for chuck-mcp v2

- **N=132 pulls**: full Shapley needs `132·2^132` evaluations. Even SIM-Shapley needs **thousands** of coalition forward-renders to get a reasonable estimate.
- One forward render of 132 K-M overprint pulls at print resolution is ~50-200ms in JAX. SIM-Shapley with ~5000 coalitions = 250-1000 seconds. **Misses the 5-second budget by 50-200x.**
- The SLA-style **gradient × mask attribution** approximates Shapley to first order at *O(1)* cost.

## When Shapley would actually be worth it

- One-shot final verification of the solver output before sending to carving.
- User clicks "give me the rigorous load-bearing analysis" in the UI → run SIM-Shapley with 1000-5000 coalitions over ~5 min as a background job, produce a definitive report.
- Discrepancy detection: if gradient attribution says pull k matters but Shapley says it doesn't → flag for review.

## Verdict

- **Inner loop / hot path**: don't use Shapley. Use gradient × mask.
- **Final report / UI verification**: SIM-Shapley with a small coalition budget is reasonable.
- **Cancellation detection** (pulls A & B fight each other): Shapley naturally handles this via pairwise marginals; gradient attribution does not. So for a once-per-design audit pass, this is worth it.

## Code reference

`https://github.com/nliulab/SIM-Shapley` — Python, scikit-learn-style API, can probably be vmap'd in JAX with modest porting effort.
