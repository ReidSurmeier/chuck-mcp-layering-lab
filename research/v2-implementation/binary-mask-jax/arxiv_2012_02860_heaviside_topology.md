---
title: Revisiting Element Removal for Density-Based Structural Topology Optimization with Reintroduction by Heaviside Projection
authors: [Reza Behrou, Reza Lotfi, Josephine Voigt Carstensen, Federico Ferrari, James K. Guest]
arxiv_id: 2012.02860
year: 2020
url: https://arxiv.org/abs/2012.02860
applicability_to_chuck_mcp: HIGHLY relevant — this is from the structural topology optimization community, where the problem **is literally** "given a continuous density field rho(x) ∈ [0,1], drive it to binary {0,1} via differentiable optimization, where 1 = material, 0 = void". This is mathematically equivalent to chuck-mcp's "1 = carved region, 0 = uncarved" problem. The **Heaviside projection with continuation in β** is the dominant technique in TopOpt for binarization. It is a smooth differentiable binarizer that approaches a step function as β → ∞. Strong third candidate behind STE and hard concrete.
---

## Why TopOpt prior art is the most directly applicable

Chuck-mcp's binary masks are *exactly* the same problem as 2D structural topology optimization:

| TopOpt                          | Chuck-mcp v2                          |
|----------------------------------|----------------------------------------|
| Density `ρ(x, y) ∈ [0, 1]`       | Mask `z(x, y) ∈ [0, 1]`                |
| 1 = material, 0 = void           | 1 = carved (printed) region, 0 = not   |
| Minimize compliance              | Minimize printed-vs-target color ΔE    |
| Volume constraint                | Optional area / pixel-count constraint |
| Density filter (Helmholtz / blur) | Mask smoothing / mokuhanga blur        |
| Heaviside projection             | **Binarization step** (this paper)     |

**TopOpt has been doing differentiable binary masks for 30 years.** Bendsøe (1989), Sigmund (1997), Guest et al. (2004) all use density-based methods that culminate in a Heaviside projection. The technique is mature, well-understood, and has a rich body of "what-can-go-wrong" knowledge.

## The Heaviside projection formula

Smoothed Heaviside (Guest, Prevost, Belytschko 2004; refined by Wang, Lazarov, Sigmund 2011):

```
H_β,η(ρ̃) = (tanh(β · η) + tanh(β · (ρ̃ - η))) / (tanh(β · η) + tanh(β · (1 - η)))
```

where:
- `ρ̃` ∈ [0, 1] is the (filtered) density / mask
- `η` ∈ (0, 1) is the threshold (typically `η = 0.5`)
- `β` is the projection sharpness (typically continuated from 1 → 64)

Properties:
- As `β → 0`, `H_β,η(ρ̃) ≈ ρ̃` (no binarization, smooth identity).
- As `β → ∞`, `H_β,η(ρ̃) → 1{ρ̃ > η}` (hard step).
- `H_β,η(0) = 0` and `H_β,η(1) = 1` for all β > 0 — exact at the endpoints, smooth in between.
- Differentiable everywhere. Closed-form gradient.

## The continuation schedule (β annealing)

The standard TopOpt move is **continuation**: start at `β = 1` (almost no binarization, smooth optimization), then double or increment β every N iterations:

```
β_k = 1, 2, 4, 8, 16, 32, 64        # every K outer iterations, e.g. K=25
# or
β_k+1 = β_k + Δβ                     # additive, e.g. Δβ = 0.5
```

By the final outer iteration, `β ≈ 64` means the Heaviside projection is essentially a step function — output is `≥ 0.999` if `ρ̃ > 0.5`, `≤ 0.001` otherwise. Threshold and you have a clean binary mask.

This is **conceptually identical** to Gumbel-Softmax temperature annealing — start smooth, end sharp — but **without stochasticity** and **without any gradient hack**. Pure deterministic smooth-to-sharp transition.

## Properties relevant to chuck-mcp

- **Deterministic forward.** No Gumbel noise, no Logistic noise. L-BFGS-B compatible out of the box.
- **Exact gradients.** Closed-form differentiable, no STE bias. Gradients of `H_β,η(ρ̃)` w.r.t. `ρ̃`:
  ```
  ∂H/∂ρ̃ = β · sech²(β · (ρ̃ - η)) / (tanh(β · η) + tanh(β · (1 - η)))
  ```
  Sharp peak at `ρ̃ = η`, near-zero elsewhere (for large β). Same dead-zone behavior as hard-tanh STE but with a smooth gradient profile that the optimizer can actually navigate.
- **No final thresholding needed (almost).** With `β = 64`, the output is binary to 3 decimal places. A final `> 0.5` threshold gives the print mask.
- **Combines naturally with filtering.** TopOpt convention is `ρ̃ = filter(ρ)` (a Helmholtz blur or convolution) before Heaviside projection. For chuck-mcp, this is the mokuhanga ink-bleed kernel — exactly what we want.

## What TopOpt teaches us about failure modes

Decades of TopOpt experience with Heaviside projection:

1. **Too aggressive β early → local minima.** If β = 64 from iteration 0, the optimization gets stuck because gradients are zero everywhere except in a thin band around `η`. **Continuation is mandatory.**
2. **Checkerboarding.** Without a smoothing filter before Heaviside, the optimizer produces 1-pixel checkerboard patterns that satisfy the objective but are unmanufacturable. Fix: always blur `ρ` before projection. For chuck-mcp, this is the mokuhanga kernel.
3. **Disconnected regions.** A region might emerge as several disconnected blobs. TopOpt handles this with morphological constraints (e.g. minimum length scale). For chuck-mcp, this is downstream in the printability-repair stage (mill-sized area-opening, see vectorization-cnc/ folder).
4. **Grayness in the final design.** Some pixels stay in `(η - ε, η + ε)` because the optimizer can't decide. Fix: add a penalty `(ρ̃ · (1 - ρ̃))` or just hard-threshold at the end.

All four failure modes apply directly to chuck-mcp.

## JAX implementation

```python
def heaviside_projection(rho_tilde, beta=1.0, eta=0.5):
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho_tilde - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den

def make_binary_mask(logit_a, beta, eta=0.5):
    rho = jax.nn.sigmoid(logit_a)                    # logit → [0, 1]
    # (optional) rho_tilde = mokuhanga_blur(rho)      # smoothing filter
    return heaviside_projection(rho, beta=beta, eta=eta)
```

This is **deterministic** and **fully differentiable**. The gradient is exact (no STE surrogate). The output asymptotes to binary as β grows.

## Verdict for chuck-mcp

Heaviside projection is the **strongest dark-horse candidate** behind STE. It has several advantages over STE:

- **No biased surrogate gradient.** Heaviside has an exact, closed-form gradient.
- **No stochasticity.** Plays well with L-BFGS-B.
- **Mature failure-mode literature** (TopOpt has been doing this since 1997).
- **Natural integration** with smoothing filters, area constraints, minimum-length-scale rules — all of which chuck-mcp's downstream printability-repair needs anyway.

The downside vs STE:

- **Not exactly binary during training.** Even at β = 64, a few pixels will be in `(0.001, 0.999)`. Forward render uses these as soft masks, not the true binary masks. So the loss reflects "what we'd print if we did final thresholding" but not exactly. STE forward is exactly binary every iteration.

For chuck-mcp v2 the recommendation is:
1. **Primary: STE** (simplest, smallest LOC, exact binary forward)
2. **Strong alternative: Heaviside continuation** (cleaner gradients, TopOpt heritage)

If you run into STE failure modes (frozen masks, dead regions, gradient saturation), Heaviside is the natural first escalation — and likely *better* than ST-Gumbel-Softmax for the chuck-mcp use case because it's deterministic.
