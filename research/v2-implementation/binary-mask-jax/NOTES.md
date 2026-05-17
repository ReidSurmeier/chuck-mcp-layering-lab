# Binary Mask Production in JAX — Verdict and Implementation Plan for chuck-mcp v2

**Agent:** BINARY-MASK-JAX
**Date:** 2026-05-16
**Output of:** chuck-mcp v2 implementation swarm `swarm-1778969836247-ys4o7z`

## VERDICT

**Ship STE (saturating, hard-tanh variant) as the primary method. Use Heaviside continuation as the v2.5 escalation if STE shows pathologies.**

Specifically: **hard-sigmoid STE with sigmoid-Jacobian backward + latent logit clipping**, applied per-pixel per-region-mask, in the form recommended by JAX issue #9032 (`x - stop_grad(x) + stop_grad(f(x))`).

### Method ranking

| Rank | Method | LOC | Forward exactly binary? | Deterministic? | L-BFGS-B compat | When to use |
|------|--------|-----|--------------------------|----------------|-----------------|-------------|
| 1 | **Hard-sigmoid STE** (with clip) | ~5 | yes | yes | excellent | **MVP — primary** |
| 2 | Heaviside-continuation (TopOpt) | ~20 | no (asymptotic) | yes | excellent | Pathologies of STE; cleaner gradient signal |
| 3 | ST-Gumbel-Sigmoid (Decoupled-ST) | ~10 | yes | no (stochastic) | needs fixed-key trick | Frozen / dead-mask collapse only |
| 4 | Hard Concrete + L0 | ~30 | no (mostly) | no | requires fixed-key | Mask-area objective required (CNC cost) |
| 5 | REINFORCE / score-function | ~5 | yes | no | poor (high var) | **Never** for chuck-mcp |

### Why STE wins for chuck-mcp specifically

1. **Forward output is exactly binary every iteration.** Crucial — the v2 design locks "forward render must use the binarized masks (not continuous α) so the loss reflects what will actually print". STE is the only method in the ranking with `forward = hard {0, 1}` and `gradient flows`. Heaviside-projection is *asymptotically* binary but still has soft pixels until `β >> 64`.

2. **Deterministic.** L-BFGS-B builds a Hessian approximation; it requires a deterministic objective. STE has zero forward stochasticity. ST-GS and hard-concrete inject noise — they require the "fix the random key for the duration of the inner loop" workaround.

3. **Cheapest.** STE adds ~5 LOC and ~3% wall-clock to the inner loop. Heaviside adds ~20 LOC + a continuation schedule that must be coordinated with the staged 3-batch outer loop (from `inverse-rendering-diff/NOTES.md`). Hard concrete adds ~30 LOC + 4 hyperparameters.

4. **Battle-tested at scale exactly like chuck-mcp's.** BinaryConnect (2015), BNN (2016), TopOpt with Heaviside (1997-present) have all converged STE-style methods on problems with ~10⁹ binary parameters. Chuck-mcp's ~10⁹ mask pixels are in the same regime.

5. **The structural topology optimization community has been doing this for 30 years.** Density-based TopOpt with smoothing + thresholding is mathematically identical to chuck-mcp's mask layer. Every TopOpt code starts with continuous densities and converges to binary; that exact recipe (with STE substituting for Heaviside in step 1, see code below) is the chuck-mcp v2 implementation.

## ~150-line JAX implementation

```python
"""
chuck_mcp/v2/binary_mask.py

Binary mask layer for chuck-mcp v2. Produces per-region {0, 1} masks
from per-pixel logits, optimizable via JAX autodiff + JAXopt L-BFGS-B.

Default: hard-sigmoid STE with latent logit clipping.
Optional: Heaviside continuation for v2.5 escalation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt


# ============================================================================
# Binary mask production methods (interchangeable — same signature)
# ============================================================================

def ste_binary_mask(logit: jax.Array, *, sigmoid_backward: bool = True) -> jax.Array:
    """
    Hard-sigmoid STE (PRIMARY METHOD).

    Forward:  z = (sigmoid(logit) > 0.5).astype(float32)   exactly binary
    Backward: ∂z/∂logit = sigmoid'(logit) = p * (1-p)       (sigmoid_backward=True)
              or ∂z/∂logit = 1                              (sigmoid_backward=False, identity STE)

    Uses the numerically-stable JAX recipe (issue #9032):
        x - stop_grad(x) + stop_grad(f(x))
    instead of  x + stop_grad(f(x) - x)  which has float-precision issues.

    No stochasticity. Deterministic. L-BFGS-B compatible.
    """
    p = jax.nn.sigmoid(logit)
    z_hard = (p > 0.5).astype(p.dtype)
    surrogate = p if sigmoid_backward else logit          # backward path
    return surrogate - jax.lax.stop_gradient(surrogate) + jax.lax.stop_gradient(z_hard)


def heaviside_binary_mask(
    logit: jax.Array, *, beta: float = 16.0, eta: float = 0.5
) -> jax.Array:
    """
    Smoothed Heaviside projection (TopOpt convention, Guest et al. 2004 + Wang/Lazarov/Sigmund 2011).

    Forward:  H_β(σ(logit)) ∈ [0, 1]   — asymptotically binary as β → ∞
    Backward: closed-form smooth gradient. No STE bias.

    For chuck-mcp the recommendation is β-continuation:
        β = 1, 2, 4, 8, 16, 32, 64    (anneal between outer-loop batches)

    Returns soft mask in (0,1); final binarization by `> 0.5` threshold.
    """
    rho = jax.nn.sigmoid(logit)
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


def st_gumbel_sigmoid_mask(
    logit: jax.Array, key: jax.Array, *, tau_f: float = 0.1, tau_b: float = 0.7
) -> jax.Array:
    """
    Straight-Through Gumbel-Sigmoid with decoupled temperatures (Shah et al. 2024).

    Escalation method — use only if STE shows dead-mask collapse.
    Requires a fixed `key` per L-BFGS-B inner iteration; otherwise the
    function is not deterministic and curvature estimates break.

    Defaults τ_f=0.1, τ_b=0.7 from Shah et al. SBN experiments.
    """
    u = jax.random.uniform(key, logit.shape, minval=1e-6, maxval=1 - 1e-6)
    L = jnp.log(u) - jnp.log1p(-u)                        # Logistic(0, 1)
    p_f = jax.nn.sigmoid((logit + L) / tau_f)
    z_hard = (p_f > 0.5).astype(logit.dtype)
    p_b = jax.nn.sigmoid(logit / tau_b)                   # decoupled backward path
    return p_b - jax.lax.stop_gradient(p_b) + jax.lax.stop_gradient(z_hard)


# ============================================================================
# Region-mask container — wraps the chosen method
# ============================================================================

@dataclass(frozen=True)
class MaskConfig:
    method: str = "ste"            # "ste" | "heaviside" | "stgs"
    logit_clip: float = 5.0        # BinaryConnect-style latent clipping
    beta: float = 1.0              # Heaviside only; updated by continuation schedule
    eta: float = 0.5               # Heaviside threshold

    # ST-GS only
    tau_f: float = 0.1
    tau_b: float = 0.7


def make_binary_masks(
    mask_logits: jax.Array,        # shape (n_regions, H, W) per block
    config: MaskConfig,
    key: jax.Array | None = None,
) -> jax.Array:
    """Dispatch to the chosen method."""
    # latent clip — always applied, regardless of method
    logit = jnp.clip(mask_logits, -config.logit_clip, config.logit_clip)
    if config.method == "ste":
        return ste_binary_mask(logit, sigmoid_backward=True)
    elif config.method == "heaviside":
        return heaviside_binary_mask(logit, beta=config.beta, eta=config.eta)
    elif config.method == "stgs":
        if key is None:
            raise ValueError("ST-GS requires `key`; fix per outer-loop step.")
        return st_gumbel_sigmoid_mask(logit, key, tau_f=config.tau_f, tau_b=config.tau_b)
    else:
        raise ValueError(f"unknown method: {config.method}")


# ============================================================================
# Integration with JAXopt L-BFGS-B inner loop
# ============================================================================

def build_loss_fn(
    forward_render: Callable[[jax.Array, dict], jax.Array],
    target_image: jax.Array,
    mask_config: MaskConfig,
    fixed_key: jax.Array | None = None,
) -> Callable[[dict], jax.Array]:
    """
    Return a loss_fn(params) -> scalar suitable for JAXopt L-BFGS-B.

    `params` is a dict containing:
        - "mask_logits": shape (n_regions, H, W), the optimization variable
        - "block_colors": shape (n_blocks, 3) — pigment-space block colors
        - ... other chuck-mcp params (paper color, pull schedule, etc.)
    """
    def loss_fn(params: dict) -> jax.Array:
        masks = make_binary_masks(params["mask_logits"], mask_config, fixed_key)
        rendered = forward_render(masks, params)          # uses BINARY masks (v2 lock)
        return jnp.mean((rendered - target_image) ** 2)   # or full ΔE2000 loss
    return loss_fn


def solve_one_outer_batch(
    init_params: dict,
    forward_render: Callable,
    target_image: jax.Array,
    mask_config: MaskConfig,
    maxiter: int = 200,
):
    """One inner L-BFGS-B run with frozen mask method + (if needed) frozen RNG key."""
    # ST-GS needs a fixed key during the inner loop
    key = jax.random.PRNGKey(0) if mask_config.method == "stgs" else None
    loss_fn = build_loss_fn(forward_render, target_image, mask_config, fixed_key=key)
    bounds = jax.tree_util.tree_map(
        lambda p: (jnp.full_like(p, -mask_config.logit_clip),
                   jnp.full_like(p,  mask_config.logit_clip))
        if p.shape == init_params["mask_logits"].shape
        else (jnp.full_like(p, -jnp.inf), jnp.full_like(p, jnp.inf)),
        init_params,
    )
    solver = jaxopt.LBFGSB(fun=loss_fn, maxiter=maxiter)
    return solver.run(init_params, bounds=bounds)


# ============================================================================
# Outer-loop schedule (3-phase, integrates with inverse-rendering-diff staging)
# ============================================================================

def run_chuck_mcp_v2(
    init_params: dict, forward_render: Callable, target_image: jax.Array
):
    """
    Recommended 3-phase schedule:
      Phase 1: continuous warmup (no binarization) — let geometry settle
      Phase 2: STE binarization with logit-clip — masks now hard {0,1}
      Phase 3: (optional) Heaviside refinement with β continuation
    """
    # ---- Phase 1: smooth warmup, no binarization ----
    # Trick: use Heaviside with β=0.5 ≈ identity sigmoid
    cfg_p1 = MaskConfig(method="heaviside", beta=0.5, logit_clip=5.0)
    res, _ = solve_one_outer_batch(init_params, forward_render, target_image, cfg_p1, 100)

    # ---- Phase 2: switch to STE, exact binary forward ----
    cfg_p2 = MaskConfig(method="ste", logit_clip=5.0)
    res, _ = solve_one_outer_batch(res, forward_render, target_image, cfg_p2, 200)

    # ---- Phase 3 (optional): Heaviside continuation refinement ----
    # Only if STE final masks look noisy; β = 4, 16, 64 (annealed)
    for beta in [4.0, 16.0, 64.0]:
        cfg = MaskConfig(method="heaviside", beta=beta, logit_clip=5.0)
        res, _ = solve_one_outer_batch(res, forward_render, target_image, cfg, 50)

    # ---- Final hard-threshold for export ----
    final_masks = (jax.nn.sigmoid(res["mask_logits"]) > 0.5).astype(jnp.uint8)
    return res, final_masks
```

---

## Failure modes and how to detect them

| Failure mode | How to detect | Mitigation |
|--------------|---------------|------------|
| **Masks freeze early** (all `|logit| >> clip_val` after few iters) | Plot histogram of `mask_logits` every N iters; if >80% are at clip bounds within first 20 iters, masks are frozen | Lower learning rate (or smaller L-BFGS-B step size); add Phase-1 warmup with Heaviside β=0.5; if persistent, switch to ST-GS |
| **Dead masks** (whole region's logit collapses to all-zero or all-one) | Track per-region `mean(z)` over iterations; if a region's mean is < 1% or > 99% and not changing, it's dead | Re-initialize that region's logits with topo-derivative spawn (see inverse-rendering-diff/NOTES.md); or switch to ST-GS for exploration noise |
| **Soft pixels in "binary" output** | After Phase 2 (STE), `unique(masks)` should be exactly `{0.0, 1.0}`. If not, you've used Heaviside or the buggy `x + stop_grad(f(x) - x)` STE | Use the issue-#9032 form `x - stop_grad(x) + stop_grad(f(x))`. Always final-threshold at export. |
| **Loss plateau (v12→v13 dE floor)** | Same loss for 20+ iterations | This is the structural problem from inverse-rendering-diff/NOTES.md, not a binarization problem. Use staged 3-batch outer loop (Wang+Mehta+Worchel). |
| **Gradient explosion at clip boundary** | `nan` or `inf` in `mask_logits` updates | Reduce clip range (5 → 3) or use sigmoid-backward STE (built-in saturation), not identity STE |
| **Checkerboard artifacts** | High-frequency 1-pixel patterns in final masks | Apply mokuhanga blur kernel to `sigmoid(logit)` **before** the STE step — i.e. mask = STE(blur(sigmoid(logit))). TopOpt convention. |
| **L-BFGS-B Hessian degeneracy** | Solver reports "small step" or "line search failed" | STE bias accumulates in the L-BFGS-B Hessian approximation. Reset the L-BFGS-B history every 50 iters, or use plain GD/Adam for the final 50 iters. |

---

## Integration with existing chuck-mcp v2 design

### Connects to other research domains

- **inverse-rendering-diff/NOTES.md:** staged 3-batch outer loop. Each outer batch calls `solve_one_outer_batch` above with its own L-BFGS-B inner run. Binary mask method (STE / Heaviside β) is held constant *within* a batch.
- **mokuhanga-methodology/NOTES.md:** `(block, pass_idx, mask)` impression key. Each `(block, pass_idx, region)` triple gets its own logit field. Total: 27 blocks × ~5 regions × ~3 pulls ≈ 405 mask tensors per image. All optimized jointly by L-BFGS-B.
- **color-science-km-mixbox/NOTES.md:** forward render uses binarized masks to gate K/S accumulation. This is the v2 "loss reflects what will actually print" lock.
- **segmentation-cellgraph/NOTES.md:** SNIC polygons provide *initial* region mask logits — set logit = +3 inside the SNIC polygon, -3 outside, then optimize.
- **vectorization-cnc/NOTES.md:** binarized masks → mill-sized morphology → Potrace SVG → CNC. The export step is post-optimization; no gradients needed.

### Annealing schedule (full pipeline)

```
Outer batch 1 (geometry settles):
    inner: Heaviside β=0.5  (smooth, identity-like)
    50-100 L-BFGS-B steps
Outer batch 2 (binarization):
    inner: STE (hard-sigmoid, sigmoid backward)
    200 L-BFGS-B steps
Outer batch 3 (refinement, optional):
    inner: Heaviside β=4 → 16 → 64
    50 L-BFGS-B steps per β
    (only if STE final isn't clean enough)
Export:
    masks = (sigmoid(logit) > 0.5).astype(uint8)
```

---

## Top-3 must-reads (in priority order)

1. **arxiv_1308_3432_bengio_ste.md** — Bengio's original STE paper. Knowing the four families and why three of them are wrong for chuck-mcp is the foundation.
2. **arxiv_2012_02860_heaviside_topology.md** — Behrou et al. (and via them the entire TopOpt heritage). The strongest alternative to STE and the source of the continuation-in-β technique used in the v2.5 escalation.
3. **arxiv_1511_00363_binaryconnect.md** — Courbariaux et al. BinaryConnect. The "shadow weight + clipping + hard-tanh STE" pattern *is* the chuck-mcp v2 mask layer.

(Honorable mentions: `web_jax_ste_recipe.md` for the JAX-issue-#9032 numerically-stable form; `arxiv_2410_13331_decoupled_ste.md` for ST-GS temperature tuning if STE fails.)

## File count

7 artifacts:
- arxiv_1308_3432_bengio_ste.md (foundational STE)
- arxiv_1611_01144_gumbel_softmax.md (ST-GS escalation)
- arxiv_1611_00712_concrete_distribution.md (theoretical sibling of Gumbel-Softmax)
- arxiv_1712_01312_louizos_hard_concrete_l0.md (hard concrete + L0, future v3)
- arxiv_1511_00363_binaryconnect.md (engineering pattern: shadow weights + clip)
- arxiv_2012_02860_heaviside_topology.md (TopOpt — strongest alternative to STE)
- arxiv_2410_13331_decoupled_ste.md (Decoupled-ST temperature tuning)
- web_jax_ste_recipe.md (numerically-stable JAX implementation)
- NOTES.md (this file — verdict + impl + failure modes)
