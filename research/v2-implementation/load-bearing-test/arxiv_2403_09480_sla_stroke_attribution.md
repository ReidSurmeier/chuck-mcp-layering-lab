---
arxiv_id: 2403.09480
title: "What Sketch Explainability Really Means for Downstream Tasks"
authors: Bandyopadhyay, Chowdhury, Bhunia, Sain, Xiang, Song
venue: CVPR 2024
published: 2024-03-14
url: https://arxiv.org/abs/2403.09480
relevance_to_chuck_mcp: HIGH — direct analogue: per-stroke attribution via differentiable rasterisation
---

# SLA / P-SLA — Stroke-Level Attribution (sketch domain)

## Why this paper matters for load-bearing-test

We have `N≈132` pulls (sketches have `T` strokes — same structure). They have to compute "how much does this stroke contribute to the final result?" while rasterisation is non-differentiable. They solve it. Same problem, different domain.

## Core trick (the load-bearing analogue)

Standard pipeline: vector sketch → rasterise (non-diff) → render → loss. Gradient can't flow back to strokes.

Their fix — **rasterise each stroke independently, then sum**:
```
For each stroke s_k: X_k = rasterise(s_k)  # non-diff, single-stroke canvas
X = Σ_k X_k                                 # diff (addition)
```

Now `∂L/∂X_k` exists, and `X_k` is the pixel footprint of stroke k. The attribution score for stroke k is:

```
A(stroke_k) = Σ_pixels |∂L/∂X · X_k|        # stroke-level attribution (SLA)
```

It's literally **integrated gradients restricted to the pixels touched by stroke k**, with the rest of the rasterised canvas held fixed.

## Why it works for us (mokuhanga overprint = additive in opacity, multiplicative in K-M)

Mokuhanga forward render is *not* additive in RGB — it's K-M overprint recursion (Curtis 1997). But:

- The **K-M recursion is differentiable end-to-end** (Mixbox is differentiable too, and so is the 36-wavelength K/S table from Berns 2016, just per-band).
- So we don't even need their non-diff trick. We *already* have `∂L/∂pull_k` for free if pull_k's mask is one of the optimisation variables.
- What we *do* need is their **attribution formulation**: project the pixel-gradient back onto the spatial footprint of each pull.

## Translation to chuck-mcp v2

In our case `s_k → (block_id, pass_idx, mask_k, color_k)`. The "stroke" is a pull. Load-bearing score is:

```python
# Forward render: composite = km_overprint(pulls[0..N], paper)
# loss = ΔE(composite, target)

# Per-pull attribution:
def load_bearing_score(pull_k, composite, target, mask_k):
    # Method 1: ablation gradient (sensitivity)
    grad = jax.grad(lambda c: ΔE(c, target))(composite)          # pixel grad
    s_k = jnp.sum(jnp.abs(grad) * mask_k)                         # restricted to pull k's footprint
    return s_k

# Method 2: counterfactual (true ablation)
def load_bearing_counterfactual(pulls, k):
    composite_with = forward_render(pulls)
    composite_without = forward_render(pulls[:k] + pulls[k+1:])
    return jnp.mean(deltaE(composite_with, composite_without))
```

Method 1 is **gradient-based, O(1 forward + 1 backward)** — cheap.
Method 2 is **true Shapley-like leave-one-out, O(N) forward renders** — expensive.

## Performance per their paper

- Single forward + single backward pass gives all `T` stroke attributions simultaneously (one gradient call vmaps across strokes via the additive decomposition).
- They run on CPU sketch demos at interactive rates.

## What we should steal

1. **Use the gradient × mask trick** as the default load-bearing score. Differentiable through the K-M render via JAX autodiff. **O(1) per attribution map.**
2. **Reserve counterfactual ablation for verification only**, not for the inner loop.
3. **Use the partial-stroke variant (P-SLA)** if we want sub-pull localisation — distance-transform-based, useful when a single pull is partially load-bearing (only the inked region matters).

## Caveats for overprint physics

- K-M is **multiplicative in transmittance space**, so the linear "sum of single-stroke canvases" trick from SLA doesn't directly apply. But gradient × mask still does.
- For pulls where mask is mostly black (no ink), `mask_k` is small → score is small → naturally non-load-bearing → solver can drop them without explicit penalty.

## Failure modes to watch

- **Cancellation:** Pull A and pull B might both be load-bearing only because they fight each other (over-print correction). Single-pull gradient says both matter; removing either makes things worse; removing both might be neutral. Ablation pairs would catch this; gradient alone wouldn't.
- **Saturation:** If a region is already maxed (full coverage), additional pulls there have zero gradient → score 0 → flagged as non-load-bearing → correctly dropped.

## Code structure (JAX)

```python
import jax, jax.numpy as jnp

def attribution_per_pull(pulls, target, render_fn, deltaE_fn):
    """O(1) attribution: one forward + one backward."""
    def loss(pulls_var):
        composite = render_fn(pulls_var)
        return deltaE_fn(composite, target).sum()

    grad_pulls = jax.grad(loss)(pulls)   # gradient w.r.t. all pull params at once
    # grad_pulls.mask_k tells us how much loss would change if mask_k changed
    # |grad| in the mask's spatial footprint = load-bearing score
    scores = jnp.array([jnp.sum(jnp.abs(g.mask)) for g in grad_pulls])
    return scores

def attribution_counterfactual(pulls, target, render_fn, deltaE_fn):
    """Verification: O(N) forward renders, no backward."""
    base = render_fn(pulls)
    base_loss = deltaE_fn(base, target).mean()
    def drop_k(k):
        composite = render_fn(pulls[:k] + pulls[k+1:])
        return deltaE_fn(composite, target).mean() - base_loss
    scores = jax.vmap(drop_k)(jnp.arange(len(pulls)))
    return scores
```
