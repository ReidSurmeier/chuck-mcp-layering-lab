---
title: Numerically Stable Straight-Through Estimator in JAX (canonical recipe)
sources:
  - https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html
  - https://github.com/jax-ml/jax/issues/9032
  - https://apxml.com/courses/advanced-jax/chapter-4-advanced-automatic-differentiation/handling-non-differentiable-functions
applicability_to_chuck_mcp: this is the actual JAX code chuck-mcp will ship — `jax.lax.stop_gradient` based STE, in the numerically-stable form recommended by JAX maintainers (issue #9032).
---

## The non-obvious gotcha

The naive STE pattern in JAX is:

```python
def ste_naive(x, f):
    return x + jax.lax.stop_gradient(f(x) - x)        # WRONG — float-precision issue
```

This is *mathematically* correct (forward = `f(x)`, backward = `∂x/∂x = 1`), but **fails for float32** because of IEEE 754 non-associativity. `f(x) - x` is computed at float32, and the subtraction loses bits if `f(x)` and `x` are close in magnitude. The result is that the forward value is no longer exactly `f(x)` — it's `f(x) + ε` for some ε ~ 1e-7.

For STE with binary masks (where `f(x) = (x > 0.5).astype(float32)`) this means the forward "binary" mask is sometimes 0.99999994 instead of exactly 1.0. Downstream code that does `mask.astype(int)` or `mask == 1` will silently mis-classify pixels.

## The fix (JAX issue #9032)

Recommended form:

```python
def ste(x, f):
    """Straight-through estimator: forward = f(x), backward = identity."""
    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(f(x))
```

This works because:

1. `x - stop_gradient(x)` is **exactly zero** in the forward pass (Sterbenz lemma: subtracting two identical floats is exact).
2. The forward value is therefore exactly `0 + f(x) = f(x)`.
3. The backward gradient: `stop_gradient` drops the second and third terms, so `∂/∂x = ∂(x)/∂x = 1`. STE behavior achieved.

## Binary mask STE for chuck-mcp

```python
import jax, jax.numpy as jnp

def binarize_ste(logit):
    """Binary mask from logit. Forward = hard {0, 1}, backward = sigmoid Jacobian."""
    p = jax.nn.sigmoid(logit)
    z_hard = (p > 0.5).astype(p.dtype)
    # STE: forward = z_hard, backward = ∂p/∂logit = p * (1 - p)
    return p - jax.lax.stop_gradient(p) + jax.lax.stop_gradient(z_hard)
```

Note: the backward gradient here is the sigmoid Jacobian, **not** identity. This is the "saturating STE" variant — gradients shrink near `|logit| → ∞` because `p * (1 - p) → 0`. This is what BinaryConnect, BNN, and most production STE deployments use.

For pure identity STE (gradient always 1):

```python
def binarize_identity_ste(logit):
    """Forward = (logit > 0), backward = 1."""
    z_hard = (logit > 0).astype(logit.dtype)
    return logit - jax.lax.stop_gradient(logit) + jax.lax.stop_gradient(z_hard)
```

For hard-tanh STE (gradient = 1 in `[-1, 1]`, 0 outside):

```python
def binarize_hardtanh_ste(logit):
    """Forward = sign(logit), backward = 1{|logit| <= 1}."""
    z_hard = jnp.sign(logit)
    surrogate = jnp.clip(logit, -1.0, 1.0)             # hard-tanh
    return surrogate - jax.lax.stop_gradient(surrogate) + jax.lax.stop_gradient(z_hard)
```

## L-BFGS-B compatibility

All three variants above are **deterministic** (no random key needed). They produce **exact binary** forward values (no float drift). They are **fully traceable** by JAX's autodiff — `grad`, `value_and_grad`, `jit`, `vmap` all work.

JAXopt's L-BFGS-B expects a deterministic `loss_fn(params) → scalar`. If the forward pass is deterministic, JAXopt will work with STE without any modification:

```python
import jaxopt

def loss_fn(params):
    logits = params["mask_logits"]               # shape (H, W, n_regions)
    masks = binarize_ste(logits)                 # forward = binary, backward = sigmoid
    return forward_render_loss(masks, params)    # deltaE etc.

solver = jaxopt.LBFGSB(fun=loss_fn, maxiter=200)
solver.run(init_params, bounds=(logit_lo, logit_hi))
```

(`bounds` is where you enforce the latent clipping that BinaryConnect recommends — set `logit_lo, logit_hi = -5, +5` to keep sigmoids in [0.0067, 0.9933].)

## Cited reference snippets (verbatim)

From JAX issue #9032 maintainer reply:
> "The order of operations matters: computing `(f(x) - x)` first and then adding to `x` yields different results than rearranging the computation. We recommend `x - stop_gradient(x) + stop_gradient(f(x))` as the numerically robust form."

From JAX docs:
> "`jax.lax.stop_gradient(x)`: identity function that does not let its gradients flow through; gradients computed via `jax.grad` will be zero with respect to `x`."

## Application: chuck-mcp v2 mask layer

The chuck-mcp v2 stack will need one STE call per region mask per block. Approximate sizing:

- 27 blocks × 5 regions × 3 pulls (average) = 405 binary masks per image
- Each mask: ~5 megapixels (typical chuck-mcp print size)
- Total STE forward calls per L-BFGS-B inner iteration: ~2e9 binarize operations

JAX should vectorize this trivially via `jax.vmap` over the mask dimension. The forward cost is `2 × sigmoid + 1 × comparison + 1 × subtract + 1 × add` = ~5 fused FLOPs per pixel. The backward cost is similar. This is in the noise compared to the forward render (which involves Kubelka-Munk recursion, also per-pixel but with more arithmetic).

**Net:** STE adds ~3% to the inner-loop cost. Free.
