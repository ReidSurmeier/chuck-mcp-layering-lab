---
title: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
authors: [Yoshua Bengio, Nicholas Léonard, Aaron Courville]
arxiv_id: 1308.3432
year: 2013
url: https://arxiv.org/abs/1308.3432
applicability_to_chuck_mcp: foundational — defines the straight-through estimator (STE) family. Most direct prior art for chuck-mcp v2 binary masks. STE = pass binary sample forward, pretend it was the sigmoid in backward. Cheapest, lowest-overhead choice. Works for "binary stochastic neuron with Bernoulli p = sigmoid(a)" — exactly the chuck-mcp v2 region-mask use case (each pixel is in or out of a region).
---

## Why this paper

Bengio's STE is the foundational technique for "I want a binary output forward, but I want gradients to flow backward as if it were continuous". It is **the** cheapest, simplest, most-widely-deployed way to produce a binary mask from a sigmoid logit while still optimizing with gradient descent. For chuck-mcp v2, where each of up to ~135 region masks must be pixel-wise binary {0, 1} at print time, STE is the default candidate.

The paper enumerates four families of solutions for backprop through stochastic / hard non-linearities:

1. **REINFORCE / score-function** — unbiased but high variance. Each binary mask pixel gets only a "reward signal" gradient. For chuck-mcp's 135 masks × millions of pixels, the variance is catastrophic. Reject.
2. **Decomposed stochastic+smooth** (Bengio's novel contribution) — split a stochastic binary neuron into a stochastic part + smooth differentiable expectation. Sound theoretically, but adds complexity.
3. **Additive/multiplicative noise injection** — gives "soft" binarization. Useful for warmup, not for the final output.
4. **Straight-through estimator (STE)** — heuristic: copy `∂L/∂z` directly to `∂L/∂a` where `z = (sigma(a) > 0.5)`. Biased but low-variance. Dominant in practice for binary weights, activations, masks.

## STE in chuck-mcp form

```
forward:  z = (sigmoid(a) > 0.5).astype(float32)        # binary mask pixel
backward: ∂L/∂a := ∂L/∂z                                 # pretend z = sigmoid(a)
```

In JAX, the numerically-stable identity-STE trick is:

```python
def ste_binary(a):
    p   = jax.nn.sigmoid(a)
    hard = (p > 0.5).astype(p.dtype)
    return p + jax.lax.stop_gradient(hard - p)
```

(Equivalently: `hard - stop_gradient(hard) + stop_gradient(p)` per the JAX issue-9032 recommendation, see `web_jax_ste_recipe.md`.)

## Properties relevant to chuck-mcp

- **Bias:** STE is biased — backward "gradient" is not a real gradient of the forward (hard) function. Bias is largest near the decision boundary `sigmoid(a) ≈ 0.5`, smallest when `|a| >> 0`.
- **Variance:** Low. Deterministic forward (no sampling noise), so the only randomness is over training-batch noise, identical to standard SGD.
- **Convergence:** Empirically excellent on BinaryConnect, BNN, BinaryGAN, MaskGAN. Theoretical analysis Yin et al. 2019, Jeong et al. 2025 (arxiv 2505.18113) shows STE converges to global minimum under sample-complexity bounds on two-layer binary nets.
- **Saturation problem:** if `a → ±∞`, the *forward* output is fine but the *learning signal* is also still being copied through — which can blow up. Standard mitigation: **clip the latent `a` to [-1, +1]** (Courbariaux et al. BinaryConnect / Hubara et al. BNN, also see hard-tanh straight-through which only passes gradient when `|a| < 1`).

## The "saturating STE" variant

Bengio et al. note that **identity-STE always passes gradient** is suboptimal because once a binary neuron is saturated (very large `|a|`), gradient should stop arriving. They recommend the "hard-tanh STE":

```
forward:   z = sign(a)            # or (a > 0)
backward:  ∂L/∂a := ∂L/∂z * 1{|a| <= 1}    # gradient only inside the linear region
```

This is the variant that BinaryConnect and BNN use in practice and is the most battle-tested. **Recommended default for chuck-mcp v2.**

## Pseudocode (binary stochastic neuron variant)

```
# stochastic forward
p = sigmoid(a)
u ~ Uniform(0, 1)
z = (u < p).astype(float32)        # Bernoulli(p) sample

# backward (straight-through)
dL/da := dL/dz                      # identity STE
# or hard-tanh:
dL/da := dL/dz * 1{|a| <= 1}
```

## Chuck-mcp application notes

- Each region mask is parameterized by a per-pixel logit `a_i`. Forward: `z_i = sign(a_i)`. Backward: hard-tanh STE.
- The L-BFGS-B inner loop is **deterministic** — there's no batch noise — so the only "variance" is from the bias of the STE gradient surrogate. This is *better* than the typical SGD regime where STE was first studied.
- L-BFGS-B is second-order; it builds a Hessian approximation from gradients. STE gradients are biased, so the Hessian approximation will also be biased. Mitigation: **lock the binary masks (`stop_gradient` on `z`) after warmup**, and let L-BFGS-B optimize *only the colors/heights* in the final phase. See NOTES.md for the recommended 3-phase schedule.

## Verdict for chuck-mcp

STE is the **primary recommendation**. It is the cheapest method, the simplest JAX implementation, the best-understood theoretically, and the one with the most prior-art in similar settings (BinaryConnect, BNN, topology optimization with thresholding).

The only candidate that might beat STE on quality is **hard concrete + L0** (Louizos), but at 5-10x the implementation complexity and an extra hyperparameter (stretch range `(γ, ζ)`) to tune. Reserve hard concrete for if STE shows pathological behavior (region masks that won't binarize, dead-mask collapse).
