---
arxiv_id: 2511.13191
title: "Birth of a Painting: Differentiable Brushstroke Reconstruction"
authors: Jiang, Lu, Chen, He, Wu, Yang, Jiang
published: 2025-11-17
url: https://arxiv.org/abs/2511.13191
relevance_to_chuck_mcp: VERY HIGH — already a t1 must-read for inverse-rendering. Stroke reconstruction = pull reconstruction.
---

# Birth of a Painting — relevance to load-bearing pull scoring

## Core relevance

This paper does **differentiable stroke reconstruction** of an *image* into a stack of B-spline strokes with smudge operator. The "stroke" abstraction is structurally identical to our "pull":
- `Θ_stroke = {Bezier control points, color, alpha, radius}` ↔ `pull = {mask, block_id, pass_idx, ink_color}`
- Both have **forward render** (differentiable)
- Both have **stroke ordering** that matters (smudge requires ordered blending; mokuhanga overprint is order-sensitive in K-M)

## Key technical move that helps load-bearing-test

**Parallel differentiable paint renderer** (their Section 3.2):
> "Removes the need for sequential blending and allows efficient stroke construction on GPUs."

Their `Algorithm 2` decomposes strokes into independent stamps along the Bézier curve. **All stamps render in parallel**, then a single ordered alpha-composite step composes them. This means:
- Forward render of N strokes is **O(1) in stroke count** with vmap.
- Backward render is also **O(1) in stroke count** — JAX gives gradient w.r.t. all stroke params simultaneously.

For chuck-mcp: same trick. If we vmap the K-M render across pulls (each producing a per-pull transmittance contribution), we get **load-bearing gradient for all 132 pulls in a single backward pass**.

## "Coarse-to-fine optimisation strategy"

Their pipeline matches Wang 2024 layered vectorisation:
1. Initialise stroke skeleton from coarse semantic structure.
2. Add strokes to reduce residual error (visual refinement).
3. Continue until residual is below threshold.

Same as our **staged 3-batch outer loop** from inverse-rendering NOTES. They confirm that staged optimisation with growth-and-prune is the right structure for stroke-based reconstruction.

## Stroke pruning via Optimize & Reduce (hirschorn2024optimize)

They cite Hirschorn 2024 "Optimize & Reduce" as the canonical stroke pruning reference:
> "top-down strategies with redundant stroke pruning (hirschorn2024optimize) and semantic simplification methods (wang2025layered) to obtain semantically consistent SVGs."

This is the SOTA recipe:
- Start with K strokes.
- Optimise.
- Score strokes (per-stroke loss contribution).
- Remove bottom-N% scored strokes.
- Repeat.

**Score = pixel-domain loss change when stroke is hidden** = our load-bearing score.

## Performance reference

Their parallel paint renderer hits **interactive rates** on a single GPU for hundreds of strokes. Translates to chuck-mcp:
- 132 pulls × paper-resolution canvas ≈ 8MP × 132 = 1Gpx forward render
- On A100/4090 in JAX with vmap'd K-M overprint: estimated **50-200ms forward, 100-400ms backward** per gradient call.
- **Budget for one load-bearing pass: ~600ms total**. Comfortably under 5-second budget.

## Verdict for load-bearing-test

Reuse the parallel-differentiable-render pattern. Each pull contributes an independent transmittance term, then ordered K-M composition merges them. Gradient through this gives per-pull score in **one backward pass**.

```python
# Pseudocode for parallel differentiable forward render
@jax.jit
def parallel_forward_render(pulls, paper):
    # vmap render single pull -> single-pull canvas
    per_pull_canvas = jax.vmap(render_single_pull)(pulls)  # (N, H, W, 3)
    # Ordered K-M composite
    composite = paper
    for k in range(len(pulls)):  # cant vmap order-sensitive op
        composite = km_overprint(composite, per_pull_canvas[k], pulls[k].mask)
    return composite

@jax.jit
def load_bearing_scores(pulls, target):
    def loss(p):
        composite = parallel_forward_render(p, paper)
        return jnp.mean(deltaE(composite, target))
    grads = jax.grad(loss)(pulls)
    # one entry per pull's mask
    return jnp.array([jnp.sum(jnp.abs(g.mask)) for g in grads])
```
