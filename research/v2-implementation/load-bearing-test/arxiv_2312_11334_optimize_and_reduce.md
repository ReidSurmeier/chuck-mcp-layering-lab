---
arxiv_id: 2312.11334
title: "Optimize & Reduce: A Top-Down Approach for Image Vectorization"
authors: Hirschorn, Jevnisek, Avidan
venue: AAAI 2024
published: 2023-12-18
url: https://arxiv.org/abs/2312.11334
code: https://github.com/ajevnisek/optimize-and-reduce
relevance_to_chuck_mcp: VERY HIGH — this is THE canonical reference for "load-bearing shape pruning"
---

# Optimize & Reduce — the reference implementation for load-bearing

## Why this is the most directly applicable paper found

This paper does *exactly* what chuck-mcp v2 needs to do for pulls, but for SVG shapes. Their rank-score formula is the load-bearing score I'd recommend adopting unchanged.

## The importance measure (formal, paste-into-code ready)

> "For each shape `P_i`, the ranking loss measures the degradation caused by not including that specific shape when the image is rendered. The loss is measured against the original raster image `I`. Therefore, the loss for the i-th shape, includes rendering all shapes but the i-th shape:
> 
> Î = render(P \ P_i)
> 
> rank-score[i] = L_reduce(I, Î)"

This is **literally counterfactual ablation**:
- Render everything except shape i.
- Measure how bad the result is vs original target.
- Shape i with the smallest degradation = least important = first to prune.

`L_reduce` can be MSE, L1, perceptual, or any pixel-wise loss. For chuck-mcp: **ΔE76 or ΔE2000** in CIE L*a*b*.

## The schedule

> "We initialize Bézier curves... we then iterate between optimization and reduction steps for the following schedule: 256 → 128 → 64."

**Top-down halving**. Start with too many primitives, halve and re-optimise, halve and re-optimise. Each halving removes the lower-scored half.

For chuck-mcp:
- Start with `N=200` candidate pulls (over-budget by ~50%).
- Optimise inner L-BFGS-B.
- Score each pull via counterfactual ablation.
- Drop bottom-N pulls.
- Re-optimise.
- Repeat until N ≤ 132 OR ΔE penalty exceeds tolerance.

This gives the **soft-with-recourse** dynamic the user wants: pulls compete for the budget, weak ones fall out.

## Speedup: 10× faster than LIVE

> "We show that our algorithm is ×10 faster than the state-of-the-art optimization-based method (Ma et al. 2022)" [LIVE].

The reason: **top-down with reduction is much cheaper than bottom-up**. Bottom-up has to optimise N shapes from scratch every time. Top-down only re-optimises shapes that survived.

For chuck-mcp: directly maps to "start with 200 pulls, reduce to 132" being **dramatically faster** than "start with 132 from scratch and add more as needed".

## Cost of one ranking pass

O&R does `N` forward renders per reduction step (one for each shape removed). For `N=132`:
- 132 forward renders × ~50ms each = ~6.6 seconds. **Over the 5s budget for a single ranking pass.**

But two tricks make this affordable:
1. **Cache the base render**; only re-render the *delta* when one shape is removed. For pure additive compositing (alpha blend) this is exact. For K-M overprint it's an approximation; for our purposes (ranking, not absolute scoring) the approximation is good.
2. **vmap the ablation** across pulls — JAX can ablate all 132 pulls in parallel, sharing the base intermediate.

Estimated chuck-mcp cost: ~1 second for full 132-pull counterfactual ranking pass. **Under budget.**

## Loss term recommendations from O&R

```
L_optimize = L_2(I, render(P)) + λ · L_geometric
```

For chuck-mcp:
- `L_2` → ΔE76 or ΔE2000 in L*a*b*.
- `L_geometric` → Wang 2024's overlap penalty (mokuhanga blocks should not overlap).
- Add `L_loadbearing = Σ_k max(0, threshold - score_k)` as a soft hinge on per-pull importance.

## Algorithm 1 (paraphrased for chuck-mcp)

```python
def optimize_and_reduce_pulls(target, N_init=200, N_final=132, halving_steps=2):
    pulls = initialize_pulls_from_blocks(target, N=N_init)
    
    for step in range(halving_steps + 1):
        # Optimize
        pulls = inner_lbfgs(pulls, target, L_optimize, n_iters=200)
        
        # Skip reduce on last step
        if step == halving_steps:
            break
        
        # Reduce
        base = forward_render(pulls)
        scores = jnp.zeros(len(pulls))
        for i in range(len(pulls)):
            ablated = forward_render(pulls[:i] + pulls[i+1:])
            scores = scores.at[i].set(deltaE(ablated, target).mean() - deltaE(base, target).mean())
        
        # Keep top N_target
        N_target = max(N_final, len(pulls) // 2)
        keep = jnp.argsort(-scores)[:N_target]
        pulls = [pulls[i] for i in keep]
    
    return pulls
```

## Verdict: this is the implementation template

- **Soft vs hard penalty**: O&R does **hard removal at scheduled checkpoints** (between optimisation passes), with **continuous optimisation in between**. This is the right hybrid.
- **Importance measure**: counterfactual ablation, exactly as recommended in the SLA artifact.
- **Performance**: ~1s per full ranking pass via JAX vmap; well under 5s budget.

This is the paper. Implement chuck-mcp v2 load-bearing as a direct port of O&R, substituting:
- DiffVG → JAX K-M overprint renderer.
- Bézier shapes → mokuhanga pulls.
- MSE / CLIP → ΔE2000.
- L_geometric → block-overlap penalty.
