---
arxiv_id: 2406.05404
title: "Layered Image Vectorization via Semantic Simplification"
authors: Wang et al.
published: 2024-06-09
url: https://arxiv.org/abs/2406.05404
venue: SIGGRAPH 2024
relevance_to_chuck_mcp: HIGH — uses progressive simplification + low-contribution pruning; same idea as load-bearing
---

# Layered Vectorization via Semantic Simplification (Wang 2024)

## Why this is on the load-bearing radar

This paper builds **compact layered representation** by:

1. Generate a progressively simplified target image sequence (via Score Distillation Sampling) — coarse first, fine last.
2. Two-stage optimisation: **structural buildup** (large semantic shapes), then **visual refinement** (small detail vectors).
3. **Periodic pruning** of redundant vectors with low-ranking scores after optimisation:
   > "Optimize & Reduce (O&R)... adds primitives based on pixel clustering, and removes primitives with low-ranking scores after optimization."

Pull k in chuck-mcp = vector primitive in their pipeline. **"Low-ranking score"** is their version of our load-bearing score. They remove low-contributing vectors at periodic intervals. We should do the same.

## Already in INDEX.md

Already flagged as a top-3 must-read for `inverse-rendering-diff/`. Their **progressive-simplified targets** (Stage I → Stage II → ... → final) was adopted in the v2 plan as the **staged 3-batch outer loop**.

## What's new for load-bearing-test specifically

- Pruning criterion: **rasterise each vector, measure pixel-area visibility and pixel-domain contribution to the loss**. Vectors below a threshold are removed.
- They prune periodically (every N steps), not once at the end.
- Their pipeline confirms: **structural pulls (Stage I) become load-bearing first, then visual-refinement pulls (Stage II) added on top must justify themselves via reducing residual error.**

## Direct chuck-mcp mapping

```python
# Inside staged 3-batch outer loop (from inverse-rendering NOTES)
for batch in [structural_pulls, midtone_pulls, detail_pulls]:
    # Inner L-BFGS-B optimization
    pulls = optimize(pulls + batch, target, n_steps=100)
    
    # Load-bearing pruning every 50 steps:
    if step % 50 == 0:
        scores = load_bearing_score_cheap(pulls, target)  # SLA-style gradient
        pulls = [p for p, s in zip(pulls, scores) if s > threshold]
        # Spawn new pulls to fill the gap (Mehta topo-derivative)
```

## Sparsity philosophy alignment

Wang's overall message:
> "Compact layered representation" is achieved by **iterative growth + pruning** with semantic-aware simplification targets, *not* by fixed-N a-priori budgeting.

This directly justifies the chuck-mcp v2 design philosophy: **don't pre-budget 132 pulls and hope they all work — start coarse, add pulls only where residual error demands, prune pulls that stop earning their place**.

## Lesson for soft-vs-hard load-bearing penalty

Wang uses a hard removal (binary prune). But the prune is **applied periodically inside an optimisation that can re-add primitives**, so a wrongly-pruned vector can be respawned by topo-derivative spawning. Effectively this is **hard prune + retry**, which approximates soft penalty without the gradient-pathology issues of L0-like penalties.

Recommendation for chuck-mcp v2: **hard prune at outer-loop boundaries, soft `||mask||₁` regulariser inside the inner L-BFGS-B**. Gets the differentiability benefit of soft penalty and the cleanliness of hard prune.
