---
arxiv_id: 2308.09865
title: "Topological Derivatives for Inverse Rendering"
authors: Mehta, Lombardi, et al.
published: 2023-08-19
url: https://arxiv.org/abs/2308.09865
relevance_to_chuck_mcp: HIGH — already in inverse-rendering NOTES. Counterpart to load-bearing: where to ADD a pull
---

# Topological Derivatives — the dual problem of load-bearing

## Why this is the mirror of load-bearing-test

Load-bearing answers: **"which existing pulls earn their place?"** (remove the bad ones).
Topological derivatives answer: **"where should a NEW pull go to most reduce residual error?"** (add the right ones).

Both are needed for the staged outer loop. The combination is the **add-prune lifecycle**:
1. Compute topological derivative map → spawn new pulls where it's high.
2. Optimise.
3. Compute load-bearing scores → drop pulls where they're low.
4. Repeat until residual ΔE plateau or pull budget hit.

## Math (paraphrased from abstract)

A "topological derivative" `T(x)` at pixel `x` measures: **what would the loss decrease if we introduced an infinitesimally small new primitive at `x`?**

Formally: `T(x) = lim_{ε → 0} [L(I_with_hole(x,ε)) - L(I_no_hole)] / ε^d` where `d` is the spatial dimension.

For chuck-mcp:
- `T(x) = ∂L/∂(introduce_new_pull_at_x)` — gradient w.r.t. spawning a new pull centred at `x`.
- High `T(x)` = a new pull at `x` would help → spawn here.

## Important property: it's a *pixel map*, not a scalar

The topological derivative is a **dense spatial field**. Single forward+backward pass gives `T(x)` for every pixel. Then:
- Threshold → candidate spawn locations.
- Cluster → consolidated new-pull centres.

For chuck-mcp v2: integrates cleanly with SNIC segmentation (already adopted) — cluster `T(x)` peaks within SNIC superpixel boundaries to get a discrete spawn candidate set.

## Performance

The paper validates on image vectorisation (small budget vector-graphics generation), 3D reconstruction, and ambigrams — all with primitive counts of 10-1000 range, similar to chuck-mcp's 132. The topological derivative computation **is one forward+backward render through the existing solution**, so it's O(1) in primitive count — fast.

## How this composes with load-bearing in the v2 plan

```python
for outer_iter in range(MAX_OUTER):
    # Inner: optimise current pull set
    pulls = inner_lbfgs(pulls, target, n_iters=200)
    
    # Score load-bearing (counterfactual ablation)
    scores = load_bearing(pulls, target)
    
    # Compute topological derivative (residual gradient map)
    residual = target - render(pulls)
    T = topological_derivative(residual, pulls)
    
    # Prune low-score pulls
    pulls = [p for p, s in zip(pulls, scores) if s > prune_threshold]
    
    # Spawn new pulls at top-K T(x) peaks (one per SNIC superpixel)
    new_pulls = spawn_from_topo_derivative(T, k=spawn_count)
    pulls.extend(new_pulls)
    
    # Convergence check
    if residual.mean() < convergence_threshold:
        break
```

This is the staged 3-batch outer loop already in the v2 plan; load-bearing-test and topo-derivative are the two complementary scoring primitives.
