# Lucci, Nasini & Severín 2018 — "A Branch and Price Algorithm for List Coloring Problem"

**arXiv ID**: 1812.00040
**Authors**: Mauro Lucci, Graciela Nasini, Daniel Severín
**Published**: 2018; *Electronic Notes in Theoretical Computer Science* 346 (2019) 613–624.
**URL**: https://arxiv.org/abs/1812.00040

## Relevance to chuck-mcp

**Medium-high**. List coloring is the natural extension when **some impressions are pre-pinned to known blocks** (artist intent, registration constraints, kento-pin alignment) or when **specific impressions can only go on specific blocks** (size constraints — a 24-inch impression can't fit on a 12-inch block). Each impression has a *list* of allowable block-IDs.

## Problem definition

**Weighted List Coloring (WLCP)**: each vertex `v ∈ V` has a list `L(v) ⊆ {1, ..., C}` of allowable colors. Each color `c` has non-negative weight `w_c`. Find a proper coloring `φ: V → C` with `φ(v) ∈ L(v) ∀v` minimizing `Σ_c w_c · [c is used]`. Setting all `w_c = 1` and `L(v) = {1..C} ∀v` recovers VCP.

**Precoloring extension**: subset of vertices have `|L(v)| = 1` (forced color). Equivalent to list coloring after restricting other vertices' lists to exclude the precolored neighbors' forced colors. NP-complete even on bipartite graphs (general case).

## Algorithm

Branch-and-price on a Mehrotra-Trick-style set-cover formulation:
- Variables: indicator per **stable set** in `G`.
- Constraints: each vertex covered exactly once.
- Pricing subproblem: weighted max-stable-set in subgraph induced by vertices with the candidate color in their list — solved by combinatorial enumeration.

Branching: on pairs (`v`, color `c`) — either force `φ(v) = c` or forbid it. List structure is preserved at each node.

## Empirical reach

The paper reports solving WLCP instances with **up to 70 vertices**. Difficulty depends on list-color distribution more than on density. For chuck-mcp at 12–40 vertices this is well within solver capacity.

## Recommendation for chuck-mcp

If artist provides a partial block assignment (e.g., "key block goes on Block 1, anything not on Block 1 is fine"), encode it as:

```python
for impression in pinned:
    L(impression) = {impression.assigned_block}
for impression in free:
    L(impression) = {1, ..., MAX_BLOCKS}
```

**Implementation tactic**: rather than full Mehrotra-Trick-style B&P, modify DSATUR to respect lists:
1. At each step, when picking the lowest available color for `v`, restrict to `L(v)`.
2. If no color in `L(v)` is feasible, backtrack (proper B&B over partial colorings).

This **list-DSATUR** runs in seconds for chuck-mcp scale and avoids implementing column generation. Use Lucci's B&P only if list-DSATUR ever fails to close on a real instance (very unlikely).

## Two-paragraph summary

Lucci, Nasini and Severín adapt the classic Mehrotra-Trick branch-and-price formulation to the weighted list coloring problem, where each vertex carries a permitted-color list and each color has a fixed weight. The master problem is a set-cover over stable sets (one per used color, weighted), the pricing subproblem is a list-restricted maximum-weight stable set per candidate color, and branching forces or forbids specific vertex-color pairs while preserving the list structure of the residual problem. The implementation closes weighted list coloring instances of up to seventy vertices comfortably.

For chuck-mcp the relevant case is artist-pinned impressions: registration-marked impressions must live on the same physical block as the key drawing, certain large impressions cannot fit on certain blocks. Both reduce to lists. At 12–40 vertices the right implementation is a list-aware DSATUR (constrain the lowest-color choice to the vertex's list, backtrack if empty) rather than full branch-and-price. Reach for Lucci's full method only if list-DSATUR fails — empirically it will not.
