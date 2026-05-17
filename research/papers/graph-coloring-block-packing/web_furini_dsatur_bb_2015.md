# Furini, Gabrel & Ternier 2015 — "An improved DSATUR-based Branch and Bound for the Vertex Coloring Problem"

**Authors**: Fabio Furini, Virginie Gabrel, Ian-Christopher Ternier (PSL, Université Paris Dauphine, CNRS, LAMSADE UMR 7243)
**Source**: https://optimization-online.org/wp-content/uploads/2015/10/5159.pdf
**Year**: 2015 (preprint; journal version published in *European Journal of Operational Research*)

## Relevance to chuck-mcp

**Highest possible** — this is the modern reference implementation of DSATUR-based exact graph coloring. Chuck-MCP's post-solve block packing stage is exactly the problem this paper solves, at sizes well below the paper's empirical limit. If we implement only one exact algorithm, it should be Furini-style DSATUR-B&B with Reduced-Graph lower bounds.

## Abstract (verbatim)

> Given an undirected graph, the Vertex Coloring Problem (VCP) consists of assigning a color to each vertex of the graph in such a way that two adjacent vertices do not share the same color and the total number of colors is minimized. DSATUR-based Branch and Bound (DSATUR) is an effective exact algorithm for the VCP. One of its main drawback is that a lower bound is computed only once and it is never updated. We introduce a reduced graph which allows the computation of lower bounds at nodes of the branching tree. We compare the effectiveness of different classical VCP bounds, plus a new lower bound based on the 1-to-1 mapping between VCPs and Stable Set Problems. Our new DSATUR outperforms the state of the art for random VCP instances with high density, significantly increasing the size of instances solved to proven optimality. Similar results can be achieved for a subset of high density DIMACS instances.

## Key contributions

### 1. Recap of the classic DSATUR B&B (Brélaz 1979 → Sewell → San Segundo)

The exact algorithm runs `DSATURh` to compute initial `LB` (heuristic max-clique from the cliquish prefix of the greedy coloring) and `UB` (number of colors used by the heuristic). It then enumerates partial colorings by branching on `k̃+1` color choices at each non-colored vertex, pruning when `max(k̃, LB) ≥ UB`.

The classical defect: **`LB` is computed exactly once at the root and never refreshed**. A trivial second bound is the count `k̃` of colors in the current partial coloring, but this is weak.

### 2. The Reduced Graph

Given a partial coloring `C̃` with stable sets `Ṽ₁ … Ṽ_k̃`, the Reduced Graph `G_R(C̃)` contracts each color class into a super-node connected to every vertex it conflicts with, plus the uncolored vertices. Any lower bound on the chromatic number of `G_R(C̃)` is a valid lower bound on `χ(G | C̃)`, where the conditional chromatic is the minimum total colors needed to complete `C̃`. This permits **re-running lower-bound oracles at every node** of the B&B tree.

### 3. Lower bounds compared

- **Clique number ω(G_R)** — useful for node-count reduction but slow per call.
- **Stability-number bound** — `LB ≥ ⌈n / α(G)⌉`.
- **Fractional chromatic number** — tighter, harder to compute exactly.
- **Hoffman bound** — eigenvalue-based.
- **New: Stable-Set-bijection bound** — exploits the 1-to-1 correspondence between feasible colorings and stable-set covers of the complement graph; computed via a partial column-generation pass. This is the bound that reduces both *node count* and *wall-time*, whereas the classical bounds only reduce nodes.

### 4. Empirical results

- For **random high-density graphs**, the Furini B&B significantly increases the size of instances solved to proven optimality vs. San Segundo's prior best DSATUR B&B.
- For DIMACS instances of high density (subset), similar wins.
- For low-density DIMACS instances the winner remains Branch-and-Price on the Mehrotra-Trick ILP — but those instances are also vastly larger than chuck-mcp scale.

## Algorithmic skeleton (paraphrased from the PDF)

```
DSATUR-Furini(G):
  (LB, UB, C̃₀) ← DSATURh(G)            # initial coloring + clique LB
  C* ← C̃₀
  recurse(∅)
  return C*

recurse(C̃):
  if all vertices colored:
    if k̃ < UB: UB ← k̃; C* ← C̃
    return
  # ↓↓ KEY DIFFERENCE: refresh LB at every node
  LB_local ← max(k̃, lowerBound(ReducedGraph(C̃)))
  if LB_local ≥ UB: prune
  v ← argmax_{u uncolored} DSAT(u, C̃)
  for c in {1..k̃} feasible for v ∪ {k̃+1}:
    color v with c
    recurse(C̃)
    uncolor v
```

## How this applies to chuck-mcp

Chuck-MCP impression conflict graphs are 12–40 vertices. At that size, the cost of computing even the strongest Furini bound at every B&B node is negligible (< 1 ms total). The expected outcome on real instances:

1. `DSATURh` alone almost always produces the optimal block count, because the conflict graph is close to an interval graph (overlap of 2D rectangles projected on the carving plane). Yekezare et al. 2024 (ORL 57) proves DSATUR is optimal on chordal graphs, which strictly contain interval graphs.
2. The B&B closes immediately with `LB = UB` at the root for almost every instance.
3. Even pathological cases where the greedy is one color off the optimum close in < 100 ms.

**Recommendation**: implement plain DSATUR heuristic first. If any instance comes back where `UB > LB`, run the Furini B&B with Reduced-Graph lower bounds. Don't even consider ILP or column generation at this scale.
