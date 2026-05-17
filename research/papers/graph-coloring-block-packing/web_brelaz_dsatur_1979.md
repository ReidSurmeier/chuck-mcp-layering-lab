# Brélaz 1979 — "New methods to color the vertices of a graph"

**Citation**
Brélaz, Daniel (1979-04-01). "New methods to color the vertices of a graph." *Communications of the ACM* 22(4): 251–256. DOI: 10.1145/359094.359101.

**Sources**
- Canonical CACM landing page: https://cacm.acm.org/research/new-methods-to-color-the-vertices-of-a-graph/ (403 to anon WebFetch)
- Algorithm reproduced verbatim by Furini, Gabrel & Ternier (2015) "An improved DSATUR-based Branch and Bound for the Vertex Coloring Problem" — Section 2 — and by Wikipedia/HandWiki DSatur pages.

## What the paper introduces

Two artifacts: (1) a heuristic vertex-coloring rule called **DSATUR**, and (2) an exact branch-and-bound (B&B) procedure built on it that, in 1979, beat the then-state-of-the-art Randall-Brown algorithm on benchmark graphs.

### Saturation degree

For an uncolored vertex `u` under a partial coloring `C̃`, the **saturation degree**
`DSAT(u, C̃) := | { color(v) : v ∈ N(u), v is colored } |`
is the count of *distinct* colors already appearing in `u`'s neighborhood. Saturation degree drives the heuristic: the most-constrained vertex (highest DSAT) is colored next. Ties are broken by the largest degree in the uncolored subgraph.

### Heuristic DSATUR (`DSATURh`)

```
1. Order vertices by descending degree; color the maximum-degree vertex with color 1.
2. While uncolored vertices remain:
     a. Select uncolored vertex u with max DSAT(u, C̃).
        Ties broken by max degree in the uncolored subgraph.
     b. Assign u the smallest color index ≥ 1 not used by any colored neighbor of u.
        (Open a new color only if all existing colors clash.)
3. Return C̃ and k̃ = number of colors used.
```

Complexity: O(n²) basic, O((n+m) log n) with binary heap, O(m + n log n) with Fibonacci heap. Optimal on bipartite, cycle, and wheel graphs (Brélaz proved it for bipartite, the rest are immediate). A 2024 result (Yekezare, Zohrehbandian, Maghasedi, Bonomo-Braberman, *ORL* vol. 57) extends optimality to **chordal graphs** — therefore also to **interval graphs**, since intervals ⊂ chordal.

### Exact DSATUR-B&B

Use `DSATURh` to compute an initial upper bound `UB = k̃` and a max-clique lower bound `LB` (the first `LB` vertices `DSATURh` colors with different colors form a clique, since each was selected by highest saturation). Then branch:

```
DSATUR(partial coloring C̃):
  if all vertices colored:
    if k̃ < UB: UB ← k̃; C* ← C̃
    return
  if max(k̃, LB) >= UB: prune
  select uncolored v by saturation rule
  for each color c ∈ {1..k̃} feasible for v, plus a new color k̃+1:
    color v with c; recurse; uncolor v
```

Branching factor at each node is at most `k̃+1`. The classical weakness is that **LB is computed once and never refreshed** — Furini, Gabrel & Ternier (2015) fix this by introducing a Reduced Graph that allows fresh lower-bound computation at every node (clique number, fractional chromatic, Hoffman bound, or a Stable-Set-bijection bound). This pushes solvable random VCP instance sizes upward significantly.

## Why this matters for chuck-mcp

Chuck-MCP's spatial-conflict graph for impressions has properties that match DSATUR's known sweet spot:
- **Small** (12–40 impressions = vertices).
- **Spatially derived**, so edges arise from rectangle/mask overlap on a 2D plane — the conflict graph is very close to an **intersection graph** of 2D regions, which when the regions are intervals on a single sweep direction collapses to an **interval graph** (DSATUR is provably optimal on chordal/interval graphs per the 2024 ORL result).
- **Few colors** (target 20–30 blocks per Yasu Shibata's Emma at 27).

For this size and structure, the Brélaz 1979 exact B&B (with Furini 2015 Reduced-Graph LB updates) finishes in milliseconds to seconds — there is essentially no reason to reach for ILP, column generation, tabu search, or RL-driven heuristics. The single heuristic `DSATURh` will usually already produce the chromatic number on a near-interval graph, and the B&B closure proves it.

## Two-paragraph algorithm summary

DSATUR colors vertices one at a time. At each step, it picks the uncolored vertex with the highest **saturation degree** — the number of distinct colors already in its neighborhood — breaking ties by degree in the uncolored subgraph. The chosen vertex is given the smallest color index not yet used by any of its colored neighbors; a new color is opened only when every existing color clashes. The intuition is myopic constraint propagation: the vertex whose neighborhood has already "burned" the most palette slots is the one most likely to force a new color, so paint it now while you still have flexibility on the rest. On random graphs DSATUR beats classical greedy by a wide margin but is in turn beaten by Recursive Largest First (RLF, Leighton 1979) which constructs entire color classes (maximal independent sets) one at a time. On chordal graphs — and therefore on interval graphs — DSATUR is provably optimal (Yekezare et al., ORL 57, 2024).

Brélaz also presents an exact branch-and-bound built around DSATUR. The heuristic computes an initial upper bound `UB` and, via the cliquish prefix of its coloring sequence, a lower bound `LB`. The B&B then expands partial colorings by trying every feasible existing color plus one new color for the next saturation-selected vertex, pruning whenever `max(k̃, LB) ≥ UB`. With ~80–100 vertex inputs this exact procedure was competitive with the Randall-Brown algorithm in 1979; with Furini, Gabrel & Ternier's 2015 Reduced-Graph lower-bound updates at every node it scales further into the high-density DIMACS regime. For chuck-mcp's 12–40 impression instances, the exact B&B closes in well under a second on commodity hardware.
