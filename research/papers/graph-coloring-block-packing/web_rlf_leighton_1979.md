# Leighton 1979 / Adegbindin et al. — Recursive Largest First (RLF)

**Source**: F. T. Leighton, "A graph coloring algorithm for large scheduling problems", *Journal of Research of the National Bureau of Standards*, 84(6), 1979, pp. 489–506. https://nvlpubs.nist.gov/nistpubs/jres/84/jresv84n6p489_a1b.pdf

**Modern reference**: Adegbindin, Hertz & Bellaïche, "A new efficient RLF-like algorithm for the vertex coloring problem", https://www.gerad.ca/~alainh/RLFPaper.pdf

**Survey-level analysis**: G. Palubeckis, "On the recursive largest first algorithm for graph colouring", *International Journal of Computer Mathematics* 85(2), 2008.

## Algorithm sketch

Unlike DSATUR which colors one vertex at a time, RLF builds **one color class at a time** by constructing a maximal independent set (stable set) greedily, then removes those vertices and recurses on the remainder.

```
RLF(G):
  k ← 0
  while G is non-empty:
    k ← k+1
    V_k ← {} (the next color class)
    pick v in G with max degree, add v to V_k
    U ← N(v)   (forbidden for V_k — already adjacent to v)
    W ← V(G) - {v} - U   (still candidate for V_k)
    while W is non-empty:
      pick u in W with the most neighbors in U
      add u to V_k
      move N(u) ∩ W into U
      remove u from W
    G ← G - V_k
  return color classes V_1 ... V_k
```

Original Leighton 1979 complexity: O(n³). Adegbindin et al. tighten to O(n²) for sparse graphs where `k·e ≤ n²`.

## Performance vs DSATUR

Lewis 2021 empirical study (in *A Guide to Graph Colouring*, 2nd ed, Springer): on random graphs with edge probability `p = 0.5`, **RLF produces significantly better colorings than DSATUR**. On chordal/interval graphs both produce optimal colorings.

The intuition: RLF's "pack a maximal IS first" rule mirrors how a human would solve packing-by-color — fill a class as densely as possible before opening the next. For chuck-mcp's block packing (where each color = one block, and we want as many impressions packed onto one block as possible) **RLF's objective is structurally aligned with the deliverable**: it directly maximizes per-block utilization at each step.

## Relevance to chuck-mcp

RLF is the heuristic that actually matches the printmaker's mental model: "Pack everything I can onto one block (one color class), then start a new block." DSATUR will produce the same minimum *count* of blocks but RLF tends to produce more *evenly utilized* blocks. For a printmaker whose carving cost scales per block, the optimal block count is what matters — DSATUR is enough. For a printmaker who also cares about per-block carving cost (each block has surface area to carve), RLF's denser packing helps.

## Recommendation

- **Primary**: DSATUR (Brélaz). Provably optimal on near-interval graphs (Yekezare 2024).
- **Secondary**: run RLF as a second candidate; if RLF returns the same block count, prefer RLF's assignment (denser per-block packing).
- **Tertiary**: if both differ from a known clique-number lower bound, run Furini B&B to close the gap.

## Two-paragraph summary

RLF builds color classes greedily, one at a time. To start a class it picks the maximum-degree uncolored vertex `v`, then iteratively adds vertices that are non-adjacent to everything already in the class, breaking ties by "most neighbors among the currently forbidden set" — a forward-looking rule that tries to leave the remaining graph as easy to color as possible. Once no more vertices can be added, that color class is closed and the class's vertices are removed from the graph; the procedure recurses on what remains. Each iteration produces a maximal independent set.

Empirically RLF beats DSATUR on random graphs and on dense DIMACS benchmarks. For chuck-mcp's setting — impression-to-block assignment where each block is one color class and we want few blocks — RLF's "fill one block to capacity, then next" strategy directly mirrors the printmaker's intuition. On the chordal/interval-like conflict graphs chuck-mcp produces, RLF will tie DSATUR on block count, but typically gives a more evenly-loaded packing. Implementation cost is O(n²) and the algorithm is < 50 lines of code.
