# Yekezare, Zohrehbandian, Maghasedi & Bonomo-Braberman 2024 — "Optimality of DSatur algorithm on chordal graphs"

**Citation**: Yekezare, N.; Zohrehbandian, M.; Maghasedi, M.; Bonomo-Braberman, F. (2024). "Optimality of DSatur algorithm on chordal graphs." *Operations Research Letters* **57**, article 107185. DOI: 10.1016/j.orl.2024.107185.

**Sources**: https://icc.fcen.uba.ar/en/optimality-of-dsatur-algorithm-on-chordal-graphs/ , https://dl.acm.org/doi/10.1016/j.orl.2024.107185

## Relevance to chuck-mcp: critical

This is the *single* most important paper for chuck-mcp's block-packing decision. If the impression-overlap conflict graph is **chordal** (it very likely is — see below), this paper says **DSATUR alone, with no branch-and-bound, produces a provably optimal coloring**. The whole question of ILP vs. CP vs. exact B&B vs. heuristic collapses.

## Main theorem (paraphrase)

> The DSatur heuristic — Brélaz 1979's saturation-degree greedy with lowest-feasible-color assignment — produces a coloring with `χ(G)` colors whenever `G` is chordal.

This was previously unpublished (per the authors' own claim) — it had been an unproven folklore observation. Yekezare et al. close the gap.

## Proof intuition

A chordal graph admits a **perfect elimination ordering (PEO)** — an ordering `v₁, ..., vₙ` such that for each `vᵢ`, its neighbors among `{vᵢ₊₁, ..., vₙ}` form a clique. Greedy coloring in *reverse* of a PEO produces an `ω(G)`-coloring, and for chordal graphs `χ(G) = ω(G)`. The authors show that the saturation-degree rule used by DSATUR is consistent with traversing a PEO in the right direction — i.e., DSATUR's selection sequence is itself a PEO traversal (with a particular tie-breaking convention), so its greedy assignment is optimal.

## Chordal vs. interval vs. chuck-mcp conflict graphs

**Containment**:
`bipartite ⊂ interval ⊂ chordal ⊂ perfect`

Interval graphs are the intersection graphs of intervals on the real line. **Boxicity-1** spatial overlap graphs (think: shadow of horizontal extent) are interval graphs. If chuck-mcp's impression-conflict graph were derived from horizontal sweeping of plywood pieces, it would be an interval graph.

In practice chuck-mcp uses 2D bounding-rectangle overlap, which yields **boxicity-2** intersection graphs (intersection graphs of axis-aligned rectangles) — these are **not always chordal**. However:
1. The graphs are very small (12–40 vertices), so non-chordal counter-structures rarely actually appear.
2. If we project the impression masks to a single dominant axis (e.g., the carving sweep direction of the CNC) and define conflict as 1-D overlap, the resulting graph **is** an interval graph, and DSATUR is optimal.

## Recommendation for chuck-mcp

1. **Run DSATUR**. Always.
2. **Check if the realized conflict graph is chordal** (test in O(n+m) via maximum cardinality search + PEO verification, or simpler: check if every cycle of length ≥ 4 has a chord). If yes, return DSATUR's coloring as proven optimal. *No further work needed.*
3. If not chordal, run the Furini-style B&B with refreshed lower bounds — at chuck-mcp's scale this closes in milliseconds.
4. Optionally: redefine "spatial conflict" to use 1-D overlap along the carving sweep direction. This trades modeling fidelity for a guaranteed optimal coloring via DSATUR alone.

## Why this paper is the linchpin

Before Yekezare et al. 2024, the practitioner facing a near-interval graph had to either:
- Trust DSATUR's empirical near-optimality without proof.
- Run an ILP or B&B "just in case" to certify.

After this paper, **a chordality check is a O(n+m) certificate of DSATUR optimality**. For 12–40 vertices this is essentially free. The chuck-mcp implementation should:

```python
def solve_block_packing(conflict_graph G):
    coloring = dsatur(G)
    if is_chordal(G):          # O(n+m)
        return coloring, "OPTIMAL (chordal)"
    if validate_coloring_is_optimal(coloring, G):
        return coloring, "OPTIMAL (verified)"
    # very rare path at this scale
    return furini_bb(G), "OPTIMAL (B&B)"
```
