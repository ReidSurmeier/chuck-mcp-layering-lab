# Jabrayilov & Mutzel 2017 — "New Integer Linear Programming Models for the Vertex Coloring Problem"

**arXiv ID**: 1706.10191
**Authors**: Adalat Jabrayilov, Petra Mutzel
**Published**: arXiv June 2017; later journal version
**URL**: https://arxiv.org/abs/1706.10191

## Relevance to chuck-mcp

**Medium-low**. ILP formulations are overkill at chuck-mcp's 12–40 vertex scale where DSATUR is provably optimal on chordal graphs. But this paper is worth knowing if (a) the conflict graph becomes non-chordal *and* (b) we want a certificate of optimality stronger than a B&B closure — e.g., if we ever extend chuck-mcp to multi-criterion coloring (block count *and* per-block area, or *and* color budget).

## Contributions

The paper introduces two new ILP formulations for VCP based on **partial orderings** and compares them to the two classical formulations:

1. **Assignment formulation (ASS-VCP)** — classical. Binary `x_{v,c}` = "vertex v gets color c"; binary `y_c` = "color c is used". Minimize `Σ y_c` subject to each vertex getting exactly one color, no clash on edges. Symmetry-breaking helps but the formulation has many symmetric optima.

2. **Representatives formulation (REP-VCP)** — Campelo, Campos, Corrêa (2008). One representative per color class.

3. **Partial-Ordering-Based Model (POP)** — Jabrayilov-Mutzel new contribution. Encode coloring via a partial order on vertices, with binary `y_{u,c} = 1` iff `u`'s color `≤ c` and `z_{u,c} = 1` iff `u`'s color `≥ c`. Linking constraints give exactly one color per vertex.

4. **POP with auxiliary variables** — variant.

## Empirical results

- **POP excels on sparse graphs.**
- **REP-VCP excels on dense graphs.**
- ASS-VCP (classical) is consistently worst.
- Compared on DIMACS benchmarks via Gurobi.

The paper does not report explicit small-instance (under-100-vertex) tables but the trend is monotonic — at small sizes all formulations are fast, and the choice matters only as instances grow.

## Recommendation for chuck-mcp

If chuck-mcp ever needs a generic ILP fallback (very unlikely at 12–40 vertices), use **POP for sparse** and **REP for dense** conflict graphs. The break-even is at edge density ~0.3.

**Don't use ASS** — it's a teaching formulation, not a production one.

For the chuck-mcp deliverable specifically, **prefer not to use ILP at all**. The graph-coloring stage is a 200-line python function with DSATUR + chordality check + Furini-style B&B fallback. ILP buys nothing here and adds a Gurobi/CPLEX dependency.

## Key formulation excerpt (POP)

For vertex `v` and color index `c ∈ {1..H}` where `H` is an upper bound on the chromatic number (e.g., `Δ(G)+1`):

```
minimize    Σ_v z_{v,1}                # = # colors used
subject to  y_{v,c} + z_{v,c+1} = 1    ∀v, c
            y_{u,c} + y_{v,c} ≥ z_{u,c} + z_{v,c} - 1     ∀(u,v) ∈ E, ∀c
            y_{v,c} ≤ y_{v,c+1}        ∀v, c
            (z monotone too)
```

This is simpler than column-generation Mehrotra-Trick and runs in plain Gurobi/CPLEX with no special pricing logic.
