# Zhu & Zhou 2025 — "A Near-Real-Time Reduction-Based Algorithm for Coloring Massive Graphs" (RECOL)

**arXiv ID**: 2509.23606
**Authors**: Chenghao Zhu, Yi Zhou
**Year**: 2025
**URL**: https://arxiv.org/abs/2509.23606

## Relevance to chuck-mcp

**Very low**. RECOL is built for SNAP / Network Repository / DIMACS10 / DIMACS2 instances — millions of vertices. Chuck-MCP has 12–40. Wrong by 5 orders of magnitude.

Listed here as a sanity check: the modern field is increasingly focused on massive graphs because that's where the field's open problems are. For small dense problems like ours, **the field considers DSATUR and exact B&B essentially solved** — no SOTA paper in the last 5 years targets the 10–100 vertex regime.

## Algorithm sketch

RECOL alternates:
1. Fast estimation of lower bound (clique-based) and upper bound (greedy).
2. **Graph reductions**: domination reduction (remove vertices whose neighborhoods are subsumed), complement crown reduction, independent-set reduction.
3. **Heuristic coloring** on the reduced graph.

Delivers results in under one minute on graphs with millions of vertices. Outperforms FastColor, HyColor, etc., on this regime.

## What chuck-mcp can borrow

The reduction techniques themselves are useful even at small scale, *if* they happen to apply:

- **Domination reduction**: if impression A's spatial neighborhood is a subset of impression B's, and B is on block X, A can also go on block X without expanding the color set. Cheap O(n²) preprocessing.
- **Independent set reduction**: a *clique* of impressions that mutually overlap must each be on a different block — pre-assign them to distinct colors and remove from the search.

Both shrink the conflict graph before DSATUR runs. For 12–40 vertices the shrinkage usually buys nothing (DSATUR is already fast), but it can simplify visualization and debugging — "these 6 impressions form a clique, they each get a dedicated block".

## Recommendation

**Don't implement RECOL**. Don't even read it. The lesson is meta: when designing chuck-mcp's coloring stage, do *not* over-engineer it. The field's heavy machinery is built for problems we don't have. A 200-line DSATUR + 50-line Furini-B&B fallback covers the entire chuck-mcp use case forever, given the impression-count is bounded by carving feasibility.
