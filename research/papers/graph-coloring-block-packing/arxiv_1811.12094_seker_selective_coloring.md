# Şeker, Ekim & Taşkın 2018 — "An Exact Cutting Plane Algorithm to Solve the Selective Graph Coloring Problem in Perfect Graphs"

**arXiv ID**: 1811.12094
**Authors**: Oylum Şeker, Tınaz Ekim, Z. Caner Taşkın
**URL**: https://arxiv.org/abs/1811.12094

## Relevance to chuck-mcp

**Medium-high**. The selective coloring problem is a structural generalization that may match a future chuck-mcp scenario: when impressions come in **alternative variants** (e.g., 2 candidate masks per impression, only one to be carved), we want to pick one variant per cluster *and* assign minimum blocks. This is exactly Selective Graph Coloring.

## Problem definition

**Selective Graph Coloring (SGC)** generalizes vertex coloring:

> Given graph G = (V, E) with V partitioned into clusters `V₁ ∪ … ∪ Vₚ`, choose **exactly one vertex per cluster** so that the number of colors needed to properly color the chosen vertex set is minimized.

Equivalently: pick a *transversal* of the clusters that minimizes the chromatic number of the induced subgraph.

Standard VCP is the special case `|Vᵢ| = 1` for all `i`. SGC is NP-hard in general but tractable on perfect graphs via this paper's cutting-plane algorithm.

## Algorithm

The paper develops a branch-and-cut algorithm for SGC on perfect graphs. Key ingredients:
1. Compact IP formulation with cluster-selection variables + color-assignment variables.
2. Cutting planes from the **clique polytope** (since perfect graphs admit polynomial-time max-clique).
3. Branch-and-bound on selection variables first.

The authors also publish a benchmark set of synthetic perfect-graph instances (no prior benchmark existed).

## Relevance scenarios for chuck-mcp

1. **Variant selection**: each impression has 2–3 mask candidates from the upstream solver (different `confidence` labels). Pick one per cluster, minimize block count. Direct SGC instance.
2. **Order-tier branching**: when the impression-order solver returns alternative orderings with different impression sets, pick one ordering and block-pack it. Lift to SGC.
3. **Underprint inclusion choice**: each visible impression has an optional underprint partner. Pick or skip. Cluster = `{visible_only, visible+underprint}`. SGC.

## Recommendation

Don't implement SGC until a concrete cluster-of-variants scenario actually arrives — chuck-mcp v23 does not have this yet. When it does:
- Run DSATUR for each variant-selection independently as a heuristic upper bound.
- If the cluster count is small (< 8), brute-force all `Π |Vᵢ|` selections, color each with DSATUR, return the minimum.
- Only fall back to Şeker's cutting-plane approach if clusters proliferate (> 10) AND brute force becomes too expensive.

The paper's cutting-plane method handles graphs of "varying size and densities" but does not specify exact size limits — for chuck-mcp's 12–40 vertex regime it would close instantly.
