# Zhu et al. 2025 — "HyColor: An Efficient Heuristic Algorithm for Graph Coloring"

**arXiv ID**: 2506.07373
**Authors**: Enqiang Zhu, Yu Zhang, Haopeng Sun, Ziqi Wei, Witold Pedrycz, Chanjuan Liu, Jin Xu
**Year**: 2025
**URL**: https://arxiv.org/abs/2506.07373

## Relevance to chuck-mcp

**Low**. HyColor is built for graphs with up to 10⁷ vertices. Chuck-MCP has 12–40 vertices. Wrong scale. Cited here only because it represents the modern state of the art for general graph coloring and confirms that DSATUR-class heuristics remain the foundation that newer methods build on.

## Abstract (paraphrased)

HyColor combines:
1. **Local decision strategy** for improved lower bounds on χ(G).
2. **Graph reduction** to shrink the working graph (domination, complement crown, independent-set reductions).
3. **k-core + mixed-degree greedy** as the core heuristic — a DSATUR-flavor with degeneracy ordering.

Evaluated on 209 instances against 3 SOTA algorithms:
- Best solutions on 194 instances (>93%)
- 34 solutions significantly surpassing competitors
- Proven optimal on 128 instances

## Architectural lesson for chuck-mcp

Even at massive scale, the pipeline is the same shape:
1. Compute a fast lower bound (clique, fractional).
2. Reduce the graph (drop dominated vertices, etc.).
3. Greedy color with a degree-style rule (DSATUR or k-core).
4. (If exact) Branch-and-bound.

For chuck-mcp's 12–40-vertex regime steps 1 and 2 are unnecessary — the graph is already small. The pipeline collapses to step 3 = DSATUR.

## When would chuck-mcp actually need HyColor

Never, unless chuck-mcp's scope changes radically — e.g., the system grows to scheduling ten thousand parallel prints across hundreds of distinct impression candidates. That is not chuck-mcp v23's mandate.

## Citation kept for: "what does the modern field look like?"

This is the right paper to cite if a reviewer asks "is your DSATUR-only approach state-of-the-art for graph coloring?" Answer: at chuck-mcp's scale, yes — newer methods like HyColor outperform DSATUR only on graphs orders of magnitude larger than ours, where reductions and tight lower bounds are needed to make the problem tractable. At 12–40 vertices DSATUR is provably optimal on chordal graphs (Yekezare et al. 2024) and exact on near-chordal graphs with Furini's B&B in well under a second.
