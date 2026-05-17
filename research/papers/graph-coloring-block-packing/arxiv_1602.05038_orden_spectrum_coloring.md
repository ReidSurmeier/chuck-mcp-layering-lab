# Orden, Gimenez-Guzman, Marsa-Maestre & de la Hoz 2016 — "Spectrum graph coloring and applications to WiFi channel assignment"

**arXiv ID**: 1602.05038
**Authors**: David Orden, Jose Manuel Gimenez-Guzman, Ivan Marsa-Maestre, Enrique de la Hoz
**URL**: https://arxiv.org/abs/1602.05038

## Relevance to chuck-mcp

**Medium**. Frame this as the **soft-conflict** generalization. In chuck-mcp the conflict graph is currently binary (overlap → edge, no overlap → no edge). But spatial overlap is graded: 5% overlap is much less problematic than 50% overlap. Spectrum coloring lets each "color" (block) carry a graded interference rather than a hard yes/no.

## Problems introduced

1. **Threshold Spectrum Coloring (TSC)**: Given a fixed number of colors and a pairwise interference matrix over colors, minimize the maximum total interference at any vertex (max over `v` of `Σ_{(u,v) ∈ E} interference(color(u), color(v))`).

2. **Chromatic Spectrum Coloring (CSC)**: Given an allowed maximum interference threshold, minimize the number of colors needed to keep all vertices under threshold.

When the interference matrix is `1` if colors are equal and `0` otherwise, both problems collapse to standard VCP.

## Algorithm

Both problems are NP-hard. The authors propose **DSATUR-based heuristics** with the saturation rule generalized to "saturation = sum of interferences across used colors in neighborhood", and the color-pick rule generalized to "pick the color minimizing the new interference contribution".

## Relevance scenarios for chuck-mcp

1. **Graded spatial overlap**: instead of "overlap → edge", define `interference(impression_i, impression_j) = area_of_overlap / total_area`. Run TSC with `K = current best block count`. Two impressions with 5% overlap on the same block is acceptable, two with 50% is not.

2. **Block face budget**: each block has a "carving complexity budget" (total visible-ink area). Assign impressions to blocks not just to avoid hard conflicts but to balance carving load. TSC variant.

3. **Pull-group congestion**: impressions in the same pull group share an ink-up step. Too many in one pull → ink dries before printing finishes. Soft penalty per impression-per-pull.

## Recommendation for chuck-mcp

**Stage 1 (v23)**: hard binary conflicts only. Standard DSATUR.

**Stage 2 (future)**: when artist requests "balance carving complexity per block" or "soft overlap is OK if scored low", lift to TSC. Implementation: replace DSATUR's `lowest-feasible-color` rule with `color-minimizing-new-interference`. Identical structure, drop-in.

## Key formulation detail

TSC objective: `min_φ max_{v ∈ V} Σ_{u ∈ N(v)} w(φ(u), φ(v))`

For chuck-mcp this becomes: `min over block assignment of max over impressions of (sum of overlap-area with same-block neighbors)`. Equivalent to **min-makespan scheduling on m identical machines** when interferences are uniform, **min-max-weighted-degree coloring** otherwise.
