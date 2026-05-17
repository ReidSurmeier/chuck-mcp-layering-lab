# Iori, de Lima, Martello, Miyazawa & Monaci 2020 — "Exact Solution Techniques for Two-dimensional Cutting and Packing"

**arXiv ID**: 2004.12619
**Authors**: Manuel Iori, Vinícius L. de Lima, Silvano Martello, Flávio K. Miyazawa, Michele Monaci
**Year**: 2020 (published in *EJOR* 2021)
**URL**: https://arxiv.org/abs/2004.12619

## Relevance to chuck-mcp

**Medium**. Survey reference for the 2D packing question — *if* chuck-mcp grows a per-block layout module to physically place multiple impressions on a single plywood face. This is the canonical exact-methods survey.

## Coverage

Surveys exact methods for four core 2D rectangular packing problems:

1. **Minimum-height strip packing (2SP)** — items into a fixed-width strip, minimize height. Closest analog to "pack impressions onto a plywood face".
2. **Bin packing (2BP)** — items into the fewest fixed-size bins. Closest analog if blocks are pre-cut to standard plywood sizes.
3. **Maximum-value knapsack (2KP)** — select a subset of items into a single bin, maximize value.
4. **Feasibility (OPP)** — can these items fit in this bin? Y/N.

## Methodological tools surveyed

- **Pseudo-polynomial discretization ILPs** (Beasley's xy-coordinate models).
- **Position-and-distance ILPs** (Onodera, Murata, etc.).
- **Column generation** (one variable per feasible packing pattern; pricing = small knapsack).
- **Branch-and-cut** with item-pair non-overlap cuts.
- **Branch-and-price**.
- **Logic-based Benders decomposition**.
- **Decomposition by guillotine stages** (for guillotine-only cuts, relevant for paper/plywood industrial cutters).

## Practical guidance from the survey

- For **small instances** (< 30 items) any formulation works.
- For **medium** (30–100 items) column generation dominates.
- For **large** (> 100 items) heuristics + warm-started B&P are necessary.

Solver choice: Gurobi marginally outperforms CPLEX; per recent comparative paper (Becker et al. 2023, arXiv 2308.04965), differences across formulations matter more than solver choice.

## Recommendation for chuck-mcp

If chuck-mcp v24 adds a per-block 2D layout pass, the **right formulation depends on what "fit" means**:

- **Bounding rectangles only, axis-aligned**: use **MaxRects** or **Skyline-BL** heuristics. Code ~50 lines. No solver dependency. Suitable for impressions sized 5–30 cm in a 60×60 cm block.
- **Bounding rectangles, want optimality**: use **Beasley-style pseudo-polynomial ILP** in Gurobi. Solves to optimality at chuck-mcp scale.
- **Irregular polygon nesting**: use **sparrow** (arxiv 2509.13329). Outperforms classical heuristics.
- **Guillotine cuts** (if the CNC must use straight passes across the plywood): use the 2D guillotine ILP from Becker et al. 2023 (arxiv 2308.04965).

This survey is the right reference for any of these decisions. Save the PDF and check it when the time comes.

## Why this matters for the front/back face decision

A plywood block has two faces (front + back). If a color class has impressions that *together* don't fit on a 60×60 cm face but *do* fit when split front+back, the graph coloring is correct but needs a face-assignment subroutine. This is a **bin packing** problem with 2 bins (front, back) and capacity 60×60 each.

Algorithmically:
1. Color class C is decided by DSATUR.
2. Try **MaxRects pack** of C onto Face 1 (60×60). Place items that fit.
3. Remaining items → Face 2 (60×60).
4. If items still remain unplaced after both faces, the color class is **infeasible** — split it and re-run DSATUR with an additional edge between the displaced items.

This is a tight 2-bin special case of 2D bin packing, solvable in O(n²) by a single MaxRects pass with backtracking. No ILP needed at chuck-mcp scale.
