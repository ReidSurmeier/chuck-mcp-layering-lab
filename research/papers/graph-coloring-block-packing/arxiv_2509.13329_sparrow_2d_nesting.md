# Gardeyn, Vanden Berghe & Wauters 2025 — "An open-source heuristic to reboot 2D nesting research" (sparrow)

**arXiv ID**: 2509.13329
**Authors**: Jeroen Gardeyn, Greet Vanden Berghe, Tony Wauters
**Year**: 2025 (v3 Feb 2026)
**URL**: https://arxiv.org/abs/2509.13329

## Relevance to chuck-mcp

**Medium**. Relevant *only* if chuck-mcp expands its post-solve stage from "block-count minimization via graph coloring" to "layout multiple impressions on a single block face physically without overlap". That is a **2D irregular packing** problem (the impressions are polygon-shaped masks). Sparrow is the modern open-source baseline for that.

## Problem solved

2D irregular **strip packing**: arrange a set of irregular polygonal items inside a strip (fixed width, infinite height, minimize height used). The state of the art in 2D nesting had stagnated for a decade due to closed-source heuristics and lack of reproducibility.

## Sparrow's approach

Decomposes the optimization into a **sequence of feasibility problems**: at each step, items are placed and overlaps (collisions) are detected; the heuristic resolves them by repositioning items via gradient-style moves. Iterates until feasible, then tries a smaller strip and repeats.

Open-source code in Rust. Consistently outperforms prior state-of-the-art (Best-Match Sequence, BLF, etc.) — sometimes by wide margins.

## What this would buy chuck-mcp

Currently chuck-mcp's block packing decides **which impressions share a block**, not **where on the block each impression sits**. If a block is 60×60 cm of plywood and two impressions are 30×30 cm each, they fit easily side-by-side. If they're each 40×40, they don't — but the conflict graph wouldn't know.

A 2D packing post-pass would:
1. Take each color class (set of impressions assigned to one block).
2. Try to nest them physically on a 60×60 face (front + 60×60 back face = 2 faces per block).
3. If nesting fails, split the class — open a new block.

This refines the graph-coloring upper bound into a physically realizable plan.

## Recommendation for chuck-mcp

**Defer this**. v23 outputs block IDs without per-block layout. If artist feedback says "block 5 has too many impressions to fit on one face", add a sparrow-style pass:

1. After graph coloring, for each color class:
   - Attempt 2D rectangle/polygon packing on a `BLOCK_WIDTH × 2·BLOCK_HEIGHT` strip (= front + back).
   - If items don't fit, split the class — create a new color, move the displaced items.
2. Repeat coloring + packing until stable.

For an MVP, even **bounding-rectangle packing** (Skyline algorithm, MaxRects, BLF) suffices — chuck-mcp's masks are shaped but the artist can also work with bounding rectangles for first-pass layout decisions. Sparrow becomes worth it only when the artist asks for tight irregular nesting to minimize plywood usage.

## Code pointer

sparrow source: https://github.com/JeroenGar/sparrow (Rust, MIT/Apache-2.0 dual licensed)
