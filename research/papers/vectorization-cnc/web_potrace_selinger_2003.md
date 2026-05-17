---
title: "Potrace: a polygon-based tracing algorithm"
authors: ["Peter Selinger"]
year: 2003
url: "http://potrace.sourceforge.net/potrace.pdf"
mirror: "https://www.semanticscholar.org/paper/Potrace-:-a-polygon-based-tracing-algorithm-Selinger/e60b78c1fed3fa907411bb189767e868e4a421f8"
source_type: "technical_report"
relevance: "MUST-READ — Potrace is the de facto baseline for binary-mask vectorization. chuck-mcp S6.c emits binary plate masks per layer; Potrace is the natural exporter. Understanding turdsize / alphamax / opttolerance unlocks the ΔE-guarded printability repair → SVG handoff."
tags: [potrace, polygon-tracing, bezier-fitting, binary-vectorization, corner-detection]
---

# Potrace — Polygon-Based Tracing (Selinger, 2003)

## Why this matters for chuck-mcp S6.c → CNC

chuck-mcp's S6.c stage produces per-plate binary masks after Delta-E-guarded
printability repair. The current SVG exporter is gated by printability
pressures (component, island, partial-cell, overlap, low-alpha). Potrace is
the canonical algorithm for turning each printable binary plate into a clean
SVG with closed cubic Bézier paths suitable for ShopBot consumption.

The Potrace pipeline aligns extremely well with the printability constraints
S6.c already enforces, because it directly exposes:

1. **`turdsize`** — drops connected components below N pixels (S6.c's island
   pressure tunable made formal at the vectorizer boundary).
2. **`alphamax`** — controls corner penalty; lower = more corners preserved,
   higher = smoother curves (matters for end-mill diameter compatibility).
3. **`opttolerance`** — Bézier curve-fitting tolerance; affects path length
   and toolpath segment count downstream.
4. **`turnpolicy`** — disambiguates ambiguous corner crossings (relevant when
   adjacent printability components share a 1-pixel-thin boundary).

## Algorithm (four stages)

### Stage 1 — Path decomposition

Operates on the binary bitmap. For every closed boundary at the pixel
lattice, the tracer walks the contour and emits a sequence of pixel-edge
moves (up/down/left/right). The path is **closed** by construction.

The `turdsize` parameter discards any closed path enclosing fewer than
`turdsize` pixels. **This is precisely the area-opening operation chuck-mcp's
S6.c performs as the "island pressure" filter** — except Potrace runs it on
the contour rather than the mask. Running `area_opening(min_feature_px)`
upstream of Potrace (i.e. inside S6.c) means Potrace doesn't have to delete
spurs the printability repair has already certified as legal.

`turnpolicy` resolves the four possible decisions when the contour walker
meets an ambiguous "saddle" (4-connectivity diagonal): `black`, `white`,
`left`, `right`, `minority`, `majority`, `random`. The default `minority`
breaks ties toward the less common color in the local neighborhood, which
tends to preserve fine features.

### Stage 2 — Polygon optimization

Replaces the staircase pixel path with the optimal straight-line polygon.
Optimality is defined lexicographically:

1. Minimize segment count.
2. Among polygons with equal segment count, minimize a per-segment penalty
   that measures the squared distance between the polygon edge and the
   original pixel path.

This is essentially Douglas-Peucker-style simplification driven by an
exact penalty rather than a heuristic tolerance, and operates on closed
polygons (so topology is preserved by construction).

### Stage 3 — Vertex adjustment

Each polygon vertex is allowed to move within its pixel-cell square. The
algorithm picks the in-cell position that minimizes a smoothness measure
of the adjacent edges. This step never changes topology because vertices
are bounded to their own pixel cells.

### Stage 4 — Smoothing & curve fitting

For each interior vertex of the adjusted polygon, compute the "alpha"
value — a number in [0, 4/3] measuring how sharp the corner is:

- `α = 0`  → maximally smooth
- `α = 4/3` → straight pass-through
- `α ≥ alphamax` → declared a corner; emit a straight-line segment around
  the vertex
- `α < alphamax` → emit a cubic Bézier curve through the vertex

Default `alphamax = 1.0`. Lower values (e.g. 0.5) preserve more corners
which is useful for engineered geometry like the layered plates emitted by
chuck-mcp.

After corner classification, an optimization pass fits cubic Béziers
across runs of non-corner vertices, with `opttolerance` controlling the
maximum allowed error between the Bézier and the underlying polygon.
Default `opttolerance = 0.2` pixel.

## Topology guarantees

- **Closed paths**: every output path is closed by construction (Stage 1).
- **Hole containment**: outer and inner contours of a region are emitted as
  separate paths; standard SVG `fill-rule="evenodd"` resolves containment.
  Potrace does **not** reorder paths to enforce parent/child trees; that is
  caller responsibility. For chuck-mcp this matters when a plate has
  speckle holes — they survive Stage 1 unless killed by `turdsize`.
- **No self-intersection** in the polygon output. The Bézier output can
  *visually* self-intersect for very high `opttolerance`, but the control
  polygon stays within the simplified polygon's region.

## Recommended Potrace flags for chuck-mcp S6.c output

| Flag                | Recommended | Rationale |
|---------------------|-------------|-----------|
| `--turdsize N`      | `min_feature_px²` (e.g. `9` for a 3×3 minimum feature) | Matches S6.c's island pressure |
| `--alphamax 0.6`    | Preserves more corners than default | Layered plates have rectilinear features |
| `--opttolerance 0.1`| Tighter than default | Yields more Bezier segments but closer fit, lower toolpath drift after offset compensation |
| `--turnpolicy minority` | Default | Best speckle preservation |
| `--blacklevel 0.5`  | n/a (mask is binary) | |
| `--unit 10`         | SVG unit scaling | Match downstream cnc-woodblock-tools EPS export |

## Limitations vs. CNC requirements

1. **No native minimum-feature-size enforcement on positive features**.
   `turdsize` removes small components, but does not enforce a minimum
   width on retained features. A 1-pixel-wide spur survives. The CNC-side
   tool diameter compensation will erode such spurs to zero — better to
   run a morphological opening before Potrace.
2. **No tool-radius offsetting**. Potrace emits the visual boundary; CAM
   must offset by tool radius (Clipper2, JTS, ShapelyOps).
3. **No V-carve depth field**. Potrace is 2D-only. For V-carve relief
   carving, chuck-mcp would need to emit a depth field per pixel and a
   distinct module to derive V-bit paths (medial-axis / SDF inversion).
4. **Binary only**. Color/grayscale must be quantized upstream — chuck-mcp
   S6.c already does this per plate so this is moot.

## Citation

Selinger, Peter. "Potrace: a polygon-based tracing algorithm." Technical
report, September 2003. http://potrace.sourceforge.net/potrace.pdf
