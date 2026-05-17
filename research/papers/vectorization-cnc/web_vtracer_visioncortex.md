---
title: "VTracer — Visioncortex raster-to-vector"
authors: ["Visioncortex / Tsang Sai Kit"]
year: 2020
url: "https://github.com/visioncortex/vtracer"
docs: "https://www.visioncortex.org/vtracer-docs"
source_type: "open_source_tool"
relevance: "MUST-READ — color-aware alternative to Potrace. Stack-mode is interesting for chuck-mcp because the layered plate model naturally maps to VTracer's stacking strategy. O(n) curve fitting vs Potrace's O(n²). However, no topology guarantees and no minimum-feature-size enforcement; needs pre-processing."
tags: [vtracer, raster-vector, color-quantization, hierarchical-clustering, bezier-fitting]
---

# VTracer (visioncortex)

## What VTracer is

A Rust + WASM raster-to-vector tracer that handles **color, high-resolution
scans** end-to-end. The image processing pipeline includes:

1. Hierarchical clustering with adjustable color quantization
   (`color_precision` — bits per RGB channel, default 8).
2. Speckle filtering via `filter_speckle` — discards patches smaller than
   N px (default 4). This is **structurally equivalent to the area-opening
   chuck-mcp's S6.c performs** on each plate.
3. Per-cluster contour extraction.
4. Curve fitting in one of three modes:
   - `pixel` — raw pixel-edge polygon
   - `polygon` — Visvalingam-Whyatt-style polygon simplification
   - `spline` — iterative spline subdivision with corner detection
5. SVG emission, either **stacked** (default — paths drawn in order, no
   holes) or **cutout** (with `clip-path` for explicit holes).

## How it differs from Potrace

| Dimension                | Potrace                   | VTracer                                    |
|--------------------------|---------------------------|--------------------------------------------|
| Input                    | Binary only               | Color (auto-quantized)                     |
| Complexity (fitting)     | O(n²) polygon optimization | O(n) iterative subdivision                 |
| Output topology          | Closed paths, separate holes | Stacked layers (no explicit holes by default) |
| Curve type               | Cubic Bézier              | Cubic Bézier (or polygon)                  |
| Minimum-feature filter   | `turdsize` (area)         | `filter_speckle` (area)                    |
| Corner detection         | `alphamax` (continuous penalty) | `corner_threshold` (angular deg)     |
| Best fit for             | Engineered binary masks   | Photographs, illustrations, manga          |

## All CLI parameters

| Parameter           | Default | Range          | Effect |
|---------------------|---------|----------------|--------|
| `--colormode`       | `color` | `color`/`bw`   | Color tracing vs. Potrace-style binary |
| `--hierarchical`    | `stacked` | `stacked`/`cutout` | Stacking strategy |
| `--mode`            | `spline`| `pixel`/`polygon`/`spline` | Curve fitting mode |
| `--filter_speckle`  | `4`     | `0..16`        | Min cluster area in px (area opening) |
| `--color_precision` | `6`     | `1..8`         | Bits per RGB channel |
| `--gradient_step`   | `16`    | `0..255`       | Color delta between hierarchical layers |
| `--corner_threshold`| `60°`   | `0..180`       | Angular threshold for corner detection |
| `--segment_length`  | `4.0`   | `3.5..10`      | Iterative subdivision stop length |
| `--splice_threshold`| `45°`   | `0..180`       | Angular threshold to splice splines |
| `--path_precision`  | `8`     | `1..12`        | Decimal places in SVG path data |

## Why it matters for chuck-mcp

The **stacking strategy** maps 1:1 onto the layered-printmaking model:
chuck-mcp emits one plate per ink layer; VTracer's stacked output draws
upper plates over lower plates without explicit hole geometry. This avoids
the parent/child contour ordering bug that Potrace exposes to callers.

**But** VTracer was tuned for visual fidelity, not machinability:

- `filter_speckle` is the **only** size-related parameter. No enforcement
  of minimum **width** on retained features. A 1-pixel spur survives, then
  vanishes under tool-radius offset.
- `splice_threshold` can break curve continuity at sharp angles — fine
  for visual SVG but produces redundant G-code stops if not smoothed
  downstream.
- No native tool-radius offset.

## Recommended VTracer flags for chuck-mcp S6.c plates

If S6.c already binarized + cleaned the mask, run VTracer in binary mode
to bypass clustering:

```
vtracer \
  --colormode bw \
  --mode spline \
  --filter_speckle <min_feature_px²> \
  --corner_threshold 30 \      # tighter for engineered geometry
  --segment_length 4 \
  --splice_threshold 30 \      # avoid spurious splices
  --path_precision 4
```

## Limitations vs. CNC

1. No minimum **stroke width** check — same issue as Potrace.
2. `cutout` mode emits SVG `clip-path` which most CAM tools can't read.
3. No DXF/EPS export — must post-process with a CAM-aware converter
   (cnc-woodblock-tools handles this downstream).
4. Spline output uses cubic Bézier; CAM tools that only accept G2/G3
   arcs need an additional approximation step.

## License

VTracer is MIT — safe to embed in chuck-mcp's SVG exporter.

## Citation

visioncortex. "VTracer: raster image tracing tool." Open source, 2020+.
https://github.com/visioncortex/vtracer
