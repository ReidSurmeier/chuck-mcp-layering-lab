---
title: "Polygon simplification — Douglas-Peucker, Visvalingam-Whyatt, topology-preserving variants"
sources:
  - "Visvalingam, M. and Whyatt, J.D. (1993). Line generalisation by repeated elimination of points. Cartographic Journal 30(1)"
  - "Douglas, D. and Peucker, T. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica 10(2)"
  - "https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm"
  - "https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm"
  - "https://tschaub.net/blog/2014/03/04/topology-preserving-simplification.html"
relevance: "MUST-READ — polygon simplification is the next stage after Potrace/VTracer for getting the path count down without breaking topology. Vanilla RDP can self-intersect; topology-preserving variants (NetTopologySuite TopologyPreservingSimplifier, JTS) are the safe choice. Visvalingam-Whyatt with area-tolerance maps cleanly to chuck-mcp's area-based island pressure."
tags: [polygon-simplification, douglas-peucker, visvalingam-whyatt, topology, jts]
---

# Polygon Simplification — Algorithm Reference

## Why chuck-mcp needs this

After Potrace/VTracer emit SVG paths from a per-plate mask, the path
geometry is dense (Bezier control points every few pixels). For CNC:

- Too-dense paths → too many G-code blocks → controller stalls
- Tool-radius offset of a dense path can produce micro self-loops
- DXF/EPS export bloats

Simplifying paths before tool-radius offset is essential.

## Algorithms

### Douglas-Peucker (1973) / Ramer-Douglas-Peucker

Pick the two endpoints, find the vertex with max perpendicular distance
from the chord. If max distance > ε, retain that vertex and recurse on
two halves. Else discard all intermediate vertices.

- Tolerance: linear distance ε
- Complexity: O(n log n) average, O(n²) worst-case
- Topology: **NOT preserved**. Two simplified lines that didn't
  originally cross can cross after simplification; closed polygons can
  self-intersect.

### Visvalingam-Whyatt (1993)

Compute the "effective area" of every vertex = the triangle formed by
the vertex and its two neighbors. Iteratively remove the vertex with the
smallest effective area, recompute affected neighbor areas.

- Tolerance: triangle area threshold (units²)
- Complexity: O(n log n) with a priority queue
- Topology: **NOT preserved by default**. Same caveat — a self-loop
  can form when adjacent vertices are removed.
- Visual character: spikes get **removed** (small area triangles),
  gentle bends get **preserved**. Opposite of RDP.

For chuck-mcp's CNC output, **Visvalingam-Whyatt with a topology-safe
variant** is the right default: spikes are unprintable anyway (smaller
than end-mill), so we want them gone, and gentle bends are the
ink-bearing edges we want preserved.

### Topology-preserving Douglas-Peucker (JTS / NetTopologySuite)

The JTS Topology Suite implements
`TopologyPreservingSimplifier`: runs RDP per-segment but
**rejects any candidate simplification that would change topology**
(introduce a self-intersection or cross another input feature).

- Tolerance: linear distance ε (same as RDP)
- Complexity: ~O(n² log n)
- Topology: **provably preserved**
- Output: identical to RDP on isolated curves; slightly less aggressive
  on curves near other features

### Whirlpool generalization (Wang & Müller, 1998)

Coarser tolerance bands, multiple iterations, designed for cartographic
generalization. Less relevant for our use case but available in QGIS
`PolygonSimplifier` plugin.

## Which to use for chuck-mcp

**Visvalingam-Whyatt with topology check** is the right default:

- Area tolerance ≈ (min_feature_px)² matches the island-pressure budget
- Spikes < end-mill diameter get removed (good)
- Topology check prevents self-loops in the offset toolpath downstream
- Cheap to implement (priority queue + JTS-style topology guard)

Concrete recommendation (see NOTES.md):

```
import shapely
from shapely.geometry import Polygon
from shapely import simplify

# Visvalingam-Whyatt, topology-preserving
simple = simplify(plate_polygon,
                  tolerance=min_feature_mm,
                  preserve_topology=True)
```

Shapely's `preserve_topology=True` routes to JTS's
`TopologyPreservingSimplifier`. It's not strictly Visvalingam-Whyatt
(Shapely uses RDP under the hood), but the topology guard is the
important part. For true VW area-based simplification, use the
`visvalingamwyatt` Python package or `mapshaper`'s `-simplify` with
`-method visvalingam`.

## Topology-safe simplification — checklist

- After simplification, verify each polygon is still **simple**
  (no self-intersection): `polygon.is_simple`
- Verify holes are still inside outer rings: `polygon.is_valid`
- Verify total area change is ≤ ΔE budget allows
  (chuck-mcp's S6.c already measures this)

## Citation

- Visvalingam, M. and Whyatt, J.D. "Line generalisation by repeated
  elimination of points." The Cartographic Journal 30(1), 1993.
- Douglas, D.H. and Peucker, T.K. "Algorithms for the reduction of the
  number of points required to represent a digitized line or its
  caricature." Cartographica 10(2), 1973.
- JTS Topology Suite, TopologyPreservingSimplifier.
