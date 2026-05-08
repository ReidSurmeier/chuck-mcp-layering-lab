"""SVG post-processing for CNC machining readiness.

Converts polyline SVG paths (M/L commands) to smooth cubic bezier curves (C commands)
suitable for CNC toolpath generation. Uses Schneider's algorithm for curve fitting.

Preserves detail while producing clean, machine-ready geometry.

Usage:
    python svg_postprocess.py input.svg output.svg [--tolerance 2.0] [--smooth 0.5]
    python svg_postprocess.py input.svg output.svg --min-length 3.0

As module:
    from svg_postprocess import postprocess_svg
    clean_svg = postprocess_svg(raw_svg_string, tolerance=2.0)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# SVG parsing
# ---------------------------------------------------------------------------

def parse_svg_paths(svg_string: str) -> tuple[list[list[tuple[float, float]]], dict]:
    """Extract point sequences from SVG path d attributes.

    Returns:
        (paths, svg_meta) where paths is list of point lists and
        svg_meta holds width/height/viewBox/extra attributes.
    """
    root = ET.fromstring(svg_string)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    meta = {
        "width": root.get("width", "100"),
        "height": root.get("height", "100"),
        "viewBox": root.get("viewBox"),
        "ns": ns,
        "extra_attribs": {k: v for k, v in root.attrib.items()
                          if k not in ("width", "height", "viewBox")},
    }

    paths: list[list[tuple[float, float]]] = []
    for elem in root.iter(f"{ns}path"):
        d = elem.get("d", "")
        points = _parse_d_attribute(d)
        if points and len(points) >= 3:
            paths.append(points)

    return paths, meta


_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parse_d_attribute(d: str) -> list[tuple[float, float]]:
    """Parse SVG path d attribute, handling M/L/Z commands (polyline paths)."""
    tokens = _CMD_RE.findall(d)
    points: list[tuple[float, float]] = []
    cmd = ""
    nums: list[float] = []
    cx, cy = 0.0, 0.0

    def _flush():
        nonlocal cx, cy
        if cmd in ("M", "m"):
            i = 0
            while i + 1 < len(nums):
                x, y = nums[i], nums[i + 1]
                if cmd == "m" and points:
                    x += cx
                    y += cy
                cx, cy = x, y
                points.append((x, y))
                i += 2
        elif cmd in ("L", "l"):
            i = 0
            while i + 1 < len(nums):
                x, y = nums[i], nums[i + 1]
                if cmd == "l":
                    x += cx
                    y += cy
                cx, cy = x, y
                points.append((x, y))
                i += 2
        elif cmd in ("H", "h"):
            for v in nums:
                x = v if cmd == "H" else cx + v
                cx = x
                points.append((cx, cy))
        elif cmd in ("V", "v"):
            for v in nums:
                y = v if cmd == "V" else cy + v
                cy = y
                points.append((cx, cy))

    for c, n in tokens:
        if c:
            _flush()
            cmd = c
            nums = []
            if c in ("Z", "z"):
                # Close path — don't duplicate first point if already there
                if points and points[-1] != points[0]:
                    points.append(points[0])
        else:
            nums.append(float(n))

    _flush()
    return points


# ---------------------------------------------------------------------------
# Schneider's bezier curve fitting algorithm
# ---------------------------------------------------------------------------

def fit_bezier_curves(
    points: list[tuple[float, float]],
    tolerance: float = 2.0,
    smooth_factor: float = 0.5,
) -> list[tuple[tuple[float, float], ...]]:
    """Fit cubic bezier curves to point sequence using Schneider's algorithm.

    Args:
        points: Sequence of (x, y) points defining the polyline.
        tolerance: Maximum allowed error (pixels) between original and fitted curve.
        smooth_factor: 0.0 = sharp corners preserved, 1.0 = max smoothing.

    Returns:
        List of bezier segments, each a 4-tuple of (x,y) control points:
        (p0, p1, p2, p3) where p0/p3 are on-curve and p1/p2 are control points.
    """
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 2:
        return []

    # Remove duplicate consecutive points
    mask = np.ones(len(pts), dtype=bool)
    for i in range(1, len(pts)):
        if np.allclose(pts[i], pts[i - 1], atol=1e-6):
            mask[i] = False
    pts = pts[mask]

    if len(pts) < 2:
        return []

    # Detect corners — points with sharp angle changes
    corner_threshold = math.radians(30 + 60 * (1.0 - smooth_factor))
    corners = [0]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-10 or len2 < 1e-10:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        if angle > corner_threshold:
            corners.append(i)
    corners.append(len(pts) - 1)

    # Fit bezier segments between corners
    beziers: list[tuple[tuple[float, float], ...]] = []
    for seg_idx in range(len(corners) - 1):
        start = corners[seg_idx]
        end = corners[seg_idx + 1]
        segment = pts[start:end + 1]
        if len(segment) < 2:
            continue

        # Compute tangents at endpoints
        left_tangent = _compute_tangent(segment, 0, forward=True)
        right_tangent = _compute_tangent(segment, len(segment) - 1, forward=False)

        seg_beziers = _fit_cubic(segment, left_tangent, right_tangent, tolerance)
        beziers.extend(seg_beziers)

    return beziers


def _compute_tangent(pts: np.ndarray, index: int, forward: bool) -> np.ndarray:
    """Compute unit tangent at a point."""
    if forward:
        # Average of first few segments for stability
        end = min(index + 4, len(pts) - 1)
        if end == index:
            return np.array([1.0, 0.0])
        t = pts[end] - pts[index]
    else:
        start = max(index - 3, 0)
        if start == index:
            return np.array([-1.0, 0.0])
        t = pts[start] - pts[index]

    norm = np.linalg.norm(t)
    if norm < 1e-10:
        return np.array([1.0, 0.0]) if forward else np.array([-1.0, 0.0])
    return t / norm


def _chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
    """Assign parameter values to points based on chord length."""
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumulative[-1]
    if total < 1e-10:
        return np.linspace(0, 1, len(pts))
    return cumulative / total


def _fit_cubic(
    pts: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    tolerance: float,
    max_iterations: int = 4,
) -> list[tuple[tuple[float, float], ...]]:
    """Recursively fit cubic bezier(s) to points (Schneider's algorithm core)."""
    if len(pts) == 2:
        dist = np.linalg.norm(pts[1] - pts[0]) / 3.0
        bezier = (
            tuple(pts[0]),
            tuple(pts[0] + left_tangent * dist),
            tuple(pts[1] + right_tangent * dist),
            tuple(pts[1]),
        )
        return [bezier]

    # Initial parameterization
    u = _chord_length_parameterize(pts)

    # Iterative refinement
    for iteration in range(max_iterations):
        bezier_pts = _generate_bezier(pts, u, left_tangent, right_tangent)
        max_error, split_point = _compute_max_error(pts, bezier_pts, u)

        if max_error < tolerance:
            return [tuple(tuple(p) for p in bezier_pts)]

        # Reparameterize for better fit
        u = _reparameterize(pts, u, bezier_pts)

    # If error still too large, split at worst point and recurse
    if split_point <= 0:
        split_point = len(pts) // 2
    if split_point >= len(pts) - 1:
        split_point = len(pts) // 2

    center_tangent = _compute_tangent(pts, split_point, forward=True)
    left_beziers = _fit_cubic(
        pts[:split_point + 1], left_tangent, -center_tangent, tolerance
    )
    right_beziers = _fit_cubic(
        pts[split_point:], center_tangent, right_tangent, tolerance
    )
    return left_beziers + right_beziers


def _generate_bezier(
    pts: np.ndarray,
    u: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> np.ndarray:
    """Generate bezier control points using least-squares fitting."""
    n = len(pts)
    p0 = pts[0]
    p3 = pts[-1]

    # Compute A matrix (tangent-scaled Bernstein basis)
    A = np.zeros((n, 2, 2))
    for i in range(n):
        t = u[i]
        b1 = 3 * t * (1 - t) ** 2
        b2 = 3 * t ** 2 * (1 - t)
        A[i, 0] = left_tangent * b1
        A[i, 1] = right_tangent * b2

    # Solve for alpha (tangent lengths)
    C = np.zeros((2, 2))
    X = np.zeros(2)
    for i in range(n):
        C[0, 0] += np.dot(A[i, 0], A[i, 0])
        C[0, 1] += np.dot(A[i, 0], A[i, 1])
        C[1, 0] = C[0, 1]
        C[1, 1] += np.dot(A[i, 1], A[i, 1])

        t = u[i]
        tmp = (
            pts[i]
            - p0 * ((1 - t) ** 3 + 3 * t * (1 - t) ** 2)
            - p3 * (3 * t ** 2 * (1 - t) + t ** 3)
        )
        X[0] += np.dot(A[i, 0], tmp)
        X[1] += np.dot(A[i, 1], tmp)

    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if abs(det) < 1e-12:
        dist = np.linalg.norm(p3 - p0) / 3.0
        alpha1 = alpha2 = dist
    else:
        alpha1 = (C[1, 1] * X[0] - C[0, 1] * X[1]) / det
        alpha2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det

    # Sanity check: alphas should be positive for reasonable curves
    seg_length = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_length
    if alpha1 < epsilon or alpha2 < epsilon:
        alpha1 = alpha2 = seg_length / 3.0

    bezier = np.array([
        p0,
        p0 + left_tangent * alpha1,
        p3 + right_tangent * alpha2,
        p3,
    ])
    return bezier


def _compute_max_error(
    pts: np.ndarray, bezier: np.ndarray, u: np.ndarray
) -> tuple[float, int]:
    """Compute maximum squared distance between points and fitted bezier."""
    max_dist = 0.0
    split_point = len(pts) // 2

    for i in range(len(pts)):
        p = _bezier_point(bezier, u[i])
        dist = np.linalg.norm(pts[i] - p)
        if dist > max_dist:
            max_dist = dist
            split_point = i

    return max_dist, split_point


def _bezier_point(bezier: np.ndarray, t: float) -> np.ndarray:
    """Evaluate cubic bezier at parameter t using De Casteljau."""
    s = 1.0 - t
    return (
        s ** 3 * bezier[0]
        + 3 * s ** 2 * t * bezier[1]
        + 3 * s * t ** 2 * bezier[2]
        + t ** 3 * bezier[3]
    )


def _reparameterize(
    pts: np.ndarray, u: np.ndarray, bezier: np.ndarray
) -> np.ndarray:
    """Newton-Raphson reparameterization for better parameter values."""
    new_u = np.copy(u)
    for i in range(len(pts)):
        new_u[i] = _newton_raphson_root(bezier, pts[i], u[i])
    return new_u


def _newton_raphson_root(
    bezier: np.ndarray, point: np.ndarray, u: float
) -> float:
    """Find closer parameter via Newton-Raphson iteration."""
    # Q(u)
    q = _bezier_point(bezier, u)
    # Q'(u) - first derivative
    d1 = 3.0 * np.array([
        (1 - u) ** 2 * (bezier[1] - bezier[0])
        + 2 * (1 - u) * u * (bezier[2] - bezier[1])
        + u ** 2 * (bezier[3] - bezier[2])
    ]).flatten()
    # Q''(u) - second derivative
    d2 = 6.0 * np.array([
        (1 - u) * (bezier[2] - 2 * bezier[1] + bezier[0])
        + u * (bezier[3] - 2 * bezier[2] + bezier[1])
    ]).flatten()

    # f(u) = (Q(u) - P) . Q'(u)
    diff = q - point
    numerator = np.dot(diff, d1)
    denominator = np.dot(d1, d1) + np.dot(diff, d2)

    if abs(denominator) < 1e-12:
        return u

    new_u = u - numerator / denominator
    return np.clip(new_u, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Geometry cleaning
# ---------------------------------------------------------------------------

def clean_geometry(
    paths: list[list[tuple[float, float]]],
    min_length: float = 3.0,
    merge_distance: float = 1.0,
) -> list[list[tuple[float, float]]]:
    """Remove short segments, fix winding order, merge close points.

    Args:
        paths: List of polyline point sequences.
        min_length: Minimum segment length in pixels.
        merge_distance: Points closer than this are merged.

    Returns:
        Cleaned paths with proper winding order.
    """
    cleaned = []
    for path in paths:
        path = _merge_close_points(path, merge_distance)
        path = _remove_short_segments(path, min_length)
        if len(path) < 3:
            continue
        cleaned.append(path)

    # Fix winding order: largest area path (outer) should be clockwise,
    # inner paths (holes) counter-clockwise.
    if not cleaned:
        return cleaned

    areas = [(_signed_area(p), i) for i, p in enumerate(cleaned)]
    areas.sort(key=lambda x: abs(x[0]), reverse=True)

    result = []
    for signed_area, idx in areas:
        path = cleaned[idx]
        is_outer = (idx == areas[0][1])
        if is_outer:
            # Outer path: clockwise (negative signed area in SVG coords)
            if signed_area > 0:
                path = list(reversed(path))
        else:
            # Hole: counter-clockwise (positive signed area in SVG coords)
            if signed_area < 0:
                path = list(reversed(path))
        result.append(path)

    return result


def _merge_close_points(
    points: list[tuple[float, float]], distance: float
) -> list[tuple[float, float]]:
    """Merge points that are closer than distance threshold."""
    if len(points) < 2:
        return points

    merged = [points[0]]
    for i in range(1, len(points)):
        dx = points[i][0] - merged[-1][0]
        dy = points[i][1] - merged[-1][1]
        if math.sqrt(dx * dx + dy * dy) >= distance:
            merged.append(points[i])

    return merged


def _remove_short_segments(
    points: list[tuple[float, float]], min_length: float
) -> list[tuple[float, float]]:
    """Remove segments shorter than min_length, keeping path shape."""
    if len(points) < 3:
        return points

    kept = [points[0]]
    for i in range(1, len(points)):
        dx = points[i][0] - kept[-1][0]
        dy = points[i][1] - kept[-1][1]
        if math.sqrt(dx * dx + dy * dy) >= min_length:
            kept.append(points[i])

    # Always keep last point for closed paths
    if len(kept) >= 2 and points[-1] != kept[-1]:
        kept.append(points[-1])

    return kept


def _signed_area(points: list[tuple[float, float]]) -> float:
    """Compute signed area using the shoelace formula.

    Positive = counter-clockwise, negative = clockwise (in screen coords
    where Y increases downward).
    """
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return area / 2.0


def check_self_intersection(points: list[tuple[float, float]]) -> bool:
    """Check if a polygon self-intersects using brute-force segment test.

    Returns True if self-intersecting.
    """
    n = len(points)
    if n < 4:
        return False

    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2:
                continue  # adjacent segments share endpoint
            if _segments_intersect(
                points[i], points[i + 1], points[j], points[j + 1]
            ):
                return True
    return False


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Test if line segment p1-p2 intersects p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases
    if abs(d1) < 1e-10 and _on_segment(p3, p4, p1):
        return True
    if abs(d2) < 1e-10 and _on_segment(p3, p4, p2):
        return True
    if abs(d3) < 1e-10 and _on_segment(p1, p2, p3):
        return True
    if abs(d4) < 1e-10 and _on_segment(p1, p2, p4):
        return True

    return False


def _cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


# ---------------------------------------------------------------------------
# Bezier path area computation (for area preservation tests)
# ---------------------------------------------------------------------------

def bezier_path_area(
    beziers: list[tuple[tuple[float, float], ...]],
    samples_per_segment: int = 20,
) -> float:
    """Approximate area of a closed bezier path via polygon sampling."""
    all_points = []
    for bez in beziers:
        p0, p1, p2, p3 = [np.array(p) for p in bez]
        for k in range(samples_per_segment):
            t = k / samples_per_segment
            s = 1.0 - t
            pt = s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3
            all_points.append((float(pt[0]), float(pt[1])))

    if len(all_points) < 3:
        return 0.0
    return abs(_signed_area(all_points))


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------

def paths_to_svg(
    bezier_paths: list[list[tuple[tuple[float, float], ...]]],
    width: str,
    height: str,
    viewBox: Optional[str] = None,
) -> str:
    """Convert bezier paths back to SVG string.

    Args:
        bezier_paths: List of paths, each path is a list of bezier segments.
        width, height: SVG dimensions.
        viewBox: Optional viewBox attribute.

    Returns:
        Complete SVG string.
    """
    vb = viewBox or f"0 0 {width} {height}"
    svg = '<?xml version="1.0" encoding="UTF-8"?>\n'
    svg += (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="{vb}">\n'
    )

    for beziers in bezier_paths:
        if not beziers:
            continue
        d = f"M {beziers[0][0][0]:.2f},{beziers[0][0][1]:.2f}"
        for seg in beziers:
            _, p1, p2, p3 = seg
            d += (
                f" C {p1[0]:.2f},{p1[1]:.2f}"
                f" {p2[0]:.2f},{p2[1]:.2f}"
                f" {p3[0]:.2f},{p3[1]:.2f}"
            )
        d += " Z"
        svg += f'  <path d="{d}" fill="black" stroke="none"/>\n'

    svg += "</svg>\n"
    return svg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def postprocess_svg(
    input_svg: str,
    tolerance: float = 2.0,
    min_segment_length: float = 3.0,
    smooth_factor: float = 0.5,
) -> str:
    """Main entry: polyline SVG in, bezier SVG out.

    Args:
        input_svg: SVG string with M/L polyline paths.
        tolerance: Max error for bezier fitting (pixels).
        min_segment_length: Min path segment length (pixels).
        smooth_factor: 0.0 = corners preserved, 1.0 = max smoothing.

    Returns:
        SVG string with cubic bezier (C) commands.
    """
    paths, meta = parse_svg_paths(input_svg)

    # Clean geometry
    paths = clean_geometry(paths, min_length=min_segment_length)

    # Fit bezier curves to each path
    bezier_paths = []
    for path in paths:
        beziers = fit_bezier_curves(path, tolerance=tolerance, smooth_factor=smooth_factor)
        if beziers:
            bezier_paths.append(beziers)

    return paths_to_svg(
        bezier_paths,
        width=meta["width"],
        height=meta["height"],
        viewBox=meta.get("viewBox"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert polyline SVG to smooth bezier SVG for CNC machining."
    )
    parser.add_argument("input", help="Input SVG file path")
    parser.add_argument("output", help="Output SVG file path")
    parser.add_argument(
        "--tolerance", type=float, default=2.0,
        help="Max bezier fitting error in pixels (default: 2.0)",
    )
    parser.add_argument(
        "--min-length", type=float, default=3.0,
        help="Minimum segment length in pixels (default: 3.0)",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.5,
        help="Smoothing factor 0.0-1.0 (default: 0.5)",
    )

    args = parser.parse_args()

    with open(args.input) as f:
        input_svg = f.read()

    output_svg = postprocess_svg(
        input_svg,
        tolerance=args.tolerance,
        min_segment_length=args.min_length,
        smooth_factor=args.smooth,
    )

    with open(args.output, "w") as f:
        f.write(output_svg)

    # Report stats
    in_count = input_svg.count(" L ")
    out_count = output_svg.count(" C ")
    print(f"Converted {in_count} line segments -> {out_count} bezier curves")


if __name__ == "__main__":
    main()
