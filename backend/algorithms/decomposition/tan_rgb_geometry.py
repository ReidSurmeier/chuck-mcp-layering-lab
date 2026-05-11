"""Tan et al. SIGGRAPH Asia 2017 — RGBXY-Space Geometry palette decomposition.

Pure-function palette extraction + per-pixel barycentric weights, restricted
to the RGB convex-hull simplification of Section 3 of:

    Jianchao Tan, Jose Echevarria, Yotam Gingold.
    "Decomposing Images into Layers via RGB-space Geometry."
    SIGGRAPH Asia 2017.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.spatial import ConvexHull, Delaunay

__all__ = ["extract_palette_hull", "barycentric_decompose", "decompose_image"]

_FloatArr = NDArray[np.float64]
_F32Arr = NDArray[np.float32]
_U8Arr = NDArray[np.uint8]

_SUBSAMPLE_FOR_HULL: int = 8000
_MAX_HULL_VERTICES: int = 50  # cap before greedy reduction
_EPS: float = 1e-9


def _to_linear_rgb(img: _U8Arr) -> _FloatArr:
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {img.shape}")
    return img.astype(np.float64) / 255.0


def _subsample(pixels: _FloatArr, n: int, seed: int) -> _FloatArr:
    if pixels.shape[0] <= n:
        return pixels
    rng = np.random.default_rng(seed)
    return pixels[rng.choice(pixels.shape[0], size=n, replace=False)]


def _cap_hull(pixels: _FloatArr, verts: _FloatArr, cap: int) -> _FloatArr:
    """Greedy farthest-point selection to cap an initial hull vertex set."""
    if verts.shape[0] <= cap:
        return verts
    centroid = pixels.mean(axis=0)
    chosen = [int(np.argmax(np.sum((verts - centroid) ** 2, axis=1)))]
    min_d = np.sum((verts - verts[chosen[0]]) ** 2, axis=1)
    while len(chosen) < cap:
        idx = int(np.argmax(min_d))
        chosen.append(idx)
        np.minimum(min_d, np.sum((verts - verts[idx]) ** 2, axis=1), out=min_d)
    return verts[chosen]


def _greedy_remove_to_k(
    pixels: _FloatArr, verts: _FloatArr, target_k: int
) -> _FloatArr:
    """Drop the vertex whose removal least increases nearest-vertex MSE.

    Vectorized via the 1st/2nd-nearest trick: removing vertex v changes the
    cost only for pixels whose current nearest IS v (they switch to 2nd).
    Tan et al. Sec. 3.2 — same idea, simplified surrogate.
    """
    current = verts.copy()
    while current.shape[0] > target_k:
        p2 = np.sum(pixels * pixels, axis=1, keepdims=True)
        v2 = np.sum(current * current, axis=1)[None, :]
        d2 = p2 + v2 - 2.0 * pixels @ current.T
        np.clip(d2, 0.0, None, out=d2)
        order = np.argsort(d2, axis=1)
        nearest, second = order[:, 0], order[:, 1]
        d_n = np.take_along_axis(d2, nearest[:, None], 1).ravel()
        d_s = np.take_along_axis(d2, second[:, None], 1).ravel()
        sums = np.full(current.shape[0], np.sum(d_n), dtype=np.float64)
        np.add.at(sums, nearest, d_s - d_n)
        current = np.delete(current, int(np.argmin(sums)), axis=0)
    return current


def extract_palette_hull(
    image_rgb_uint8: _U8Arr,
    target_palette_size: int = 13,
    rng_seed: int = 42,
) -> _FloatArr:
    """Build RGB convex hull, simplify to K vertices.

    Returns ``(K, 3)`` float64 in ``[0, 1]``. Tan et al. SIGGRAPH Asia 2017.
    """
    if target_palette_size < 1:
        raise ValueError("target_palette_size must be >= 1")
    pixels = _to_linear_rgb(image_rgb_uint8).reshape(-1, 3)
    sample = _subsample(pixels, _SUBSAMPLE_FOR_HULL, rng_seed)

    if np.ptp(sample, axis=0).max() < _EPS:  # constant image
        return np.tile(sample[0], (target_palette_size, 1))
    if sample.shape[0] < 4:
        return sample[:target_palette_size].astype(np.float64)

    try:
        hull = ConvexHull(sample)
        verts = sample[hull.vertices]
    except Exception:
        uniq = np.unique(sample, axis=0)
        verts = uniq[:target_palette_size]

    if verts.shape[0] <= target_palette_size:
        return verts.astype(np.float64)
    if verts.shape[0] > _MAX_HULL_VERTICES:
        verts = _cap_hull(sample, verts, _MAX_HULL_VERTICES)
    return _greedy_remove_to_k(sample, verts, target_palette_size)


def _project_to_triangle(
    pts: _FloatArr, a: _FloatArr, b: _FloatArr, c: _FloatArr
) -> tuple[_FloatArr, _FloatArr]:
    """Vectorized closest-point-on-triangle (Eberly). Returns (d2, bary)."""
    ab, ac = b - a, c - a
    ap = pts - a
    d1, d2v = ap @ ab, ap @ ac
    bp = pts - b
    d3, d4 = bp @ ab, bp @ ac
    cp = pts - c
    d5, d6 = cp @ ab, cp @ ac
    vc, vb, va = d1 * d4 - d3 * d2v, d5 * d2v - d1 * d6, d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < _EPS, 1.0, denom)
    v, w = vb / safe, vc / safe
    u = 1.0 - v - w
    # Voronoi-region clamps.
    ra = (d1 <= 0) & (d2v <= 0)
    rb = (d3 >= 0) & (d4 <= d3)
    rc = (d6 >= 0) & (d5 <= d6)
    rab = (~ra) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    rac = (~ra) & (vb <= 0) & (d2v >= 0) & (d6 <= 0)
    rbc = (~rb) & (~rc) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    u = np.where(ra, 1.0, np.where(rb | rc, 0.0, u))
    v = np.where(rb, 1.0, np.where(ra | rc, 0.0, v))
    w = np.where(rc, 1.0, np.where(ra | rb, 0.0, w))
    t = np.clip(d1 / np.where((d1 - d3) == 0, 1.0, d1 - d3), 0.0, 1.0)
    u = np.where(rab, 1.0 - t, u); v = np.where(rab, t, v); w = np.where(rab, 0.0, w)
    t = np.clip(d2v / np.where((d2v - d6) == 0, 1.0, d2v - d6), 0.0, 1.0)
    u = np.where(rac, 1.0 - t, u); v = np.where(rac, 0.0, v); w = np.where(rac, t, w)
    dn = (d4 - d3) + (d5 - d6)
    t = np.clip((d4 - d3) / np.where(dn == 0, 1.0, dn), 0.0, 1.0)
    u = np.where(rbc, 0.0, u); v = np.where(rbc, 1.0 - t, v); w = np.where(rbc, t, w)
    closest = u[:, None] * a + v[:, None] * b + w[:, None] * c
    diff = pts - closest
    return np.sum(diff * diff, axis=1), np.stack([u, v, w], axis=1)


def _nnls_fallback(pixels: _FloatArr, palette: _FloatArr) -> _FloatArr:
    """Per-pixel NNLS — last-resort for degenerate (non-3D) palettes."""
    k = palette.shape[0]
    big = 1e6
    a_mat = np.vstack([palette.T, np.full((1, k), big)])
    out = np.empty((pixels.shape[0], k), dtype=np.float64)
    for i in range(pixels.shape[0]):
        w_i, _ = nnls(a_mat, np.concatenate([pixels[i], [big]]))
        s = w_i.sum()
        out[i] = w_i / s if s > _EPS else np.full(k, 1.0 / k)
    return out


def _project_outside(pixels: _FloatArr, palette: _FloatArr) -> _FloatArr:
    """Out-of-hull pixels → nearest hull face, then barycentric on it."""
    n, k = pixels.shape[0], palette.shape[0]
    weights = np.zeros((n, k), dtype=np.float64)
    if n == 0:
        return weights
    try:
        simplices = ConvexHull(palette).simplices
    except Exception:
        return _nnls_fallback(pixels, palette)
    best_d2 = np.full(n, np.inf)
    best_bary = np.zeros((n, 3), dtype=np.float64)
    best_face = np.zeros(n, dtype=np.int64)
    for i, face in enumerate(simplices):
        d2_f, bary = _project_to_triangle(
            pixels, palette[face[0]], palette[face[1]], palette[face[2]]
        )
        better = d2_f < best_d2
        if np.any(better):
            best_d2 = np.where(better, d2_f, best_d2)
            best_bary = np.where(better[:, None], bary, best_bary)
            best_face = np.where(better, i, best_face)
    rows = np.repeat(np.arange(n), 3)
    cols = simplices[best_face].ravel()
    np.add.at(weights, (rows, cols), best_bary.ravel())
    return weights


def barycentric_decompose(
    image_rgb_uint8: _U8Arr, palette: _FloatArr
) -> _F32Arr:
    """Per-pixel barycentric weights against ``palette``.

    Inside-hull pixels: Delaunay tetrahedralization (4 non-zero per pixel).
    Outside-hull pixels: project to nearest hull face (3 non-zero per pixel).
    Returns ``(H, W, K)`` float32, non-negative, sums to ~1.
    """
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must be (K, 3); got {palette.shape}")
    pixels = _to_linear_rgb(image_rgb_uint8)
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 3)
    k = palette.shape[0]
    weights = np.zeros((flat.shape[0], k), dtype=np.float64)

    if k == 1:
        weights[:, 0] = 1.0
        return weights.reshape(h, w, k).astype(np.float32)
    if k == 2:
        v0, v1 = palette[0], palette[1]
        d = v1 - v0
        denom = float(np.dot(d, d))
        if denom < _EPS:
            weights[:, 0] = 1.0
        else:
            t = np.clip((flat - v0) @ d / denom, 0.0, 1.0)
            weights[:, 0], weights[:, 1] = 1.0 - t, t
        return weights.reshape(h, w, k).astype(np.float32)

    inside = np.zeros(flat.shape[0], dtype=bool)
    if k >= 4:
        try:
            tri = Delaunay(palette)
            simp = tri.find_simplex(flat)
            inside = simp >= 0
            if np.any(inside):
                pts = flat[inside]
                sim = simp[inside]
                tr = tri.transform[sim]  # (N, 4, 3)
                b_first = np.einsum("nij,nj->ni", tr[:, :3], pts - tr[:, 3])
                bary = np.concatenate(
                    [b_first, 1.0 - b_first.sum(axis=1, keepdims=True)], axis=1
                )
                sv = tri.simplices[sim]
                rows = np.repeat(np.arange(pts.shape[0]), 4)
                w_in = np.zeros((pts.shape[0], k), dtype=np.float64)
                np.add.at(w_in, (rows, sv.ravel()), bary.ravel())
                weights[inside] = w_in
        except Exception:
            inside = np.zeros(flat.shape[0], dtype=bool)

    outside = ~inside
    if np.any(outside):
        weights[outside] = _project_outside(flat[outside], palette)

    np.clip(weights, 0.0, None, out=weights)
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums < _EPS] = 1.0
    weights /= sums
    return weights.reshape(h, w, k).astype(np.float32)


def decompose_image(
    image_rgb_uint8: _U8Arr,
    palette: _FloatArr | None = None,
    target_palette_size: int = 13,
) -> tuple[_FloatArr, _F32Arr]:
    """Top-level: extract palette (if None) + decompose. Tan et al. 2017."""
    if palette is None:
        palette = extract_palette_hull(
            image_rgb_uint8, target_palette_size=target_palette_size
        )
    return palette, barycentric_decompose(image_rgb_uint8, palette)
