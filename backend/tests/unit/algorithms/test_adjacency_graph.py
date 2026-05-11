"""Unit tests for backend.algorithms.decomposition.adjacency_graph."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest

from algorithms.decomposition.adjacency_graph import (
    absorb_slivers,
    build_adjacency_graph,
    labels_from_masks,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------------------
# adjacency cases
# ---------------------------------------------------------------------------

def test_two_adjacent_rectangles_one_edge() -> None:
    """Two horizontally-touching rectangles share a single edge."""
    h, w = 64, 128
    left = _rect_mask(h, w, 0, h, 0, 64)
    right = _rect_mask(h, w, 0, h, 64, 128)
    graph = build_adjacency_graph(
        [left, right], ["red", "blue"], bleed_tolerance_px=0
    )
    assert set(graph.nodes) == {"red", "blue"}
    assert graph.number_of_edges() == 1
    assert graph.has_edge("red", "blue")
    # Boundary length = 64 pixel pairs along the shared column seam.
    assert graph["red"]["blue"]["weight"] == 64


def test_three_islands_far_apart_no_edges() -> None:
    """Disjoint masks separated by large gaps produce zero edges."""
    h, w = 200, 200
    a = _rect_mask(h, w, 0, 20, 0, 20)
    b = _rect_mask(h, w, 0, 20, 180, 200)
    c = _rect_mask(h, w, 180, 200, 90, 110)
    graph = build_adjacency_graph(
        [a, b, c],
        ["a", "b", "c"],
        bleed_tolerance_px=4,
        sliver_threshold_px=0,
    )
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 0


def test_checkerboard_two_colors_one_edge() -> None:
    """A checkerboard of two colors collapses to a single edge between them."""
    h, w = 32, 32
    yy, xx = np.indices((h, w))
    a = ((yy + xx) % 2 == 0)
    b = ((yy + xx) % 2 == 1)
    graph = build_adjacency_graph(
        [a, b], ["a", "b"], bleed_tolerance_px=0, sliver_threshold_px=0
    )
    assert graph.number_of_edges() == 1
    assert graph.has_edge("a", "b")


def test_four_corners_background_separates() -> None:
    """Four corner masks with background gaps produce zero edges."""
    h, w = 200, 200
    tl = _rect_mask(h, w, 0, 40, 0, 40)
    tr = _rect_mask(h, w, 0, 40, 160, 200)
    bl = _rect_mask(h, w, 160, 200, 0, 40)
    br = _rect_mask(h, w, 160, 200, 160, 200)
    graph = build_adjacency_graph(
        [tl, tr, bl, br],
        ["tl", "tr", "bl", "br"],
        bleed_tolerance_px=4,
        sliver_threshold_px=0,
    )
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 0


def test_bleed_bridges_small_gap() -> None:
    """Two masks 5 px apart get bridged by bleed=8."""
    h, w = 64, 200
    left = _rect_mask(h, w, 0, h, 0, 80)
    right = _rect_mask(h, w, 0, h, 85, 200)  # 5-pixel gap
    graph = build_adjacency_graph(
        [left, right],
        ["red", "blue"],
        bleed_tolerance_px=8,
        sliver_threshold_px=0,
    )
    assert graph.has_edge("red", "blue")
    assert graph["red"]["blue"]["weight"] > 0


def test_bleed_does_not_bridge_large_gap() -> None:
    """Two masks 20 px apart stay disconnected with bleed=8."""
    h, w = 64, 200
    left = _rect_mask(h, w, 0, h, 0, 80)
    right = _rect_mask(h, w, 0, h, 100, 200)  # 20-pixel gap
    graph = build_adjacency_graph(
        [left, right],
        ["red", "blue"],
        bleed_tolerance_px=8,
        sliver_threshold_px=0,
    )
    assert not graph.has_edge("red", "blue")


def test_sliver_absorbed_at_default_threshold() -> None:
    """A 10-pixel mask is dropped at default threshold (bleed×10 = 80)."""
    h, w = 100, 100
    big = _rect_mask(h, w, 0, 50, 0, 100)
    sliver = _rect_mask(h, w, 60, 63, 60, 63)  # 9 pixels
    assert int(np.count_nonzero(sliver)) < 80

    graph = build_adjacency_graph(
        [big, sliver], ["big", "sliver"], bleed_tolerance_px=8
    )
    assert "big" in graph.nodes
    assert "sliver" not in graph.nodes


def test_disjoint_same_color_collapses_to_one_node() -> None:
    """Two physically-separate masks sharing a color id are one node."""
    h, w = 100, 200
    patch_a = _rect_mask(h, w, 0, 40, 0, 40)
    patch_b = _rect_mask(h, w, 60, 100, 160, 200)
    other = _rect_mask(h, w, 0, 100, 90, 110)

    graph = build_adjacency_graph(
        [patch_a, patch_b, other],
        ["red", "red", "blue"],
        bleed_tolerance_px=2,
        sliver_threshold_px=0,
    )
    assert set(graph.nodes) == {"red", "blue"}


def test_absorb_slivers_drops_small_masks() -> None:
    """absorb_slivers filters by total pixel count."""
    h, w = 50, 50
    big = _rect_mask(h, w, 0, 25, 0, 25)
    small = _rect_mask(h, w, 0, 2, 0, 2)
    out = absorb_slivers([big, small], min_area_px=10)
    assert len(out) == 1
    assert np.array_equal(out[0], big)


def test_labels_from_masks_basic_assignment() -> None:
    """labels_from_masks returns 1-indexed labels with 0 = background."""
    h, w = 10, 20
    a = _rect_mask(h, w, 0, h, 0, 10)
    b = _rect_mask(h, w, 0, h, 10, 20)
    labels = labels_from_masks([a, b], bleed_tolerance_px=0)
    assert labels.dtype == np.int16
    assert labels.shape == (h, w)
    assert int(labels[0, 0]) == 1
    assert int(labels[0, 15]) == 2


def test_labels_from_masks_requires_nonempty() -> None:
    """Empty mask list is a usage error."""
    with pytest.raises(ValueError):
        labels_from_masks([])


def test_graph_returns_networkx_graph_type() -> None:
    """Public API returns a real networkx Graph."""
    h, w = 32, 32
    graph = build_adjacency_graph(
        [_rect_mask(h, w, 0, h, 0, 16), _rect_mask(h, w, 0, h, 16, w)],
        ["a", "b"],
        bleed_tolerance_px=0,
    )
    assert isinstance(graph, nx.Graph)


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------

def test_performance_2000_square_eight_colors_under_1s() -> None:
    """2000×2000 image with 8 stripe colors must build in <1 s."""
    h, w = 2000, 2000
    n_colors = 8
    band = w // n_colors
    masks = []
    for i in range(n_colors):
        m = np.zeros((h, w), dtype=bool)
        x0 = i * band
        x1 = w if i == n_colors - 1 else (i + 1) * band
        m[:, x0:x1] = True
        masks.append(m)
    color_ids = [f"c{i}" for i in range(n_colors)]

    start = time.perf_counter()
    graph = build_adjacency_graph(
        masks, color_ids, bleed_tolerance_px=4, sliver_threshold_px=0
    )
    elapsed = time.perf_counter() - start

    assert graph.number_of_nodes() == n_colors
    # Stripes touch their neighbors → exactly n_colors - 1 edges.
    assert graph.number_of_edges() == n_colors - 1
    assert elapsed < 1.0, f"build_adjacency_graph took {elapsed:.3f}s (>1.0s)"
