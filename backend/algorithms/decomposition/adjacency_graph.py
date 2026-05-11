"""Region adjacency graph for woodblock pigment decomposition.

Builds a networkx graph where nodes are pigment color IDs and edges
represent "these two colors touch in the print output" — accounting for
ink bleed via a tolerance radius. Used downstream by DSATUR for plate
assignment.

Design notes:
    * One node per color. Disjoint same-color patches collapse to a single
      node (they are the same ink on the same block).
    * Adjacency is computed on a single H×W int16 labels array using a
      vectorized north/east neighbor scan — O(HW), no Python loops over
      pixel pairs.
    * Bleed tolerance is implemented with a single
      ``scipy.ndimage.distance_transform_edt`` call: every background
      pixel within ``bleed_tolerance_px`` of any mask is assigned to its
      nearest mask's label. Two masks separated by ≤ 2×bleed therefore
      meet in the middle and register an edge.
    * Edge weights are the count of shared boundary pixel pairs (after
      bleed expansion).
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = ["labels_from_masks", "absorb_slivers", "build_adjacency_graph"]


def absorb_slivers(
    masks: list[np.ndarray], min_area_px: int
) -> list[np.ndarray]:
    """Drop masks with total area below ``min_area_px``.

    Mutates nothing — returns a new list. Order is preserved.
    """
    if min_area_px <= 0:
        return list(masks)
    return [m for m in masks if int(np.count_nonzero(m)) >= min_area_px]


def labels_from_masks(
    masks: Sequence[np.ndarray], bleed_tolerance_px: int = 8
) -> np.ndarray:
    """Convert N binary masks to a single ``(H, W)`` int16 labels array.

    Pixel value = ``color_id`` where ``color_id ∈ [1, N]`` matches the
    1-indexed position of the mask in ``masks``. Background = 0.

    When ``bleed_tolerance_px > 0``, background pixels within that
    distance of any mask are assigned to the *nearest* mask's label. This
    is what lets two masks separated by a small gap register as adjacent.

    Overlap policy: when multiple raw masks claim the same pixel, the
    *lowest-indexed* mask wins (deterministic and cheap).
    """
    if len(masks) == 0:
        raise ValueError("labels_from_masks requires at least one mask")

    first = masks[0]
    if first.ndim != 2:
        raise ValueError(
            f"masks must be 2-D, got shape {first.shape}"
        )
    height, width = first.shape

    # Build labels via reverse-order putmask: the lowest-indexed mask
    # wins ties because it gets written last. This avoids materializing
    # an ``(N, H, W)`` stack and an argmax over it, which dominate cost
    # at 4k×4k. Equivalent semantics, ~7× faster on N=13 / 16 Mpx.
    labels: np.ndarray = np.zeros((height, width), dtype=np.int16)
    for idx in range(len(masks) - 1, -1, -1):
        mask = masks[idx]
        if mask.shape != (height, width):
            raise ValueError(
                f"mask {idx} shape {mask.shape} != expected {(height, width)}"
            )
        np.putmask(labels, mask, np.int16(idx + 1))
    any_set: np.ndarray = labels != 0

    if bleed_tolerance_px <= 0:
        return labels

    # Bleed: for every background pixel, find the nearest non-background
    # pixel. If it's within bleed_tolerance, adopt that label.
    background: np.ndarray = np.logical_not(any_set)
    # distance_transform_edt computes distance to the nearest 0 within
    # the *input* array. We want distance to the nearest non-bg pixel
    # from each bg pixel, so feed it the background mask. With both
    # return_distances and return_indices, the call returns a 2-tuple
    # ``(distances, indices)`` where ``indices`` has shape (ndim, H, W).
    edt_result = distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    # The (True, True) overload returns (distances, indices); narrow the
    # Optional/overload union for mypy.
    assert isinstance(edt_result, tuple)
    distances, indices = edt_result
    yi = indices[0]
    xi = indices[1]
    within = background & (distances <= float(bleed_tolerance_px))
    if np.any(within):
        labels[within] = labels[yi[within], xi[within]]

    return labels


def build_adjacency_graph(
    masks: Sequence[np.ndarray],
    color_ids: Sequence[str],
    bleed_tolerance_px: int = 8,
    sliver_threshold_px: int | None = None,
) -> nx.Graph[str]:
    """Build a weighted region adjacency graph keyed by ``color_ids``.

    Args:
        masks: ``N`` binary masks, all the same shape. ``masks[i]``
            belongs to ``color_ids[i]``.
        color_ids: External color identifiers used as node names. May
            contain duplicates — duplicates collapse to a single node
            (per the "one node per color" rule).
        bleed_tolerance_px: Radius (pixels) by which each mask is
            implicitly dilated when checking adjacency. Models physical
            ink bleed in a print.
        sliver_threshold_px: Masks with area below this are dropped
            before graph construction. Defaults to
            ``bleed_tolerance_px * 10``.

    Returns:
        ``networkx.Graph`` with one node per *unique* color id present
        after sliver absorption. Each edge has a ``weight`` attribute =
        count of touching pixel pairs (north + east neighbors) after
        bleed expansion.
    """
    if len(masks) != len(color_ids):
        raise ValueError(
            f"masks ({len(masks)}) and color_ids ({len(color_ids)}) length mismatch"
        )

    threshold = (
        sliver_threshold_px
        if sliver_threshold_px is not None
        else bleed_tolerance_px * 10
    )

    # Sliver absorption — keep masks + matching color_ids aligned.
    kept: list[tuple[np.ndarray, str]] = [
        (m, cid)
        for m, cid in zip(masks, color_ids)
        if int(np.count_nonzero(m)) >= threshold
    ]
    graph: nx.Graph[str] = nx.Graph()

    if not kept:
        return graph

    kept_masks = [m for m, _ in kept]
    kept_ids = [cid for _, cid in kept]

    # One node per unique color id (in stable first-seen order).
    seen: dict[str, None] = {}
    for cid in kept_ids:
        if cid not in seen:
            seen[cid] = None
            graph.add_node(cid)

    # Build labels array. Labels are 1-indexed positions into kept_masks.
    labels = labels_from_masks(kept_masks, bleed_tolerance_px=bleed_tolerance_px)

    # Map label int -> color_id string for edge lookup. Position 0 is
    # background; positions 1..N map to kept_ids[0..N-1].
    label_to_cid: list[str | None] = [None] + list(kept_ids)

    _add_edges_from_labels(labels, label_to_cid, graph)
    return graph


def _add_edges_from_labels(
    labels: np.ndarray,
    label_to_cid: list[str | None],
    graph: nx.Graph[str],
) -> None:
    """Vectorized north/east neighbor scan to count touching pixel pairs.

    Pixel ``(y, x)`` and ``(y, x+1)`` form an east-pair; ``(y, x)`` and
    ``(y+1, x)`` form a south-pair. We accumulate counts for unordered
    label pairs ``{a, b}`` where ``a != b`` and both are non-zero.
    Self-pairs (same label) and any pair involving background (0) are
    dropped before ``np.unique`` to keep it cheap.

    For speed, pairs are encoded as a single int64 key
    ``lo * stride + hi`` instead of a 2-column int16 array — ``np.unique``
    on 1-D is significantly faster than ``axis=0`` on 2-D arrays.
    """
    if labels.size == 0 or labels.shape[0] < 2 and labels.shape[1] < 2:
        return

    chunks_a: list[np.ndarray] = []
    chunks_b: list[np.ndarray] = []

    if labels.shape[1] >= 2:
        chunks_a.append(labels[:, :-1].ravel())
        chunks_b.append(labels[:, 1:].ravel())

    if labels.shape[0] >= 2:
        chunks_a.append(labels[:-1, :].ravel())
        chunks_b.append(labels[1:, :].ravel())

    if not chunks_a:
        return

    all_a = np.concatenate(chunks_a)
    all_b = np.concatenate(chunks_b)

    mask = (all_a != 0) & (all_b != 0) & (all_a != all_b)
    if not np.any(mask):
        return

    la = all_a[mask].astype(np.int64, copy=False)
    lb = all_b[mask].astype(np.int64, copy=False)
    lo = np.minimum(la, lb)
    hi = np.maximum(la, lb)

    # Encode unordered (lo, hi) as one int64 key. stride must exceed any
    # label value; len(label_to_cid) already does.
    stride = np.int64(len(label_to_cid))
    keys = lo * stride + hi

    unique_keys, counts = np.unique(keys, return_counts=True)
    decoded_lo = (unique_keys // stride).astype(np.int64)
    decoded_hi = (unique_keys % stride).astype(np.int64)

    for la_i, hi_i, weight in zip(
        decoded_lo.tolist(), decoded_hi.tolist(), counts.tolist()
    ):
        cid_a = label_to_cid[la_i]
        cid_b = label_to_cid[hi_i]
        if cid_a is None or cid_b is None or cid_a == cid_b:
            # cid_a == cid_b happens when two masks share a color id;
            # those pixels live on the same node, not an edge.
            continue
        if graph.has_edge(cid_a, cid_b):
            graph[cid_a][cid_b]["weight"] += int(weight)
        else:
            graph.add_edge(cid_a, cid_b, weight=int(weight))
