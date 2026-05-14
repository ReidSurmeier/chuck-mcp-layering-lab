"""S6.b jigsaw organization tests."""
from __future__ import annotations

import numpy as np


def _target() -> np.ndarray:
    target = np.zeros((64, 64, 3), dtype=np.float32)
    target[:, :22] = (0.82, 0.46, 0.38)
    target[:, 22:44] = (0.36, 0.58, 0.70)
    target[:, 44:] = (0.42, 0.65, 0.38)
    return target


def _stack() -> np.ndarray:
    alpha = np.zeros((8, 64, 64), dtype=np.float32)
    alpha[0:3] = 0.12
    alpha[3, :, 8:42] = 0.55
    alpha[4, :, 22:56] = 0.52
    alpha[5, 10:54, 12:52] = 0.48
    alpha[6, 20:44, 20:44] = 0.35
    alpha[7, 28:36, 28:36] = 0.80
    return alpha


def test_jigsaw_organization_reduces_middle_overlap() -> None:
    from backend.services.v23.stages.s6b_jigsaw_organize import organize_jigsaw_regions

    alpha = _stack()
    result = organize_jigsaw_regions(
        alpha,
        np.array([13, 21, 17, 10, 8, 20, 15, 12], dtype=np.int32),
        target_rgb=_target(),
        n_segments=90,
        min_region_px=12,
    )

    assert result.diagnostics["applied"] is True
    assert (
        result.diagnostics["mid_pairwise_overlap_after"]
        < result.diagnostics["mid_pairwise_overlap_before"]
    )


def test_jigsaw_organization_preserves_under_and_detail_roles() -> None:
    from backend.services.v23.stages.s6b_jigsaw_organize import organize_jigsaw_regions

    alpha = _stack()
    result = organize_jigsaw_regions(
        alpha,
        np.array([13, 21, 17, 10, 8, 20, 15, 12], dtype=np.int32),
        target_rgb=_target(),
        n_segments=90,
        min_region_px=12,
    )

    np.testing.assert_allclose(result.alpha_stack[:3], alpha[:3])
    np.testing.assert_allclose(result.alpha_stack[-2:], alpha[-2:])


def test_jigsaw_organization_recovers_near_paper_tint_cells() -> None:
    from backend.services.v23.stages.s6b_jigsaw_organize import organize_jigsaw_regions

    alpha = np.zeros((8, 64, 64), dtype=np.float32)
    target = np.full((64, 64, 3), [0.965, 0.945, 0.890], dtype=np.float32)
    target[10:54, 10:54] = (0.88, 0.91, 0.94)
    pigment_idx = np.array([13, 21, 23, 17, 10, 8, 20, 12], dtype=np.int32)

    result = organize_jigsaw_regions(
        alpha,
        pigment_idx,
        target_rgb=target,
        n_segments=80,
        min_region_px=12,
    )

    assert result.diagnostics["tint_recovery_cells"] > 0
    assert float(result.alpha_stack[1, 10:54, 10:54].mean()) > 0.0
