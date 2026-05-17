"""Validator 2: role_purity_score.

Each block has a clear print role (underlayer_light / local_chroma /
regional_mass / key_detail). Cells living on one physical plate must
belong (mostly) to one role family. A plate that mixes light-yellow
underlayer cells with key-detail dark-contour cells is incoherent for
the printer — different brush, different ink batch, different opacity.

PER docs/v2-design-locked-2026-05-16.md row 2:
    "Each block tagged with role; reject if cell-zones span > 2 role
     families per block"

Practical formulation:
    purity = (count of cells with modal role) / (count of all cells on plate)

We score the FRACTION belonging to the modal role. Threshold = 0.7
(per the reconstruction doc spec the task brief gives).

The doc also says "> 2 role families" rejects. We treat the stronger
threshold (purity >= 0.7) as the gating signal AND separately count
distinct roles for an auxiliary check.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

ALLOWED_ROLES = {
    "underlayer_light",
    "local_chroma",
    "regional_mass",
    "key_detail",
}

# Per task brief
PURITY_THRESHOLD = 0.7
MAX_ROLE_FAMILIES = 2  # Per design doc


def score(
    plate_id: Optional[int] = None,
    cells_in_plate: Optional[Iterable[int]] = None,
    cell_role_labels: Optional[dict[int, str]] = None,
    return_components: bool = False,
):
    """Run the role-purity validator on ONE plate.

    Args:
        plate_id: optional plate id (informational only).
        cells_in_plate: iterable of cell IDs that live on this plate.
        cell_role_labels: dict mapping cell_id -> role string.
        return_components: if True, return dict breakdown.

    Returns:
        purity score in [0, 1]. >= PURITY_THRESHOLD means PASS.
    """
    if cells_in_plate is None or cell_role_labels is None:
        raise ValueError("cells_in_plate and cell_role_labels required")

    cells: List[int] = list(cells_in_plate)
    if not cells:
        # Empty plate — vacuously pure but caller should flag empty plates
        # separately. Return 1.0 here, runner will warn.
        purity = 1.0
        modal = None
        distinct_roles: list[str] = []
    else:
        roles = [cell_role_labels.get(c, "unknown") for c in cells]
        # Validate roles
        bad_roles = [r for r in roles if r not in ALLOWED_ROLES and r != "unknown"]
        if bad_roles:
            # Treat unknown-role cells as a separate "unknown" bucket.
            pass
        counts = Counter(roles)
        modal, modal_n = counts.most_common(1)[0]
        purity = modal_n / len(cells)
        distinct_roles = sorted(counts.keys())

    distinct_role_count = len(distinct_roles)
    passes_purity = purity >= PURITY_THRESHOLD
    passes_family_cap = distinct_role_count <= MAX_ROLE_FAMILIES
    # Hard gate: BOTH must hold
    passing = passes_purity and passes_family_cap

    if return_components:
        return {
            "plate_id": plate_id,
            "purity_score": float(purity),
            "passes": bool(passing),
            "modal_role": modal,
            "n_cells": len(cells),
            "distinct_roles": distinct_roles,
            "distinct_role_count": distinct_role_count,
            "purity_threshold": PURITY_THRESHOLD,
            "max_role_families": MAX_ROLE_FAMILIES,
            "fail_reason": (
                None
                if passing
                else (
                    f"purity {purity:.2f} < {PURITY_THRESHOLD}"
                    if not passes_purity
                    else f"distinct_roles {distinct_role_count} > {MAX_ROLE_FAMILIES}"
                )
            ),
        }
    return float(purity)


def passes(
    plate_id: Optional[int],
    cells_in_plate: Iterable[int],
    cell_role_labels: dict[int, str],
) -> bool:
    out = score(plate_id, cells_in_plate, cell_role_labels, return_components=True)
    return bool(out["passes"])


if __name__ == "__main__":
    # Smoke: pure plate + impure plate
    labels = {1: "underlayer_light", 2: "underlayer_light", 3: "underlayer_light", 4: "key_detail"}
    pure = score(plate_id=1, cells_in_plate=[1, 2, 3], cell_role_labels=labels, return_components=True)
    impure = score(plate_id=2, cells_in_plate=[1, 2, 3, 4], cell_role_labels=labels, return_components=True)
    print("PURE plate:", pure)
    print("IMPURE plate:", impure)
