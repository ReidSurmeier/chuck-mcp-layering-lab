"""TDD CYCLE 2 + 4 — mokuhanga_emma adapter picks concrete pigments.

Cycle 2: the adapter wraps the v3 underlayer_proposer and returns plates
whose pigment_name is a real PigmentEntry (not a default placeholder).

Cycle 4: mid/dark plates pick concrete pigments by minimizing ΔE_76 to
their plate's target Lab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mokuhanga_emma import (
    EmmaPigmentPlate,
    plan_underlayer_pigments,
    select_pigment_for_role_plate,
    delta_e_76,
)
from pigment_library import load_pigment_library, EMMA_PIGMENT_LIBRARY_PATH


# Reuse the existing v3 emma synthesizer (it builds the same color profile
# Reid annotated against).
from test_emma_annotation import (  # noqa: E402  — sibling test module
    synthesize_emma_cellgraph_and_landmarks,
    REID_ANNOTATION,
    EMMA_REGION_MEAN_RGB,
)


# --------------------------------------------------------------------------- #
# CYCLE 2: underlayer proposer picks pigments from library                    #
# --------------------------------------------------------------------------- #
def test_underlayer_proposer_on_emma_picks_yellow_for_cheek_temple() -> None:
    """The mokuhanga_emma adapter must return underlayer plates for
    cheek/temple regions and assign concrete pigments from the Emma library
    whose family is light_yellow OR pale_pink (per Reid's annotation +
    rule_table neighbor swap)."""
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    cg, fl = synthesize_emma_cellgraph_and_landmarks()
    plates = plan_underlayer_pigments(
        target_image=np.zeros((2048, 1658, 3), dtype=np.float32),
        cell_graph=cg,
        face_landmarks=fl,
        pigment_library=library,
    )

    assert len(plates) >= 4, f"expected at least 4 underlayer plates, got {len(plates)}"

    by_region = {p.region_label: p for p in plates}

    # Cheek MUST be light_yellow family (canonical Pace first-plate rule)
    assert "cheek" in by_region, "expected a cheek underlayer plate"
    cheek = by_region["cheek"]
    assert cheek.pigment_family == "light_yellow", (
        f"cheek pigment_family={cheek.pigment_family!r}, expected light_yellow"
    )
    assert library[cheek.pigment_name].family == "light_yellow", (
        f"cheek pigment_name={cheek.pigment_name!r} is not a light_yellow pigment"
    )
    assert cheek.role == "underlayer_light"
    assert cheek.provenance.startswith("algorithm"), cheek.provenance

    # Temple is the pink-extension of the cheek mass.
    assert "temple" in by_region, "expected a temple underlayer plate"
    temple = by_region["temple"]
    assert temple.pigment_family in ("pale_pink", "light_yellow"), (
        f"temple family={temple.pigment_family!r}, "
        f"expected one of (pale_pink, light_yellow)"
    )
    assert library[temple.pigment_name].family == temple.pigment_family

    # Every plate must reference a real library entry, never a placeholder.
    for p in plates:
        assert p.pigment_name in library, (
            f"plate {p.region_label}: pigment_name={p.pigment_name!r} "
            f"is not in library"
        )
        assert p.pigment_name != "", f"plate {p.region_label}: empty pigment_name"


# --------------------------------------------------------------------------- #
# CYCLE 4: mid/dark plates pick pigments by lab distance                      #
# --------------------------------------------------------------------------- #
def test_mid_dark_plates_pick_pigments_by_lab_distance() -> None:
    """For non-underlayer roles, select_pigment_for_role_plate must return
    the pigment whose Lab values are CLOSEST (smallest ΔE_76) to the plate
    target Lab, restricted to the candidate family."""
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)

    # Build a synthetic "lip mid-build" plate: target Lab matches vermilion
    # closely. The library has vermilion / scarlet / alizarin_crimson in the
    # pale_red family — vermilion is the closest hit.
    target_lab = (56.0, 60.0, 50.0)  # equals vermilion exactly
    pigment = select_pigment_for_role_plate(
        library=library,
        target_lab=target_lab,
        candidate_family="pale_red",
    )
    assert pigment.name == "vermilion", (
        f"target_lab={target_lab} should match vermilion, got {pigment.name}"
    )

    # Build a deep-red plate: closer to alizarin_crimson.
    target_lab = (38.0, 50.0, 24.0)
    pigment = select_pigment_for_role_plate(
        library=library,
        target_lab=target_lab,
        candidate_family="pale_red",
    )
    assert pigment.name == "alizarin_crimson", pigment.name

    # Dark warm-grey target should match burnt_umber or ivory_black,
    # NOT light warm_grey.
    target_lab = (18.0, 0.0, 0.0)
    pigment = select_pigment_for_role_plate(
        library=library,
        target_lab=target_lab,
        candidate_family="warm_grey",
    )
    assert pigment.name in ("ivory_black", "burnt_umber"), pigment.name

    # Verify delta_e_76 maths
    a = (50.0, 0.0, 0.0)
    b_pt = (54.0, 0.0, 3.0)
    de = delta_e_76(a, b_pt)
    assert abs(de - 5.0) < 1e-6, de


# --------------------------------------------------------------------------- #
# CYCLE 5: Emma underlayer match score ≥ 85%                                  #
# --------------------------------------------------------------------------- #
def test_emma_underlayer_match_score_ge_85_percent() -> None:
    """The mokuhanga_emma plan_underlayer_pigments output, scored against
    Reid's annotation, must reach ≥85% per-region match (looser than the
    94.4% lab-only number)."""
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    cg, fl = synthesize_emma_cellgraph_and_landmarks()
    plates = plan_underlayer_pigments(
        target_image=np.zeros((2048, 1658, 3), dtype=np.float32),
        cell_graph=cg,
        face_landmarks=fl,
        pigment_library=library,
    )

    predicted_family: dict[str, str] = {
        p.region_label: p.pigment_family for p in plates
    }

    # Reuse the same axis-grouped scoring as the v3 harness.
    COLOR_AXIS = {
        "light_yellow":  "warm",
        "pale_pink":     "warm",
        "pale_orange":   "warm",
        "pale_red":      "warm",
        "pale_blue":     "cool",
        "pale_green":    "cool",
        "warm_grey":     "neutral",
    }
    total = 0.0
    exact = 0
    for region, annotated_family in REID_ANNOTATION.items():
        pred = predicted_family.get(region)
        if pred is None:
            continue
        if pred == annotated_family:
            total += 1.0
            exact += 1
        elif COLOR_AXIS[pred] == COLOR_AXIS[annotated_family]:
            total += 0.5
    match_rate = total / float(len(REID_ANNOTATION))
    assert match_rate >= 0.85, (
        f"emma underlayer match={match_rate:.1%} (need ≥85%), "
        f"exact={exact}/{len(REID_ANNOTATION)}"
    )
