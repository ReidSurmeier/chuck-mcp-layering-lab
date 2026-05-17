"""TDD CYCLE 1 — pigment library loads.

Failing-first test for the Emma pigment library YAML and its loader.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pigment_library import (
    PigmentEntry,
    load_pigment_library,
    EMMA_PIGMENT_LIBRARY_PATH,
)


# --------------------------------------------------------------------------- #
# CYCLE 1: pigment library loads                                              #
# --------------------------------------------------------------------------- #
def test_pigment_library_loads_15_to_25_entries() -> None:
    """The Emma pigment library YAML loads, contains 15-25 entries, and every
    entry has the required schema fields.
    """
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    n = len(library)
    assert 15 <= n <= 25, (
        f"expected 15-25 pigment entries, got {n}. "
        f"file={EMMA_PIGMENT_LIBRARY_PATH}"
    )

    # Each entry parsed into a PigmentEntry with required fields
    for name, entry in library.items():
        assert isinstance(entry, PigmentEntry), (
            f"library[{name!r}] is not a PigmentEntry"
        )
        assert entry.name == name
        assert len(entry.lab_values) == 3, f"{name}: lab_values must be (L,a,b)"
        assert isinstance(entry.opacity_curve, list)
        assert len(entry.opacity_curve) >= 2
        assert entry.uncalibrated_v1 is True, (
            f"{name}: V1 must mark itself uncalibrated"
        )
        assert entry.source_note, f"{name}: source_note missing"


def test_pigment_library_contains_required_emma_pigments() -> None:
    """The library must include the canonical Emma pigments the rule
    classifier expects (one or more per family)."""
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    required_names = {
        "light_yellow",
        "pale_pink",
        "pale_orange",
        "vermilion",
        "alizarin_crimson",
        "phthalo_blue",
        "pale_blue",
        "yellow_ochre",
        "burnt_umber",
        "ivory_black",
    }
    missing = required_names - set(library.keys())
    assert not missing, f"required Emma pigments missing: {sorted(missing)}"


def test_pigment_library_family_index_covers_all_seven_families() -> None:
    """family_to_pigments() must return at least one pigment per rule_table
    family so the underlayer proposer can pick a concrete pigment."""
    library = load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    from pigment_library import family_to_pigments

    fam_to_pigs = family_to_pigments(library)
    needed_families = {
        "light_yellow", "pale_pink", "pale_orange",
        "pale_red", "pale_blue", "pale_green", "warm_grey",
    }
    missing = needed_families - set(fam_to_pigs.keys())
    assert not missing, (
        f"families missing concrete pigments: {sorted(missing)}; "
        f"covered={sorted(fam_to_pigs.keys())}"
    )
    for fam, pigs in fam_to_pigs.items():
        assert len(pigs) >= 1, f"family={fam!r} has no concrete pigment"
