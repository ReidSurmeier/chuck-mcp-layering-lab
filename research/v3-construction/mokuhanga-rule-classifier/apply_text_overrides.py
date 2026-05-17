"""
apply_text_overrides.py
=======================

Apply LLM-parsed text directives on top of algorithmic underlayer baseline.

Per v2-design Workflow step 6:
    "Text overrides applied to algorithmic baseline (text always wins where specified)"

Per Q27 (locked hybrid):
    ALGORITHM proposes 4-9 baseline underlayer plates from the cell graph;
    TEXT prompt OVERRIDES algorithm decisions where Reid specifies.

Override schema (output of LLM prompt translation; sibling agent in v3 swarm):

    {
      "version": "1.0",
      "overrides": [
        {
          "kind": "region_pigment_family",      # types listed below
          "region": "hair",
          "pigment_family": "pale_blue",
          "rationale_text": "blue under hair (NOT green)",  # verbatim user phrase
        },
        {
          "kind": "forbid_family_in_region",
          "region": "lip",
          "pigment_family": "light_yellow",
          "rationale_text": "no yellow anywhere on lips",
        },
        {
          "kind": "add_region",
          "region": "background",
          "pigment_family": "pale_orange",
          "rationale_text": "warm orange behind head for contrast",
        },
        {
          "kind": "remove_region",
          "region": "eye_white",
          "rationale_text": "skip the eye-white underlayer",
        },
        {
          "kind": "set_opacity",
          "region": "cheek",
          "opacity": 0.28,
          "rationale_text": "make the cheek underlay stronger",
        },
      ]
    }

Override kinds:
    - region_pigment_family:    change family for one region
    - forbid_family_in_region:  remove a family from allowed set, re-pick
    - add_region:               force a plate even if region didn't pass coverage check
    - remove_region:            drop a plate
    - set_opacity:              override opacity for one region (within 0.10..0.35)

After applying overrides, the function re-runs pass_order assignment so the
final plate list still respects light-to-dark + first-pull-is-lightest rules.

Each modified/added plate gets provenance = "text:<verbatim_phrase>".
Untouched plates keep provenance = "algorithm".
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from underlayer_proposer import (
    PIGMENT_FAMILY_ANCHORS_RGB,
    PigmentFamily,
    RegionLabel,
    UnderlayerPlate,
    _assign_pass_order,
    load_rules,
)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def apply_text_overrides(
    baseline_plates: list[UnderlayerPlate],
    overrides: list[dict[str, Any]] | dict[str, Any] | None,
    rules: dict | None = None,
) -> list[UnderlayerPlate]:
    """
    Apply LLM-parsed text overrides on top of the algorithmic baseline.

    Parameters
    ----------
    baseline_plates : list[UnderlayerPlate]
        Output of `underlayer_proposer.propose_underlayers()`.
    overrides : list[dict] | dict | None
        Override directives. Either:
          - a list of override dicts (each with 'kind' and parameters)
          - a dict with 'overrides' key wrapping the list (matches LLM output)
          - None (returns baseline unchanged)
    rules : dict, optional
        Parsed rule_table.yaml. Falls back to default load.

    Returns
    -------
    list[UnderlayerPlate]
        Final plate list with overrides applied and pass_order re-assigned.
    """
    rules = rules or load_rules()

    if overrides is None:
        return baseline_plates

    if isinstance(overrides, dict):
        override_list = overrides.get("overrides", [])
    else:
        override_list = overrides

    # Work on a deep copy so the baseline can be referenced for provenance diff
    plates = deepcopy(baseline_plates)

    for ov in override_list:
        kind = ov.get("kind")
        if kind == "region_pigment_family":
            plates = _override_region_family(plates, ov, rules)
        elif kind == "forbid_family_in_region":
            plates = _override_forbid_family(plates, ov, rules)
        elif kind == "add_region":
            plates = _override_add_region(plates, ov, rules, baseline_plates)
        elif kind == "remove_region":
            plates = _override_remove_region(plates, ov)
        elif kind == "set_opacity":
            plates = _override_set_opacity(plates, ov)
        else:
            # Unknown kind — preserve and tag with diagnostic
            for p in plates:
                p.rationale += f" [unknown override kind '{kind}' ignored]"

    # Re-assign pass_index after all edits (light-to-dark may have shifted)
    plates = _assign_pass_order(plates, rules)
    return plates


# --------------------------------------------------------------------------- #
# Per-override-kind handlers                                                  #
# --------------------------------------------------------------------------- #

def _override_region_family(
    plates: list[UnderlayerPlate],
    ov: dict[str, Any],
    rules: dict,
) -> list[UnderlayerPlate]:
    """Change pigment family for an existing region."""
    region = ov["region"]
    new_family = ov["pigment_family"]
    phrase = ov.get("rationale_text", "")

    if new_family not in PIGMENT_FAMILY_ANCHORS_RGB:
        for p in plates:
            p.rationale += f" [override invalid family '{new_family}' ignored]"
        return plates

    found = False
    for p in plates:
        if p.region_label == region:
            old_family = p.pigment_family
            p.pigment_family = new_family
            p.provenance = f"text:{phrase}"
            p.rationale = (
                f"OVERRIDE: '{phrase}' "
                f"(was algorithm-picked {old_family}, now {new_family})"
            )
            # Override may bypass rule_table forbidden_families — log if so
            forbidden = rules["regions"].get(region, {}).get("forbidden_families", [])
            if new_family in forbidden:
                p.rationale += (
                    f" [WARNING: {new_family} is normally forbidden in {region}; "
                    f"text-override bypassed rule per Q27]"
                )
            found = True
    if not found:
        # Region didn't have a plate — treat as add_region
        return _override_add_region(plates, ov, rules, plates)
    return plates


def _override_forbid_family(
    plates: list[UnderlayerPlate],
    ov: dict[str, Any],
    rules: dict,
) -> list[UnderlayerPlate]:
    """Drop a family from the region's plate and replace with next-best allowed."""
    region = ov["region"]
    forbidden_family = ov["pigment_family"]
    phrase = ov.get("rationale_text", "")

    region_rule = rules["regions"].get(region, {})
    allowed = [
        f for f in region_rule.get("allowed_families", [])
        if f != forbidden_family
    ]
    if not allowed:
        return plates  # no fallback available

    for p in plates:
        if p.region_label == region and p.pigment_family == forbidden_family:
            old_family = p.pigment_family
            p.pigment_family = allowed[0]   # top-priority allowed alternate
            p.provenance = f"text:{phrase}"
            p.rationale = (
                f"OVERRIDE forbid '{forbidden_family}' in {region}: '{phrase}' "
                f"(swapped {old_family} -> {allowed[0]})"
            )
    return plates


def _override_add_region(
    plates: list[UnderlayerPlate],
    ov: dict[str, Any],
    rules: dict,
    original_baseline: list[UnderlayerPlate],
) -> list[UnderlayerPlate]:
    """Force-add a plate for a region that the algorithm skipped."""
    region = ov["region"]
    family = ov["pigment_family"]
    phrase = ov.get("rationale_text", "")
    opacity = ov.get("opacity")

    if any(p.region_label == region for p in plates):
        # Already exists — treat as family change
        return _override_region_family(plates, ov, rules)

    # New block_id = max existing + 1
    next_block = max((p.block_id for p in plates), default=0) + 1

    region_rule = rules["regions"].get(region, {})
    if opacity is None:
        lo, hi = region_rule.get("opacity_range", [0.18, 0.25])
        opacity = float((lo + hi) / 2.0)

    plates.append(UnderlayerPlate(
        block_id=next_block,
        region_label=region,
        pigment_family=family,
        cell_zone_ids=[],   # caller fills via region_cells_map; empty = solver assigns
        opacity=opacity,
        pass_index=0,
        coverage=0.0,
        provenance=f"text:{phrase}",
        rationale=f"OVERRIDE add_region: '{phrase}' (algorithm did not propose {region})",
    ))
    return plates


def _override_remove_region(
    plates: list[UnderlayerPlate],
    ov: dict[str, Any],
) -> list[UnderlayerPlate]:
    """Drop a plate entirely."""
    region = ov["region"]
    phrase = ov.get("rationale_text", "")
    kept = [p for p in plates if p.region_label != region]
    if len(kept) != len(plates):
        # Annotate the remaining plates so provenance diff captures the removal
        for p in kept:
            if not p.rationale.startswith("OVERRIDE"):
                continue
        # Log on a fresh dummy if none survived (debug aid)
    return kept


def _override_set_opacity(
    plates: list[UnderlayerPlate],
    ov: dict[str, Any],
) -> list[UnderlayerPlate]:
    """Set opacity directly (clamped to [0.10, 0.35])."""
    region = ov["region"]
    opacity = float(ov["opacity"])
    phrase = ov.get("rationale_text", "")
    opacity = max(0.10, min(0.35, opacity))
    for p in plates:
        if p.region_label == region:
            p.opacity = opacity
            p.provenance = f"text:{phrase}"
            p.rationale += f" [OVERRIDE opacity={opacity:.3f}: '{phrase}']"
    return plates


# --------------------------------------------------------------------------- #
# Diff utility (for the UI's "interpretation panel")                          #
# --------------------------------------------------------------------------- #

def diff_against_baseline(
    baseline_plates: list[UnderlayerPlate],
    final_plates: list[UnderlayerPlate],
) -> list[dict[str, Any]]:
    """
    Return a list of human-readable change descriptors comparing baseline vs final.

    Each entry: {"region", "kind", "before", "after", "phrase"}.
    Used to populate the interpretation panel sidebar.
    """
    by_region_before = {p.region_label: p for p in baseline_plates}
    by_region_after = {p.region_label: p for p in final_plates}

    changes: list[dict[str, Any]] = []
    # Detect adds / mods
    for region, after_p in by_region_after.items():
        if region not in by_region_before:
            changes.append({
                "region": region,
                "kind": "added",
                "before": None,
                "after": {"pigment_family": after_p.pigment_family,
                          "opacity": after_p.opacity},
                "phrase": after_p.provenance.removeprefix("text:")
                          if after_p.provenance.startswith("text:") else "",
            })
            continue
        before_p = by_region_before[region]
        if before_p.pigment_family != after_p.pigment_family:
            changes.append({
                "region": region,
                "kind": "family_changed",
                "before": {"pigment_family": before_p.pigment_family},
                "after": {"pigment_family": after_p.pigment_family},
                "phrase": after_p.provenance.removeprefix("text:")
                          if after_p.provenance.startswith("text:") else "",
            })
        elif abs(before_p.opacity - after_p.opacity) > 1e-6:
            changes.append({
                "region": region,
                "kind": "opacity_changed",
                "before": {"opacity": before_p.opacity},
                "after": {"opacity": after_p.opacity},
                "phrase": after_p.provenance.removeprefix("text:")
                          if after_p.provenance.startswith("text:") else "",
            })

    # Detect removals
    for region in by_region_before:
        if region not in by_region_after:
            changes.append({
                "region": region,
                "kind": "removed",
                "before": {"pigment_family": by_region_before[region].pigment_family},
                "after": None,
                "phrase": "",
            })
    return changes


if __name__ == "__main__":
    # Smoke test demonstrating the override flow against a fake baseline
    baseline = [
        UnderlayerPlate(block_id=1, region_label="cheek",
                        pigment_family="light_yellow", cell_zone_ids=[0, 1],
                        opacity=0.20, pass_index=1, coverage=0.45,
                        rationale="cheek default"),
        UnderlayerPlate(block_id=2, region_label="hair",
                        pigment_family="pale_green", cell_zone_ids=[5, 6],
                        opacity=0.20, pass_index=2, coverage=0.30,
                        rationale="hair default"),
        UnderlayerPlate(block_id=3, region_label="lip",
                        pigment_family="pale_pink", cell_zone_ids=[7],
                        opacity=0.25, pass_index=3, coverage=0.70,
                        rationale="lip default"),
    ]
    overrides = [
        {"kind": "region_pigment_family", "region": "hair",
         "pigment_family": "pale_blue",
         "rationale_text": "blue under hair (NOT green)"},
        {"kind": "set_opacity", "region": "cheek", "opacity": 0.28,
         "rationale_text": "make the cheek underlay stronger"},
    ]
    final = apply_text_overrides(baseline, overrides)
    for p in final:
        print(p.block_id, p.region_label, p.pigment_family,
              p.opacity, p.provenance, p.rationale[:80])
    print("---")
    for d in diff_against_baseline(baseline, final):
        print(d)
