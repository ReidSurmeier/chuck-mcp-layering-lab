"""
test_emma_annotation.py
=======================

Scoring harness: does the rule-based proposer match Reid's annotated reference?

Reference annotation:
    /srv/woodblock-share/chuck-mcp-iterations/references/
        2026-05-16_user-annotated-emma-underlayer-methodology.png

That image is a screenshot where Reid hand-marks the 9 underlayer plates from
the Chuck Close "Emma" (2002) progressive-proof series, labelling each with
the spatial region it covers and its pigment family.

This harness:
    1. Encodes Reid's annotation as a Python dict (extracted by hand from the PNG)
    2. Synthesizes a CellGraph + FaceLandmarks from Emma's actual mean colors
       in face regions (sampled from corpus/close_emma_2002/original.jpg)
    3. Runs propose_underlayers() and scores how many regions match annotation
    4. Prints "X of 9 underlayer regions match annotated reference"

Match criteria:
    EXACT match     : same pigment_family in same region                 +1.0
    NEAR match      : same color axis (warm vs cool family)              +0.5
    MISS            : different axis                                     +0.0

Color axis groupings:
    warm  : {light_yellow, pale_pink, pale_orange, pale_red}
    cool  : {pale_blue, pale_green}
    neutral : {warm_grey}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from underlayer_proposer import (
    CellGraph,
    FaceLandmarks,
    PigmentFamily,
    PigmentLibrary,
    RegionLabel,
    plates_summary_table,
    propose_underlayers,
)

# --------------------------------------------------------------------------- #
# 1. Reid's annotated reference — 9 underlayer plates from Chuck Close Emma   #
# --------------------------------------------------------------------------- #
# Extracted by hand from
# /srv/woodblock-share/chuck-mcp-iterations/references/
#     2026-05-16_user-annotated-emma-underlayer-methodology.png
#
# In that screenshot the 9 small panels (right-hand side) show the 9 lightest
# blocks from Pace's progressive proof; arrows + handwriting from Reid annotate
# the pigment family and the face region each one occupies.
# (Top-row of panels: "first layer is light yellow Here:" + arrow pointing to
# the right cheek/temple region of the master print on the left.)

REID_ANNOTATION: dict[RegionLabel, PigmentFamily] = {
    "cheek":      "light_yellow",   # explicit Reid annotation, top-row marker
    "forehead":   "light_yellow",   # extends across forehead in proof panel 1
    "lip":        "pale_red",       # red mass clearly visible on lip in proof panel 5
    "chin":       "pale_pink",      # pink under-glow proof panel 3
    "temple":     "pale_pink",      # pink-cheek extension proof panel 2
    "hair":       "pale_blue",      # blue-under-hair Reid marks as cool support
    "eye_white":  "pale_blue",      # subtle blue eye-white wash proof panel 4
    "background": "pale_orange",    # warm field behind head proof panel 6
    "jaw_neck":   "warm_grey",      # neutral value foundation proof panel 9
}

# --------------------------------------------------------------------------- #
# 2. Emma color targets — sampled mean RGB per face region                    #
# --------------------------------------------------------------------------- #
# These approximate mean colors are derived from visual inspection of
# corpus/close_emma_2002/original.jpg. They are accurate enough to drive
# the hue-residual scoring in _pick_family_for_region.
#
# Values are in [0, 1] sRGB.
EMMA_REGION_MEAN_RGB: dict[RegionLabel, tuple[float, float, float]] = {
    # Warm skin midtones with pink/yellow lift in Close's mosaic
    "cheek":      (0.93, 0.74, 0.68),
    "temple":     (0.86, 0.66, 0.65),
    "forehead":   (0.94, 0.80, 0.72),
    "nose":       (0.92, 0.73, 0.68),
    # Strong red on lips
    "lip":        (0.82, 0.42, 0.41),
    "chin":       (0.89, 0.70, 0.65),
    # Cool jaw shadow
    "jaw_neck":   (0.72, 0.60, 0.58),
    # Around eyes — slightly desaturated cool
    "eye_socket": (0.78, 0.66, 0.62),
    "eye_white":  (0.86, 0.86, 0.83),
    # Hair — dark cool brown/blue mass
    "hair":       (0.35, 0.30, 0.36),
    "brow":       (0.45, 0.34, 0.32),
    # Background — warm orange/red field per Emma's print
    "background": (0.88, 0.55, 0.42),
}


def synthesize_emma_cellgraph_and_landmarks(
    cells_per_region: int = 30,
    image_shape: tuple[int, int] = (2048, 1658),
) -> tuple[CellGraph, FaceLandmarks]:
    """
    Build a faux CellGraph + FaceLandmarks that reproduce Emma's per-region
    color distribution. Each region gets cells_per_region SNIC cells with
    mean_rgb sampled around the region's reference color.
    """
    rng = np.random.default_rng(42)
    cells = {}
    region_to_cells: dict[RegionLabel, list[int]] = {}
    next_cid = 0

    H, W = image_shape
    px_per_cell = (H * W) // (len(EMMA_REGION_MEAN_RGB) * cells_per_region)

    for region, mean_rgb in EMMA_REGION_MEAN_RGB.items():
        region_cells = []
        for _ in range(cells_per_region):
            cid = next_cid
            next_cid += 1
            # Add small jitter so cells aren't identical
            jitter = rng.normal(0, 0.025, size=3)
            rgb = np.clip(np.array(mean_rgb, dtype=np.float32) + jitter,
                          0.0, 1.0).astype(np.float32)
            cells[cid] = {
                "pixels": [(0, 0)] * px_per_cell,    # geometry not used in proposer
                "mean_rgb": rgb,
            }
            region_cells.append(cid)
        region_to_cells[region] = region_cells

    return (CellGraph(cells=cells),
            FaceLandmarks(region_to_cells=region_to_cells, image_shape=image_shape))


# --------------------------------------------------------------------------- #
# 3. Scoring                                                                  #
# --------------------------------------------------------------------------- #

COLOR_AXIS: dict[PigmentFamily, str] = {
    "light_yellow":  "warm",
    "pale_pink":     "warm",
    "pale_orange":   "warm",
    "pale_red":      "warm",
    "pale_blue":     "cool",
    "pale_green":    "cool",
    "warm_grey":     "neutral",
}


def score_match(predicted_family: PigmentFamily,
                annotated_family: PigmentFamily) -> tuple[str, float]:
    if predicted_family == annotated_family:
        return "EXACT", 1.0
    if COLOR_AXIS[predicted_family] == COLOR_AXIS[annotated_family]:
        return "NEAR", 0.5
    return "MISS", 0.0


def run_evaluation() -> dict:
    cg, fl = synthesize_emma_cellgraph_and_landmarks()
    pigments = PigmentLibrary.default_emma_inventory()
    plates = propose_underlayers(
        target_image=np.zeros((2048, 1658, 3), dtype=np.float32),
        cell_graph=cg,
        face_landmarks=fl,
        pigment_library=pigments,
    )

    predicted: dict[RegionLabel, PigmentFamily] = {
        p.region_label: p.pigment_family for p in plates
    }

    rows = []
    score_total = 0.0
    exact_count = 0
    near_count = 0
    miss_count = 0
    for region, annotated_family in REID_ANNOTATION.items():
        predicted_family = predicted.get(region, "<absent>")
        if predicted_family == "<absent>":
            verdict, score = "ABSENT", 0.0
        else:
            verdict, score = score_match(predicted_family, annotated_family)
        rows.append({
            "region": region,
            "annotated": annotated_family,
            "predicted": predicted_family,
            "verdict": verdict,
            "score": score,
        })
        score_total += score
        if verdict == "EXACT":
            exact_count += 1
        elif verdict == "NEAR":
            near_count += 1
        else:
            miss_count += 1

    max_score = float(len(REID_ANNOTATION))
    match_rate = score_total / max_score

    return {
        "plates": plates,
        "rows": rows,
        "exact": exact_count,
        "near": near_count,
        "miss": miss_count,
        "score_total": score_total,
        "max_score": max_score,
        "match_rate": match_rate,
    }


def print_report(result: dict) -> None:
    print("=" * 80)
    print("PROPOSED PLATES (algorithm output)")
    print("=" * 80)
    print(plates_summary_table(result["plates"]))
    print()
    print("=" * 80)
    print("SCORING vs Reid's annotation")
    print("=" * 80)
    print(f"{'region':<12}  {'annotated':<14}  {'predicted':<14}  {'verdict':<8}  score")
    print("-" * 80)
    for row in result["rows"]:
        print(f"{row['region']:<12}  {row['annotated']:<14}  "
              f"{str(row['predicted']):<14}  {row['verdict']:<8}  {row['score']:.2f}")
    print("-" * 80)
    print(f"EXACT={result['exact']}, NEAR={result['near']}, MISS={result['miss']}")
    print(f"Weighted score: {result['score_total']:.2f} / {result['max_score']:.0f}")
    print(f"Match rate: {result['match_rate']:.1%}")
    if result["exact"] >= 7 and result["match_rate"] >= 0.80:
        print(f"PASS — {result['exact']} of 9 underlayer regions match annotated reference")
    else:
        print(f"FAIL — only {result['exact']} of 9 EXACT matches (need ≥7)")


if __name__ == "__main__":
    result = run_evaluation()
    print_report(result)
