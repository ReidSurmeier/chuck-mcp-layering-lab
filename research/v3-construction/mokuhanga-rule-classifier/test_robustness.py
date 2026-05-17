"""
test_robustness.py
==================

Stress-test the proposer on non-Emma portrait inputs to confirm the rule table
generalizes.

Three test cases:
  1. Pale-skin Caucasian portrait (similar to Emma — should pass at >7/9 EXACT)
  2. Dark-skin portrait (skin hue ~30-50° but darker; tests value-foundation logic)
  3. Adversarial: cool-toned portrait with warm background (inversion test)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from test_emma_annotation import (
    REID_ANNOTATION,
    score_match,
    synthesize_emma_cellgraph_and_landmarks,
)
from underlayer_proposer import (
    CellGraph, FaceLandmarks, PigmentFamily, PigmentLibrary, RegionLabel,
    plates_summary_table, propose_underlayers,
)


def synthesize_portrait(region_mean_rgb: dict[RegionLabel, tuple[float, float, float]],
                        cells_per_region: int = 30,
                        image_shape: tuple[int, int] = (2048, 1658)
                        ) -> tuple[CellGraph, FaceLandmarks]:
    rng = np.random.default_rng(123)
    cells = {}
    region_to_cells = {}
    next_cid = 0
    H, W = image_shape
    px_per_cell = (H * W) // (len(region_mean_rgb) * cells_per_region)
    for region, mean_rgb in region_mean_rgb.items():
        region_cells = []
        for _ in range(cells_per_region):
            cid = next_cid; next_cid += 1
            jitter = rng.normal(0, 0.025, size=3)
            rgb = np.clip(np.array(mean_rgb, dtype=np.float32) + jitter, 0.0, 1.0).astype(np.float32)
            cells[cid] = {"pixels": [(0, 0)] * px_per_cell, "mean_rgb": rgb}
            region_cells.append(cid)
        region_to_cells[region] = region_cells
    return CellGraph(cells=cells), FaceLandmarks(region_to_cells=region_to_cells, image_shape=image_shape)


# --- Test case 1: Caucasian pale-skin (similar to Emma but slightly different palette)
CASE_PALE_SKIN: dict[RegionLabel, tuple[float, float, float]] = {
    "cheek":      (0.95, 0.80, 0.72),
    "temple":     (0.90, 0.72, 0.66),
    "forehead":   (0.96, 0.85, 0.75),
    "nose":       (0.93, 0.76, 0.70),
    "lip":        (0.75, 0.40, 0.45),
    "chin":       (0.91, 0.74, 0.68),
    "jaw_neck":   (0.74, 0.65, 0.62),
    "eye_socket": (0.80, 0.68, 0.64),
    "eye_white":  (0.88, 0.88, 0.86),
    "hair":       (0.42, 0.32, 0.25),   # warm brown hair
    "brow":       (0.40, 0.30, 0.25),
    "background": (0.65, 0.72, 0.85),   # cool blue background
}

# --- Test case 2: Dark-skin portrait (warm brown skin, darker values)
CASE_DARK_SKIN: dict[RegionLabel, tuple[float, float, float]] = {
    "cheek":      (0.55, 0.38, 0.28),
    "temple":     (0.50, 0.34, 0.26),
    "forehead":   (0.58, 0.40, 0.30),
    "nose":       (0.54, 0.37, 0.28),
    "lip":        (0.45, 0.22, 0.20),
    "chin":       (0.52, 0.36, 0.27),
    "jaw_neck":   (0.40, 0.30, 0.24),
    "eye_socket": (0.42, 0.28, 0.22),
    "eye_white":  (0.85, 0.82, 0.76),
    "hair":       (0.18, 0.15, 0.13),
    "brow":       (0.20, 0.16, 0.14),
    "background": (0.80, 0.75, 0.55),   # warm yellow background
}

# --- Test case 3: Cool-toned portrait, warm background (adversarial)
CASE_COOL_SKIN: dict[RegionLabel, tuple[float, float, float]] = {
    "cheek":      (0.78, 0.74, 0.80),   # subtle cool-pink
    "temple":     (0.72, 0.68, 0.76),
    "forehead":   (0.82, 0.78, 0.82),
    "nose":       (0.76, 0.72, 0.78),
    "lip":        (0.70, 0.40, 0.50),
    "chin":       (0.74, 0.70, 0.76),
    "jaw_neck":   (0.62, 0.60, 0.66),
    "eye_socket": (0.66, 0.62, 0.68),
    "eye_white":  (0.86, 0.86, 0.85),
    "hair":       (0.38, 0.34, 0.40),
    "brow":       (0.32, 0.28, 0.30),
    "background": (0.92, 0.62, 0.40),   # strong warm orange
}


# Expected underlayer choices for each (informed reasonable expectations,
# NOT ground-truth annotations — just sanity that the rule table behaves rationally)
EXPECTED_PALE_SKIN: dict[RegionLabel, PigmentFamily] = {
    "cheek": "light_yellow", "lip": "pale_red", "hair": "pale_blue",
    "background": "pale_blue", "eye_white": "pale_blue",
    "jaw_neck": "warm_grey",
}

EXPECTED_DARK_SKIN: dict[RegionLabel, PigmentFamily] = {
    "cheek": "light_yellow",     # canonical first plate still fires
    "lip": "pale_red", "hair": "pale_blue",
    "background": "pale_blue",   # complementary cool against warm skin
    "jaw_neck": "warm_grey",
}

EXPECTED_COOL_SKIN: dict[RegionLabel, PigmentFamily] = {
    "cheek": "light_yellow",     # always canonical first plate
    "lip": "pale_red",
    "hair": "pale_blue",
    "background": "pale_orange",  # intrinsic-saturation suppression triggers
}


def evaluate_case(name: str,
                  region_palette: dict[RegionLabel, tuple[float, float, float]],
                  expected: dict[RegionLabel, PigmentFamily]) -> None:
    print(f"\n=== CASE: {name} ===")
    cg, fl = synthesize_portrait(region_palette)
    plates = propose_underlayers(np.zeros((2048, 1658, 3), dtype=np.float32), cg, fl)
    print(plates_summary_table(plates))
    predicted = {p.region_label: p.pigment_family for p in plates}
    rows = []
    total = 0.0
    for region, exp_fam in expected.items():
        pred = predicted.get(region, "<absent>")
        if pred == "<absent>":
            verdict, score = "ABSENT", 0.0
        else:
            verdict, score = score_match(pred, exp_fam)
        rows.append((region, exp_fam, pred, verdict, score))
        total += score
    print(f"\n{'region':<12}  {'expected':<14}  {'predicted':<14}  {'verdict':<8}  score")
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:<12}  {row[1]:<14}  {str(row[2]):<14}  {row[3]:<8}  {row[4]:.2f}")
    print(f"Total: {total:.2f} / {len(expected)} ({total/len(expected):.1%} match)")


if __name__ == "__main__":
    evaluate_case("Pale-skin Caucasian (cool background)",
                  CASE_PALE_SKIN, EXPECTED_PALE_SKIN)
    evaluate_case("Dark-skin (warm background)",
                  CASE_DARK_SKIN, EXPECTED_DARK_SKIN)
    evaluate_case("Cool-toned face + warm background (adversarial)",
                  CASE_COOL_SKIN, EXPECTED_COOL_SKIN)
