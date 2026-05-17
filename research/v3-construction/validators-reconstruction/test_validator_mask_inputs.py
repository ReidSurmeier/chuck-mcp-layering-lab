from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plate_not_composite  # noqa: E402
import run_all_validators  # noqa: E402


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _save_l(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def test_plate_not_composite_scores_white_ink_binary_mask_as_sparse(tmp_path: Path) -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[24:64, 32:72] = 255
    mask_path = tmp_path / "inked_mask.png"
    _save_l(mask_path, mask)

    final = np.full((256, 256, 3), 245, dtype=np.uint8)
    result = plate_not_composite.score(str(mask_path), final, return_components=True)

    assert result["passes"]
    assert 0.01 < result["inked_area_fraction"] < 0.05
    assert result["coverage_spread_badness"] == 0.0


def test_runner_prefers_inked_mask_path_over_plate_preview(tmp_path: Path) -> None:
    final_path = tmp_path / "final.png"
    proof_0 = tmp_path / "proof_0.png"
    proof_1 = tmp_path / "proof_1.png"
    preview_path = tmp_path / "wood_preview.png"
    mask_path = tmp_path / "inked_mask.png"

    final = np.full((256, 256, 3), 245, dtype=np.uint8)
    full_dark_preview = np.full((256, 256, 3), 40, dtype=np.uint8)
    sparse_mask = np.zeros((256, 256), dtype=np.uint8)
    sparse_mask[32:72, 32:72] = 255

    _save_rgb(final_path, final)
    _save_rgb(proof_0, final)
    _save_rgb(proof_1, final)
    _save_rgb(preview_path, full_dark_preview)
    _save_l(mask_path, sparse_mask)

    plan = {
        "plan_id": "mask-input-regression",
        "final_composite": str(final_path),
        "proof_states": [str(proof_0), str(proof_1)],
        "plates": [
            {
                "block_id": 1,
                "plate_preview": str(preview_path),
                "inked_mask": str(mask_path),
                "cells_in_plate": [1],
                "role": "regional_mass",
                "dpi": 300,
            }
        ],
        "cell_role_labels": {1: "regional_mass"},
        "cell_pixel_positions": {},
        "cell_adjacency": {},
    }

    report = run_all_validators.run_all_validators(plan)
    pnc = report["validators"]["plate_not_composite"]["per_plate"][0]

    assert pnc["input_source"] == "inked_mask"
    assert pnc["passes"]
    assert pnc["inked_area_fraction"] < 0.05
