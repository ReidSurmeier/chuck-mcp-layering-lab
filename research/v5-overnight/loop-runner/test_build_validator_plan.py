from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parent / "build_validator_plan.py"


def _save_l(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((16, 16), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_validator_plan_maps_alpha_masks_by_sorted_pull_order(tmp_path: Path) -> None:
    job = tmp_path / "job"
    artifacts = job / "artifacts"
    hybrid_result = job / "hybrid_result.json"
    production_plan = job / "production_plan.json"
    input_image = tmp_path / "input.png"
    output = job / "validator_plan.json"

    _save_rgb(input_image, 245)
    _save_rgb(artifacts / "final_composite.png", 240)
    _save_rgb(artifacts / "plates" / "block_07.preview.png", 100)
    _save_rgb(artifacts / "plates" / "block_17.preview.png", 100)
    _save_l(artifacts / "alphas" / "pull_001_alpha.png", 255)
    _save_l(artifacts / "alphas" / "pull_002_alpha.png", 128)
    _save_rgb(artifacts / "pulls" / "pull_001.png", 220)
    _save_rgb(artifacts / "pulls" / "pull_002.png", 210)

    hybrid_result.write_text(json.dumps({
        "plates": [
            {
                "block_id": 17,
                "pass_index": 37,
                "cell_zone_ids": [17],
                "role": "regional_mass",
            },
            {
                "block_id": 7,
                "pass_index": 7,
                "cell_zone_ids": [7],
                "role": "local_chroma",
            },
        ]
    }))
    production_plan.write_text(json.dumps({"plates": []}))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-result",
            str(hybrid_result),
            "--production-plan",
            str(production_plan),
            "--artifacts-dir",
            str(artifacts),
            "--job-dir",
            str(job),
            "--input-image",
            str(input_image),
            "--output",
            str(output),
        ],
        check=True,
    )

    plan = json.loads(output.read_text())
    by_id = {p["block_id"]: p for p in plan["plates"]}

    assert by_id[7]["pull_index"] == 1
    assert by_id[7]["inked_mask"].endswith("pull_001_alpha.png")
    assert by_id[7]["pull_preview"].endswith("pull_001.png")
    assert by_id[17]["pull_index"] == 2
    assert by_id[17]["inked_mask"].endswith("pull_002_alpha.png")
    assert by_id[17]["pull_preview"].endswith("pull_002.png")
