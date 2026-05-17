"""Integration tests: plan_emma must invoke the alpha-proof dumper and
produce a directory that the acceptance_harness can render rows 2/3/4 from.

These are SLOWER than the unit tests above (they exercise the real hybrid
optimizer) so they're a separate file you can `-k integration` skip.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO_ROOT / "research" / "v4-build" / "example-harness"
EMMA_IMAGE = Path("/srv/woodblock-share/input-images/close_emma_2002_2048.jpg")


@pytest.mark.integration
def test_plan_emma_emits_artifacts_dir_consumed_by_harness(tmp_path: Path) -> None:
    """The whole stack: run plan_emma --synthetic with --artifacts-dir set,
    then point the acceptance harness at the dir and assert rows 2/3/4 are
    populated.
    """
    artifacts = tmp_path / "synth_run"
    hybrid_out = tmp_path / "hybrid.json"
    plan_out = tmp_path / "plan.json"

    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + os.pathsep
        + str(REPO_ROOT / "research" / "v4-build" / "hybrid-optimizer")
        + os.pathsep
        + str(REPO_ROOT / "research" / "v4-build" / "production-solver")
        + os.pathsep
        + str(REPO_ROOT / "research" / "v5-overnight" / "alpha-proof-dumper")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    cmd = [
        sys.executable,
        "-m",
        "chuck_mcp_v2.plan_emma",
        "--synthetic",
        "--output",
        str(hybrid_out),
        "--plan-output",
        str(plan_out),
        "--artifacts-dir",
        str(artifacts),
        "--size",
        "64",
        "--cells",
        "16",
        "--plate-count",
        "6",
        "--target-pulls",
        "6",
        "--max-outer-iters",
        "1",
        "--max-inner-iters",
        "3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=240)
    assert proc.returncode == 0, (
        f"plan_emma failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # The dumper writes its outputs to <artifacts>/. Check the new directories
    # exist.
    assert (artifacts / "plates").is_dir(), (
        f"plan_emma did not invoke alpha-proof-dumper (no plates/ dir).\n"
        f"contents: {sorted(p.name for p in artifacts.iterdir() if p.is_dir())}"
    )
    assert (artifacts / "alphas").is_dir()
    assert (artifacts / "pulls").is_dir()
    assert (artifacts / "proofs").is_dir()

    # Now run the acceptance harness on it.
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    from acceptance_harness.acceptance_harness import render_acceptance_sheet

    sheet = artifacts / "acceptance_sheet.png"
    result = render_acceptance_sheet(plan_output_dir=artifacts, output_path=sheet)
    assert sheet.exists()
    # Row 3: at least 1 plate rendered (we asked for plate-count=6).
    assert result.plate_count_rendered >= 1, (
        f"row 3 not populated: warnings={result.warnings}"
    )
    # Row 4: at least 1 alpha rendered.
    assert result.alpha_count_rendered >= 1, (
        f"row 4 not populated: warnings={result.warnings}"
    )
