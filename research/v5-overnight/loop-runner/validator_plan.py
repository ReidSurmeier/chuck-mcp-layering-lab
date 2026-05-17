"""Typed validator-plan construction for v5 Emma runs.

This module owns the seam between run artifacts and validator input. Its main
job is to keep **Validator truth** separate from **Review preview** files:
geometry validators receive authoritative mask paths, while plate/contact-sheet
previews remain human-facing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidatorPlanInputs:
    hybrid_result: Path
    production_plan: Path
    artifacts_dir: Path
    job_dir: Path
    input_image: Path


def build_validator_plan(inputs: ValidatorPlanInputs) -> dict[str, Any]:
    """Build the dict consumed by `run_all_validators`.

    The alpha dumper writes `alphas/pull_NNN_alpha.png` in sorted pull order.
    That file is the authoritative printed-area Mask for a solved
    Block/Impression entry. Review previews under `plates/` are attached for
    humans but never used as geometry truth when an `inked_mask` exists.
    """
    plan: dict[str, Any] = {
        "plan_id": f"v5-overnight-{inputs.job_dir.name}",
        "target_image": str(inputs.input_image),
        "plates": [],
        "cell_role_labels": {},
        "cell_pixel_positions": {},
        "cell_adjacency": {},
        "proof_states": [],
    }

    _merge_production_plan(plan, _read_json(inputs.production_plan))
    _merge_hybrid_result(plan, _read_json(inputs.hybrid_result))
    _assign_pull_indices(plan)
    _attach_role_labels(plan)
    _attach_artifacts(plan, inputs.artifacts_dir)

    return plan


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text())
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge_production_plan(plan: dict[str, Any], prod: dict[str, Any]) -> None:
    for plate in prod.get("plates", []):
        block_id = int(plate.get("block_id", 0))
        first_pull = (plate.get("pulls") or [{}])[0]
        plan["plates"].append(
            {
                "block_id": block_id,
                "cells_in_plate": list(plate.get("cell_zone_ids", [])),
                "role": plate.get("role", "regional_mass"),
                "order_step": int(first_pull.get("order_step", block_id) or block_id),
                "pass_index": int(first_pull.get("pass_index", block_id) or block_id),
                "dpi": 300,
            }
        )


def _merge_hybrid_result(plan: dict[str, Any], hyb: dict[str, Any]) -> None:
    plates_by_id = {int(p["block_id"]): p for p in plan["plates"]}
    for plate in hyb.get("plates", []):
        bid = int(plate.get("block_id", 0))
        entry = plates_by_id.get(bid)
        if entry is None:
            entry = {
                "block_id": bid,
                "cells_in_plate": list(plate.get("cell_zone_ids", [])),
                "role": plate.get("role", "regional_mass"),
                "dpi": 300,
            }
            plan["plates"].append(entry)
            plates_by_id[bid] = entry

        if plate.get("cell_zone_ids"):
            entry["cells_in_plate"] = list(plate["cell_zone_ids"])
        if plate.get("role"):
            entry["role"] = plate["role"]
        if plate.get("pass_index") is not None:
            entry["pass_index"] = int(plate["pass_index"])
        if plate.get("pigment_id"):
            entry["pigment_id"] = plate["pigment_id"]
        if plate.get("opacity") is not None:
            entry["opacity"] = plate["opacity"]
        if plate.get("dilution") is not None:
            entry["dilution"] = plate["dilution"]


def _assign_pull_indices(plan: dict[str, Any]) -> None:
    for pull_index, plate in enumerate(sorted(plan["plates"], key=_sort_key), start=1):
        plate["pull_index"] = pull_index


def _attach_role_labels(plan: dict[str, Any]) -> None:
    role_labels: dict[int, str] = {}
    for plate in plan["plates"]:
        for cid in plate.get("cells_in_plate", []):
            role_labels[int(cid)] = plate["role"]
    plan["cell_role_labels"] = role_labels


def _attach_artifacts(plan: dict[str, Any], artifacts_dir: Path) -> None:
    _attach_plate_previews(plan, artifacts_dir)
    _attach_masks(plan, artifacts_dir)
    proofs = _proof_paths(artifacts_dir)
    plan["proof_states"] = proofs
    _attach_pull_previews(plan, artifacts_dir, proofs)
    _attach_final_composite(plan, artifacts_dir, proofs)


def _attach_plate_previews(plan: dict[str, Any], artifacts_dir: Path) -> None:
    plates_dir = artifacts_dir / "plates"
    if not plates_dir.is_dir():
        return
    for plate in plan["plates"]:
        bid = int(plate["block_id"])
        preview = _first_existing(
            [
                plates_dir / f"block_{bid:02d}.preview.png",
                plates_dir / f"plate_{bid:02d}.preview.png",
                plates_dir / f"block_{bid}.preview.png",
                plates_dir / f"plate_{bid}.preview.png",
            ]
        )
        if preview:
            plate["plate_preview"] = str(preview)

        svg = _first_existing(
            [plates_dir / f"block_{bid:02d}.svg", plates_dir / f"plate_{bid:02d}.svg"]
        )
        if svg:
            plate["plate_svg"] = str(svg)


def _attach_masks(plan: dict[str, Any], artifacts_dir: Path) -> None:
    alpha_dir = artifacts_dir / "alpha_masks"
    alphas_dir = artifacts_dir / "alphas"
    for plate in plan["plates"]:
        bid = int(plate["block_id"])
        pull_index = int(plate.get("pull_index", bid) or bid)
        mask = _first_existing(
            [
                alphas_dir / f"pull_{pull_index:03d}_alpha.png",
                alphas_dir / f"pull_{bid:03d}_alpha.png",
                alpha_dir / f"alpha_{pull_index:02d}.png",
                alpha_dir / f"alpha_{bid:02d}.png",
                alpha_dir / f"alpha_{bid}.png",
            ]
        )
        if mask:
            plate["alpha_preview"] = str(mask)
            plate["inked_mask"] = str(mask)


def _proof_paths(artifacts_dir: Path) -> list[str]:
    return (
        sorted(str(x) for x in artifacts_dir.glob("cumulative_pull_*.png"))
        or sorted(str(x) for x in artifacts_dir.glob("pull_*.png"))
        or sorted(str(x) for x in artifacts_dir.glob("proof_*.png"))
    )


def _attach_pull_previews(
    plan: dict[str, Any], artifacts_dir: Path, proofs: list[str]
) -> None:
    pulls_dir = artifacts_dir / "pulls"
    ordered = sorted(plan["plates"], key=lambda x: int(x.get("pull_index", 0) or 0))
    for i, plate in enumerate(ordered):
        pull_index = int(plate.get("pull_index", i + 1) or (i + 1))
        pull = _first_existing(
            [
                pulls_dir / f"pull_{pull_index:03d}.png",
                artifacts_dir / f"pull_{pull_index:03d}.png",
            ]
        )
        if pull:
            plate["pull_preview"] = str(pull)
        elif i < len(proofs):
            plate["pull_preview"] = proofs[i]


def _attach_final_composite(
    plan: dict[str, Any], artifacts_dir: Path, proofs: list[str]
) -> None:
    final_composite = artifacts_dir / "final_composite.png"
    if final_composite.exists():
        plan["final_composite"] = str(final_composite)
    elif proofs:
        plan["final_composite"] = proofs[-1]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _sort_key(plate: dict[str, Any]) -> tuple[int, int]:
    order = plate.get("pass_index") or plate.get("order_step") or plate.get("block_id") or 0
    return int(order), int(plate.get("block_id", 0) or 0)
