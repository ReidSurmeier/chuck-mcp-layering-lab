#!/usr/bin/env python3
"""Visual methodology gate for Chuck Close-style progressive proofs.

This is intentionally stricter than the numerical validators. It answers the
question the user actually asked: does the current proof sequence visually
develop like the reference methodology, or is it just dot-cell output?
"""
from __future__ import annotations

import argparse
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage import feature, measure, morphology
except Exception:  # pragma: no cover - test env includes skimage
    feature = None
    measure = None
    morphology = None


DEFAULT_REFERENCE = Path(
    "/srv/woodblock-share/chuck-mcp-iterations/references/"
    "2026-05-14_chuck-close-progressive-proof-screenshot.png"
)


@dataclass
class TileMetrics:
    active_fraction: float
    component_count: int
    median_component_area: float
    component_area_cv: float
    dot_like_fraction: float
    edge_density: float


@dataclass
class MethodologyReport:
    passes: bool
    verdict: str
    gates: dict[str, dict[str, Any]]
    reference_final: TileMetrics
    current_final: TileMetrics
    alpha_mask_summary: dict[str, float]
    notes: list[str]


def run_gate(artifacts_dir: Path, reference_sheet: Path = DEFAULT_REFERENCE) -> MethodologyReport:
    reference_tiles = _load_reference_tiles(reference_sheet)
    current_tiles = _load_current_proof_tiles(artifacts_dir)
    if not reference_tiles:
        raise ValueError(f"no reference tiles loaded from {reference_sheet}")
    if not current_tiles:
        raise ValueError(f"no current proof tiles loaded from {artifacts_dir}")

    ref_final = _metrics(reference_tiles[-1])
    cur_final = _metrics(current_tiles[-1])
    alpha_summary = _alpha_mask_summary(artifacts_dir)

    ref_mask = _foreground_mask(reference_tiles[-1])
    cur_mask = _foreground_mask(current_tiles[-1])
    silhouette_iou = _mask_iou(_resize_mask(cur_mask, ref_mask.shape), ref_mask)

    ref_progression = [_metrics(t).active_fraction for t in reference_tiles]
    cur_progression = [_metrics(t).active_fraction for t in current_tiles]
    progression_ratio = _safe_ratio(cur_progression[-1], ref_progression[-1])

    dot_artifact_score = _dot_artifact_score(cur_final)

    gates = {
        "final_coverage_floor": {
            "passes": progression_ratio >= 0.55,
            "current_vs_reference_ratio": progression_ratio,
            "threshold": 0.55,
            "current_active_fraction": cur_final.active_fraction,
            "reference_active_fraction": ref_final.active_fraction,
        },
        "reference_silhouette_overlap": {
            "passes": silhouette_iou >= 0.35,
            "iou": silhouette_iou,
            "threshold": 0.35,
        },
        "dot_cell_artifact_rejection": {
            "passes": dot_artifact_score <= 0.45,
            "dot_artifact_score": dot_artifact_score,
            "threshold": 0.45,
            "dot_like_fraction": cur_final.dot_like_fraction,
            "component_area_cv": cur_final.component_area_cv,
        },
        "final_edge_complexity_floor": {
            "passes": _safe_ratio(cur_final.edge_density, ref_final.edge_density) >= 0.45,
            "current_vs_reference_ratio": _safe_ratio(cur_final.edge_density, ref_final.edge_density),
            "threshold": 0.45,
            "current_edge_density": cur_final.edge_density,
            "reference_edge_density": ref_final.edge_density,
        },
        "alpha_mask_dot_cell_rejection": {
            "passes": alpha_summary["mean_dot_artifact_score"] <= 0.40,
            "mean_dot_artifact_score": alpha_summary["mean_dot_artifact_score"],
            "threshold": 0.40,
            "mean_dot_like_fraction": alpha_summary["mean_dot_like_fraction"],
            "mask_count": alpha_summary["mask_count"],
        },
        "alpha_mask_connected_region_floor": {
            "passes": alpha_summary["mean_largest_component_fraction"] >= 0.35,
            "mean_largest_component_fraction": alpha_summary["mean_largest_component_fraction"],
            "threshold": 0.35,
            "mask_count": alpha_summary["mask_count"],
        },
    }
    passes = all(g["passes"] for g in gates.values())
    notes = _failure_notes(gates)
    verdict = "PASS" if passes else "FAIL"
    return MethodologyReport(
        passes=passes,
        verdict=verdict,
        gates=gates,
        reference_final=ref_final,
        current_final=cur_final,
        alpha_mask_summary=alpha_summary,
        notes=notes,
    )


def _load_reference_tiles(path: Path) -> list[Image.Image]:
    path = _resolve_path(path)
    img = Image.open(path).convert("RGB")
    cols, rows = 4, 2
    cell_w = img.width // cols
    cell_h = img.height // rows
    tiles = []
    for row in range(rows):
        for col in range(cols):
            tiles.append(
                img.crop(
                    (
                        col * cell_w,
                        row * cell_h,
                        (col + 1) * cell_w,
                        (row + 1) * cell_h,
                    )
                )
            )
    return tiles


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.parent.exists():
        return path
    target = _normalized_filename(path.name)
    for candidate in path.parent.iterdir():
        if candidate.is_file() and _normalized_filename(candidate.name) == target:
            return candidate
    return path


def _normalized_filename(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).split())


def _load_current_proof_tiles(artifacts_dir: Path, count: int = 8) -> list[Image.Image]:
    candidates = sorted((artifacts_dir / "pulls").glob("pull_*.png"))
    if not candidates:
        candidates = sorted(artifacts_dir.glob("cumulative_pull_*.png"))
    if not candidates:
        candidates = sorted(artifacts_dir.glob("proof_*.png"))
    chosen = _evenly_spaced(candidates, count)
    return [Image.open(p).convert("RGB") for p in chosen]


def _load_alpha_tiles(artifacts_dir: Path) -> list[Image.Image]:
    candidates = sorted((artifacts_dir / "alphas").glob("pull_*_alpha.png"))
    if not candidates:
        candidates = sorted((artifacts_dir / "alpha_masks").glob("alpha_*.png"))
    return [Image.open(p).convert("RGB") for p in candidates]


def _evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    idxs = np.linspace(0, len(paths) - 1, num=count).round().astype(int).tolist()
    seen = []
    for idx in idxs:
        if idx not in seen:
            seen.append(idx)
    return [paths[i] for i in seen]


def _metrics(img: Image.Image) -> TileMetrics:
    mask = _foreground_mask(img)
    active_fraction = float(mask.mean())
    component_areas: list[float] = []
    dot_like = 0
    if measure is not None:
        labels = measure.label(mask)
        for region in measure.regionprops(labels):
            area = float(region.area)
            if area < 12:
                continue
            component_areas.append(area)
            perimeter = max(float(region.perimeter), 1.0)
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.62 and region.extent > 0.45:
                dot_like += 1
    component_count = len(component_areas)
    median_area = float(np.median(component_areas)) if component_areas else 0.0
    area_cv = _coefficient_of_variation(component_areas)
    dot_like_fraction = float(dot_like / component_count) if component_count else 0.0
    edge_density = _edge_density(img)
    return TileMetrics(
        active_fraction=active_fraction,
        component_count=component_count,
        median_component_area=median_area,
        component_area_cv=area_cv,
        dot_like_fraction=dot_like_fraction,
        edge_density=edge_density,
    )


def _alpha_mask_summary(artifacts_dir: Path) -> dict[str, float]:
    tiles = _load_alpha_tiles(artifacts_dir)
    if not tiles:
        return {
            "mask_count": 0.0,
            "mean_dot_artifact_score": 1.0,
            "mean_dot_like_fraction": 1.0,
            "mean_largest_component_fraction": 0.0,
        }
    metrics = [_metrics(t) for t in tiles]
    largest = [_largest_component_fraction(_foreground_mask(t)) for t in tiles]
    return {
        "mask_count": float(len(tiles)),
        "mean_dot_artifact_score": float(np.mean([_dot_artifact_score(m) for m in metrics])),
        "mean_dot_like_fraction": float(np.mean([m.dot_like_fraction for m in metrics])),
        "mean_largest_component_fraction": float(np.mean(largest)),
    }


def _largest_component_fraction(mask: np.ndarray) -> float:
    if measure is None or not mask.any():
        return 0.0
    labels = measure.label(mask)
    areas = [float(region.area) for region in measure.regionprops(labels)]
    total = float(mask.sum())
    return max(areas) / total if areas and total else 0.0


def _foreground_mask(img: Image.Image) -> np.ndarray:
    small = img.copy()
    small.thumbnail((384, 384), Image.Resampling.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    border = np.concatenate(
        [
            arr[:8].reshape(-1, 3),
            arr[-8:].reshape(-1, 3),
            arr[:, :8].reshape(-1, 3),
            arr[:, -8:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=-1)
    threshold = max(0.055, float(np.percentile(dist, 82)) * 0.55)
    mask = dist > threshold
    if morphology is not None:
        mask = morphology.remove_small_objects(mask, min_size=12)
    return mask.astype(bool)


def _edge_density(img: Image.Image) -> float:
    small = img.convert("L")
    small.thumbnail((384, 384), Image.Resampling.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    if feature is not None:
        edges = feature.canny(arr, sigma=1.2)
        return float(edges.mean())
    gy, gx = np.gradient(arr)
    return float((np.sqrt(gx * gx + gy * gy) > 0.08).mean())


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else 0.0


def _coefficient_of_variation(vals: list[float]) -> float:
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float32)
    mean = float(arr.mean())
    return float(arr.std() / mean) if mean > 1e-9 else 0.0


def _dot_artifact_score(metrics: TileMetrics) -> float:
    uniform_component_score = 1.0 - min(metrics.component_area_cv, 1.0)
    sparse_score = 1.0 if metrics.active_fraction < 0.35 else 0.0
    return float(
        np.clip(
            0.55 * metrics.dot_like_fraction
            + 0.30 * uniform_component_score
            + 0.15 * sparse_score,
            0.0,
            1.0,
        )
    )


def _failure_notes(gates: dict[str, dict[str, Any]]) -> list[str]:
    notes = []
    if not gates["final_coverage_floor"]["passes"]:
        notes.append("final proof coverage is far below the reference proof sequence")
    if not gates["reference_silhouette_overlap"]["passes"]:
        notes.append("final proof foreground does not overlap the reference portrait silhouette")
    if not gates["dot_cell_artifact_rejection"]["passes"]:
        notes.append("current proof reads as dot/cell centroid artifacts")
    if not gates["final_edge_complexity_floor"]["passes"]:
        notes.append("final proof lacks reference-like edge/detail complexity")
    if not gates["alpha_mask_dot_cell_rejection"]["passes"]:
        notes.append("individual masks read as dot/cell centroid artifacts")
    if not gates["alpha_mask_connected_region_floor"]["passes"]:
        notes.append("individual masks are not organized as connected carved regions")
    return notes


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--reference-sheet", default=DEFAULT_REFERENCE, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    report = run_gate(args.artifacts_dir, args.reference_sheet)
    data = asdict(report)
    text = json.dumps(data, indent=2, default=_json_default)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(report.verdict)
    print(text)
    return 0 if report.passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
