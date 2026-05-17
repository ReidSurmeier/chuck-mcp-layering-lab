"""Production ingestion + validation for user-painted chuck-mcp v2 PSDs.

Parses a Photoshop-painted PSD, validates kento integrity + binary-mask quality,
returns Dict[slot_name, np.ndarray[uint8]] suitable for hand-off to the JAX solver.

Usage (CLI):
    python ingest_masks.py --psd user.psd --manifest template.json --out masks.npz

Usage (programmatic):
    from ingest_masks import ingest_painted_psd, ValidationError
    masks, report = ingest_painted_psd('user.psd', manifest_json='template.json')
    solver.fit_underlayers(masks=masks)

Tested: psd-tools==1.17.0 on Linux 6.6.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage


class ValidationError(Exception):
    """Raised when the painted PSD fails a hard validation rule."""


@dataclass
class ValidationReport:
    """Per-slot validation results + global warnings/errors."""

    canvas: dict = field(default_factory=dict)
    kento: dict = field(default_factory=dict)
    slots: Dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "canvas": self.canvas,
            "kento": self.kento,
            "slots": self.slots,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _layer_to_canvas_array(layer, canvas_w: int, canvas_h: int) -> Optional[np.ndarray]:
    """Render a layer into a canvas-sized grayscale numpy array.

    Returns None if the layer is empty (no painted pixels).
    """
    pil = layer.composite()
    if pil is None:
        return None
    arr = np.array(pil.convert("L"))
    h, w = arr.shape
    # Layer may be smaller than canvas; place at layer.offset.
    if (w, h) == (canvas_w, canvas_h):
        return arr
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    ox, oy = layer.offset
    # Clip to canvas bounds (defensive)
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(canvas_w, ox + w)
    y1 = min(canvas_h, oy + h)
    if x1 > x0 and y1 > y0:
        sx0 = x0 - ox
        sy0 = y0 - oy
        canvas[y0:y1, x0:x1] = arr[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    return canvas


def _is_binary(arr: np.ndarray, tolerance: int = 2, min_pct: float = 99.0) -> tuple[bool, float]:
    """Check if a grayscale array clusters at 0 and 255.

    Returns (passed, percentage_clustered).
    """
    flat = arr.flatten()
    clustered = ((flat <= tolerance) | (flat >= 255 - tolerance)).mean() * 100.0
    return clustered >= min_pct, clustered


def _overlap_with_zones(
    mask_bool: np.ndarray, zones: list[dict]
) -> tuple[int, list[str]]:
    """Count painted pixels that fall inside any kento no-paint zone.

    Returns (total_overlap_pixels, list_of_violated_zone_labels).
    """
    H, W = mask_bool.shape
    total = 0
    violated = []
    for zone in zones:
        x0, y0, x1, y1 = zone["bbox"]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = mask_bool[y0:y1, x0:x1]
        if region.any():
            count = int(region.sum())
            total += count
            violated.append(f"{zone['label']}({count}px)")
    return total, violated


def ingest_painted_psd(
    psd_path: str | Path,
    manifest: Optional[dict] = None,
    manifest_json: Optional[str | Path] = None,
    strict: bool = True,
) -> tuple[Dict[str, np.ndarray], ValidationReport]:
    """Open a user-painted PSD and return binary masks + validation report.

    Args:
        psd_path: Path to the painted .psd.
        manifest: Pre-loaded manifest dict (see gen_template.make_manifest_dict).
        manifest_json: Or path to a manifest JSON file.
        strict: If True, raise ValidationError on any hard violation.

    Returns:
        (masks, report) where masks is {slot_name: uint8 array, 1=painted, 0=unpainted}
        and report is a ValidationReport with details.

    Raises:
        ValidationError: If strict=True and any hard rule fails.
    """
    if manifest is None and manifest_json is not None:
        manifest = json.loads(Path(manifest_json).read_text())
    if manifest is None:
        raise ValueError("Must provide manifest or manifest_json")

    report = ValidationReport()
    psd = PSDImage.open(str(psd_path))

    # --- Canvas dimension check ---
    expected_w = manifest["canvas"]["width_px"]
    expected_h = manifest["canvas"]["height_px"]
    actual_w, actual_h = psd.width, psd.height
    report.canvas = {
        "expected": [expected_w, expected_h],
        "actual": [actual_w, actual_h],
        "match": (actual_w, actual_h) == (expected_w, expected_h),
    }
    if not report.canvas["match"]:
        report.errors.append(
            f"canvas dimensions changed: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
        )

    # --- Kento layer presence ---
    layers = list(psd)
    kento_layer = next(
        (L for L in layers if "kento_marks" in L.name), None
    )
    if kento_layer is None:
        report.errors.append("kento_marks reference layer missing or renamed")
        report.kento = {"present": False}
    else:
        report.kento = {"present": True, "name": kento_layer.name}
        # Optional: bitwise comparison against original kento. Skipped by default
        # because user may have moved or hidden it but not painted on it. We
        # detect paint-over-kento via the no_paint_zones check on slot masks.

    # --- Slot extraction + per-slot validation ---
    slot_specs = {s["name"]: s for s in manifest["slots"]}
    threshold_pct = manifest["validation"]["binary_threshold_pct"]
    tolerance = manifest["validation"]["binary_tolerance"]
    min_cov = manifest["validation"]["min_coverage_pct"]
    max_cov = manifest["validation"]["max_coverage_pct"]
    no_paint_zones = manifest["kento"]["no_paint_zones"]

    masks: Dict[str, np.ndarray] = {}
    for layer in layers:
        if not layer.name.startswith("slot_"):
            continue
        if layer.name not in slot_specs:
            report.warnings.append(f"unknown slot {layer.name!r} (not in manifest)")
            continue

        slot_report: dict = {"name": layer.name}
        arr = _layer_to_canvas_array(layer, expected_w, expected_h)
        # Treat as empty if: layer.composite() is None, OR layer was the 1x1
        # placeholder we wrote in gen_template (untouched by user), OR the
        # entire canvas is uniform zero (user erased all pixels).
        is_placeholder_size = layer.size == (1, 1)
        is_uniform_zero = arr is not None and not arr.any()
        if arr is None or is_placeholder_size or is_uniform_zero:
            slot_report["status"] = "empty"
            report.slots[layer.name] = slot_report
            report.warnings.append(f"{layer.name}: empty (no painted pixels)")
            continue

        # 1. Binary check
        is_bin, clustered_pct = _is_binary(arr, tolerance=tolerance, min_pct=threshold_pct)
        slot_report["binary_clustered_pct"] = round(clustered_pct, 3)
        if not is_bin:
            slot_report["status"] = "non_binary"
            report.errors.append(
                f"{layer.name}: not binary ({clustered_pct:.2f}% clustered at extremes, "
                f"need ≥{threshold_pct}%). User likely used soft brush — switch to hard-edge."
            )
            report.slots[layer.name] = slot_report
            continue

        # Threshold at 127 → boolean mask
        mask_bool = arr > 127
        coverage_pct = mask_bool.mean() * 100.0
        slot_report["coverage_pct"] = round(coverage_pct, 3)

        # 2. Coverage sanity
        if coverage_pct < min_cov:
            slot_report["status"] = "underpainted"
            report.errors.append(
                f"{layer.name}: coverage {coverage_pct:.3f}% below minimum {min_cov}%. "
                "Probably an accidental dot — please re-mask."
            )
            report.slots[layer.name] = slot_report
            continue
        if coverage_pct > max_cov:
            slot_report["status"] = "overpainted"
            report.errors.append(
                f"{layer.name}: coverage {coverage_pct:.3f}% above maximum {max_cov}%. "
                "Probably forgot to mask off — please clip."
            )
            report.slots[layer.name] = slot_report
            continue

        # 3. Kento no-paint-zone overlap
        overlap_px, violated = _overlap_with_zones(mask_bool, no_paint_zones)
        slot_report["kento_overlap_px"] = overlap_px
        if overlap_px > 0:
            slot_report["status"] = "kento_violation"
            report.errors.append(
                f"{layer.name}: {overlap_px}px painted over kento marks "
                f"({', '.join(violated)}). Registration will fail — please clear corners."
            )
            report.slots[layer.name] = slot_report
            continue

        # All checks passed
        slot_report["status"] = "ok"
        report.slots[layer.name] = slot_report
        masks[layer.name] = mask_bool.astype(np.uint8)

    # --- Missing-slot check ---
    expected_slot_names = set(slot_specs.keys())
    received_slot_names = {n for n in masks.keys()}
    for missing in sorted(expected_slot_names - received_slot_names - {
        n for n, s in report.slots.items() if s.get("status") == "empty"
    }):
        # Already flagged as empty or error above; just compile missing-or-errored
        pass

    missing_or_empty = sorted(expected_slot_names - received_slot_names)
    if missing_or_empty:
        report.warnings.append(
            f"slots without valid masks: {missing_or_empty}. "
            "Solver will skip these unless they were declared essential."
        )

    if strict and not report.ok:
        raise ValidationError(
            "PSD validation failed:\n  - " + "\n  - ".join(report.errors)
        )

    return masks, report


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest chuck-mcp v2 painted PSD")
    p.add_argument("--psd", required=True, help="Path to user-painted PSD")
    p.add_argument("--manifest", required=True, help="Path to template manifest JSON")
    p.add_argument("--out", required=True, help="Output .npz with masks")
    p.add_argument("--report", default=None, help="Optional report JSON output")
    p.add_argument("--no-strict", action="store_true", help="Do not raise on errors")
    args = p.parse_args()

    try:
        masks, report = ingest_painted_psd(
            psd_path=args.psd,
            manifest_json=args.manifest,
            strict=not args.no_strict,
        )
    except ValidationError as exc:
        print(f"VALIDATION FAILED:\n{exc}")
        return 2

    np.savez_compressed(args.out, **masks)
    print(f"Wrote {len(masks)} masks → {args.out}")
    for name, mask in masks.items():
        coverage = mask.mean() * 100
        print(f"  {name}: {mask.shape}, coverage={coverage:.2f}%")

    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Wrote report → {args.report}")
    if report.warnings:
        print(f"Warnings: {len(report.warnings)}")
        for w in report.warnings:
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
