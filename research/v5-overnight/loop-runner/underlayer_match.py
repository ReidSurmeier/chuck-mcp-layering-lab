#!/usr/bin/env python3
"""Underlayer match: compare our rendered underlayer plate(s) against Reid's
hand-annotated 2026-05-16 underlayer methodology PNG.

Strategy:
1. Load reference annotation. Reduce to grayscale, threshold to a binary mask
   of "where underlayer ink should land" (dark pixels = annotated regions).
2. From the artifacts dir, find every plate with role starting "underlayer".
3. Build a union mask of inked area (from plate preview alpha or alpha_masks/).
4. Align both to the same shape (downscale to 256 short side).
5. Report IoU(reference_mask, union_inked_mask) as match_pct (0..100).

This is approximate. If the reference image isn't readable, returns 0.0.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image


def _to_binary(arr: np.ndarray, *, dark_is_signal: bool = True) -> np.ndarray:
    if arr.ndim == 3:
        # luminance
        arr = arr[..., :3].mean(axis=-1)
    arr = arr.astype(np.float32)
    # adaptive: threshold at otsu-ish midpoint
    thr = np.percentile(arr, 50)
    mask = (arr < thr) if dark_is_signal else (arr > thr)
    return mask.astype(np.uint8)


def _resize(arr: np.ndarray, short_side: int = 256) -> np.ndarray:
    img = Image.fromarray(arr if arr.dtype == np.uint8 else arr.astype(np.uint8))
    h, w = arr.shape[:2]
    if min(h, w) == 0:
        return arr
    scale = short_side / float(min(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return np.asarray(img.resize((nw, nh), Image.Resampling.NEAREST))


def _load_mask_from(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        # alpha-only PNGs: prefer alpha; else luminance < bg
        if img.mode == "RGBA":
            a = np.asarray(img)[:, :, 3]
            return (a > 32).astype(np.uint8)
        arr = np.asarray(img.convert("RGB"))
        return _to_binary(arr, dark_is_signal=True)
    except Exception as e:
        print(f"WARN load {path}: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    out = {"match_pct": 0.0, "iou": 0.0, "ref_path": str(args.reference),
           "n_underlayer_plates": 0, "notes": []}

    if not args.reference.exists():
        out["notes"].append(f"reference missing: {args.reference}")
        args.output.write_text(json.dumps(out, indent=2))
        return 0

    try:
        ref_img = np.asarray(Image.open(args.reference).convert("RGB"))
    except Exception as e:
        out["notes"].append(f"ref load failed: {e}")
        args.output.write_text(json.dumps(out, indent=2))
        return 0

    ref_mask = _to_binary(ref_img, dark_is_signal=True)
    ref_mask = _resize(ref_mask, short_side=256)

    # gather underlayer plate masks from validator_plan.json if present
    vp_path = args.job_dir / "validator_plan.json"
    union = np.zeros_like(ref_mask)
    n_under = 0
    if vp_path.exists():
        try:
            plan = json.loads(vp_path.read_text())
        except Exception as e:
            out["notes"].append(f"validator_plan parse: {e}")
            plan = {}
        for p in plan.get("plates", []):
            role = str(p.get("role", ""))
            if not role.startswith("underlayer"):
                continue
            for key in ("plate_preview", "alpha_preview"):
                cand = p.get(key)
                if not cand:
                    continue
                m = _load_mask_from(Path(cand))
                if m is None:
                    continue
                m_resized = _resize(m, short_side=256)
                # pad/crop to ref shape
                rh, rw = ref_mask.shape
                mh, mw = m_resized.shape
                if (mh, mw) != (rh, rw):
                    new = np.zeros_like(ref_mask)
                    h = min(rh, mh); w = min(rw, mw)
                    new[:h, :w] = m_resized[:h, :w]
                    m_resized = new
                union = np.maximum(union, m_resized)
                n_under += 1
                break

    # if no underlayer plates found, try first 3 cumulative proofs as a proxy
    if n_under == 0:
        proofs = sorted((args.artifacts_dir).glob("cumulative_pull_*.png"))[:3]
        for pp in proofs:
            m = _load_mask_from(pp)
            if m is None:
                continue
            m_resized = _resize(m, short_side=256)
            rh, rw = ref_mask.shape
            mh, mw = m_resized.shape
            if (mh, mw) != (rh, rw):
                new = np.zeros_like(ref_mask)
                h = min(rh, mh); w = min(rw, mw)
                new[:h, :w] = m_resized[:h, :w]
                m_resized = new
            union = np.maximum(union, m_resized)
            n_under += 1
        if n_under:
            out["notes"].append(f"fell back to first {n_under} cumulative proofs")

    out["n_underlayer_plates"] = int(n_under)
    if n_under == 0:
        out["notes"].append("no underlayer plates or proofs found")
        args.output.write_text(json.dumps(out, indent=2))
        return 0

    inter = int((union & ref_mask).sum())
    uni = int((union | ref_mask).sum())
    iou = (inter / uni) if uni > 0 else 0.0
    out["iou"] = float(iou)
    out["match_pct"] = float(iou * 100.0)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"underlayer match: {out['match_pct']:.2f}% iou={iou:.4f} n_under={n_under}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
