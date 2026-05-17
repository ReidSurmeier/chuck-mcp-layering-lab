"""
opus_cell_id_extractor.py — call Opus 4.7 vision via `claude -p`.

Single public entry point:

    extract_cell_ids_from_overlay(
        overlay_png_path, region_names, max_cell_id, ...,
    ) -> ExtractionResult

We delegate to the existing claude_p.py transport (shared with chuck-mcp's
intent translator) so this benchmark is billed against Reid's Max plan,
not against an API key.

The prompt asks Opus to look at a labeled SNIC-cell overlay of a face
portrait and emit, for each named region, the list of integer cell IDs
that anatomically belong to that region. We pass the overlay image as a
file path that the CLI loads (claude -p accepts `<image:path>` markers in
the prompt body via the `[image]` form), and we constrain output via a
strict JSON schema (one array<int> per region).

Failure modes are explicit:
    OpusExtractionError       — caller treats it as a single-overlay miss
    (subprocess failed, schema invalid, etc.)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_V3_TRANSPORT = _HERE.parent.parent / "v3-construction" / "claude-p-transport"
sys.path.insert(0, str(_V3_TRANSPORT))

import claude_p  # noqa: E402
from claude_p import (  # noqa: E402
    DEFAULT_MAX_TURNS,
    ClaudePError,
    ClaudeResult,
    translate_intent_prompt,
)


# Monkeypatch the transport so Opus can use the Read tool to load the
# overlay image off disk (we cannot inline base64 — overlays are ~2MB
# even after downsizing, which exceeds practical prompt size). The Read
# tool is read-only and has no side effects.
_ORIG_BUILD_ARGV = claude_p._build_argv


def _vision_build_argv(prompt: str, schema: dict, *,
                       max_turns: int, system_prompt: str) -> list[str]:
    argv = _ORIG_BUILD_ARGV(
        prompt, schema, max_turns=max_turns, system_prompt=system_prompt,
    )
    # Strip the global Read-disallowance + add Read to the allow-list.
    #
    # IMPORTANT: --allowedTools is variadic and will swallow the trailing
    # positional prompt if placed too close to it. Insert near the front
    # of the argv (right after `-p`) so it can only bind to "Read".
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--disallowedTools" and i + 1 < len(argv):
            i += 2
            continue
        out.append(tok)
        i += 1
    # Insert --allowedTools Read right after `-p` so it is bounded by the
    # next named flag (--output-format ...) and cannot eat the prompt.
    insert_at = 2 if len(out) >= 2 and out[1] == "-p" else len(out)
    out.insert(insert_at, "--allowedTools")
    out.insert(insert_at + 1, "Read")
    return out


claude_p._build_argv = _vision_build_argv


class OpusExtractionError(Exception):
    """Wraps any failure from claude -p when extracting cell IDs."""


@dataclass
class ExtractionResult:
    image_id: str
    predictions: dict[str, list[int]]
    cost_usd: float
    duration_ms: int
    session_id: str | None
    raw_envelope: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema + prompt builders
# ---------------------------------------------------------------------------


def _build_schema(region_names: list[str]) -> dict[str, Any]:
    """One required int-array per region — strict, no extras."""
    properties = {
        name: {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "description": f"SNIC cell IDs that lie inside the '{name}' region.",
        }
        for name in region_names
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(region_names),
        "additionalProperties": False,
    }


_SYSTEM_PROMPT = (
    "You are looking at a portrait photograph that has been segmented into "
    "labeled superpixel cells. Each cell is outlined in black and has a "
    "small integer ID drawn at its centroid. Your job is to assign each "
    "anatomical region (left_cheek, hair, etc.) the list of cell IDs that "
    "physically lie inside that region. Use the cell ID labels you can read "
    "in the overlay. Return ONLY a strict JSON object matching the supplied "
    "schema — one integer array per region name. Empty arrays are allowed "
    "for regions that are not visible. Do not invent cell IDs. Do not write "
    "prose.\n\n"
    "EFFICIENCY: Use the Read tool ONCE to load the overlay. Then output the "
    "JSON immediately. Do not Read the overlay multiple times. Do not Read "
    "any other files. Do not ask clarifying questions."
)


def _build_prompt(overlay_path: Path, region_names: list[str],
                  max_cell_id: int) -> str:
    """File-path prompt — Opus uses the Read tool to load the overlay.

    We pass an absolute path; the system prompt already tells Opus to use
    the Read tool. (We can't inline base64 — overlays exceed practical
    prompt size after encoding.)
    """
    if not overlay_path.exists():
        raise FileNotFoundError(overlay_path)

    region_block = ", ".join(region_names)
    # Absolute path — claude's cwd may differ from ours.
    abs_path = overlay_path.resolve()

    return (
        f"Use the Read tool to load this overlay image:\n"
        f"  {abs_path}\n\n"
        f"It is a face portrait segmented into labeled superpixel cells. "
        f"Each cell is outlined and has a small integer ID drawn at its "
        f"centroid.\n\n"
        f"Assign SNIC cell IDs to the following anatomical regions:\n"
        f"  {region_block}\n\n"
        f"Cell IDs in this overlay range from 0 to {max_cell_id} inclusive. "
        f"Each cell appears at most once in the overlay. Return one JSON "
        f"object whose keys are the region names above and whose values are "
        f"sorted lists of integer cell IDs that lie inside that region. "
        f"If a region is not visible (for example 'hair' on a bald portrait), "
        f"return an empty list for it. Output JSON only, no prose."
    )


def downsize_overlay_for_opus(overlay_path: Path, *,
                              max_edge_px: int = 1280) -> Path:
    """Write a downsized copy of the overlay for Opus to read.

    Returns the path of the smaller overlay (sibling file with '_small'
    suffix). If the overlay is already small enough, returns the input
    unchanged.

    Lazy import of cv2 so unit tests that mock the extractor don't pay
    for the heavy import.
    """
    import cv2  # local import

    img = cv2.imread(str(overlay_path))
    if img is None:
        raise FileNotFoundError(overlay_path)
    h, w = img.shape[:2]
    if max(h, w) <= max_edge_px:
        return overlay_path

    scale = max_edge_px / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_path = overlay_path.with_name(overlay_path.stem + "_small.png")
    cv2.imwrite(str(small_path), small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return small_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_cell_ids_from_overlay(
    overlay_png_path: str | Path,
    region_names: list[str],
    *,
    image_id: str = "",
    max_cell_id: int = 2500,
    timeout_s: int = 300,
    max_turns: int = 8,
    max_retries: int = 1,
) -> ExtractionResult:
    """Ask Opus to read the labeled overlay and return cell-IDs per region.

    Returns ExtractionResult. Raises OpusExtractionError on any failure
    (subprocess crash, schema validation miss, structured_output missing).
    """
    overlay = Path(overlay_png_path)
    if not overlay.exists():
        raise FileNotFoundError(overlay)
    if not region_names:
        raise ValueError("region_names must be non-empty")

    # Downsize the overlay if it's larger than Opus needs (saves tokens).
    try:
        opus_overlay = downsize_overlay_for_opus(overlay, max_edge_px=1280)
    except (ImportError, FileNotFoundError):
        opus_overlay = overlay

    schema = _build_schema(region_names)
    prompt = _build_prompt(opus_overlay, region_names, max_cell_id)

    t0 = time.time()
    try:
        result: ClaudeResult = translate_intent_prompt(
            prompt,
            schema,
            timeout_s=timeout_s,
            max_turns=max_turns,
            max_retries=max_retries,
            cwd=str(_HERE),
            extra_system_prompt=_SYSTEM_PROMPT,
        )
    except ClaudePError as exc:
        wall_s = time.time() - t0
        raise OpusExtractionError(
            f"claude -p failed after {wall_s:.1f}s: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    preds = result.structured_output
    # Force the contract: ints, sorted, deduped, clamped to [0, max_cell_id].
    clean: dict[str, list[int]] = {}
    for name in region_names:
        raw = preds.get(name, [])
        if not isinstance(raw, list):
            raw = []
        ids = sorted({int(c) for c in raw
                      if isinstance(c, (int, float))
                      and 0 <= int(c) <= max_cell_id})
        clean[name] = ids

    return ExtractionResult(
        image_id=image_id,
        predictions=clean,
        cost_usd=result.total_cost_usd,
        duration_ms=result.duration_ms,
        session_id=result.session_id,
        raw_envelope=result.raw_envelope,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay")
    parser.add_argument("--regions", default="face,hair,background,left_cheek,right_cheek")
    parser.add_argument("--max-cell-id", type=int, default=2500)
    args = parser.parse_args()

    region_names = [r.strip() for r in args.regions.split(",") if r.strip()]
    try:
        out = extract_cell_ids_from_overlay(
            args.overlay,
            region_names,
            image_id="cli",
            max_cell_id=args.max_cell_id,
        )
    except OpusExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps({
        "predictions": out.predictions,
        "cost_usd": out.cost_usd,
        "duration_ms": out.duration_ms,
    }, indent=2))
