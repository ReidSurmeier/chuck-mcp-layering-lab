"""
underlayer_proposer.py
======================

Rule-based underlayer-plate proposer for chuck-mcp v3.

Given (target_image, SNIC cell graph, face landmarks, pigment library), this module
returns 4-9 Plate proposals matching the locked v2 design:

    Workflow step 5: Algorithm proposes baseline underlayers (4-9 plates from cell
    graph + face landmarks + hue rules)

Each plate carries:
    - block_id (1..27)
    - cell_zone_ids (which SNIC cells are inked)
    - pigment_family (light_yellow | pale_pink | pale_orange | pale_red |
                      pale_blue | pale_green | warm_grey)
    - opacity (0.15..0.30 underlayer range)
    - role = "underlayer_light"
    - pass_index (1..~20, ordered light-to-dark)
    - region_label (cheek | temple | forehead | nose | lip | chin | jaw_neck |
                    eye_socket | eye_white | hair | brow | background)
    - provenance = "algorithm" (overrides flip this to "text:<phrase>")
    - rationale (one-line explanation pulled from rule_table.yaml)

Rule source: ./rule_table.yaml  (canonical chuck-mcp underlayer rules).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Domain objects (mirror v2-design Plate dataclass; underlayer-specialized)   #
# --------------------------------------------------------------------------- #

PigmentFamily = Literal[
    "light_yellow",
    "pale_pink",
    "pale_orange",
    "pale_red",
    "pale_blue",
    "pale_green",
    "warm_grey",
]

RegionLabel = Literal[
    "cheek", "temple", "forehead", "nose", "lip", "chin", "jaw_neck",
    "eye_socket", "eye_white", "hair", "brow", "background",
]


@dataclass
class UnderlayerPlate:
    """Underlayer-specialized Plate. Compatible with v2-design Plate dataclass."""

    block_id: int
    region_label: RegionLabel
    pigment_family: PigmentFamily
    cell_zone_ids: list[int]
    opacity: float
    pass_index: int
    role: Literal["underlayer_light"] = "underlayer_light"
    mirror: bool = True
    provenance: str = "algorithm"
    rationale: str = ""
    coverage: float = 0.0          # fraction of THIS region the plate covers (0..1)
    image_area_fraction: float = 0.0  # fraction of whole image the region occupies (0..1)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Color-space helpers                                                         #
# --------------------------------------------------------------------------- #

# Reference "paper white" for Shiramine washi (warm off-white). sRGB ~ (245, 240, 225).
PAPER_WHITE_RGB = np.array([245, 240, 225], dtype=np.float32) / 255.0

# Canonical pigment-family RGB anchors (approx. perceptual lift direction).
# These are PRE-K-M approximations — they encode "where on the color wheel does
# this family push a pixel when overprinted at low opacity". Calibration in V2
# (Mixbox/Curtis K-M) will replace these with measured values from Reid's swatches.
PIGMENT_FAMILY_ANCHORS_RGB: dict[PigmentFamily, np.ndarray] = {
    "light_yellow": np.array([252, 233, 138], dtype=np.float32) / 255.0,
    "pale_pink":    np.array([248, 196, 196], dtype=np.float32) / 255.0,
    "pale_orange":  np.array([250, 200, 150], dtype=np.float32) / 255.0,
    "pale_red":     np.array([232, 140, 130], dtype=np.float32) / 255.0,
    "pale_blue":    np.array([170, 200, 230], dtype=np.float32) / 255.0,
    "pale_green":   np.array([180, 220, 180], dtype=np.float32) / 255.0,
    "warm_grey":    np.array([200, 188, 175], dtype=np.float32) / 255.0,
}


def _rgb_to_hsl_hue_deg(rgb: np.ndarray) -> float:
    """Return hue in degrees [0, 360). Works on a 3-vector in [0, 1]."""
    r, g, b = rgb
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        return 0.0
    if cmax == r:
        h = ((g - b) / delta) % 6
    elif cmax == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    return float(h * 60.0)


def _hue_distance_deg(h1: float, h2: float) -> float:
    """Circular distance on the hue ring."""
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _hue_lift_alignment(pixel_rgb: np.ndarray, family: PigmentFamily) -> float:
    """
    Score how well pigment FAMILY's expected hue-lift direction matches the
    residual pixel - paper_white.

    Returns a score in [0, 1] — 1 = perfect alignment, 0 = orthogonal/opposed.
    """
    residual = pixel_rgb - PAPER_WHITE_RGB
    pigment_lift = PIGMENT_FAMILY_ANCHORS_RGB[family] - PAPER_WHITE_RGB

    n_res = np.linalg.norm(residual)
    n_lift = np.linalg.norm(pigment_lift)
    if n_res < 1e-6 or n_lift < 1e-6:
        return 0.0

    cos = float(np.dot(residual, pigment_lift) / (n_res * n_lift))
    # Map [-1, 1] -> [0, 1]
    return max(0.0, (cos + 1.0) / 2.0)


# Canonical hue bands (degrees on the HSL ring) → preferred underlayer family.
# Built from Pace progressive-proof forensic + Hokusai role-axis (see rule_table).
HUE_BAND_TO_FAMILY: list[tuple[tuple[float, float], PigmentFamily]] = [
    # (hue_lo, hue_hi), family preference
    ((350, 360), "pale_red"),     # pink-reds
    ((0, 15),    "pale_red"),     # warm reds (lip core)
    ((15, 35),   "pale_orange"),  # warm orange (forehead-to-hairline transition, background warm)
    ((35, 70),   "light_yellow"), # yellow-skin warmth (the "Pace first plate" band)
    ((70, 150),  "pale_green"),   # green (background complementary)
    ((150, 250), "pale_blue"),    # blues (eye-white, hair-cool, jaw shadow)
    ((250, 320), "pale_blue"),    # violets fall to blue family at underlayer level
    ((320, 350), "pale_pink"),    # pink-magenta (cheek/temple warmth lift)
]


def _hue_band_family(hue_deg: float) -> PigmentFamily:
    """Map a hue angle to its canonical pigment family per the band table."""
    h = hue_deg % 360.0
    for (lo, hi), fam in HUE_BAND_TO_FAMILY:
        if lo <= h < hi:
            return fam
    return "warm_grey"


def _saturation(rgb: np.ndarray) -> float:
    """HSL saturation in [0, 1]."""
    cmax = float(rgb.max())
    cmin = float(rgb.min())
    delta = cmax - cmin
    if delta == 0:
        return 0.0
    L = (cmax + cmin) / 2.0
    return delta / (1.0 - abs(2.0 * L - 1.0)) if (1.0 - abs(2.0 * L - 1.0)) > 1e-6 else 0.0


# --------------------------------------------------------------------------- #
# Inputs (lightweight protocols — full geometry comes from sibling agents)    #
# --------------------------------------------------------------------------- #

@dataclass
class CellGraph:
    """Output of SNIC superpixel pass (sibling agent: cell-zone-renderer).

    cell_id -> {pixels: list[(y, x)], mean_rgb: ndarray[3]}.
    """

    cells: dict[int, dict]  # cell_id -> {"pixels": list[tuple[int,int]], "mean_rgb": np.ndarray}

    @property
    def cell_ids(self) -> list[int]:
        return list(self.cells.keys())

    def mean_rgb(self, cell_id: int) -> np.ndarray:
        return self.cells[cell_id]["mean_rgb"]

    def pixel_count(self, cell_id: int) -> int:
        return len(self.cells[cell_id]["pixels"])


@dataclass
class FaceLandmarks:
    """Output of MediaPipe face-spatial pass (sibling agent: mediapipe-face-spatial).

    region_to_cells: maps each RegionLabel to the list of SNIC cell_ids that fall
    inside that region's MediaPipe-derived mask.
    """

    region_to_cells: dict[RegionLabel, list[int]]
    image_shape: tuple[int, int]   # (H, W)

    def cells_in(self, region: RegionLabel) -> list[int]:
        return self.region_to_cells.get(region, [])


@dataclass
class PigmentLibrary:
    """Reid's physical pigment inventory (subset relevant to underlayers).

    For V1 we just need the families that are available. The full per-pigment
    YAML schema is locked in week-3 (per v2-design open architectural questions).
    """

    available_families: set[PigmentFamily]
    family_to_pigment_id: dict[PigmentFamily, str]   # e.g. "light_yellow" -> "PY3_holbein_pale"

    @classmethod
    def default_emma_inventory(cls) -> "PigmentLibrary":
        return cls(
            available_families={
                "light_yellow", "pale_pink", "pale_orange", "pale_red",
                "pale_blue", "pale_green", "warm_grey",
            },
            family_to_pigment_id={
                "light_yellow": "PY3_holbein_pale",
                "pale_pink":    "PR122_holbein_dilute",
                "pale_orange":  "PO48_holbein_dilute",
                "pale_red":     "PR112_holbein_medium",
                "pale_blue":    "PB15_holbein_dilute",
                "pale_green":   "PG36_holbein_dilute",
                "warm_grey":    "PBr7_PBk7_blend",
            },
        )


# --------------------------------------------------------------------------- #
# Rule loader                                                                 #
# --------------------------------------------------------------------------- #

DEFAULT_RULE_PATH = Path(__file__).parent / "rule_table.yaml"


def load_rules(path: Path | str = DEFAULT_RULE_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Core algorithm                                                              #
# --------------------------------------------------------------------------- #

def _pick_family_for_region(
    region: RegionLabel,
    region_cells: list[int],
    cell_graph: CellGraph,
    rules: dict,
    pigments: PigmentLibrary,
) -> tuple[PigmentFamily, float, str]:
    """
    For one region, pick the pigment family using a TWO-AXIS classifier:

      1. HUE BAND — map aggregate hue to its canonical family via HUE_BAND_TO_FAMILY
      2. SATURATION GATE — if saturation < 0.15, region is desaturated; prefer
         warm_grey (value-foundation) over chroma families
      3. FILTER through rule_table allowed_families ∩ pigment inventory
      4. If hue-band choice violates rules, fall back to top-priority allowed
         family, then secondary, etc.

    Returns: (chosen_family, confidence_score, rationale_text).
    """
    region_rule = rules["regions"][region]
    allowed = [f for f in region_rule["allowed_families"]
               if f in pigments.available_families
               and f not in region_rule.get("forbidden_families", [])]
    if not allowed:
        return "warm_grey", 0.0, f"no allowed family in inventory for {region}; falling back to warm_grey"

    if not region_cells:
        return allowed[0], 0.0, f"no cells in {region}; default to top-priority allowed family"

    # Aggregate mean RGB
    total_px = 0
    weighted = np.zeros(3, dtype=np.float64)
    for cid in region_cells:
        n = cell_graph.pixel_count(cid)
        weighted += cell_graph.mean_rgb(cid).astype(np.float64) * n
        total_px += n
    if total_px == 0:
        return allowed[0], 0.0, f"zero pixels in {region}"
    aggregate_rgb = (weighted / total_px).astype(np.float32)

    # Compute hue + saturation
    hue_deg = _rgb_to_hsl_hue_deg(aggregate_rgb)
    sat = _saturation(aggregate_rgb)

    # PACE CANONICAL: cheek priority=1 always gets light_yellow first if allowed.
    # This is the "first plate" of every Shibata-style portrait (Sultan_Shiff_2003).
    if region == "cheek" and "light_yellow" in allowed:
        return ("light_yellow", 1.0,
                f"region={region}, Pace canonical first-plate rule: cheek gets light_yellow "
                f"(value-foundation, Shibata sequence start). "
                f"agg_rgb={tuple(np.round(aggregate_rgb, 3).tolist())}")

    # Axis 2: low-saturation regions go to warm_grey if allowed,
    # EXCEPT hair (always cool support per Reid annotation) and lip (always chroma).
    if (sat < 0.15
            and region not in ("hair", "lip", "eye_white", "background")
            and "warm_grey" in allowed):
        return ("warm_grey", 0.9,
                f"region={region}, low-saturation ({sat:.2f}) → warm_grey value-foundation. "
                f"agg_rgb={tuple(np.round(aggregate_rgb, 3).tolist())}")

    # Hair: dark + low-saturation but the canonical underlayer is pale_blue (cool support)
    if region == "hair" and "pale_blue" in allowed:
        return ("pale_blue", 0.92,
                f"region={region}, hair canonical underlayer = pale_blue "
                f"(cool support per Reid annotation + Sultan_Shiff_2003_p138). "
                f"agg_rgb={tuple(np.round(aggregate_rgb, 3).tolist())}")

    # Axis 1: hue band → canonical family
    hue_pick = _hue_band_family(hue_deg)

    # If hue_pick is allowed, use it
    if hue_pick in allowed:
        return (hue_pick, 0.95,
                f"region={region}, hue={hue_deg:.1f}° → {hue_pick} via hue-band classifier. "
                f"agg_rgb={tuple(np.round(aggregate_rgb, 3).tolist())} per rule: "
                f"{region_rule['rationale'][:100]}")

    # hue_pick not in allowed (e.g., lip region hue says light_yellow but rule forbids).
    # Fall back to alignment-scored choice among allowed.
    scored = sorted(
        ((fam, _hue_lift_alignment(aggregate_rgb, fam)) for fam in allowed),
        key=lambda x: x[1], reverse=True,
    )
    best_family, best_score = scored[0]
    # Priority-rank tie-breaker (allowed_families is priority-ordered)
    if len(scored) > 1 and abs(scored[0][1] - scored[1][1]) < 0.05:
        for fam in allowed:
            if fam in (scored[0][0], scored[1][0]):
                best_family = fam
                best_score = dict(scored)[fam]
                break

    rationale = (
        f"region={region}, hue={hue_deg:.1f}° suggested {hue_pick} "
        f"(not in allowed={allowed}); fallback alignment-score → {best_family} "
        f"({best_score:.3f}). agg_rgb={tuple(np.round(aggregate_rgb, 3).tolist())}"
    )
    return best_family, best_score, rationale


def _coverage(region_cells: list[int], cell_graph: CellGraph,
              image_shape: tuple[int, int]) -> float:
    """
    Coverage = fraction of THIS region's cells that the underlayer plate covers.

    For the baseline proposer this is always 1.0 (we always cover the whole region
    on the underlayer). The plate's role in the final composite uses image-area
    via separate metrics. The reason this is fraction-of-region (not fraction-of-
    image) is the rule_table.coverage_min spec: `coverage_min: 0.40` reads as
    "underlayer must cover ≥40% of cheek SNIC cells".
    """
    if not region_cells:
        return 0.0
    # Default: full coverage. Solver may decimate later (partial underlay).
    return 1.0


def _image_area_fraction(region_cells: list[int], cell_graph: CellGraph,
                         image_shape: tuple[int, int]) -> float:
    """Fraction of the WHOLE IMAGE these cells occupy (for first-pull rule)."""
    H, W = image_shape
    total = H * W
    if total == 0:
        return 0.0
    region_px = sum(cell_graph.pixel_count(cid) for cid in region_cells)
    return region_px / float(total)


def _opacity_for(family: PigmentFamily, region: RegionLabel, rules: dict) -> float:
    """Pick the mid-point of the opacity range for this region. Refined later by solver."""
    lo, hi = rules["regions"][region]["opacity_range"]
    return float((lo + hi) / 2.0)


def _enforce_global_no_double_underlay(
    region_choices: dict[RegionLabel, tuple[PigmentFamily, float, str]],
    rules: dict,
) -> dict[RegionLabel, tuple[PigmentFamily, float, str]]:
    """
    Apply global rule: no two ADJACENT regions should share the same pigment family
    underlayer. Adjacency map is hardcoded for portrait domain.
    """
    adjacency: dict[RegionLabel, list[RegionLabel]] = {
        "cheek":       ["temple", "nose", "jaw_neck", "chin"],
        "temple":      ["cheek", "forehead", "hair"],
        "forehead":    ["temple", "nose", "brow", "hair"],
        "nose":        ["cheek", "forehead", "lip"],
        "lip":         ["nose", "chin"],
        "chin":        ["lip", "jaw_neck", "cheek"],
        "jaw_neck":    ["cheek", "chin"],
        "eye_socket":  ["brow", "eye_white", "nose"],
        "eye_white":   ["eye_socket"],
        "hair":        ["forehead", "temple", "background"],
        "brow":        ["forehead", "eye_socket"],
        "background":  ["hair"],
    }

    # Greedy de-duplication: for each region in priority order, if its choice
    # collides with an already-assigned adjacent region using same family,
    # try the next-best family from its allowed list.
    rule_regions = rules["regions"]
    ordered = sorted(region_choices.keys(),
                     key=lambda r: rule_regions[r]["priority_rank"])
    fixed: dict[RegionLabel, tuple[PigmentFamily, float, str]] = {}
    for region in ordered:
        fam, score, rationale = region_choices[region]
        clash = any(
            n in fixed and fixed[n][0] == fam
            for n in adjacency.get(region, [])
        )
        if not clash:
            fixed[region] = (fam, score, rationale)
            continue
        # Try alternates from allowed_families
        allowed = rule_regions[region]["allowed_families"]
        forbidden = set(rule_regions[region].get("forbidden_families", []))
        replaced = False
        for alt in allowed:
            if alt in forbidden:
                continue
            if alt == fam:
                continue
            alt_clash = any(
                n in fixed and fixed[n][0] == alt
                for n in adjacency.get(region, [])
            )
            if not alt_clash:
                fixed[region] = (alt, score * 0.9,  # slight score penalty for swap
                                 rationale + f" [swapped to {alt} to avoid {fam} clash with adjacent]")
                replaced = True
                break
        if not replaced:
            # No swap possible — keep original (rule allows it; print will be slightly flat there)
            fixed[region] = (fam, score, rationale + " [global no-double-underlay violation accepted]")
    return fixed


def _assign_pass_order(plates: list[UnderlayerPlate], rules: dict) -> list[UnderlayerPlate]:
    """
    Assign pass_index 1..N to plates.

    Rules:
      - light_to_dark_strict: warmest, lowest-opacity families first
      - first_pull_is_lightest_largest: pass_index=1 must have opacity<0.20 AND coverage>0.40
    """
    # Family pass-order priority (lower = earlier in stack)
    family_pass_order: dict[PigmentFamily, int] = {
        "light_yellow":  1,   # lightest, broadest first
        "pale_pink":     2,
        "pale_orange":   3,
        "pale_green":    3,   # comparable opacity to orange; background-typical
        "pale_blue":     4,
        "warm_grey":     5,
        "pale_red":      6,   # most opaque of underlayer families
    }

    # Sort: (family rank ascending, opacity ascending, image_area_fraction descending)
    plates_sorted = sorted(
        plates,
        key=lambda p: (
            family_pass_order.get(p.pigment_family, 99),
            p.opacity,
            -p.image_area_fraction,
        ),
    )

    # If the first plate violates first_pull_is_lightest_largest, swap with one that does
    # (rule: pass_index=1 must have opacity<0.20 AND image_area_fraction>0.40)
    if plates_sorted:
        head = plates_sorted[0]
        if not (head.opacity < 0.20 and head.image_area_fraction > 0.40):
            for i, candidate in enumerate(plates_sorted[1:], start=1):
                if candidate.opacity < 0.20 and candidate.image_area_fraction > 0.40:
                    plates_sorted[0], plates_sorted[i] = plates_sorted[i], plates_sorted[0]
                    break

    for i, p in enumerate(plates_sorted, start=1):
        p.pass_index = i
    return plates_sorted


def propose_underlayers(
    target_image: np.ndarray,
    cell_graph: CellGraph,
    face_landmarks: FaceLandmarks,
    pigment_library: PigmentLibrary | None = None,
    rules: dict | None = None,
    starting_block_id: int = 1,
    max_plates: int = 9,
    min_plates: int = 4,
) -> list[UnderlayerPlate]:
    """
    Top-level entry point.

    Parameters
    ----------
    target_image : np.ndarray of shape (H, W, 3), values in [0, 1]
        Target image (not used directly — cell_graph carries the per-cell color).
        Retained in the signature for API compatibility / future use.
    cell_graph : CellGraph
        SNIC superpixel cell graph from the cell-zone-renderer.
    face_landmarks : FaceLandmarks
        Region → cells map from the MediaPipe face-spatial agent.
    pigment_library : PigmentLibrary, optional
        Reid's inventory. Defaults to default_emma_inventory().
    rules : dict, optional
        Parsed rule_table.yaml. Defaults to loading from sibling file.
    starting_block_id : int
        First block_id to assign. Underlayer plates use 1..N; solver assigns the rest.
    max_plates : int
    min_plates : int

    Returns
    -------
    list[UnderlayerPlate]
        4..9 plates, ordered by pass_index ascending, ready to feed the v3 solver
        as the algorithmic baseline.
    """
    pigments = pigment_library or PigmentLibrary.default_emma_inventory()
    rules = rules or load_rules()

    # Step 1: per-region family choice
    region_choices: dict[RegionLabel, tuple[PigmentFamily, float, str]] = {}
    region_cells_map: dict[RegionLabel, list[int]] = {}

    for region in rules["regions"]:
        cells = face_landmarks.cells_in(region)  # type: ignore[arg-type]
        if not cells:
            continue
        # Image-area sanity check: a face region occupying <0.5% of the image is
        # too small to warrant a dedicated underlayer plate (likely segmentation error).
        img_frac = _image_area_fraction(cells, cell_graph, face_landmarks.image_shape)
        if img_frac < 0.005:
            continue
        family, score, rationale = _pick_family_for_region(
            region, cells, cell_graph, rules, pigments,
        )
        region_choices[region] = (family, score, rationale)
        region_cells_map[region] = cells

    # Step 2: enforce global no-double-underlay rule
    region_choices = _enforce_global_no_double_underlay(region_choices, rules)

    # Step 3: enforce complementary-background rule
    region_choices = _apply_complementary_background_rule(
        region_choices, region_cells_map, cell_graph, rules,
    )

    # Step 4: trim to max_plates if too many.
    # Always KEEP the 9 Reid-annotated load-bearing underlayer regions:
    #   cheek, temple, forehead, chin, lip, hair, eye_white, background, jaw_neck
    # Drop nose / brow / eye_socket if over budget — those are typically merged
    # into adjacent regions in mokuhanga underlayer practice.
    if len(region_choices) > max_plates:
        reid_canonical: list[RegionLabel] = [
            "cheek", "lip", "hair", "background",   # 4 load-bearing
            "temple", "forehead", "chin", "eye_white", "jaw_neck",  # round out to 9
        ]
        kept = set(r for r in reid_canonical if r in region_choices)
        slots_left = max_plates - len(kept)
        candidates = sorted(
            [r for r in region_choices if r not in kept],
            key=lambda r: rules["regions"][r]["priority_rank"],
        )
        for r in candidates:
            if slots_left <= 0:
                break
            kept.add(r)
            slots_left -= 1
        region_choices = {r: v for r, v in region_choices.items() if r in kept}
        region_cells_map = {r: v for r, v in region_cells_map.items() if r in kept}

    # Step 5: build UnderlayerPlate objects
    plates: list[UnderlayerPlate] = []
    block_id = starting_block_id
    for region, (family, score, rationale) in region_choices.items():
        cells = region_cells_map[region]
        opacity = _opacity_for(family, region, rules)
        coverage = _coverage(cells, cell_graph, face_landmarks.image_shape)
        img_frac = _image_area_fraction(cells, cell_graph, face_landmarks.image_shape)
        plates.append(UnderlayerPlate(
            block_id=block_id,
            region_label=region,
            pigment_family=family,
            cell_zone_ids=sorted(cells),
            opacity=opacity,
            pass_index=0,    # assigned next step
            coverage=coverage,
            image_area_fraction=img_frac,
            rationale=rationale,
        ))
        block_id += 1

    # Step 6: assign pass_index respecting light-to-dark + first-pull rules
    plates = _assign_pass_order(plates, rules)

    # Step 7: if below min_plates, that's a quality signal — caller may want to
    # relax coverage_min or include lower-priority regions.
    if len(plates) < min_plates:
        # Synthesize a forced background underlayer if absent and we have inventory
        if "background" not in {p.region_label for p in plates}:
            bg_cells = face_landmarks.cells_in("background")
            if bg_cells:
                fam = "pale_blue" if "pale_blue" in pigments.available_families else "warm_grey"
                plates.append(UnderlayerPlate(
                    block_id=block_id,
                    region_label="background",
                    pigment_family=fam,
                    cell_zone_ids=sorted(bg_cells),
                    opacity=0.20,
                    pass_index=len(plates) + 1,
                    coverage=_coverage(bg_cells, cell_graph, face_landmarks.image_shape),
                    image_area_fraction=_image_area_fraction(
                        bg_cells, cell_graph, face_landmarks.image_shape,
                    ),
                    rationale=(
                        "forced background plate to satisfy min_plates=4; "
                        "see rule_table.yaml total_underlayer_plates"
                    ),
                ))
                plates = _assign_pass_order(plates, rules)

    return plates


def _apply_complementary_background_rule(
    region_choices: dict[RegionLabel, tuple[PigmentFamily, float, str]],
    region_cells_map: dict[RegionLabel, list[int]],
    cell_graph: CellGraph,
    rules: dict,
) -> dict[RegionLabel, tuple[PigmentFamily, float, str]]:
    """
    If face is dominantly warm, push background to a cool family; vice versa.
    """
    if "background" not in region_choices:
        return region_choices

    # Compute aggregate face hue (any non-background region with cells)
    face_regions = [r for r in region_choices if r != "background"]
    if not face_regions:
        return region_choices

    total_px = 0
    weighted = np.zeros(3, dtype=np.float64)
    for region in face_regions:
        for cid in region_cells_map.get(region, []):
            n = cell_graph.pixel_count(cid)
            weighted += cell_graph.mean_rgb(cid).astype(np.float64) * n
            total_px += n
    if total_px == 0:
        return region_choices

    mean_face_rgb = (weighted / total_px).astype(np.float32)
    mean_face_hue = _rgb_to_hsl_hue_deg(mean_face_rgb)

    # Skip the complementary push if the background is itself highly saturated —
    # in that case Reid (or the artist) clearly INTENDS a chromatic background,
    # and we shouldn't override their intent to a complementary cool/warm wash.
    bg_cells = region_cells_map["background"]
    bg_total_px = 0
    bg_weighted = np.zeros(3, dtype=np.float64)
    for cid in bg_cells:
        n = cell_graph.pixel_count(cid)
        bg_weighted += cell_graph.mean_rgb(cid).astype(np.float64) * n
        bg_total_px += n
    if bg_total_px > 0:
        bg_mean_rgb = (bg_weighted / bg_total_px).astype(np.float32)
        bg_sat = _saturation(bg_mean_rgb)
        if bg_sat >= 0.40:
            # Background is intrinsically saturated — keep the hue-band pick
            region_choices["background"] = (
                region_choices["background"][0],
                region_choices["background"][1],
                region_choices["background"][2]
                + f" [complementary_background SUPPRESSED: bg intrinsically saturated "
                  f"(sat={bg_sat:.2f}); honoring intrinsic hue choice]"
            )
            return region_choices

    is_warm = (mean_face_hue <= 60.0) or (mean_face_hue >= 330.0)
    bg_allowed = rules["regions"]["background"]["allowed_families"]
    current_bg = region_choices["background"][0]

    preferred_cool = ["pale_blue", "pale_green"]
    preferred_warm = ["pale_orange", "light_yellow"]
    desired = preferred_cool if is_warm else preferred_warm
    pick = next((f for f in desired if f in bg_allowed), current_bg)

    if pick != current_bg:
        region_choices["background"] = (
            pick,
            region_choices["background"][1],
            region_choices["background"][2]
            + f" [complementary_background: face is {'warm' if is_warm else 'cool'} "
              f"(hue={mean_face_hue:.1f}), pushing background to {pick}]"
        )
    return region_choices


# --------------------------------------------------------------------------- #
# Serialization helpers                                                       #
# --------------------------------------------------------------------------- #

def plates_to_json(plates: list[UnderlayerPlate]) -> str:
    """JSON-serialize plate list. cell_zone_ids preserved as full lists."""
    return json.dumps([p.to_dict() for p in plates], indent=2)


def plates_summary_table(plates: list[UnderlayerPlate]) -> str:
    """Human-readable single-screen summary."""
    lines = [
        f"{'pass':>4}  {'block':>5}  {'region':<12}  {'family':<14}  "
        f"{'opacity':>7}  {'cov':>6}  {'cells':>6}  provenance"
    ]
    lines.append("-" * 90)
    for p in plates:
        lines.append(
            f"{p.pass_index:>4}  {p.block_id:>5}  {p.region_label:<12}  "
            f"{p.pigment_family:<14}  {p.opacity:>7.3f}  {p.coverage:>6.3f}  "
            f"{len(p.cell_zone_ids):>6}  {p.provenance}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test — fabricate a tiny graph and run the proposer
    rng = np.random.default_rng(42)
    image = rng.random((256, 256, 3)).astype(np.float32)

    # Fake 20 cells with mean colors mimicking face regions
    cells = {}
    for cid in range(20):
        h, w = (rng.integers(0, 256), rng.integers(0, 256))
        cells[cid] = {
            "pixels": [(h, w)] * 500,
            "mean_rgb": rng.random(3).astype(np.float32),
        }
    cg = CellGraph(cells=cells)
    fl = FaceLandmarks(
        region_to_cells={
            "cheek":      [0, 1, 2],
            "lip":        [3],
            "forehead":   [4, 5],
            "hair":       [6, 7, 8],
            "background": [9, 10, 11, 12],
            "eye_white":  [13],
            "brow":       [14],
            "chin":       [15],
        },
        image_shape=(256, 256),
    )
    plates = propose_underlayers(image, cg, fl)
    print(plates_summary_table(plates))
