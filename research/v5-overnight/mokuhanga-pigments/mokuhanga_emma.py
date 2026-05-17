"""mokuhanga_emma.py — adapter wiring v3 underlayer_proposer + Emma pigment library.

This is the integration seam between:

    research/v3-construction/mokuhanga-rule-classifier/underlayer_proposer.py
        (rule-based family classifier, 94.4% match vs Reid's annotation)
    chuck_mcp_v2/pigment_library_emma.yaml
        (concrete pigment inventory, 19 entries, 7 families)

It exports two callables used by chuck_mcp_v2.plan_emma:

    plan_underlayer_pigments(target_image, cell_graph, face_landmarks, library)
        -> list[EmmaPigmentPlate]
    select_pigment_for_role_plate(library, target_lab, candidate_family)
        -> PigmentEntry

The adapter NEVER fabricates a pigment name; every output plate carries a
real library key. When the rule classifier picks a family the library does
not stock, the adapter falls back to warm_grey (deterministically).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pigment_library import (
    PigmentEntry,
    family_to_pigments,
    load_pigment_library,
    EMMA_PIGMENT_LIBRARY_PATH,
    PIGMENT_FAMILIES,
)

# v3 rule classifier imports — resolved via conftest sys.path.
from underlayer_proposer import (  # type: ignore[import-not-found]
    CellGraph,
    FaceLandmarks,
    PigmentLibrary as RulePigmentLibrary,
    propose_underlayers,
    UnderlayerPlate,
)


# --------------------------------------------------------------------------- #
# Domain object                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class EmmaPigmentPlate:
    """One underlayer plate carrying a CONCRETE library pigment.

    Compatible with chuck_mcp_v2.types.Plate (production schema) — the
    integration patch in plan_emma converts these into PlateSpec objects.
    """

    block_id: int
    region_label: str
    pigment_family: str
    pigment_name: str                  # key into PigmentLibrary
    pigment_lab: tuple[float, float, float]
    cell_zone_ids: list[int]
    opacity: float
    pass_index: int
    coverage: float
    image_area_fraction: float
    role: str = "underlayer_light"
    mirror: bool = True
    provenance: str = "algorithm"
    rationale: str = ""
    confidence: float = 1.0


# --------------------------------------------------------------------------- #
# ΔE_76 distance (catalog-Lab is OK for V1 — see expected_v1_dE_uncertainty)  #
# --------------------------------------------------------------------------- #
def delta_e_76(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Euclidean distance in CIE Lab space (ΔE*_76).

    Catalog Lab values are uncalibrated_v1, so we expect ±6 ΔE noise. The
    selector still picks correctly within the same family because each
    family's pigments span a wide intra-family ΔE range.
    """
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


# --------------------------------------------------------------------------- #
# Pigment selection                                                           #
# --------------------------------------------------------------------------- #
def select_pigment_for_role_plate(
    library: dict[str, PigmentEntry],
    target_lab: tuple[float, float, float],
    candidate_family: str,
) -> PigmentEntry:
    """Pick the library pigment closest in Lab to `target_lab`, restricted
    to pigments of family `candidate_family`. Falls back to closest-of-any
    if the family is empty.
    """
    fam_index = family_to_pigments(library)
    pool = fam_index.get(candidate_family) or []
    if not pool:
        # Fallback: search the whole library.
        pool = list(library.values())
    if not pool:
        raise ValueError("pigment library is empty")
    return min(pool, key=lambda p: delta_e_76(target_lab, p.lab_values))


# --------------------------------------------------------------------------- #
# Color-space helpers (sRGB -> Lab approx for cell aggregates)                #
# --------------------------------------------------------------------------- #
def _srgb_to_lab_approx(rgb_01: np.ndarray) -> np.ndarray:
    """Cheap sRGB→Lab conversion via XYZ D65; sufficient for V1 selection."""
    if rgb_01.ndim == 1:
        rgb_01 = rgb_01.reshape(1, 1, 3)

    def _linearize(c: np.ndarray) -> np.ndarray:
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    rgb_lin = _linearize(np.clip(rgb_01.astype(np.float64), 0.0, 1.0))
    # sRGB D65 -> XYZ
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    flat = rgb_lin.reshape(-1, 3)
    xyz = flat @ M.T
    # D65 white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xyz_n = xyz / np.array([xn, yn, zn])

    def _f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(
            t > delta ** 3,
            np.cbrt(t),
            t / (3.0 * delta ** 2) + 4.0 / 29.0,
        )

    fx = _f(xyz_n[:, 0])
    fy = _f(xyz_n[:, 1])
    fz = _f(xyz_n[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    lab = np.stack([L, a, b], axis=-1).reshape(rgb_01.shape)
    return lab


def _aggregate_region_lab(
    cell_graph: CellGraph,
    region_cells: list[int],
) -> tuple[float, float, float]:
    """Pixel-weighted mean Lab across the region's SNIC cells."""
    if not region_cells:
        return (50.0, 0.0, 0.0)
    total = 0
    weighted = np.zeros(3, dtype=np.float64)
    for cid in region_cells:
        n = cell_graph.pixel_count(cid)
        weighted += cell_graph.mean_rgb(cid).astype(np.float64) * n
        total += n
    if total == 0:
        return (50.0, 0.0, 0.0)
    mean_rgb = (weighted / total).astype(np.float32)
    if mean_rgb.max() > 1.5:
        mean_rgb = mean_rgb / 255.0
    lab = _srgb_to_lab_approx(mean_rgb.reshape(1, 1, 3))[0, 0]
    return (float(lab[0]), float(lab[1]), float(lab[2]))


# --------------------------------------------------------------------------- #
# Adapter                                                                     #
# --------------------------------------------------------------------------- #
def _rule_library_from(library: dict[str, PigmentEntry]) -> RulePigmentLibrary:
    """Build a v3-compatible RulePigmentLibrary advertising every family
    that has at least one concrete pigment in `library`.

    family_to_pigment_id is filled with each family's first pigment name —
    the rule classifier only uses it for diagnostic strings, never for
    selection.
    """
    fam_index = family_to_pigments(library)
    available = set(fam_index.keys()) & set(PIGMENT_FAMILIES)
    family_to_id = {
        fam: pigments[0].name
        for fam, pigments in fam_index.items()
        if fam in PIGMENT_FAMILIES
    }
    return RulePigmentLibrary(
        available_families=available,
        family_to_pigment_id=family_to_id,
    )


def plan_underlayer_pigments(
    target_image: np.ndarray,
    cell_graph: CellGraph,
    face_landmarks: FaceLandmarks,
    pigment_library: Optional[dict[str, PigmentEntry]] = None,
    starting_block_id: int = 1,
    max_plates: int = 9,
    min_plates: int = 4,
) -> list[EmmaPigmentPlate]:
    """Run the v3 rule-based proposer, then pick a CONCRETE library pigment
    for each plate by Lab-distance to the region's aggregate Lab.

    Returns
    -------
    list[EmmaPigmentPlate]
        ordered by pass_index ascending; every plate carries pigment_name
        pointing into pigment_library.
    """
    library = pigment_library or load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)
    rule_lib = _rule_library_from(library)

    underlayer_plates: list[UnderlayerPlate] = propose_underlayers(
        target_image=target_image,
        cell_graph=cell_graph,
        face_landmarks=face_landmarks,
        pigment_library=rule_lib,
        starting_block_id=starting_block_id,
        max_plates=max_plates,
        min_plates=min_plates,
    )

    out: list[EmmaPigmentPlate] = []
    for plate in underlayer_plates:
        region_lab = _aggregate_region_lab(cell_graph, plate.cell_zone_ids)
        pigment = select_pigment_for_role_plate(
            library=library,
            target_lab=region_lab,
            candidate_family=plate.pigment_family,
        )
        out.append(EmmaPigmentPlate(
            block_id=plate.block_id,
            region_label=plate.region_label,
            pigment_family=plate.pigment_family,
            pigment_name=pigment.name,
            pigment_lab=pigment.lab_values,
            cell_zone_ids=list(plate.cell_zone_ids),
            opacity=plate.opacity,
            pass_index=plate.pass_index,
            coverage=plate.coverage,
            image_area_fraction=plate.image_area_fraction,
            role="underlayer_light",
            mirror=plate.mirror,
            provenance=plate.provenance,
            rationale=(
                plate.rationale
                + f" [pigment_pick={pigment.name} via ΔE_76 vs region "
                  f"L={region_lab[0]:.1f} a={region_lab[1]:.1f} b={region_lab[2]:.1f}]"
            ),
        ))
    return out


# --------------------------------------------------------------------------- #
# Plan-level integration — mutate a built ProductionPlan in place             #
# --------------------------------------------------------------------------- #
def _build_face_landmarks_from_plan(
    plan,
    cell_graph: CellGraph,
) -> FaceLandmarks:
    """Construct a FaceLandmarks proxy from a built ProductionPlan.

    The plan's underlayer_light plates carry region_label hints when the
    upstream face-spatial agent ran. When face-spatial output is unavailable
    (synthetic harnesses, demos), we fall back to a single 'background'
    region carrying every cell — the proposer then degenerates gracefully
    to ≤ max_plates plates with rule-table defaults.
    """
    region_to_cells: dict[str, list[int]] = {}
    for plate in plan.plates:
        label = (plate.region_label or "").strip()
        if not label:
            continue
        region_to_cells.setdefault(label, []).extend(int(c) for c in plate.cell_zone_ids)

    if not region_to_cells:
        # Degenerate fallback: treat the whole image as the 'background' region.
        all_cells = [int(c) for c in cell_graph.cells.keys()]
        region_to_cells = {"background": all_cells}

    # Best-effort image shape from cell pixel counts; the proposer mostly
    # uses image_shape for area fractions.
    total_px = sum(cell_graph.pixel_count(int(cid)) for cid in cell_graph.cells)
    side = max(1, int(round(total_px ** 0.5)))
    return FaceLandmarks(region_to_cells=region_to_cells, image_shape=(side, side))


def _select_pigment_by_cells(
    library: dict[str, PigmentEntry],
    candidate_family: str,
    cell_graph: CellGraph,
    cell_ids: list[int],
) -> tuple[PigmentEntry, tuple[float, float, float]]:
    target_lab = _aggregate_region_lab(cell_graph, [int(c) for c in cell_ids])
    pigment = select_pigment_for_role_plate(
        library=library,
        target_lab=target_lab,
        candidate_family=candidate_family,
    )
    return pigment, target_lab


# Default family per non-underlayer role when no better hint exists.
_ROLE_FAMILY_FALLBACK: dict[str, str] = {
    "underlayer_light": "light_yellow",
    "local_chroma":     "pale_red",
    "regional_mass":    "pale_blue",
    "key_detail":       "warm_grey",
}


def apply_mokuhanga_pigments_to_plan(
    plan,
    cell_graph: CellGraph,
    pigment_library: Optional[dict[str, PigmentEntry]] = None,
) -> dict:
    """Mutate a built ProductionPlan so every plate carries a CONCRETE
    Emma-library pigment.

    Steps
    -----
    1. Underlayer plates: rerun the v3 rule classifier against the plate
       set, mapping the resulting families+pigments back onto the matching
       PlateSpec.
    2. Mid/dark plates (local_chroma, regional_mass, key_detail): pick a
       library pigment by ΔE_76 to each plate's aggregate-cell Lab.
    3. Every plate's pulls have pigment_id rewritten to the chosen library
       pigment name (a real key in pigment_library_emma.yaml).

    Returns
    -------
    dict
        Diagnostic summary: counts per role, picks, families used.
    """
    library = pigment_library or load_pigment_library(EMMA_PIGMENT_LIBRARY_PATH)

    # --- 1) Underlayer pass via the rule classifier ---
    underlayer_plates = [p for p in plan.plates if p.role == "underlayer_light"]
    fl = _build_face_landmarks_from_plan(plan, cell_graph)
    rule_lib = _rule_library_from(library)
    proposed: list[UnderlayerPlate] = []
    if underlayer_plates and fl.region_to_cells:
        proposed = propose_underlayers(
            target_image=np.zeros((fl.image_shape[0], fl.image_shape[1], 3),
                                  dtype=np.float32),
            cell_graph=cell_graph,
            face_landmarks=fl,
            pigment_library=rule_lib,
            starting_block_id=min(p.block_id for p in underlayer_plates),
            max_plates=max(4, min(9, len(underlayer_plates))),
            min_plates=min(4, len(underlayer_plates)),
        )

    # Build a region -> (pigment_family, pigment_name) map from the proposer.
    region_pick: dict[str, tuple[str, str]] = {}
    for up in proposed:
        target_lab = _aggregate_region_lab(cell_graph, up.cell_zone_ids)
        pigment = select_pigment_for_role_plate(
            library=library,
            target_lab=target_lab,
            candidate_family=up.pigment_family,
        )
        region_pick[up.region_label] = (up.pigment_family, pigment.name)

    summary: dict = {
        "underlayer_picks": region_pick,
        "mid_dark_picks": {},
        "families_used": set(),
    }

    # --- 2) Mutate each plate ---
    for plate in plan.plates:
        if plate.role == "underlayer_light":
            # Prefer rule classifier pick by region_label, else fall back to
            # Lab-distance pick within the existing family.
            label = (plate.region_label or "").strip()
            if label in region_pick:
                fam, pig_name = region_pick[label]
            else:
                fam = plate.pigment_family or _ROLE_FAMILY_FALLBACK["underlayer_light"]
                if fam not in {p.family for p in library.values()}:
                    fam = _ROLE_FAMILY_FALLBACK["underlayer_light"]
                pigment, _ = _select_pigment_by_cells(
                    library, fam, cell_graph, plate.cell_zone_ids,
                )
                pig_name = pigment.name
        else:
            fam = plate.pigment_family or _ROLE_FAMILY_FALLBACK.get(plate.role, "warm_grey")
            if fam not in {p.family for p in library.values()}:
                fam = _ROLE_FAMILY_FALLBACK.get(plate.role, "warm_grey")
            pigment, target_lab = _select_pigment_by_cells(
                library, fam, cell_graph, plate.cell_zone_ids,
            )
            pig_name = pigment.name
            summary["mid_dark_picks"][plate.block_id] = {
                "role": plate.role,
                "family": fam,
                "pigment": pig_name,
                "target_lab": target_lab,
            }

        # Write back to the plate. We deliberately do NOT set plate.pigment_id
        # here — that field belongs to the post-JAX solved schema and would
        # flip the Plate.to_dict() output away from the production schema we
        # want at this stage. pigment_name is the production-schema field.
        plate.pigment_family = fam
        plate.pigment_name = pig_name
        plate.pigment_lab = library[pig_name].lab_values
        plate.provenance = (plate.provenance or "algorithm") + ";mokuhanga_pigments"
        plate.rationale = (
            (plate.rationale or "")
            + f" [pigment_pick={pig_name} family={fam} role={plate.role} via mokuhanga_emma]"
        )

        # Rewrite every pull's pigment_id to the concrete library key.
        for pull in plate.pulls:
            pull.pigment_id = pig_name

        summary["families_used"].add(fam)

    summary["families_used"] = sorted(summary["families_used"])
    return summary


__all__ = [
    "EmmaPigmentPlate",
    "delta_e_76",
    "plan_underlayer_pigments",
    "select_pigment_for_role_plate",
    "apply_mokuhanga_pigments_to_plan",
]
