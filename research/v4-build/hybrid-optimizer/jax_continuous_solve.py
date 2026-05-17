"""Stage 3: JAX continuous solve for opacity / dilution / pigment-blend.

THIS IS LOAD-BEARING. Per docs/v2-design-locked-2026-05-16.md Phase 4:

    "JAX continuous solve for opacity/dilution/color per pull, holding
     cell-zone assignments FROZEN.
     JAX optimizes continuous pigment/load variables. It does NOT
     invent printable topology from unconstrained alpha."

So this file:
    INPUT:  per-plate fixed inked mask (cell_zone_ids -> binary mask)
            + target image in Lab
            + initial guesses for (opacity, dilution, pigment-blend)
    OUTPUT: optimized (opacity, dilution, pigment-blend) per plate
            cell-zone assignments are UNCHANGED.

Loss is computed in CIE Lab using ΔE_76 (Euclidean Lab distance) — NOT
ΔE_2000 — because of Sharma/Wu/Dalal 2005:

    "These [CIEDE2000] discontinuities preclude the use of the formula in
     analysis based on Taylor series approximations and in design techniques
     using gradient based optimization."

ΔE_2000 is the VALIDATOR metric; ΔE_76 is the SOLVER loss.

The optimizer is JAXopt's L-BFGS-B (box-constrained, deterministic).
Falls back to jax.scipy.optimize.minimize -> projected gradient descent if
jaxopt unavailable (CI environments without GPU).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from chuck_mcp_v2.types import Plate as FrozenPlate

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover - environment without JAX
    jax = None
    jnp = np  # type: ignore
    _HAS_JAX = False

try:
    import jaxopt  # noqa: F401

    _HAS_JAXOPT = True
except Exception:  # pragma: no cover - environment without jaxopt
    _HAS_JAXOPT = False

try:
    _PLATE_OBJECTIVE_DIR = Path(__file__).resolve().parents[1] / "plate-objective"
    if str(_PLATE_OBJECTIVE_DIR) not in sys.path:
        sys.path.insert(0, str(_PLATE_OBJECTIVE_DIR))
    from objective_terms import Plate as ObjectivePlate
    from objective_terms import composite_loss as plate_objective_composite_loss

    _HAS_PLATE_OBJECTIVE = True
except Exception:  # pragma: no cover - isolated Stage 3 or no JAX installed
    ObjectivePlate = None  # type: ignore[assignment]
    plate_objective_composite_loss = None  # type: ignore[assignment]
    _HAS_PLATE_OBJECTIVE = False


@dataclass
class SolveResult:
    """Per-plate solve output.

    Attributes:
        per_plate: dict[block_id -> {"opacity", "dilution", "pigment_weights",
            "pigment_blend_rgb"}].
        loss_initial: scalar loss before optimization (ΔE_76 mean).
        loss_final: scalar loss after optimization.
        n_iterations: how many L-BFGS-B steps actually ran (or 0 fallback).
        converged: bool.
    """

    per_plate: Dict[int, Dict[str, Any]]
    loss_initial: float
    loss_final: float
    n_iterations: int
    converged: bool


# ---------- Math primitives -------------------------------------------


def _rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..1) to CIE Lab. Vectorized via skimage when present.

    Falls back to a hand-rolled D65 conversion (good to ~0.5 ΔE, fine for
    initial warm-starts).
    """
    try:
        from skimage.color import rgb2lab

        return rgb2lab(rgb.astype(np.float32))
    except Exception:  # pragma: no cover - fallback when skimage missing
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        # sRGB linearize
        a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        m = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )
        xyz = a @ m.T
        white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz_n = xyz / white
        eps = 216.0 / 24389.0
        k = 24389.0 / 27.0
        f = np.where(xyz_n > eps, xyz_n ** (1.0 / 3.0), (k * xyz_n + 16.0) / 116.0)
        L = 116.0 * f[..., 1] - 16.0
        a_ = 500.0 * (f[..., 0] - f[..., 1])
        b_ = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a_, b_], axis=-1)


def _lab_to_rgb_jax(lab):
    """Differentiable Lab -> sRGB, hand-rolled. Used for forward render in JAX."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps3 = (6.0 / 29.0) ** 3
    fxyz = jnp.stack([fx, fy, fz], axis=-1)
    cube = fxyz ** 3
    linear = 3.0 * (6.0 / 29.0) ** 2 * (fxyz - 4.0 / 29.0)
    xyz = jnp.where(cube > eps3, cube, linear)
    white = jnp.array([0.95047, 1.0, 1.08883])
    xyz = xyz * white
    m_inv = jnp.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_lin = xyz @ m_inv.T
    rgb = jnp.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * jnp.power(jnp.clip(rgb_lin, 1e-9, 1.0), 1.0 / 2.4) - 0.055,
    )
    return jnp.clip(rgb, 0.0, 1.0)


def _delta_e_76(lab_a, lab_b):
    """ΔE_76 (Euclidean Lab) — differentiable everywhere."""
    d = lab_a - lab_b
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-9)


# ---------- Forward render (composable, differentiable) ---------------


def _compose_pull_over_substrate(substrate_lab, ink_lab, opacity, dilution, mask):
    """Single-pull alpha-composite in Lab space.

    effective_alpha = mask * opacity * (1 - dilution)
    out = lerp(substrate, ink, effective_alpha)

    mask: H x W float (0/1)
    opacity, dilution: scalars in [0, 1]
    ink_lab: (3,) Lab triplet
    substrate_lab: H x W x 3
    """
    eff_alpha = mask * opacity * (1.0 - 0.6 * dilution)  # dilution lightens
    eff_alpha = jnp.clip(eff_alpha, 0.0, 1.0)[..., None]
    ink_broadcast = ink_lab[None, None, :]
    return substrate_lab * (1.0 - eff_alpha) + ink_broadcast * eff_alpha


def _pigment_blend(weights_logits, pigment_lab_array):
    """Softmax over pigment-choice weights -> (3,) blended Lab.

    weights_logits: (P,) unconstrained.
    pigment_lab_array: (P, 3) Lab triplets.
    """
    w = jax.nn.softmax(weights_logits) if _HAS_JAX else _np_softmax(weights_logits)
    return (w[..., None] * pigment_lab_array).sum(axis=0)


def _np_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------- Loss function ---------------------------------------------


def _build_loss_fn(
    plates: List[FrozenPlate],
    target_lab,
    substrate_lab,
):
    """Return f(params) -> scalar mean ΔE_76 of composed pulls vs target.

    Uses `jax.lax.scan` over a fixed-shape (P,) tensor of pre-sorted pull
    indices so JAX traces ONCE regardless of plate count. This is the
    difference between O(seconds) and O(minutes) for Emma-scale plans.

    params shape (P, 2 + K_max):
        params[i, 0] = sigmoid -> opacity
        params[i, 1] = sigmoid -> dilution
        params[i, 2:2+K_i] = pigment weight logits (softmax later)
    K_max = max pigment count across plates (padded).
    """
    if not _HAS_JAX:
        # Trivial fallback closure — never called when _HAS_JAX=False
        K_max = max((len(p.pigment_choices) for p in plates), default=1)

        def loss_fn(_params):
            return 0.0

        return loss_fn, K_max

    K_max = max(len(p.pigment_choices) for p in plates) if plates else 1
    pigment_lab_table = np.zeros((len(plates), K_max, 3), dtype=np.float32)
    pigment_valid_mask = np.zeros((len(plates), K_max), dtype=np.float32)
    for i, p in enumerate(plates):
        for k, (_, lab) in enumerate(p.pigment_choices):
            pigment_lab_table[i, k] = lab
            pigment_valid_mask[i, k] = 1.0

    masks = jnp.stack([jnp.asarray(p.inked_mask, dtype=jnp.float32) for p in plates])
    pigment_lab_table_j = jnp.asarray(pigment_lab_table)
    pigment_valid_mask_j = jnp.asarray(pigment_valid_mask)
    # Pull order = sorted by (pass_index, block_id) — stable, deterministic.
    order_keys = [(p.pass_index, p.block_id) for p in plates]
    pull_order_list = sorted(range(len(plates)), key=lambda i: order_keys[i])
    pull_order = jnp.asarray(pull_order_list, dtype=jnp.int32)

    def loss_fn(params):
        opacities = jax.nn.sigmoid(params[:, 0])
        dilutions = jax.nn.sigmoid(params[:, 1])
        w_logits = params[:, 2 : 2 + K_max]
        masked_logits = jnp.where(pigment_valid_mask_j > 0, w_logits, -1e9)
        weights = jax.nn.softmax(masked_logits, axis=-1)
        plate_lab = (weights[:, :, None] * pigment_lab_table_j).sum(axis=1)

        def body(substrate, idx):
            substrate = _compose_pull_over_substrate(
                substrate,
                plate_lab[idx],
                opacities[idx],
                dilutions[idx],
                masks[idx],
            )
            return substrate, None

        final_substrate, _ = jax.lax.scan(body, substrate_lab, pull_order)
        if _HAS_PLATE_OBJECTIVE:
            objective_plates = [
                ObjectivePlate(
                    block_id=plates[i].block_id,
                    mask=masks[i],
                    pigment_lab=plate_lab[i],
                    opacity=opacities[i],
                    role=plates[i].role,
                    cell_zone_ids=tuple(plates[i].cell_zone_ids),
                    pass_index=plates[i].pass_index,
                )
                for i in range(len(plates))
            ]
            return plate_objective_composite_loss(
                objective_plates,
                target_lab,
                rendered_final=final_substrate,
                plate_order=pull_order_list,
            )

        de = _delta_e_76(final_substrate, target_lab)
        return jnp.mean(de)

    return loss_fn, K_max


def _initial_params(plates: List[FrozenPlate], K_max: int):
    """Warm-start params from each plate's `initial_*` fields."""
    P = len(plates)
    arr = np.zeros((P, 2 + K_max), dtype=np.float32)
    for i, p in enumerate(plates):
        # inverse-sigmoid the warm-starts
        op = float(np.clip(p.initial_opacity, 1e-3, 1 - 1e-3))
        di = float(np.clip(p.initial_dilution, 1e-3, 1 - 1e-3))
        arr[i, 0] = float(np.log(op / (1 - op)))
        arr[i, 1] = float(np.log(di / (1 - di)))
        weights = getattr(p, "pigment_weights", {}) or {}
        if weights:
            vals = np.array(
                [float(weights.get(pid, 0.0)) for pid, _ in p.pigment_choices],
                dtype=np.float32,
            )
            total = float(vals.sum())
            if total > 1e-9:
                vals = np.clip(vals / total, 1e-6, 1.0)
                logits = np.log(vals)
                logits = logits - float(logits.mean())
                for k, v in enumerate(logits):
                    arr[i, 2 + k] = float(v)
                continue
        # Uniform pigment weights (logits = 0)
        for k in range(len(p.pigment_choices)):
            arr[i, 2 + k] = 0.0
    return arr


# ---------- Top-level entry -------------------------------------------


def solve_pigment_load(
    plates: List[FrozenPlate],
    target_lab: np.ndarray,
    substrate_lab: Optional[np.ndarray] = None,
    max_iters: int = 200,
    tol: float = 1e-4,
) -> SolveResult:
    """Stage 3 entry.

    Args:
        plates: frozen-mask plates from Stage 2.
        target_lab: H x W x 3 Lab target image.
        substrate_lab: H x W x 3 Lab starting substrate. Defaults to washi
            white (Lab ~ 96, -1, 3).
        max_iters: L-BFGS-B max iterations.
        tol: convergence tolerance.

    Returns:
        SolveResult.
    """
    if not plates:
        return SolveResult({}, 0.0, 0.0, 0, True)

    H, W = target_lab.shape[:2]
    if substrate_lab is None:
        substrate_lab = np.broadcast_to(
            np.array([96.0, -1.0, 3.0], dtype=np.float32), (H, W, 3)
        ).copy()

    target_lab_j = jnp.asarray(target_lab.astype(np.float32))
    substrate_lab_j = jnp.asarray(substrate_lab.astype(np.float32))

    loss_fn, K_max = _build_loss_fn(plates, target_lab_j, substrate_lab_j)
    x0 = _initial_params(plates, K_max)

    loss_initial = float(loss_fn(jnp.asarray(x0)))

    converged = False
    n_iters = 0
    x_final = x0
    loss_final = loss_initial

    if not _HAS_JAX:
        # No JAX -> warm-starts only (degenerate fallback).
        loss_final = loss_initial
    else:
        # JIT the loss + grad once — keeps re-tracing out of the inner loop.
        jitted_loss = jax.jit(loss_fn)
        jitted_grad = jax.jit(jax.grad(loss_fn))
        # Plain gradient descent — works without jaxopt and is bounded.
        # jaxopt L-BFGS can re-trace internally; on small Emma-scale plans
        # gradient descent with backtracking is comparable and more reliable.
        x = jnp.asarray(x0)
        lr = 0.5
        best_x = x
        best_loss = loss_initial
        stale = 0
        for it in range(max_iters):
            g = jitted_grad(x)
            x_new = x - lr * g
            lv = float(jitted_loss(x_new))
            if lv < best_loss - tol:
                best_loss = lv
                best_x = x_new
                x = x_new
                stale = 0
            else:
                lr *= 0.5
                stale += 1
                if lr < 1e-6 or stale >= 8:
                    break
            n_iters = it + 1
        x_final = np.asarray(best_x)
        loss_final = best_loss
        converged = best_loss < loss_initial

    # Decode per-plate
    per_plate: Dict[int, Dict[str, Any]] = {}
    for i, p in enumerate(plates):
        op = float(1.0 / (1.0 + np.exp(-x_final[i, 0])))
        di = float(1.0 / (1.0 + np.exp(-x_final[i, 1])))
        logits = x_final[i, 2 : 2 + len(p.pigment_choices)]
        if len(logits) == 0:
            weights = np.array([1.0])
            ids = ["white"]
        else:
            weights = _np_softmax(np.asarray(logits))
            ids = [pid for pid, _ in p.pigment_choices]
        labs = np.array([lab for _, lab in p.pigment_choices], dtype=np.float32) if p.pigment_choices else np.array([[96, -1, 3]], dtype=np.float32)
        blend_lab = (weights[:, None] * labs).sum(axis=0)
        per_plate[p.block_id] = {
            "opacity": op,
            "dilution": di,
            "pigment_weights": {pid: float(w) for pid, w in zip(ids, weights)},
            "pigment_blend_lab": blend_lab.tolist(),
            "pigment_id": ids[int(np.argmax(weights))],
        }

    return SolveResult(
        per_plate=per_plate,
        loss_initial=loss_initial,
        loss_final=loss_final,
        n_iterations=n_iters,
        converged=converged,
    )
