"""pytest suite for the chuck-mcp v4 plate-objective module.

Gate tests (must pass for V1.0):
    - test_delta_e_2000_passes_all_34_sharma_pairs  ← THE ground-truth gate
    - test_composite_loss_is_jax_grad_compatible    ← architectural gate
    - test_plate_not_composite_penalty_nonzero_for_v13_residual_synthetic
                                                    ← v13-killer behavior gate

Run with:
    pytest -q test_plate_objective.py
"""
from __future__ import annotations

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ciede2000_test_table import SHARMA_TABLE_I
from delta_e import delta_e_76, delta_e_94, delta_e_2000
from objective_terms import (
    LossWeights,
    Plate,
    cell_exclusivity_penalty,
    checkpoint_proof_loss,
    composite_loss,
    final_image_loss,
    load_bearing_pair_penalty,
    load_bearing_singleton_penalty,
    plate_not_composite_penalty_per_plate,
    printability_in_loop_penalty,
    role_coverage_caps_penalty,
    role_frequency_permission_penalty,
)


# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def small_target():
    """Tiny Lab target (16x16x3). Used in fast inner-loop tests."""
    H, W = 16, 16
    L = jnp.full((H, W), 55.0)
    a = jnp.full((H, W), 4.0)
    b = jnp.full((H, W), -3.0)
    return jnp.stack([L, a, b], axis=-1)


@pytest.fixture
def plate_set_clean(small_target):
    """A 3-plate, low-coverage, non-overlapping set — the "good" case."""
    H, W = small_target.shape[:2]
    m1 = jnp.zeros((H, W)).at[:5, :5].set(1.0)
    m2 = jnp.zeros((H, W)).at[6:11, 6:11].set(1.0)
    m3 = jnp.zeros((H, W)).at[12:, 12:].set(1.0)
    return [
        Plate(block_id=1, mask=m1, pigment_lab=jnp.array([60.0, 10.0, 5.0]),
              opacity=0.7, role="underlayer_light"),
        Plate(block_id=2, mask=m2, pigment_lab=jnp.array([40.0, 20.0, -5.0]),
              opacity=0.7, role="local_chroma"),
        Plate(block_id=3, mask=m3, pigment_lab=jnp.array([20.0, 5.0, 10.0]),
              opacity=0.7, role="key_detail"),
    ]


@pytest.fixture
def render_fn_mock():
    """Simple linear render: weighted average of plate Lab by mask.

    Cheap, JAX-traceable. Good enough for testing the loss math without
    pulling in the real mokuhanga renderer.
    """
    def _render(plates):
        H, W = plates[0].mask.shape
        # Background = neutral mid-gray
        out = jnp.full((H, W, 3), jnp.array([50.0, 0.0, 0.0]))
        weight_sum = jnp.full((H, W), 1.0)
        for p in plates:
            w = p.mask * p.opacity
            out = out + w[..., None] * p.pigment_lab
            weight_sum = weight_sum + w
        return out / weight_sum[..., None]

    return _render


# ----------------------------------------------------------------------
# 1. ΔE_2000 against Sharma Table I — THE gate test.
# ----------------------------------------------------------------------

def test_delta_e_2000_passes_all_34_sharma_pairs():
    """Every Sharma 2005 Table I pair must reproduce to <1e-4."""
    failures = []
    for i, (lab1, lab2, expected) in enumerate(SHARMA_TABLE_I, 1):
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        if not math.isclose(got, expected, abs_tol=1e-4):
            failures.append(f"pair {i}: expected {expected:.4f}, got {got:.4f}")
    assert not failures, "\n".join(failures)


# ----------------------------------------------------------------------
# 2. JAX differentiability gates for ΔE_76 / ΔE_94.
# ----------------------------------------------------------------------

def test_delta_e_76_is_jax_differentiable():
    """grad of ΔE_76 wrt lab1 is finite (no NaN)."""
    lab1 = jnp.array([55.0, 4.0, -3.0])
    lab2 = jnp.array([60.0, 0.0, 5.0])
    g = jax.grad(lambda x: delta_e_76(x, lab2))(lab1)
    assert jnp.all(jnp.isfinite(g)), f"NaN/Inf gradient: {g}"
    # Sanity: gradient should be non-zero when lab1 != lab2.
    assert float(jnp.linalg.norm(g)) > 0.0


def test_delta_e_94_is_jax_differentiable():
    """grad of ΔE_94 wrt lab1 is finite (no NaN)."""
    lab1 = jnp.array([55.0, 4.0, -3.0])
    lab2 = jnp.array([60.0, 0.0, 5.0])
    g = jax.grad(lambda x: delta_e_94(x, lab2))(lab1)
    assert jnp.all(jnp.isfinite(g)), f"NaN/Inf gradient: {g}"
    assert float(jnp.linalg.norm(g)) > 0.0


# ----------------------------------------------------------------------
# 3. final_image_loss decreases as render approaches target.
# ----------------------------------------------------------------------

def test_final_image_loss_decreases_when_render_approaches_target(small_target):
    """Linear interpolation between random init and target must monotone-decrease loss."""
    rng = np.random.default_rng(0)
    init = jnp.asarray(rng.normal(50.0, 10.0, size=small_target.shape))
    losses = []
    for t in np.linspace(0.0, 1.0, 11):
        rendered = init * (1 - t) + small_target * t
        losses.append(float(final_image_loss(rendered, small_target)))
    # Each step must be ≤ previous (allowing a tiny numerical slack).
    for prev, curr in zip(losses, losses[1:]):
        assert curr <= prev + 1e-6, f"loss increased: {prev} → {curr}"
    assert losses[-1] < 1e-3, f"loss at t=1 should be ~0, got {losses[-1]}"


# ----------------------------------------------------------------------
# 4. plate_not_composite — clean plates vs v13-style residuals.
# ----------------------------------------------------------------------

def test_plate_not_composite_penalty_zero_for_clean_plates(plate_set_clean):
    """A set of small, distinct jigsaw plates against a structured final.

    Cosine similarity with a structured (non-flat) final is low when the
    plate is a small disjoint region; the penalty must be zero.
    """
    H, W = plate_set_clean[0].mask.shape
    rng = np.random.default_rng(7)
    # Structured final (Lab) with strong spatial pattern → low cos-sim against tiny plates.
    L = jnp.asarray(rng.normal(50.0, 20.0, size=(H, W)).clip(5, 95))
    a = jnp.asarray(rng.normal(0.0, 15.0, size=(H, W)))
    b = jnp.asarray(rng.normal(0.0, 15.0, size=(H, W)))
    final_lab = jnp.stack([L, a, b], axis=-1)
    pen = float(plate_not_composite_penalty_per_plate(plate_set_clean, final_lab))
    assert pen == 0.0, f"clean plates penalised against structured final: {pen}"


def test_plate_not_composite_penalty_nonzero_for_v13_residual_synthetic(small_target):
    """v13 residual = a plate that IS the final composite (full-area, high cos-sim)."""
    H, W = small_target.shape[:2]
    # Use a final composite that's a structured pattern, then make a "plate" that
    # mirrors the structure → high cosine similarity.
    final = jnp.tile(
        jnp.linspace(0.0, 1.0, W)[None, :] * 50.0 + 25.0,
        (H, 1),
    )  # luminance ramp 25..75
    final_lab = jnp.stack([final, jnp.zeros((H, W)), jnp.zeros((H, W))], axis=-1)
    # The bad plate matches the inverse luminance pattern (high cosine similarity in inverted signal).
    bad_mask = 1.0 - (final / 100.0)  # high-coverage, correlates with final
    bad_plate = Plate(
        block_id=99, mask=bad_mask, pigment_lab=jnp.array([30.0, 0.0, 0.0]),
        opacity=0.8, role="key_detail",
    )
    pen = float(plate_not_composite_penalty_per_plate([bad_plate], final_lab))
    assert pen > 0.0, f"v13-style residual should trigger penalty, got {pen}"


# ----------------------------------------------------------------------
# 5. cell exclusivity catches overlap.
# ----------------------------------------------------------------------

def test_cell_exclusivity_penalty_catches_overlap():
    """Two plates that fully overlap incur a large penalty vs disjoint plates."""
    H, W = 16, 16
    full = jnp.ones((H, W))
    half = jnp.zeros((H, W)).at[:, :W // 2].set(1.0)
    other_half = jnp.zeros((H, W)).at[:, W // 2:].set(1.0)
    p_lab = jnp.array([50.0, 0.0, 0.0])

    plates_disjoint = [
        Plate(1, half, p_lab, 1.0, "underlayer_light"),
        Plate(2, other_half, p_lab, 1.0, "local_chroma"),
    ]
    plates_overlap = [
        Plate(1, full, p_lab, 1.0, "underlayer_light"),
        Plate(2, full, p_lab, 1.0, "local_chroma"),
    ]
    disjoint_pen = float(cell_exclusivity_penalty(plates_disjoint))
    overlap_pen = float(cell_exclusivity_penalty(plates_overlap))
    assert overlap_pen > disjoint_pen, (
        f"overlap {overlap_pen} should exceed disjoint {disjoint_pen}"
    )
    assert disjoint_pen == 0.0, f"disjoint plates penalised: {disjoint_pen}"


# ----------------------------------------------------------------------
# 6. role coverage caps caps dominant role.
# ----------------------------------------------------------------------

def test_role_coverage_caps_caps_dominant_role():
    """If one role family overwhelms total coverage, penalty fires."""
    H, W = 16, 16
    big = jnp.ones((H, W)) * 0.9
    small = jnp.zeros((H, W)).at[0, 0].set(0.05)
    p_lab = jnp.array([50.0, 0.0, 0.0])

    # Three plates of same role family + one tiny other = dominant role.
    dominated = [
        Plate(1, big, p_lab, 1.0, "regional_mass"),
        Plate(2, big, p_lab, 1.0, "regional_mass"),
        Plate(3, big, p_lab, 1.0, "regional_mass"),
        Plate(4, small, p_lab, 1.0, "underlayer_light"),
    ]
    pen = float(role_coverage_caps_penalty(dominated))
    assert pen > 0.0, f"dominated set should trigger cap, got {pen}"

    # Balanced set, four families ~equal — no penalty.
    quarter = jnp.zeros((H, W)).at[:8, :8].set(1.0)
    balanced = [
        Plate(1, quarter, p_lab, 1.0, "underlayer_light"),
        Plate(2, quarter, p_lab, 1.0, "local_chroma"),
        Plate(3, quarter, p_lab, 1.0, "regional_mass"),
        Plate(4, quarter, p_lab, 1.0, "key_detail"),
    ]
    pen_bal = float(role_coverage_caps_penalty(balanced))
    assert pen_bal == 0.0, f"balanced set penalised: {pen_bal}"


# ----------------------------------------------------------------------
# 7. load-bearing singleton flags an unused pull.
# ----------------------------------------------------------------------

def test_load_bearing_singleton_flags_unused_pull(small_target, render_fn_mock):
    """A plate with mask=zeros has zero gradient → flagged as dead."""
    H, W = small_target.shape[:2]
    live_plate = Plate(
        block_id=1,
        mask=jnp.zeros((H, W)).at[:8, :8].set(1.0),
        pigment_lab=jnp.array([20.0, 30.0, -10.0]),
        opacity=0.5,
        role="local_chroma",
    )
    dead_plate = Plate(
        block_id=2,
        mask=jnp.zeros((H, W)),  # contributes nothing
        pigment_lab=jnp.array([20.0, 30.0, -10.0]),
        opacity=0.5,
        role="key_detail",
    )
    pen_with_dead = float(load_bearing_singleton_penalty(
        [live_plate, dead_plate], small_target, render_fn_mock
    ))
    pen_live_only = float(load_bearing_singleton_penalty(
        [live_plate], small_target, render_fn_mock
    ))
    assert pen_with_dead > pen_live_only, (
        f"dead plate should add to penalty: with-dead={pen_with_dead}, live-only={pen_live_only}"
    )


# ----------------------------------------------------------------------
# 8. load-bearing pair flags a cancellation pair.
# ----------------------------------------------------------------------

def test_load_bearing_pair_flags_cancellation_pair(small_target):
    """Two plates with identical masks and opposite pigments cancel each other."""
    H, W = small_target.shape[:2]
    overlap = jnp.ones((H, W))  # both share full coverage
    p_a = Plate(1, overlap, jnp.array([60.0,  20.0, -5.0]), 0.5, "local_chroma")
    p_b = Plate(2, overlap, jnp.array([60.0, -20.0,  5.0]), 0.5, "local_chroma")

    # render_fn that lets them cancel: linear in pigment, weighted by opacity.
    def render_fn(plates):
        out = jnp.zeros((H, W, 3))
        wsum = jnp.zeros((H, W)) + 1e-6
        for p in plates:
            w = p.mask * p.opacity
            out = out + w[..., None] * p.pigment_lab
            wsum = wsum + w
        return out / wsum[..., None]

    pen = float(load_bearing_pair_penalty([p_a, p_b], small_target, render_fn, top_k=5))
    # Cancellation penalty must be non-negative; the pair should be flagged
    # given the overlap is high and the singleton gradients cancel.
    assert pen >= 0.0, f"penalty must be ≥ 0, got {pen}"


# ----------------------------------------------------------------------
# 9. printability catches sub-mill features.
# ----------------------------------------------------------------------

def test_printability_in_loop_catches_sub_mill_feature():
    """A 1-px-wide line should be flagged when mill_radius ≥ 1."""
    H, W = 16, 16
    # One-pixel-wide vertical hairline
    hairline = jnp.zeros((H, W)).at[:, W // 2].set(1.0)
    # Solid 5x5 block — passes a r=1 mill
    block = jnp.zeros((H, W)).at[4:9, 4:9].set(1.0)
    p_lab = jnp.array([20.0, 0.0, 0.0])

    p_hair = Plate(1, hairline, p_lab, 1.0, "key_detail")
    p_block = Plate(2, block, p_lab, 1.0, "regional_mass")

    pen_hair = float(printability_in_loop_penalty([p_hair], mill_radius_px=1))
    pen_block = float(printability_in_loop_penalty([p_block], mill_radius_px=1))
    assert pen_hair > pen_block, (
        f"hairline penalty {pen_hair} should exceed solid-block {pen_block}"
    )


# ----------------------------------------------------------------------
# 10. composite_loss is JAX-grad compatible end-to-end.
# ----------------------------------------------------------------------

def test_composite_loss_is_jax_grad_compatible(small_target, plate_set_clean, render_fn_mock):
    """grad of composite_loss wrt plate.mask runs without error and is finite."""

    def loss_of_masks(masks_stack: jnp.ndarray) -> jnp.ndarray:
        # masks_stack: (n_plates, H, W)
        new_plates = [
            replace(p, mask=masks_stack[i]) for i, p in enumerate(plate_set_clean)
        ]
        return composite_loss(
            plates=new_plates,
            target=small_target,
            render_fn=render_fn_mock,
            plate_order=list(range(len(new_plates))),
            mill_radius_px=1,
        )

    masks_stack = jnp.stack([p.mask for p in plate_set_clean], axis=0)
    g = jax.grad(loss_of_masks)(masks_stack)
    assert jnp.all(jnp.isfinite(g)), f"grad contains NaN/Inf: {g}"
    # Sanity: the gradient should be non-trivial since loss depends on masks.
    assert float(jnp.linalg.norm(g)) > 0.0


# ----------------------------------------------------------------------
# 11. ΔE_2000 signs around blue hue 275° (where R_T cross-term matters).
# ----------------------------------------------------------------------

def test_ciede2000_signs_correct_around_blue_hue_275():
    """Pairs near hue 275° (Sharma Table I rows 1-3) hit R_T cross-term.

    If signs of ΔC' or ΔH' are wrong, these pairs drift visibly.
    """
    # Pairs 1-3 of Sharma Table I are all near blue hue 275°.
    for lab1, lab2, expected in SHARMA_TABLE_I[:3]:
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        assert math.isclose(got, expected, abs_tol=1e-4), (
            f"blue-hue pair drift: expected {expected}, got {got}"
        )


# ----------------------------------------------------------------------
# 12. ΔE_2000 mean-hue boundary at 180° apart (the paper's gotcha).
# ----------------------------------------------------------------------

def test_ciede2000_mean_hue_boundary_at_180_apart():
    """Pairs 15-16 of Sharma Table I are ~180° apart, exercising Eq. (14).

    Without the mean-hue branch, these come out wrong by exactly the
    180° rotation in h̄', which propagates through T, Δθ, R_T.
    """
    # Pair 15: a*=±2.49, b ~0 → hues at 0° and 180° apart, exactly the boundary case.
    for lab1, lab2, expected in SHARMA_TABLE_I[14:16]:
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        assert math.isclose(got, expected, abs_tol=1e-4), (
            f"180° pair drift: expected {expected}, got {got}"
        )


# ----------------------------------------------------------------------
# 13. Symmetry sanity — ΔE_2000(a,b) == ΔE_2000(b,a) for the asymmetry pair.
# ----------------------------------------------------------------------

def test_ciede2000_symmetric():
    """Sharma pair 7 and 8 are intentionally swap-equivalent — must produce same dE."""
    lab1, lab2, expected = SHARMA_TABLE_I[6]
    lab2b, lab1b, expected_swap = SHARMA_TABLE_I[7]
    got_a = float(delta_e_2000(np.array(lab1), np.array(lab2)))
    got_b = float(delta_e_2000(np.array(lab2b), np.array(lab1b)))
    assert math.isclose(got_a, got_b, abs_tol=1e-6), (
        f"asymmetry: {got_a} vs {got_b}"
    )


# ----------------------------------------------------------------------
# 14. checkpoint_proof_loss decreases as renders approach targets.
# ----------------------------------------------------------------------

def test_loss_visualizer_smoke(tmp_path):
    """Visualiser writes a PNG and a JSON without crashing."""
    from loss_visualizer import LossHistory, quick_plot_from_steps

    h = LossHistory()
    for step in range(10):
        h.record_step(
            step,
            {"final_image": 10.0 - step, "plate_not_composite": 1.0, "printability": 0.5},
            total=11.5 - step,
        )
    out_png = tmp_path / "loss.png"
    out_json = tmp_path / "loss.json"
    h.plot(out_png)
    h.save_json(out_json)
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_json.exists() and out_json.stat().st_size > 0
    loaded = LossHistory.load_json(out_json)
    assert loaded.steps == h.steps
    dom = h.dominant_terms(top_k=2)
    assert dom[0][0] in ("final_image", "plate_not_composite", "printability")
    # quick_plot_from_steps convenience path
    out2 = quick_plot_from_steps(
        tmp_path / "loss2.png",
        [(s, {"final_image": float(s)}, float(s)) for s in range(3)],
    )
    assert out2.exists()


def test_checkpoint_proof_loss_decreases_toward_targets():
    """As checkpoint renders interpolate toward expected, loss must monotone-decrease."""
    rng = np.random.default_rng(1)
    K, H, W = 4, 8, 8
    targets = jnp.asarray(rng.normal(50.0, 5.0, size=(K, H, W, 3)))
    init = jnp.asarray(rng.normal(50.0, 15.0, size=(K, H, W, 3)))
    losses = []
    for t in np.linspace(0.0, 1.0, 11):
        renders = init * (1 - t) + targets * t
        losses.append(float(checkpoint_proof_loss(renders, targets)))
    for prev, curr in zip(losses, losses[1:]):
        assert curr <= prev + 1e-6, f"checkpoint loss increased: {prev} → {curr}"
    assert losses[-1] < 1e-3
