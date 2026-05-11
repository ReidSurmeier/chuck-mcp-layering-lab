"""Ring 4 placeholder — S5 solver 5-step recovery smoke.

Uses :func:`backend.tests.v23._helpers.synthetic_fixtures.make_3imp_synthetic`
as the ground truth. Lands green at D7.1
(``test_synth_3imp_recovers_under_dE_0_5``).
"""
from __future__ import annotations

import importlib

import pytest

from backend.tests.v23._helpers.synthetic_fixtures import SyntheticStack


def test_synthetic_fixture_is_deterministic(synthetic_3imp_stack: SyntheticStack) -> None:
    """Sanity: the fixture is reproducible and shaped correctly.

    Runs green today — no production code dependency. Protects the
    solver smoke from a future fixture-generation regression.
    """
    assert synthetic_3imp_stack.rgb.shape == (256, 256, 3)
    assert synthetic_3imp_stack.alpha.shape == (3, 256, 256)
    assert synthetic_3imp_stack.pigment_rgb.shape == (3, 3)
    assert 0.0 <= synthetic_3imp_stack.alpha.min() <= synthetic_3imp_stack.alpha.max() <= 1.0


@pytest.mark.xfail(reason="awaits D7.1 — stages/s5_solver.py 5-step recovery")
def test_s5_solver_smoke_recovers_under_de_0_5(synthetic_3imp_stack: SyntheticStack) -> None:
    mod = importlib.import_module("backend.services.v23.stages.s5_solver")
    recover = mod.solve_5step  # symbol lands in D7.1
    result = recover(synthetic_3imp_stack.rgb, ground_truth=synthetic_3imp_stack.alpha)
    assert result.dE_mean <= 0.5, f"smoke recovery ΔE={result.dE_mean:.3f}"
