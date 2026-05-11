"""Ring 5 placeholder — full-flow ΔE regression gate.

Real gate lands incrementally:

- D10.1: 1 Tier-1 fixture (``close_emma_2002``) under ΔE 3.0
- D15.1: 5/5 Tier-1 under ΔE 1.5 mean / 3.0 p95
- D23.3: 3-corpus golden-path SHIP gate

This placeholder verifies the fixture-loading machinery (the YAML +
parametrize wiring) is itself correct so the gate test that lands in
D10 only has to focus on solver behaviour.
"""
from __future__ import annotations

import pytest

from backend.tests.v23.corpus.conftest import CorpusFixture


def test_corpus_fixture_directory_exists(corpus_fixture: CorpusFixture) -> None:
    """Run today — every fixture in corpus_tiers.yaml is on disk."""
    assert corpus_fixture.path.is_dir(), f"missing corpus dir: {corpus_fixture.path}"
    originals = list(corpus_fixture.path.glob("original.*"))
    assert originals, f"no original.* in {corpus_fixture.path}"


@pytest.mark.xfail(reason="awaits D10.1 — real propose_stack + ΔE gate")
def test_corpus_dE_under_gate(corpus_fixture: CorpusFixture) -> None:
    from backend.mcp.tools import core as _core  # lands in D9.4

    image_path = next(corpus_fixture.path.glob("original.*"))
    result = _core.propose_stack(image_path=str(image_path), solve_profile="default")
    de_mean = result.reconstruction_dE_mean
    de_p95 = result.reconstruction_dE_p95
    assert de_mean <= corpus_fixture.de_mean_gate, (
        f"{corpus_fixture.id}: ΔE mean {de_mean:.2f} > {corpus_fixture.de_mean_gate}"
    )
    assert de_p95 <= corpus_fixture.de_p95_gate, (
        f"{corpus_fixture.id}: ΔE p95 {de_p95:.2f} > {corpus_fixture.de_p95_gate}"
    )
