"""Ring 4 placeholder — S2 SAM gateway shape.

Lands green at D5.1 (``test_mocked_sam_returns_8_regions``) when the
HTTP gateway + cache wrapper ship in
``backend.services.v23.stages.s2_sam``.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.xfail(reason="awaits D5.1 — stages/s2_sam.py")
def test_s2_sam_module_present() -> None:
    mod = importlib.import_module("backend.services.v23.stages.s2_sam")
    assert hasattr(mod, "segment"), "s2_sam must expose `segment(image)`"
