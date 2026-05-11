"""Ring 4 placeholder — S1 ``ingest_reference_image`` shape.

Lands green at D4.1 (``test_loads_png_returns_rgb_array``) when
``backend.services.v23.stages.s1_ingest`` is implemented.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.xfail(reason="awaits D4.1 — stages/s1_ingest.py")
def test_s1_ingest_module_present() -> None:
    mod = importlib.import_module("backend.services.v23.stages.s1_ingest")
    assert hasattr(mod, "ingest"), "s1_ingest must expose `ingest(path)`"
