"""Unit tests for the v23 SAM HTTP endpoint (POST /api/sam).

Mocks SAM2.1 so tests run without ``ultralytics`` or CUDA. Covers:
1. happy path → 200 + valid SAMRegion list
2. invalid upload → 400
3. SAM timeout → 504 ``SAM_TIMEOUT``
4. cached call → SAM not re-invoked, ``cached=True``
5. cache key includes ``min_region_area_px``
6. regions below min_area dropped

Authority: ``/tmp/research-v23-mcp-interfaces.md`` §1.1
"""
from __future__ import annotations

import base64
import hashlib
import io
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


def _make_png(w: int = 64, h: int = 64) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = (220, 40, 40)
    arr[: h // 2, w // 2 :] = (40, 200, 60)
    arr[h // 2 :, : w // 2] = (40, 60, 220)
    arr[h // 2 :, w // 2 :] = (240, 220, 50)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class _FakeMask:
    """Mimic ``ultralytics.Results.masks.data`` torch-like tensor."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeMask":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeResults:
    def __init__(self, masks_arr: np.ndarray | None) -> None:
        if masks_arr is None:
            self.masks = None
        else:
            class _M:
                data = _FakeMask(masks_arr)
            self.masks = _M()


class _FakeSamModel:
    def __init__(self, masks_arr: np.ndarray | None, sleep_s: float = 0.0) -> None:
        self._masks = masks_arr
        self._sleep_s = sleep_s
        self.call_count = 0

    def predict(self, img, verbose: bool = False):  # noqa: ARG002, ANN001
        self.call_count += 1
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        return [_FakeResults(self._masks)]


def _two_quadrant_masks(h: int = 64, w: int = 64) -> np.ndarray:
    masks = np.zeros((2, h, w), dtype=bool)
    masks[0, : h // 2, :] = True
    masks[1, h // 2 :, :] = True
    return masks


@pytest.fixture
def app_with_route(monkeypatch):
    from routes import sam_v23  # noqa: PLC0415
    sam_v23._SAM_CACHE.clear()  # type: ignore[attr-defined]
    fake = _FakeSamModel(_two_quadrant_masks())
    monkeypatch.setattr(sam_v23, "_load_sam_model", lambda: fake)
    app = FastAPI()
    app.include_router(sam_v23.router, prefix="/api")
    return app, fake, sam_v23


def test_sam_post_returns_regions_on_valid_image(app_with_route):
    app, fake, _ = app_with_route
    png = _make_png()
    sha = hashlib.sha256(png).hexdigest()
    with TestClient(app) as cx:
        r = cx.post(
            "/api/sam",
            files={"file": ("img.png", png, "image/png")},
            data={"min_region_area_px": "100", "max_regions": "8"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_sha256"] == sha
    assert len(body["regions"]) == 2
    for i, reg in enumerate(body["regions"]):
        assert reg["region_id"] == f"rgn_{i:03d}"
        assert len(reg["bbox"]) == 4
        assert all(isinstance(v, int) for v in reg["bbox"])
        assert reg["area_px"] > 0
        Image.open(io.BytesIO(base64.b64decode(reg["mask_png_b64"]))).verify()
        assert len(reg["mean_oklab"]) == 3
        assert all(isinstance(v, float) for v in reg["mean_oklab"])
    assert body["sam_wall_s"] >= 0.0
    assert body["cached"] is False
    assert fake.call_count == 1


def test_sam_post_rejects_invalid_upload(app_with_route):
    app, fake, _ = app_with_route
    with TestClient(app) as cx:
        r = cx.post("/api/sam", files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
    assert "error" in r.json()
    assert fake.call_count == 0


def test_sam_post_caches_second_call(app_with_route):
    app, fake, _ = app_with_route
    png = _make_png()
    with TestClient(app) as cx:
        r1 = cx.post("/api/sam", files={"file": ("img.png", png, "image/png")},
                     data={"min_region_area_px": "100"})
        r2 = cx.post("/api/sam", files={"file": ("img.png", png, "image/png")},
                     data={"min_region_area_px": "100"})
    assert r1.status_code == r2.status_code == 200
    assert fake.call_count == 1
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is True
    assert r1.json()["regions"] == r2.json()["regions"]


def test_sam_post_different_min_area_busts_cache(app_with_route):
    """Cache key includes ``min_region_area_px``."""
    app, fake, _ = app_with_route
    png = _make_png()
    with TestClient(app) as cx:
        cx.post("/api/sam", files={"file": ("img.png", png, "image/png")},
                data={"min_region_area_px": "100"})
        cx.post("/api/sam", files={"file": ("img.png", png, "image/png")},
                data={"min_region_area_px": "500"})
    assert fake.call_count == 2


def test_sam_post_returns_504_on_timeout(monkeypatch):
    from routes import sam_v23  # noqa: PLC0415
    sam_v23._SAM_CACHE.clear()  # type: ignore[attr-defined]
    monkeypatch.setattr(sam_v23, "SAM_TIMEOUT_SECONDS", 0.05)
    slow = _FakeSamModel(_two_quadrant_masks(), sleep_s=0.5)
    monkeypatch.setattr(sam_v23, "_load_sam_model", lambda: slow)
    app = FastAPI()
    app.include_router(sam_v23.router, prefix="/api")
    with TestClient(app) as cx:
        r = cx.post("/api/sam", files={"file": ("img.png", _make_png(), "image/png")})
    assert r.status_code == 504
    assert r.json().get("code") == "SAM_TIMEOUT"


def test_sam_post_filters_below_min_area(app_with_route):
    """Regions below ``min_region_area_px`` are dropped."""
    app, _, sam_v23 = app_with_route
    sam_v23._SAM_CACHE.clear()  # type: ignore[attr-defined]
    with TestClient(app) as cx:
        r = cx.post("/api/sam", files={"file": ("img.png", _make_png(), "image/png")},
                    data={"min_region_area_px": "10000"})
    assert r.status_code == 200
    assert r.json()["regions"] == []
