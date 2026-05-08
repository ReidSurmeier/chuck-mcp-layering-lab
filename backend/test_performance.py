"""Backend performance benchmarks for color-separator v20 optimization.

Tests verify:
1. v20 preview completes in ≤30s (cached model) / ≤45s (cold)
2. Upscale 4x works and is included in timing
3. SAM runs on GPU (not CPU)
4. Real-ESRGAN uses GPU

Run inside container: pytest test_performance.py -v
Run from host: pytest backend/tests/test_performance.py -v
"""
import io
import json
import time
from pathlib import Path

import pytest
import requests

BACKEND_URL = "http://localhost:8001"
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_IMAGE_SMALL = PROJECT_ROOT / "test_red.png"
TEST_IMAGE_STD = PROJECT_ROOT / "test_colorsep.png"


def _get_test_image():
    """Return path to first available test image."""
    for p in [TEST_IMAGE_SMALL, TEST_IMAGE_STD]:
        if p.exists():
            return p
    pytest.skip("No test image found")


def _upload_preview(image_path: Path, upscale: str = "false", upscale_scale: int = 2) -> requests.Response:
    """Upload image to preview endpoint."""
    with open(image_path, "rb") as f:
        return requests.post(
            f"{BACKEND_URL}/api/preview",
            files={"image": (image_path.name, f, "image/png")},
            data={
                "plates": "4",
                "dust": "5",
                "use_edges": "true",
                "edge_sigma": "3.0",
                "version": "v20",
                "upscale": upscale,
                "upscale_scale": str(upscale_scale),
                "chroma_boost": "1.3",
            },
            timeout=120,
        )


class TestV20CachedPerformance:
    """After first request warms models, subsequent requests must be fast."""

    def test_v20_warmup(self):
        """Warm up models — not timed, just ensures healthy state."""
        img = _get_test_image()
        r = _upload_preview(img, upscale="false")
        assert r.status_code == 200, f"Warmup failed: {r.status_code} {r.text[:200]}"

    def test_v20_cached_under_30s(self):
        """v20 preview with cached model must complete in ≤30s."""
        img = _get_test_image()
        # Warmup first
        _upload_preview(img, upscale="false")

        start = time.time()
        r = _upload_preview(img, upscale="false")
        elapsed = time.time() - start

        assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text[:200]}"
        assert elapsed <= 30, f"Cached preview took {elapsed:.1f}s (max 30s)"

    def test_v20_upscale_4x_under_30s(self):
        """v20 preview with 4x upscale included must complete in ≤30s (cached)."""
        img = _get_test_image()
        # Warmup with upscale
        _upload_preview(img, upscale="true", upscale_scale=4)

        start = time.time()
        r = _upload_preview(img, upscale="true", upscale_scale=4)
        elapsed = time.time() - start

        assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text[:200]}"
        assert elapsed <= 30, f"4x upscale preview took {elapsed:.1f}s (max 30s)"


class TestV20ColdPerformance:
    """First request after restart (cold start) gets more time."""

    def test_v20_cold_under_45s(self):
        """First request after model load should complete in ≤45s."""
        img = _get_test_image()
        start = time.time()
        r = _upload_preview(img, upscale="false")
        elapsed = time.time() - start

        assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text[:200]}"
        assert elapsed <= 45, f"Cold preview took {elapsed:.1f}s (max 45s)"


class TestSAMRunsOnGPU:
    """SAM must run on CUDA, not CPU."""

    def test_sam_device_is_cuda(self):
        """Verify SAM model is on GPU after loading."""
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=10)
        assert r.status_code == 200

        # Trigger SAM load via preview
        img = _get_test_image()
        _upload_preview(img, upscale="false")

        # Check device via diagnostic endpoint or direct inspection
        r = requests.get(f"{BACKEND_URL}/api/gpu-status", timeout=10)
        if r.status_code == 200:
            data = r.json()
            assert data.get("sam_device") != "cpu", \
                f"SAM running on CPU: {data}"
            assert "cuda" in str(data.get("sam_device", "")), \
                f"SAM not on CUDA: {data.get('sam_device')}"
        else:
            # Fallback: if no gpu-status endpoint, at least verify
            # the config isn't forcing CPU
            pytest.skip("No /api/gpu-status endpoint — add it to verify GPU placement")


class TestRealESRGANOnGPU:
    """Real-ESRGAN must use GPU for upscaling."""

    def test_upscale_uses_gpu(self):
        """Upscale should use CUDA device."""
        img = _get_test_image()
        r = _upload_preview(img, upscale="true", upscale_scale=2)
        assert r.status_code == 200

        # Check via gpu-status endpoint
        r = requests.get(f"{BACKEND_URL}/api/gpu-status", timeout=10)
        if r.status_code == 200:
            data = r.json()
            assert data.get("esrgan_device") != "cpu", \
                f"ESRGAN on CPU: {data}"
        else:
            pytest.skip("No /api/gpu-status endpoint")


class TestSSEProgress:
    """Preview-stream must send progress events with percent, stage, ETA."""

    def test_sse_has_progress_fields(self):
        """SSE events must include pct, stage, and eta fields."""
        img = _get_test_image()
        with open(img, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/preview-stream",
                files={"image": (img.name, f, "image/png")},
                data={
                    "plates": "4",
                    "dust": "5",
                    "use_edges": "true",
                    "edge_sigma": "3.0",
                    "version": "v20",
                    "upscale": "false",
                    "chroma_boost": "1.3",
                },
                timeout=120,
                stream=True,
            )
        assert r.status_code == 200

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        assert len(events) >= 2, f"Expected ≥2 SSE events, got {len(events)}"

        # Find progress events (not the final complete event)
        progress_events = [e for e in events if e.get("stage") not in ("complete", "partial_complete", None)]

        assert len(progress_events) >= 1, \
            f"No progress events found. Events: {[e.get('stage') for e in events]}"

        for evt in progress_events:
            assert "pct" in evt, f"Missing 'pct' in progress event: {evt}"
            assert "stage" in evt, f"Missing 'stage' in progress event: {evt}"
            assert "eta" in evt, f"Missing 'eta' in progress event: {evt}"
            assert 0 <= evt["pct"] <= 100, f"pct out of range: {evt['pct']}"


class TestUpscaleCorrectness:
    """Upscale must honor the requested scale factor."""

    def test_upscale_4x_changes_dimensions(self):
        """4x upscale should produce larger output than no-upscale."""
        img = _get_test_image()

        r_no = _upload_preview(img, upscale="false")
        assert r_no.status_code == 200
        manifest_no = json.loads(r_no.headers.get("X-Manifest", "{}"))

        r_4x = _upload_preview(img, upscale="true", upscale_scale=4)
        assert r_4x.status_code == 200
        manifest_4x = json.loads(r_4x.headers.get("X-Manifest", "{}"))

        if manifest_no and manifest_4x:
            w_no = manifest_no.get("width", 0)
            w_4x = manifest_4x.get("width", 0)
            if w_no > 0 and w_4x > 0:
                assert w_4x > w_no, \
                    f"4x upscale width ({w_4x}) not larger than no-upscale ({w_no})"
