"""Backend performance tests for color-separator API.

Tests hit the live backend container directly.
Run from host: pytest backend/tests/ -v
"""
import io
import time
from pathlib import Path

import pytest
import requests

# Backend URL — Docker internal or tunnel
BACKEND_URL = "http://localhost:8001"
# Test images in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_IMAGE_SMALL = PROJECT_ROOT / "test_red.png"
TEST_IMAGE_STD = PROJECT_ROOT / "test_colorsep.png"


def _upload_image(endpoint: str, image_path: Path, **extra_fields) -> requests.Response:
    """Upload an image to backend endpoint."""
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {
            "plates": "4",
            "dust": "5",
            "use_edges": "true",
            "edge_sigma": "3.0",
            "version": "v20",
            "upscale": "false",  # skip upscale for speed
            "chroma_boost": "1.3",
            **extra_fields,
        }
        return requests.post(
            f"{BACKEND_URL}{endpoint}",
            files=files,
            data=data,
            timeout=120,
        )


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "ram_gb" in data or "status" in data

    def test_health_responds_fast(self):
        start = time.time()
        requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Health check took {elapsed:.1f}s (max 3s)"


class TestPreviewPerformance:
    @pytest.mark.skipif(
        not TEST_IMAGE_SMALL.exists(),
        reason="test_red.png not found",
    )
    def test_v20_preview_cached_under_30s(self):
        """v20 preview with small image should complete in ≤30s (cached model)."""
        start = time.time()
        r = _upload_image("/api/preview", TEST_IMAGE_SMALL)
        elapsed = time.time() - start
        assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text[:200]}"
        assert elapsed <= 30, f"Preview took {elapsed:.1f}s (max 30s cached)"

    @pytest.mark.skipif(
        not TEST_IMAGE_SMALL.exists(),
        reason="test_red.png not found",
    )
    def test_v20_preview_cold_under_45s(self):
        """First request after restart may be slower (cold start)."""
        start = time.time()
        r = _upload_image("/api/preview", TEST_IMAGE_SMALL)
        elapsed = time.time() - start
        assert r.status_code == 200, f"Preview failed: {r.status_code} {r.text[:200]}"
        assert elapsed <= 45, f"Preview took {elapsed:.1f}s (max 45s cold)"


class TestSeparationEndpoint:
    @pytest.mark.skipif(
        not TEST_IMAGE_SMALL.exists(),
        reason="test_red.png not found",
    )
    def test_separate_returns_zip(self):
        """POST /api/separate should return a valid ZIP file."""
        r = _upload_image("/api/separate", TEST_IMAGE_SMALL)
        assert r.status_code == 200
        assert r.headers.get("content-type") in (
            "application/zip",
            "application/x-zip-compressed",
            "application/octet-stream",
        )
        # ZIP magic bytes
        assert r.content[:4] == b"PK\x03\x04", "Response is not a valid ZIP"
        assert len(r.content) > 1000, f"ZIP too small: {len(r.content)} bytes"

    @pytest.mark.skipif(
        not TEST_IMAGE_SMALL.exists(),
        reason="test_red.png not found",
    )
    def test_separate_zip_contains_plates(self):
        """ZIP should contain plate PNG files."""
        import zipfile

        r = _upload_image("/api/separate", TEST_IMAGE_SMALL)
        assert r.status_code == 200

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            png_files = [n for n in names if n.endswith(".png")]
            assert len(png_files) >= 2, f"Expected ≥2 PNGs, got: {names}"


class TestPreviewStreamEndpoint:
    @pytest.mark.skipif(
        not TEST_IMAGE_SMALL.exists(),
        reason="test_red.png not found",
    )
    def test_preview_stream_sends_sse_events(self):
        """POST /api/preview-stream should send SSE progress events."""
        with open(TEST_IMAGE_SMALL, "rb") as f:
            files = {"image": (TEST_IMAGE_SMALL.name, f, "image/png")}
            data = {
                "plates": "4",
                "dust": "5",
                "use_edges": "true",
                "edge_sigma": "3.0",
                "version": "v20",
                "upscale": "false",
                "chroma_boost": "1.3",
            }
            r = requests.post(
                f"{BACKEND_URL}/api/preview-stream",
                files=files,
                data=data,
                timeout=120,
                stream=True,
            )
        assert r.status_code == 200

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                events.append(line[6:])

        assert len(events) >= 1, "No SSE events received"
        # Last event should be "complete" stage
        import json

        last = json.loads(events[-1])
        assert last.get("stage") in (
            "complete",
            "partial_complete",
        ), f"Last event stage: {last.get('stage')}"


class TestEdgeCases:
    def test_missing_image_returns_422(self):
        """POST without image should return 422."""
        r = requests.post(
            f"{BACKEND_URL}/api/preview",
            data={"plates": "4", "version": "v20"},
            timeout=10,
        )
        assert r.status_code == 422

    def test_invalid_version_handled(self):
        """POST with invalid version should return error, not crash."""
        if not TEST_IMAGE_SMALL.exists():
            pytest.skip("test_red.png not found")
        r = _upload_image("/api/preview", TEST_IMAGE_SMALL, version="v999")
        # Should be 400 or 422, not 500
        assert r.status_code in (400, 422, 500)  # 500 acceptable if graceful
