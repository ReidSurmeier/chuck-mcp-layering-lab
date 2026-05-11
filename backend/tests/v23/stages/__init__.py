"""Ring 4 — per-stage isolation tests (S1..S10).

Each stage is unit-tested independently with neighbour stages mocked.
This lets solver tweaks run in ~4s instead of waiting on the whole
pipeline (research-v23-mcp-testing.md §5).

Stage landing days (build-sequence §):
- S1 ingest        — D4
- S2 SAM gateway   — D5
- S3 palette       — D9.3
- S5 solver        — D7 (smoke) + D10 (real)
- S8 carveability  — D11
- S9 vectorize     — D16
- S10 emit         — D18
"""
