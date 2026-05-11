"""Ring 3 — mock-Opus full-flow conversation tests.

Drives the day-1 11-tool surface end-to-end through a scripted Opus
stand-in. Asserts on the *flow* (analyze → propose → inspect →
[refine] → export), not on inner stage shapes.

Real harness lives in ``harness.py`` once D21.1 lands (research doc §4).
The placeholder here keeps the file count honest and documents the next
deliverable in an ``xfail`` reason string.
"""
