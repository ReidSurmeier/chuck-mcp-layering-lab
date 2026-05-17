"""Pytest config — make this directory importable, and expose the v3 mediapipe
source dir on sys.path so the thin wrapper can reach the canonical modules.
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_V3 = _HERE.parent.parent / "v3-construction" / "mediapipe-face-spatial"
_REPO = _HERE.parent.parent.parent

for p in (_HERE, _V3, _REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
