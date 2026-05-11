"""Make ``backend/`` importable so tests can use ``from routes import …``,
``from algorithms.decomposition import …`` etc. without installing the
package.

Mirrors the convention used by ``backend/main.py`` (run with ``backend/``
as ``cwd``).
"""
from __future__ import annotations

import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
