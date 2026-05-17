"""pytest conftest — add renderer dir + project root to sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (HERE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
