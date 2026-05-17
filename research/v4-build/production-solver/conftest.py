"""Local conftest for the production-solver tests.

The directory name (production-solver, hyphenated) is not a valid Python
identifier, so pytest cannot import this folder as a package. We use
conftest to:

  1. Add this folder to sys.path so its peer modules import absolutely.
  2. Tell pytest to ignore __init__.py (it isn't a test file).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

collect_ignore = ["__init__.py"]
