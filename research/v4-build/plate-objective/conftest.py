"""pytest conftest — add module dir + project root to sys.path so flat imports work."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for path in (_HERE, _PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
