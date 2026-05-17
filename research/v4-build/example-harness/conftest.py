"""Test bootstrap: make `import acceptance_harness` resolve to the on-disk
package sitting next to this conftest, regardless of where pytest is invoked
from.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Put HERE on sys.path so `acceptance_harness/` is importable as a package.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
