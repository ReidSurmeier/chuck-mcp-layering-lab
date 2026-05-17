"""chuck-mcp v3 reconstruction validators.

Six hard-gate validators against v13's failure mode (residual alpha-maps
masquerading as plates), per docs/reconstruction-plan-2026-05-16.md
and docs/v2-design-locked-2026-05-16.md.

NOTE: the directory name uses a hyphen which is not a valid Python
package name. Two import styles work:

  1. Direct module import (recommended):
     sys.path.insert(0, "<this-dir>")
     import plate_not_composite, role_purity, ...

  2. Package import via a symlink without the hyphen, e.g.:
     ln -s validators-reconstruction validators_reconstruction
     from validators_reconstruction import plate_not_composite
"""
