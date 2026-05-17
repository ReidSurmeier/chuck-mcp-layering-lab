"""Allow `python -m acceptance_harness <plan_dir>` invocation."""

from .cli import main

raise SystemExit(main())
