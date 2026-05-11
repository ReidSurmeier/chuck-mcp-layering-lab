"""Shared test helpers for v23-MCP test rings.

Public surface:

- :func:`synthetic_fixtures.make_3imp_synthetic` — deterministic 256×256
  3-impression ground truth used by Ring 4 (stages) + solver smoke.
- :func:`pydantic_factories.make_pigment` / :func:`make_block` /
  :func:`make_plan` — minimum-viable Pydantic instances for Ring 1
  contract tests.

Helpers MUST NOT import production code that may not yet exist; they
guard imports lazily so the scaffold ring stays green pre-D9.
"""
