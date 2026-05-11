"""Ring 5 — full-flow ΔE regression gate across the 17 corpus fixtures.

Tiered gates (research-v23-mcp-testing.md §6):

- Tier-1 (5 fixtures): mean ΔE ≤ 1.5, p95 ≤ 3.0
- Tier-2 (12 fixtures): mean ΔE ≤ 3.0, p95 ≤ 6.0

Fixture list authority: ``corpus_tiers.yaml`` in this package.

Lands green at D10.1 (single Tier-1 fixture) then expands through D15.1
(``test_all_tier1_fixtures_pass``) and D23.3 (3-fixture SHIP gate).
"""
