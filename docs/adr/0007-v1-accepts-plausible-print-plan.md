# 0007 — V1 accepts a plausible print plan, not a solved reconstruction

Status: accepted (2026-05-17)

Chuck MCP V1 is a planning tool for mokuhanga-style block/proof construction.
It ingests one image and returns a visually plausible **Block**,
**Impression**, **Mask**, **Order**, proof, and print recipe surface that a
printmaker can inspect and test.

## Decision

V1 acceptance is based on physical-plan plausibility:

- validators score authoritative **Mask** data and proof states;
- **Order** and proof progression must read like an incremental mokuhanga print;
- **Block**/**Mask** geometry must be separable enough for jigsaw carving;
- underprints must be plausible support structures designed by printmaking
  rules, not claimed as recovered process;
- final-match ΔE remains reported telemetry and an improvement target, but is
  not the only shipping gate.

The dE < 8 target remains useful for future calibrated render tiers. It is not
the V1 hard gate while the renderer is still mostly Mixbox-style **Mixing** and
the topology generator is still being stabilized.

## Alternatives Considered

**Hard-gate V1 on dE < 8.** Rejected for now. The current v5 run with corrected
validator truth reaches 3/5 physical gates and dE around 18.8. Forcing the
solver to chase dE before block topology and empirical overprint are stable
encourages unprintable masks and misleading success metrics.

**Ship only qualitative contact sheets with no quantitative score.** Rejected.
ΔE and gate telemetry are still essential for comparing iterations and for
knowing whether render-tier improvements matter.

## Consequence

Future issues and docs should distinguish:

- **acceptance gates**: physical print-plan behavior;
- **telemetry**: final-match ΔE, underlayer overlap score, and other numerical
  diagnostics.

An implementation can be valuable even when dE is high if it improves physical
printability and proof readability. Conversely, a lower dE result is not a V1
success if it fails mask, jigsaw, or proof-methodology gates.
