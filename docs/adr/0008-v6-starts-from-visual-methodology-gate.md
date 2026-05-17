# 0008 — v6 starts from a visual methodology gate

Status: accepted (2026-05-17)

`sheet_iter_13.png` passed useful plumbing checks, but visually it does not
correspond to the Chuck Close proof reference or the Emma woodblock method. It
is dot-cell accumulation, not a progressive mokuhanga proof plan.

## Decision

v6 starts from a visual methodology gate that deliberately fails iter 13.

The gate compares current proof artifacts to the Chuck Close progressive proof
reference and rejects runs that read as dot/cell centroid output. This gate is
upstream of dE optimization claims: if the proof sheet does not visually build
like a plausible woodblock print, improved numeric metrics do not count as
project progress.

## Alternatives Considered

**Continue from v5 and tune topology gradually.** Rejected. The v5 output shape
is wrong enough that incremental tuning risks preserving the wrong abstraction.

**Keep relying on human review only.** Rejected. Human review is final, but the
system needs a repeatable failure signal so agents do not continue building
against obviously wrong sheets.

## Consequence

The next algorithmic work must generate connected carved-region **Masks** before
continuous color solving. Dot-cell masks are now a known failing baseline, even
when they pass some lower-level validators.
