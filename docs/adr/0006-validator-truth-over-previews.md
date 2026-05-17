# 0006 — Validators score truth objects, not review previews

Status: accepted (2026-05-17)

The v5 overnight loop reported every block as `inked_area_fraction: 1.0` for
the plate-not-composite validator, while the review sheet visibly showed sparse
dot-style masks. The validator was being fed `plate_preview` PNGs: full wood
rectangles with pigment rendered on top. That is useful for a human contact
sheet, but it is not the physical printed-area mask.

## Decision

Validators must score the authoritative object for their contract:

- geometry validators score **Mask** data (`inked_mask`, `plate_mask`,
  `alpha_preview`) before any review image;
- proof progression scores cumulative proof states;
- final-match scores the final composite against the input image;
- review/contact-sheet previews are never used as geometry truth when a mask is
  available.

`build_validator_plan.py` now maps each solved block to the alpha mask emitted
by the artifact dumper's sorted pull order and writes that path as
`inked_mask`. `run_all_validators.py` prefers that mask for both
plate-not-composite and jigsaw-separation.

## Consequence

Some v5 overnight metrics are historical only. In particular, the
plate-not-composite failures in `FINAL_REPORT.md` measured the wrong object and
must not be used to diagnose SNIC, jigsaw grouping, or block coverage. The
correct next step is to re-run validation on the same artifacts with mask-backed
plans, then decide whether the remaining failures are topology, order, color, or
optimizer feedback.
