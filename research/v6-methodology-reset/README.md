# v6 Methodology Reset

This is the new starting point after rejecting `sheet_iter_13.png` as a visual
methodology failure.

The old v5 loop proved useful plumbing:

- JAX runs on GPU.
- validators can score **Mask** truth instead of Review previews.
- the outer loop can carry solved state forward.

But the image output is not a plausible Chuck Close / mokuhanga proof sequence.
It is dot-cell accumulation. That means v6 starts from a visual methodology gate
instead of from solver metrics.

## Rule

No algorithmic change counts as progress unless it improves the visual
methodology gate against the Chuck Close proof reference.

The gate deliberately fails iter 13 for these reasons:

- proof states do not build a coherent portrait form;
- masks read as sparse circular centroid stamps;
- the final proof has low overlap with the reference proof silhouette;
- the output does not resemble the progressive block/proof methodology above
  the current run in the review sheet.

## Commands

Run the gate against the known-bad iter 13 baseline:

```bash
.venv-renderer/bin/python research/v6-methodology-reset/visual_methodology_gate.py \
  --artifacts-dir /home/reidsurmeier/cnc-carving-jobs/emma-overnight-iter-13/artifacts \
  --reference-sheet /srv/woodblock-share/chuck-mcp-iterations/references/2026-05-14_chuck-close-progressive-proof-screenshot.png \
  --output research/v6-methodology-reset/reports/iter13_methodology_gate.json
```

Expected result:

```text
FAIL
```

## Next Build

Do not continue the dot-cell path. The next implementation should generate
connected carved-region **Masks** before pigment/load solving, then rerun this
gate and visually inspect the proof sheet.
