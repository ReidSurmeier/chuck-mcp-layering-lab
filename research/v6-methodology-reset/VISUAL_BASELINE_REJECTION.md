# Visual Baseline Rejection — iter 13

Baseline rejected:

```text
/srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-13/sheet_iter_13.png
```

Gate report:

```text
research/v6-methodology-reset/reports/iter13_methodology_gate.json
```

## Verdict

`sheet_iter_13.png` is not a plausible Chuck Close / mokuhanga methodology
result. It is a plumbing and validator test output.

## Why it fails

- The reference row develops a coherent portrait through accumulated carved
  proof states.
- The current row accumulates circular cell-centroid marks.
- The current block row is mostly empty wood with a few isolated spots.
- The alpha row is debug data, not a printmaking methodology.
- The proof progression does not explain how an Emma-like woodblock print would
  be constructed.

## Consequence

No future run should be described as progress merely because validator counts or
dE improve. The first gate is visual methodology coherence against the reference
proof sheets.

## New starting point

v6 starts from a failing visual gate, not from the v5 optimizer.

The next implementation must create connected carved-region **Masks** before
continuous pigment/load solving. The output should be judged as proof batches
and block organization first, not as individual optimizer masks.
