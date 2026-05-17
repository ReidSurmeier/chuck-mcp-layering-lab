# loop-runner NOTES.md

## Role
LOOP RUNNER agent for swarm-1778989256284-xvs2l5. Runs up to 12 iterations of the v5
overnight Emma reconstruction pipeline after the 4 sibling TDD agents land their patches.

## Dependencies (NOTES.md required from)
- snic-real
- mediapipe-spatial
- alpha-proof-dumper
- mokuhanga-pigments

## State machine
- WAITING_FOR_SIBLINGS -- poll every 120s, max 4h
- LOOPING -- run up to 12 iters, 25min each
- ACCEPTED -- 6/6 validators + dE<8 + underlayer>=85%
- EXHAUSTED -- iter cap or time cap reached without accept
- ERROR -- record in iterations.csv and continue

## Workspace
- iterations.csv -- per-iter metrics
- FINAL_REPORT.md -- written at loop exit
- ~/cnc-carving-jobs/emma-overnight-iter-NN/ -- per-iter outputs
- /srv/woodblock-share/chuck-mcp-iterations/current-review/2026-05-17_v5-overnight-iter-NN/sheet_iter_NN.png

## Status: WAITING_FOR_SIBLINGS (initialized 2026-05-16 23:44 EDT)
