# Session Memory: Methodology Proof States

Date: 2026-05-14

Project: Chuck MCP / `chuck-mcp-layering-lab`

Do not touch `emma-mokuhanga-mcp` for this work. This repo is the separate
Chuck MCP experiment.

## User-Approved Direction

The Chuck Close reference screenshot should be read as cumulative proof states
after small groups of blocks have been printed. It is not one proof per single
plate and it is not a rigid `4x4x4` grid.

The reference methodology is:

- build the print through overlapping cumulative proofs
- add blocks in small batches, usually around 3-5 blocks
- keep the block count adaptive to image complexity
- do not force 27 blocks; 27 is only a useful reference scale for the Chuck
  Close/Emma production
- early blocks can contain detailed carved geometry, but individual block 1
  should not already look like a fully formed proof
- proof 1 should look formed only after the first small batch of blocks
- final proof should visually resemble the input image

## Current Accepted Prototype

Script:

```text
scripts/methodology_proof_states.py
```

Latest run:

```text
/srv/woodblock-share/chuck-methodology-proofs/methodology-adaptive-proofs-emma-v13-20260514
```

Latest symlink:

```text
/srv/woodblock-share/chuck-methodology-proofs/latest-emma
```

Clean handoff folder:

```text
/srv/woodblock-share/mcp v12
```

Latest run metrics:

- adaptive blocks: 26
- proof snapshots: 7
- proof end blocks: 4, 8, 12, 16, 20, 24, 26
- mean DeltaE76: 4.982
- p95 DeltaE76: 9.343

Important fix from user feedback: the first version made block 1 look too
fully formed. The current version splits the first proof batch into separate
pale warm, pale pink, pale cool, and pale detail scaffold blocks so B01 is only
one partial support block.

## Remaining Gap

This is a methodology/proof-state prototype. It is not yet CNC-safe final
plate geometry. The next real build step is turning these adaptive proof blocks
into printable, vectorizable region plates while preserving the accepted proof
sequence.
