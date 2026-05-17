"""alpha-proof-dumper: per-pull alpha + cumulative proof + plate preview PNG
emitter for the chuck-mcp v4 plan_emma pipeline.

The current `chuck_mcp_v2.plan_emma` only writes `production_plan.json` +
`hybrid_result.json` + a thumbnail-only `iter_NN/` artifacts dir. The
acceptance_harness (research/v4-build/example-harness/) expects per-plate
PNG previews + cumulative proof PNGs + alpha snapshots on disk; without
those, rows 2/3/4 of the contact sheet are "NOT FOUND" placeholders.

This package adds that emission step:

    from alpha_proof_dumper.dumper import dump_run_artifacts
    dump_run_artifacts(
        target_rgb=...,
        optimization_result=...,
        out_dir=Path("~/cnc-carving-jobs/emma-YYYY-MM-DD"),
    )

After the call, the directory contains:

    plates/block_NN.png          # one mirrored plate per SolvedPlate
    plates/block_NN.preview.png  # same content, harness-preferred name
    pulls/pull_NNN.png           # cumulative state after pull N
    proofs/proof_NN_after_pull_MMM.png  # 7-checkpoint proof series
    alphas/pull_NNN_alpha.png    # raw alpha snapshot for debugging
    alpha_masks/alpha_NN.png     # subset of the alphas, harness-preferred name
    cumulative_pull_NN.png       # harness-preferred name for proof checkpoints

The dumper is **pure NumPy + PIL** so it has no shapely/svgwrite/skimage
hard-deps beyond what the rest of the chuck_mcp_v2 stack already needs.
"""
