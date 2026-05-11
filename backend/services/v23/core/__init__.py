"""v23 core — math + render + solver + types.

Module layout per ``/tmp/research-v23-mcp-repo-layout.md``:

- ``forward_render_jax`` — JAX-traceable forward render (Mixbox-stack lerp)
- ``inverse_solver``    — JAX L-BFGS-B inverse stack solver (D7+)
- ``render_tier``       — t1_mixbox | t2_empirical | t3_spectral dispatch (D6.5)
- ``topology_repair``   — post-solve morph_open/close on rule 6+7 (D11)
- ``emma_priors``       — 6-family taxonomy, accent rule, keyblock rule (D8)
- ``templates``         — portrait_emma | landscape | high_chroma_graphic (D8)
- ``score``             — 5-component combined plan score (D12)
"""
