"""v23 pipeline + core services.

S1 ingest → S2 SAM → S3 hue family → S4 Tan warm-start →
S5 inverse solver → S6 three-state mask → S7 DSATUR block packing →
S8 carveability filter → S9 SVG vectorize → S10 ZIP + recipe.

See ``/tmp/research-v23-mcp-build-sequence.md`` for the day-by-day
TDD plan and ``/tmp/research-v23-mcp-interfaces.md`` for stage handoff
Pydantic types.
"""
