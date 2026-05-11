"""v23 S1-S10 pipeline stages.

Each stage owns one file. Entry signature is stage-specific but every
stage emits Pydantic results that downstream stages consume.
"""
