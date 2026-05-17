---
title: "MARVEL: Raster Manga Vectorization via Primitive-wise Deep Reinforcement Learning"
authors: ["Hao Su", "Jianwei Niu", "Xuefeng Liu", "Jiahe Cui", "Ji Wan"]
arxiv_id: "2110.04830"
year: 2021
url: "https://arxiv.org/abs/2110.04830"
code: "https://github.com/SwordHolderSH/Mang2Vec"
relevance: "LOW — manga-specific DRL approach. Mostly survey-completeness; not applicable to chuck-mcp's mask-based pipeline. Primitive-wise stroke sequence model is interesting analogically for relief carving (each stroke ↔ each toolpath segment) but the domain is binary line art, not filled regions."
tags: [vectorization, drl, manga, stroke-decomposition]
---

# MARVEL — Manga Vectorization via DRL

## What it does

Decomposes a raster manga page into a sequence of basic stroke primitives
(black-and-white line strokes) using a primitive-wise deep reinforcement
learning model. The agent emits one stroke per step; episode terminates
when reconstruction quality is high.

## Why we looked at it

Two analogies:

1. **Primitive-wise** matches the layer-by-layer plate model.
2. **Stroke sequence** is conceptually similar to a relief-carving
   toolpath sequence (each cut = each stroke).

## Why it isn't directly useful

- **Domain**: black-and-white line strokes, not filled color regions.
  chuck-mcp masks are filled regions per ink, not strokes.
- **Topology**: no closed-path guarantees; strokes are open curves.
- **No fabrication constraints**.
- **Runtime**: DRL inference is slow, doesn't fit S6.c inline.

## Takeaway

Confirms again that the "vectorize and rely on the network for sane
output" approach hasn't yet produced anything fabrication-ready.
Potrace stays the right S6.c primitive.

## Citation

Su, H. et al. "MARVEL: Raster Manga Vectorization via Primitive-wise
Deep Reinforcement Learning." arXiv:2110.04830, 2021.
