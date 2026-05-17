---
title: "Draw2Cut: Direct On-Material Annotations for CNC Milling"
authors: ["Xinyue Gui", "Ding Xia", "Wang Gao", "Mustafa Doga Dogan", "Maria Larsson", "Takeo Igarashi"]
arxiv_id: "2501.18951"
year: 2025
venue: "CHI 2025"
url: "https://arxiv.org/abs/2501.18951"
relevance: "MEDIUM — closest CHI/HCI work to the cnc-woodblock-tools downstream consumer. Demonstrates a custom drawing language → toolpath, which is exactly what chuck-mcp's SVG → CNC handoff needs. Useful for the symbol vocabulary (pocket, contour, V-carve) and real-time alignment / kento concepts."
tags: [cnc, milling, woodworking, fabrication, hci, toolpath]
---

# Draw2Cut — On-Material Annotations → CNC Toolpaths

## What it does

Lets a user sketch on the **physical workpiece** with colored markers.
A camera registers the workpiece, parses the sketches into a custom
drawing language (lines, colors, symbols → toolpath types), and emits
G-code for a CNC router. Targeted at novice / artistic users.

## Drawing language → toolpath

The paper defines a small visual vocabulary:

- Different **colors** map to different toolpath types (e.g. black =
  contour cut, red = pocket clear, blue = V-carve).
- **Symbols** annotate depth, repeat counts, tool selection.
- **Real-time alignment** keeps the virtual toolpath registered to the
  physical material despite small workpiece shifts.

## Why this matters for chuck-mcp → ShopBot

The chuck-mcp → cnc-woodblock-tools handoff is currently an SVG with
no semantics about toolpath type. Each plate goes to cherry/shina
plywood; cnc-woodblock-tools decides toolpath type per region. Draw2Cut
suggests a **richer SVG dialect** where:

- Stroke color = toolpath type (contour, pocket, V-carve)
- Stroke width = end-mill diameter requested
- An `<svg:metadata>` section carries kento (registration mark)
  positions and material thickness

This would let cnc-woodblock-tools generate G-code deterministically
without inferring intent from raw geometry.

## Registration / kento

Draw2Cut's real-time alignment ↔ the kento marks cnc-woodblock-tools
already places. The Draw2Cut paper has good camera-based alignment
math worth borrowing if we ever build a closed-loop QA station that
photographs the cut block and compares to the planned SVG.

## Limitations for chuck-mcp

- Draw2Cut is **user-facing** and **single-tool, single-pass**. Doesn't
  scale to multi-plate carved editions of an art print.
- The paper focuses on usability, not toolpath optimization.

## Cite

Gui, X. et al. "Draw2Cut: Direct On-Material Annotations for CNC
Milling." CHI 2025. arXiv:2501.18951.
