"use client";

import { useEffect, useRef } from "react";

export default function About() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const particles: { x: number; y: number; vx: number; vy: number; size: number }[] = [];
    for (let i = 0; i < 40; i++) {
      particles.push({
        x: Math.random() * 200,
        y: Math.random() * 100,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 2 + 1,
      });
    }

    let animId: number;
    const draw = () => {
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, 200, 100);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = 200;
        if (p.x > 200) p.x = 0;
        if (p.y < 0) p.y = 100;
        if (p.y > 100) p.y = 0;

        ctx.fillStyle = "#000";
        ctx.fillRect(Math.round(p.x), Math.round(p.y), p.size, p.size);
      }
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <main
      style={{
        maxWidth: 520,
        margin: "0 auto",
        padding: "2rem 1rem",
        fontFamily: "'DepartureMono', monospace",
        fontSize: 13,
        lineHeight: 1.6,
        color: "#1d2021",
        minHeight: "100vh",
      }}
    >
      {/* ASCII cat */}
      <pre
        style={{
          fontSize: 12,
          lineHeight: 1.3,
          color: "#bbb",
          marginBottom: 24,
          userSelect: "none",
        }}
      >
        {`  ∧＿∧
 (｡･ω･｡)つ━☆
 /　  つ
(　/  ﾉ
 ＼first undo）
  ＼＼  ＼`}
      </pre>

      {/* ABOUT box */}
      <div
        style={{
          border: "1px solid #ddd",
          padding: 16,
          marginBottom: 16,
        }}
      >
        <div style={{ fontSize: 11, color: "#999", marginBottom: 8, letterSpacing: "0.1em" }}>
          ABOUT
        </div>
        <p style={{ marginBottom: 8 }}>
          color.separator is a digital color separation tool for woodblock, CNC, and silkscreen
          printing. upload an image, choose how many colors, adjust parameters, and download plate
          files ready for CNC toolpath generation or screen exposure.
        </p>
        <p style={{ color: "#666" }}>
          one plate = one color = one block. the tool does the hard part of deciding which pixels
          belong to which color.
        </p>
      </div>

      {/* TECH box */}
      <div
        style={{
          border: "1px solid #ddd",
          padding: 16,
          marginBottom: 16,
        }}
      >
        <div style={{ fontSize: 11, color: "#999", marginBottom: 8, letterSpacing: "0.1em" }}>
          TECH
        </div>
        <p style={{ marginBottom: 8 }}>
          based on the taohuawu woodblock digital restoration paper (MDPI 2025) and superpixel color
          quantization research. uses K-means++ clustering in CIELAB perceptual color space with
          optional Real-ESRGAN 4x upscaling.
        </p>
        <ul style={{ listStyle: "none", color: "#666", fontSize: 12 }}>
          <li>· SLIC superpixel segmentation</li>
          <li>· CRF smoothing + bilateral filter</li>
          <li>· mean-shift spatial clustering</li>
          <li>· auto merge similar plates (ΔE)</li>
          <li>· v2 → v11 algorithm iterations</li>
        </ul>
      </div>

      {/* Animated particles */}
      <div style={{ marginBottom: 16 }}>
        <canvas
          ref={canvasRef}
          width={200}
          height={100}
          style={{ border: "1px solid #eee", display: "block" }}
        />
      </div>

      {/* References */}
      <div style={{ fontSize: 11, color: "#999", marginBottom: 24 }}>
        <div style={{ letterSpacing: "0.1em", marginBottom: 4 }}>REFERENCES</div>
        <a
          href="https://www.mdpi.com/2076-3417/15/16/9081"
          target="_blank"
          rel="noreferrer"
          style={{ color: "#666", display: "block", marginBottom: 2 }}
        >
          Digital Restoration of Taohuawu Woodblock Prints (MDPI 2025)
        </a>
        <a
          href="https://www.mdpi.com/1424-8220/22/16/6043"
          target="_blank"
          rel="noreferrer"
          style={{ color: "#666", display: "block", marginBottom: 2 }}
        >
          Efficient Color Quantization Using Superpixels (MDPI Sensors 2022)
        </a>
        <a
          href="https://colorshift.theretherenow.com/"
          target="_blank"
          rel="noreferrer"
          style={{ color: "#666", display: "block" }}
        >
          color/shift — risograph color profiles
        </a>
      </div>

      <a
        href="/color-separator"
        style={{ color: "#000", fontSize: 13, fontFamily: "'DepartureMono', monospace" }}
      >
        ← back
      </a>
    </main>
  );
}
