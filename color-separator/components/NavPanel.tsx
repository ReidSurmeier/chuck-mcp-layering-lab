import { type ChangeEvent, type RefObject } from "react";
import { rgbToHex } from "@/lib/colors";
import type { Manifest } from "@/lib/types";
import { VERSIONS } from "../constants";
import type { VersionId, PaletteColor, PlateImage, AppError } from "../constants";

interface NavPanelProps {
  navOpen: boolean;
  version: VersionId;
  onVersionChange: (v: VersionId) => void;
  upscale: boolean;
  onToggleUpscale: () => void;
  hasUpscaleToggle: boolean;
  fileName: string;
  fileInputRef: RefObject<HTMLInputElement | null>;
  onFileSelect: (e: ChangeEvent<HTMLInputElement>) => void;
  plates: number;
  colors: PaletteColor[];
  onColorChange: (index: number, hex: string) => void;
  onToggleLock: (index: number) => void;
  onRemoveColor: (index: number) => void;
  onAddColor: () => void;
  dust: number;
  useEdges: boolean;
  edgeSigma: number;
  hasCrfSliders: boolean;
  crfSpatial: number;
  crfColor: number;
  crfCompat: number;
  hasV9Sliders: boolean;
  sigmaS: number;
  sigmaR: number;
  meanshiftSp: number;
  meanshiftSr: number;
  hasV4Sliders: boolean;
  medianSize: number;
  shadowThreshold: number;
  highlightThreshold: number;
  hasSuperpixelSliders: boolean;
  nSegments: number;
  compactness: number;
  hasChromaSlider: boolean;
  chromaBoost: number;
  detailStrength: number;
  onParamChange: (key: string, value: number | boolean) => void;
  file: File | null;
  onProcess: () => void;
  onReset: () => void;
  showOriginal: boolean;
  onToggleOriginal: () => void;
  canCompare: boolean;
  compositeUrl: string | null;
  isLoading: boolean;
  downloadProgress: string | null;
  onDownload: () => void;
  mergeMode: boolean;
  onToggleMergeMode: () => void;
  manifest: Manifest | null;
  plateImages: PlateImage[];
  selectedForMerge: number[];
  isMerging: boolean;
  onMerge: () => void;
  showAbout: boolean;
  onToggleAbout: () => void;
  imageInfo: { width: number; height: number; size: string; type: string } | null;
  error: AppError | null;
  onClearError: () => void;
  onRetry: () => void;
  onCancel: () => void;
}

export default function NavPanel({
  navOpen,
  version,
  onVersionChange,
  upscale,
  onToggleUpscale,
  hasUpscaleToggle,
  fileName,
  fileInputRef,
  onFileSelect,
  plates,
  colors,
  onColorChange,
  onToggleLock,
  onRemoveColor,
  onAddColor,
  dust,
  useEdges,
  edgeSigma,
  hasCrfSliders,
  crfSpatial,
  crfColor,
  crfCompat,
  hasV9Sliders,
  sigmaS,
  sigmaR,
  meanshiftSp,
  meanshiftSr,
  hasV4Sliders,
  medianSize,
  shadowThreshold,
  highlightThreshold,
  hasSuperpixelSliders,
  nSegments,
  compactness,
  hasChromaSlider,
  chromaBoost,
  detailStrength,
  onParamChange,
  file,
  onProcess,
  onReset,
  showOriginal,
  onToggleOriginal,
  canCompare,
  compositeUrl,
  isLoading,
  downloadProgress,
  onDownload,
  mergeMode,
  onToggleMergeMode,
  manifest,
  plateImages,
  selectedForMerge,
  isMerging,
  onMerge,
  showAbout,
  onToggleAbout,
  imageInfo,
  error,
  onClearError,
  onRetry,
  onCancel,
}: NavPanelProps) {
  return (
    <div className={`nav-panel${navOpen ? " nav-open" : ""}`}>
      <h3 className="app-title">
        <span>COLOR.SEPARATOR</span>
      </h3>

      {/* Version selector */}
      <select
        value={version}
        onChange={(e) => onVersionChange(e.target.value as VersionId)}
      >
        {VERSIONS.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
      </select>

      {/* Upscale toggle */}
      {hasUpscaleToggle && (
        <>
          <h3>upscale</h3>
          <button
            data-active={upscale ? "true" : "false"}
            onClick={onToggleUpscale}
          >
            {upscale ? "on" : "off"}
          </button>
        </>
      )}

      {/* Source */}
      <h3>source</h3>
      <button
        className="source-btn"
        onClick={() => fileInputRef.current?.click()}
      >
        {fileName || "choose file"}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={onFileSelect}
      />

      {/* Plates */}
      <h3>plates {plates}</h3>
      <input
        type="range"
        min={2}
        max={35}
        step={1}
        value={plates}
        onChange={(e) => onParamChange("plates", Number(e.target.value))}
      />
      {plates > 15 && (
        <div style={{ fontSize: 11, color: "#c90", marginTop: 2 }}>
          high plate count — processing will be slower
        </div>
      )}

      {/* Colors */}
      {colors.length > 0 && (
        <>
          <h3>colors</h3>
          <div className="colors">
            {colors.map((c, i) => (
              <span className="color" key={i}>
                <input
                  type="color"
                  value={rgbToHex(c.rgb)}
                  onChange={(e) => onColorChange(i, e.target.value)}
                  onClick={() => onToggleLock(i)}
                />
                {c.locked && <span className="lock-indicator" />}
                <button
                  className="remove-btn"
                  onClick={() => onRemoveColor(i)}
                >
                  &times;
                </button>
              </span>
            ))}
            <button onClick={onAddColor}>+</button>
          </div>
        </>
      )}

      {/* Dust */}
      <h3>dust {dust}</h3>
      <input
        type="range"
        min={5}
        max={100}
        step={1}
        value={dust}
        onChange={(e) => onParamChange("dust", Number(e.target.value))}
      />

      {/* Edge Detection */}
      <h3>edge detection</h3>
      <button
        data-active={useEdges ? "true" : "false"}
        onClick={() => onParamChange("useEdges", !useEdges)}
      >
        {useEdges ? "on" : "off"}
      </button>

      {useEdges && (
        <>
          <h3>edge sigma {edgeSigma.toFixed(1)}</h3>
          <input
            type="range"
            min={0.5}
            max={3.0}
            step={0.1}
            value={edgeSigma}
            onChange={(e) => onParamChange("edgeSigma", Number(e.target.value))}
          />
        </>
      )}

      {/* CRF controls (v7/v8) */}
      {hasCrfSliders && (
        <>
          <h3>spatial {crfSpatial}</h3>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={crfSpatial}
            onChange={(e) => onParamChange("crfSpatial", Number(e.target.value))}
          />
          <h3>color {crfColor}</h3>
          <input
            type="range"
            min={5}
            max={50}
            step={1}
            value={crfColor}
            onChange={(e) => onParamChange("crfColor", Number(e.target.value))}
          />
          <h3>edge {crfCompat}</h3>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={crfCompat}
            onChange={(e) => onParamChange("crfCompat", Number(e.target.value))}
          />
        </>
      )}

      {/* V9 controls */}
      {hasV9Sliders && (
        <>
          <h3>smooth &sigma;s {sigmaS}</h3>
          <input
            type="range"
            min={20}
            max={200}
            step={5}
            value={sigmaS}
            onChange={(e) => onParamChange("sigmaS", Number(e.target.value))}
          />
          <h3>range &sigma;r {sigmaR.toFixed(1)}</h3>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={sigmaR}
            onChange={(e) => onParamChange("sigmaR", Number(e.target.value))}
          />
          <h3>shift sp {meanshiftSp}</h3>
          <input
            type="range"
            min={5}
            max={50}
            step={1}
            value={meanshiftSp}
            onChange={(e) =>
              onParamChange("meanshiftSp", Number(e.target.value))
            }
          />
          <h3>shift sr {meanshiftSr}</h3>
          <input
            type="range"
            min={10}
            max={80}
            step={1}
            value={meanshiftSr}
            onChange={(e) =>
              onParamChange("meanshiftSr", Number(e.target.value))
            }
          />
        </>
      )}

      {/* V4 tuning controls */}
      {hasV4Sliders && (
        <>
          <h3>smooth {medianSize}</h3>
          <input
            type="range"
            min={1}
            max={11}
            step={2}
            value={medianSize}
            onChange={(e) =>
              onParamChange("medianSize", Number(e.target.value))
            }
          />
          <h3>shadows {shadowThreshold}</h3>
          <input
            type="range"
            min={5}
            max={50}
            step={1}
            value={shadowThreshold}
            onChange={(e) =>
              onParamChange("shadowThreshold", Number(e.target.value))
            }
          />
          <h3>highlights {highlightThreshold}</h3>
          <input
            type="range"
            min={80}
            max={99}
            step={1}
            value={highlightThreshold}
            onChange={(e) =>
              onParamChange("highlightThreshold", Number(e.target.value))
            }
          />
        </>
      )}

      {/* Superpixel controls (v6) */}
      {hasSuperpixelSliders && (
        <>
          <h3>detail {nSegments}</h3>
          <input
            type="range"
            min={500}
            max={10000}
            step={100}
            value={nSegments}
            onChange={(e) =>
              onParamChange("nSegments", Number(e.target.value))
            }
          />
          <h3>compact {compactness}</h3>
          <input
            type="range"
            min={5}
            max={40}
            step={1}
            value={compactness}
            onChange={(e) =>
              onParamChange("compactness", Number(e.target.value))
            }
          />
        </>
      )}

      {/* Chroma slider */}
      {hasChromaSlider && (
        <>
          <h3>chroma {chromaBoost.toFixed(1)}</h3>
          <input
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={chromaBoost}
            onChange={(e) =>
              onParamChange("chromaBoost", Number(e.target.value))
            }
          />
        </>
      )}

      {/* V14 detail strength */}
      {version === "v14" && (
        <>
          <h3>detail {detailStrength.toFixed(2)}</h3>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={detailStrength}
            onChange={(e) =>
              onParamChange("detailStrength", Number(e.target.value))
            }
          />
        </>
      )}

      {/* Error display */}
      {error && (
        <div
          style={{
            margin: "8px 0",
            padding: "6px 8px",
            background: "#fee",
            border: "1px solid #fcc",
            fontSize: 12,
          }}
        >
          <div style={{ color: "#c00", marginBottom: 4 }}>{error.message}</div>
          <div style={{ display: "flex", gap: 4 }}>
            {error.retryable && (
              <button onClick={onRetry} style={{ fontSize: 11 }}>
                retry
              </button>
            )}
            <button onClick={onClearError} style={{ fontSize: 11 }}>
              dismiss
            </button>
          </div>
        </div>
      )}

      {/* Actions */}
      <h3>actions</h3>
      <button className="process-btn" onClick={onProcess} disabled={!file}>
        process
      </button>
      <button onClick={onReset}>reset</button>
      {isLoading && (
        <button onClick={onCancel} style={{ color: "#c00" }}>
          cancel
        </button>
      )}
      {canCompare && (
        <button
          data-active={showOriginal ? "true" : "false"}
          onClick={onToggleOriginal}
          title="Spacebar to toggle"
        >
          {showOriginal ? "showing original" : "compare"}
        </button>
      )}

      {/* Download */}
      <h3>download</h3>
      <button
        onClick={onDownload}
        disabled={!compositeUrl || isLoading || !!downloadProgress}
      >
        {downloadProgress ?? "ZIP"}
      </button>
      {downloadProgress && (
        <div className="download-progress">
          <div className="download-progress-bar">
            <div className="download-progress-fill" />
          </div>
        </div>
      )}

      {/* Merge plates */}
      <h3>merge plates</h3>
      <button
        data-active={mergeMode ? "true" : "false"}
        onClick={onToggleMergeMode}
        disabled={!manifest || plateImages.length === 0}
      >
        {mergeMode ? "cancel" : "select plates"}
      </button>
      {mergeMode && selectedForMerge.length >= 2 && (
        <button className="process-btn" disabled={isMerging} onClick={onMerge}>
          {isMerging ? "merging..." : `merge ${selectedForMerge.length} plates`}
        </button>
      )}
      {mergeMode && (
        <div style={{ fontSize: 11, color: "#999", marginTop: 4 }}>
          click plates to select ({selectedForMerge.length} selected)
        </div>
      )}

      <h3>about</h3>
      <button onClick={onToggleAbout}>
        {showAbout ? "hide" : "show"}
      </button>
      <a
        href="https://github.com/ReidSurmeier/color-separator"
        target="_blank"
        rel="noreferrer"
        style={{
          display: "inline-block",
          background: "#ddd",
          padding: "2px 4px",
          margin: "0 2px",
          fontSize: 14,
          textDecoration: "none",
          color: "inherit",
          cursor: "pointer",
        }}
      >
        github
      </a>

      {(imageInfo || manifest) && (
        <div className="data-box">
          <h3>data</h3>
          {imageInfo && (
            <>
              <div className="data-row">
                <span>size</span>
                <span>
                  {imageInfo.width}&times;{imageInfo.height}
                </span>
              </div>
              <div className="data-row">
                <span>file</span>
                <span>{imageInfo.size}</span>
              </div>
              <div className="data-row">
                <span>type</span>
                <span>{imageInfo.type}</span>
              </div>
            </>
          )}
          {manifest && (
            <>
              <div className="data-row">
                <span>plates</span>
                <span>{manifest.plates.length}</span>
              </div>
              {manifest.upscaled && (
                <div className="data-row">
                  <span>upscaled</span>
                  <span>{manifest.upscale_scale ?? 2}&times;</span>
                </div>
              )}
              {manifest.ai_analysis && (
                <div className="data-row">
                  <span>ai score</span>
                  <span>{manifest.ai_analysis.quality_score}/100</span>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
