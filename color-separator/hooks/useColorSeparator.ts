"use client";

import { useState, useCallback, useRef, useEffect, type ChangeEvent } from "react";
import {
  fetchPreviewStream,
  fetchPlatesStream,
  fetchSeparation,
  fetchUpscale,
  fetchMerge,
  ApiError,
} from "@/lib/api";
import type { PlateStreamEvent } from "@/lib/api";
import { rgbToHex, hexToRgb } from "@/lib/colors";
import type { SeparationParams, Manifest, PreviewResult } from "@/lib/types";
import type { VersionId, PaletteColor, PlateImage, AppError } from "../constants";

function toAppError(err: unknown): AppError {
  if (err instanceof ApiError) {
    return { message: err.message, retryable: err.retryable };
  }
  if (err instanceof DOMException && err.name === "AbortError") {
    return { message: "Request cancelled", retryable: false };
  }
  return {
    message: err instanceof Error ? err.message : "Unknown error",
    retryable: true,
  };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

export function useColorSeparator() {
  // === File state ===
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [imageInfo, setImageInfo] = useState<{
    width: number;
    height: number;
    size: string;
    type: string;
  } | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [compositeUrl, setCompositeUrl] = useState<string | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [colors, setColors] = useState<PaletteColor[]>([]);

  // === Loading / error ===
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<AppError | null>(null);

  // === Params ===
  const [plates, setPlates] = useState(4);
  const [dust, setDust] = useState(5);
  const [useEdges, setUseEdges] = useState(true);
  const [edgeSigma, setEdgeSigma] = useState(3.0);
  const [version, setVersion] = useState<VersionId>("v20");
  const [upscale, setUpscale] = useState(true);
  const [medianSize, setMedianSize] = useState(5);
  const [chromaBoost, setChromaBoost] = useState(1.3);
  const [shadowThreshold, setShadowThreshold] = useState(8);
  const [highlightThreshold, setHighlightThreshold] = useState(95);
  const [nSegments, setNSegments] = useState(3000);
  const [compactness, setCompactness] = useState(15);
  const [crfSpatial, setCrfSpatial] = useState(3);
  const [crfColor, setCrfColor] = useState(13);
  const [crfCompat, setCrfCompat] = useState(10);
  const [sigmaS, setSigmaS] = useState(100);
  const [sigmaR, setSigmaR] = useState(0.5);
  const [meanshiftSp, setMeanshiftSp] = useState(15);
  const [meanshiftSr, setMeanshiftSr] = useState(30);
  const [detailStrength, setDetailStrength] = useState(0.5);

  // === UI state ===
  const [progressStage, setProgressStage] = useState<string | null>(null);
  const [progressPct, setProgressPct] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [mergeMode, setMergeMode] = useState(false);
  const [isMerging, setIsMerging] = useState(false);
  const [selectedForMerge, setSelectedForMerge] = useState<number[]>([]);
  const [zoomedPlate, setZoomedPlate] = useState<number | null>(null);
  const [plateImages, setPlateImages] = useState<PlateImage[]>([]);
  const [isLoadingPlates, setIsLoadingPlates] = useState(false);
  const [platesLoadedCount, setPlatesLoadedCount] = useState(0);
  const [platesTotalCount, setPlatesTotalCount] = useState(0);
  const [cachedZipBlob, setCachedZipBlob] = useState<Blob | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<string | null>(null);
  const [upscaleHash, setUpscaleHash] = useState<string | null>(null);
  const [isUpscaling, setIsUpscaling] = useState(false);

  // === Refs ===
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);
  const compositeUrlRef = useRef<string | null>(null);
  const sourceUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const plateUrlsRef = useRef<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval>>(null);

  // === Computed ===
  const hasCrfSliders = version === "v7" || version === "v8";
  const hasSuperpixelSliders = version === "v6";
  const hasV4Sliders = version === "v4";
  const hasUpscaleToggle =
    version === "v4" ||
    version === "v6" ||
    version === "v9" ||
    version === "v10" ||
    version === "v11" ||
    version === "v14" ||
    version === "v15" ||
    version === "v16" ||
    version === "v17" ||
    version === "v18" ||
    version === "v19" ||
    version === "v20";
  const hasChromaSlider =
    version === "v4" ||
    version === "v6" ||
    version === "v7" ||
    version === "v8" ||
    version === "v9" ||
    version === "v10" ||
    version === "v11" ||
    version === "v14" ||
    version === "v15" ||
    version === "v16" ||
    version === "v17" ||
    version === "v18" ||
    version === "v19" ||
    version === "v20";
  const hasV9Sliders =
    version === "v9" || version === "v10" || version === "v11" || version === "v14";
  const canCompare = compositeUrl !== null && sourceUrl !== null;
  const displayImage = showOriginal && canCompare ? sourceUrl : (compositeUrl ?? sourceUrl);

  // === Helpers ===
  const cleanupPlateUrls = useCallback(() => {
    for (const url of plateUrlsRef.current) {
      URL.revokeObjectURL(url);
    }
    plateUrlsRef.current = [];
  }, []);

  const clearError = useCallback(() => setError(null), []);

  const cancelRequest = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
    setIsLoadingPlates(false);
    setProgressStage(null);
    setProgressPct(0);
  }, []);

  const resetUiOnError = useCallback(() => {
    setIsLoading(false);
    setIsLoadingPlates(false);
    setProgressStage(null);
    setProgressPct(0);
    setIsMerging(false);
    setDownloadProgress(null);
  }, []);

  // === Keyboard shortcut: spacebar toggles comparison ===
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        setShowOriginal((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // === Close nav on outside click (mobile) ===
  useEffect(() => {
    if (!navOpen) return;
    const handler = (e: MouseEvent) => {
      const nav = document.querySelector(".nav-panel");
      const burger = document.querySelector(".hamburger");
      if (
        nav &&
        !nav.contains(e.target as Node) &&
        burger &&
        !burger.contains(e.target as Node)
      ) {
        setNavOpen(false);
      }
    };
    window.addEventListener("click", handler);
    return () => window.removeEventListener("click", handler);
  }, [navOpen]);

  const getParams = useCallback(
    (overrides?: Partial<SeparationParams>): SeparationParams => ({
      plates: overrides?.plates ?? plates,
      dust: overrides?.dust ?? dust,
      useEdges: overrides?.useEdges ?? useEdges,
      edgeSigma: overrides?.edgeSigma ?? edgeSigma,
      lockedColors:
        overrides?.lockedColors ?? colors.filter((c) => c.locked).map((c) => c.rgb),
      version: overrides?.version ?? version,
      upscale: overrides?.upscale ?? upscale,
      medianSize: overrides?.medianSize ?? medianSize,
      chromaBoost: overrides?.chromaBoost ?? chromaBoost,
      shadowThreshold: overrides?.shadowThreshold ?? shadowThreshold,
      highlightThreshold: overrides?.highlightThreshold ?? highlightThreshold,
      nSegments: overrides?.nSegments ?? nSegments,
      compactness: overrides?.compactness ?? compactness,
      crfSpatial: overrides?.crfSpatial ?? crfSpatial,
      crfColor: overrides?.crfColor ?? crfColor,
      crfCompat: overrides?.crfCompat ?? crfCompat,
      sigmaS: overrides?.sigmaS ?? sigmaS,
      sigmaR: overrides?.sigmaR ?? sigmaR,
      meanshiftSp: overrides?.meanshiftSp ?? meanshiftSp,
      meanshiftSr: overrides?.meanshiftSr ?? meanshiftSr,
      detailStrength: overrides?.detailStrength ?? detailStrength,
    }),
    [
      plates, dust, useEdges, edgeSigma, colors, version, upscale,
      medianSize, chromaBoost, shadowThreshold, highlightThreshold,
      nSegments, compactness, crfSpatial, crfColor, crfCompat,
      sigmaS, sigmaR, meanshiftSp, meanshiftSr, detailStrength,
    ]
  );

  const stopProgress = useCallback(() => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    progressTimerRef.current = null;
    setProgressPct(100);
    setProgressStage(null);
  }, []);

  const fetchPlateImagesFromApi = useCallback(
    async (currentFile: File, params: SeparationParams) => {
      setIsLoadingPlates(true);
      setPlatesLoadedCount(0);
      setPlatesTotalCount(params.plates);
      cleanupPlateUrls();
      setPlateImages([]);

      try {
        await fetchPlatesStream(currentFile, params, (evt: PlateStreamEvent) => {
          if (evt.type === "count") {
            setPlatesTotalCount(evt.total);
          } else if (evt.type === "plate") {
            setPlateImages((prev) => [
              ...prev,
              {
                name: evt.name!,
                url: evt.image!,
                color: evt.color!,
                coverage: evt.coverage ?? 0,
              },
            ]);
            setPlatesLoadedCount((evt.index ?? 0) + 1);
          } else if (evt.type === "done") {
            setIsLoadingPlates(false);
          }
        });
      } catch (err) {
        console.error("Plate stream failed:", err);
      } finally {
        setIsLoadingPlates(false);
      }
    },
    [cleanupPlateUrls]
  );

  const applyPreviewResult = useCallback(
    (result: PreviewResult, currentFile: File, params: SeparationParams) => {
      if (compositeUrlRef.current) URL.revokeObjectURL(compositeUrlRef.current);
      compositeUrlRef.current = result.compositeUrl;
      setCompositeUrl(result.compositeUrl);
      setManifest(result.manifest);

      if (result.manifest.plates.length > 0) {
        setColors((prev) => {
          const locked = prev.filter((c) => c.locked);
          const detected = result.manifest.plates.map((p) => ({
            rgb: p.color,
            locked: false,
          }));
          if (locked.length === 0) return detected;
          return [...locked, ...detected.slice(locked.length)];
        });
      }

      fetchPlateImagesFromApi(currentFile, params);
    },
    [fetchPlateImagesFromApi]
  );

  const runPreview = useCallback(
    async (currentFile: File, params: SeparationParams) => {
      setIsLoading(true);
      setError(null);
      setPlateImages([]);
      setIsLoadingPlates(true);
      setPlatesLoadedCount(0);
      setPlatesTotalCount(params.plates);

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        setProgressPct(0);
        setProgressStage("Separating colors");
        const result = await fetchPreviewStream(
          currentFile,
          params,
          (stage, pct) => {
            setProgressStage(stage);
            setProgressPct(pct);
          },
          controller.signal
        );
        applyPreviewResult(result, currentFile, params);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Preview failed:", err);
        setError(toAppError(err));
        resetUiOnError();
        return;
      } finally {
        stopProgress();
        setIsLoading(false);
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [stopProgress, applyPreviewResult, resetUiOnError]
  );

  const schedulePreview = useCallback(
    (currentFile: File, params: SeparationParams) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        runPreview(currentFile, params);
      }, 200);
    },
    [runPreview]
  );

  const handleFileSelect = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setFile(f);
      setFileName(f.name);
      setError(null);
      const imgEl = new window.Image();
      const objUrl = URL.createObjectURL(f);
      imgEl.onload = () => {
        setImageInfo({
          width: imgEl.naturalWidth,
          height: imgEl.naturalHeight,
          size:
            f.size > 1024 * 1024
              ? (f.size / 1024 / 1024).toFixed(1) + "MB"
              : Math.round(f.size / 1024) + "KB",
          type: f.type.replace("image/", "") || "unknown",
        });
        URL.revokeObjectURL(objUrl);
      };
      imgEl.src = objUrl;
      setUpscaleHash(null);
      if (sourceUrlRef.current) URL.revokeObjectURL(sourceUrlRef.current);
      const url = URL.createObjectURL(f);
      sourceUrlRef.current = url;
      setSourceUrl(url);
      setCompositeUrl(null);
      setManifest(null);
      setColors([]);
      setPlateImages([]);
      cleanupPlateUrls();
      setCachedZipBlob(null);

      if (upscale) {
        setIsUpscaling(true);
        fetchUpscale(f)
          .then((result) => setUpscaleHash(result.hash))
          .catch((err) => console.error("Upscale cache failed:", err))
          .finally(() => setIsUpscaling(false));
      }
    },
    [upscale, cleanupPlateUrls]
  );

  const handleProcess = useCallback(() => {
    if (!file) return;
    runPreview(file, getParams());
  }, [file, getParams, runPreview]);

  const handleReset = useCallback(() => {
    setCompositeUrl(null);
    setManifest(null);
    setColors([]);
    setShowOriginal(false);
    setPlateImages([]);
    cleanupPlateUrls();
    setCachedZipBlob(null);
    setError(null);
  }, [cleanupPlateUrls]);

  const handleParamChange = useCallback(
    (key: string, value: number | boolean) => {
      const setters: Record<string, (v: never) => void> = {
        plates: setPlates as (v: never) => void,
        dust: setDust as (v: never) => void,
        useEdges: setUseEdges as (v: never) => void,
        edgeSigma: setEdgeSigma as (v: never) => void,
        medianSize: setMedianSize as (v: never) => void,
        chromaBoost: setChromaBoost as (v: never) => void,
        shadowThreshold: setShadowThreshold as (v: never) => void,
        highlightThreshold: setHighlightThreshold as (v: never) => void,
        nSegments: setNSegments as (v: never) => void,
        compactness: setCompactness as (v: never) => void,
        crfSpatial: setCrfSpatial as (v: never) => void,
        crfColor: setCrfColor as (v: never) => void,
        crfCompat: setCrfCompat as (v: never) => void,
        sigmaS: setSigmaS as (v: never) => void,
        sigmaR: setSigmaR as (v: never) => void,
        meanshiftSp: setMeanshiftSp as (v: never) => void,
        meanshiftSr: setMeanshiftSr as (v: never) => void,
        detailStrength: setDetailStrength as (v: never) => void,
      };
      setters[key]?.(value as never);
      if (file && compositeUrl) {
        const overrides = { [key]: value };
        schedulePreview(file, getParams(overrides));
      }
    },
    [file, compositeUrl, schedulePreview, getParams]
  );

  const handleColorChange = useCallback((index: number, hex: string) => {
    const rgb = hexToRgb(hex);
    setColors((prev) => {
      const next = [...prev];
      if (next[index]) {
        next[index] = { rgb, locked: true };
      }
      return next;
    });
  }, []);

  const handleToggleLock = useCallback((index: number) => {
    setColors((prev) => {
      const next = [...prev];
      if (next[index]) {
        next[index] = { ...next[index], locked: !next[index].locked };
      }
      return next;
    });
  }, []);

  const handleRemoveColor = useCallback((index: number) => {
    setColors((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleAddColor = useCallback(() => {
    setColors((prev) => [...prev, { rgb: [128, 128, 128], locked: true }]);
  }, []);

  const generateDiagram = useCallback(
    async (compositeImgUrl: string, plateImgs: PlateImage[]): Promise<Blob> => {
      const compositeImg = await loadImage(compositeImgUrl);
      const plateLoadedImgs: HTMLImageElement[] = [];
      for (const p of plateImgs) {
        plateLoadedImgs.push(await loadImage(p.url));
      }

      const padding = 20;
      const labelHeight = 30;
      const cols = Math.min(plateImgs.length, 4);
      const rows = Math.ceil(plateImgs.length / cols);
      const plateW = Math.floor(compositeImg.width / cols);
      const plateH = Math.floor((compositeImg.height / compositeImg.width) * plateW);

      const canvasW = compositeImg.width + padding * 2;
      const canvasH =
        compositeImg.height + padding * 3 + (plateH + labelHeight) * rows + padding;

      const canvas = document.createElement("canvas");
      canvas.width = canvasW;
      canvas.height = canvasH;
      const ctx = canvas.getContext("2d")!;

      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvasW, canvasH);
      ctx.drawImage(compositeImg, padding, padding, compositeImg.width, compositeImg.height);
      ctx.fillStyle = "#000";
      ctx.font = `${Math.max(14, Math.floor(canvasW / 40))}px monospace`;
      ctx.fillText("composite", padding, compositeImg.height + padding * 2);

      const plateStartY = compositeImg.height + padding * 2 + labelHeight;
      for (let i = 0; i < plateImgs.length; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = padding + col * (plateW + padding);
        const y = plateStartY + row * (plateH + labelHeight + padding);

        const hex = rgbToHex(plateImgs[i].color);
        ctx.fillStyle = hex;
        ctx.fillRect(x, y - labelHeight, plateW, labelHeight - 2);
        ctx.fillStyle = "#000";
        ctx.font = `${Math.max(10, Math.floor(canvasW / 60))}px monospace`;
        ctx.fillText(
          `${plateImgs[i].name} ${hex} ${plateImgs[i].coverage.toFixed(1)}%`,
          x + 4,
          y - 8
        );
        if (plateLoadedImgs[i]) {
          ctx.drawImage(plateLoadedImgs[i], x, y, plateW, plateH);
        }
      }

      return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob!), "image/png");
      });
    },
    []
  );

  const handleDownload = useCallback(async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setDownloadProgress("fetching...");
    try {
      const zipBlob = cachedZipBlob ?? (await fetchSeparation(file, getParams()));
      setDownloadProgress("building ZIP...");
      const JSZip = (await import("jszip")).default;
      const zip = await JSZip.loadAsync(zipBlob);
      const newZip = new JSZip();

      const plateColorMap = manifest?.plates ?? [];
      for (const [filename, zipEntry] of Object.entries(zip.files)) {
        if (zipEntry.dir) continue;
        const data = await zipEntry.async("blob");

        const plateMatch = filename.match(/^(plate\d+)\.png$/);
        if (plateMatch) {
          const plateName = plateMatch[1];
          const plateInfo = plateColorMap.find((p) => p.name === plateName);
          if (plateInfo) {
            const hex = rgbToHex(plateInfo.color).replace("#", "").toUpperCase();
            newZip.file(`${plateName}_${hex}.png`, data);
          } else {
            newZip.file(filename, data);
          }
        } else {
          newZip.file(filename, data);
        }
      }

      if (compositeUrl && plateImages.length > 0 && manifest) {
        const diagramBlob = await generateDiagram(compositeUrl, plateImages);
        newZip.file("diagram.png", diagramBlob);
      }

      const finalBlob = await newZip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(finalBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "color-separator-plates.zip";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed:", err);
      setError(toAppError(err));
    } finally {
      setIsLoading(false);
      setDownloadProgress(null);
    }
  }, [file, getParams, cachedZipBlob, manifest, compositeUrl, plateImages, generateDiagram]);

  const handleMerge = useCallback(async () => {
    if (!file || selectedForMerge.length < 2) return;
    setIsMerging(true);
    setError(null);
    try {
      const pairs: number[][] = [];
      for (let i = 1; i < selectedForMerge.length; i++) {
        pairs.push([selectedForMerge[0], selectedForMerge[i]]);
      }
      const result = await fetchMerge(file, getParams(), pairs, upscaleHash);
      if (compositeUrlRef.current) URL.revokeObjectURL(compositeUrlRef.current);
      compositeUrlRef.current = result.compositeUrl;
      setCompositeUrl(result.compositeUrl);
      setManifest(result.manifest);
      if (result.manifest.plates.length > 0) {
        setColors(result.manifest.plates.map((p) => ({ rgb: p.color, locked: false })));
      }
      setMergeMode(false);
      setSelectedForMerge([]);
      fetchPlateImagesFromApi(file, getParams());
    } catch (err) {
      console.error("Merge failed:", err);
      setError(toAppError(err));
    } finally {
      setIsMerging(false);
    }
  }, [file, selectedForMerge, getParams, upscaleHash, fetchPlateImagesFromApi]);

  return {
    // File
    file,
    fileName,
    imageInfo,
    sourceUrl,
    compositeUrl,
    manifest,
    colors,
    // Loading / error
    isLoading,
    error,
    clearError,
    cancelRequest,
    // Params
    version,
    setVersion,
    plates,
    dust,
    useEdges,
    edgeSigma,
    upscale,
    setUpscale,
    medianSize,
    chromaBoost,
    shadowThreshold,
    highlightThreshold,
    nSegments,
    compactness,
    crfSpatial,
    crfColor,
    crfCompat,
    sigmaS,
    sigmaR,
    meanshiftSp,
    meanshiftSr,
    detailStrength,
    // Progress
    progressStage,
    progressPct,
    // UI
    showOriginal,
    setShowOriginal,
    navOpen,
    setNavOpen,
    showAbout,
    setShowAbout,
    mergeMode,
    setMergeMode,
    isMerging,
    selectedForMerge,
    setSelectedForMerge,
    zoomedPlate,
    setZoomedPlate,
    plateImages,
    isLoadingPlates,
    platesLoadedCount,
    platesTotalCount,
    downloadProgress,
    isUpscaling,
    // Computed
    hasCrfSliders,
    hasSuperpixelSliders,
    hasV4Sliders,
    hasUpscaleToggle,
    hasChromaSlider,
    hasV9Sliders,
    canCompare,
    displayImage,
    // Handlers
    handleFileSelect,
    handleProcess,
    handleReset,
    handleParamChange,
    handleColorChange,
    handleToggleLock,
    handleRemoveColor,
    handleAddColor,
    handleDownload,
    handleMerge,
    // Refs
    fileInputRef,
  };
}
