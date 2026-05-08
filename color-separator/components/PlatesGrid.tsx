import { memo } from "react";
import { rgbToHex } from "@/lib/colors";
import type { PlateImage } from "../constants";

interface PlatesGridProps {
  plateImages: PlateImage[];
  isLoadingPlates: boolean;
  platesLoadedCount: number;
  platesTotalCount: number;
  skeletonCount: number;
  mergeMode: boolean;
  selectedForMerge: number[];
  onToggleMergeSelect: (index: number) => void;
  onZoomPlate: (index: number) => void;
  isMerging: boolean;
}

const PlatesGrid = memo(function PlatesGrid({
  plateImages,
  isLoadingPlates,
  platesTotalCount,
  skeletonCount,
  mergeMode,
  selectedForMerge,
  onToggleMergeSelect,
  onZoomPlate,
  isMerging,
}: PlatesGridProps) {
  if (plateImages.length === 0 && !isLoadingPlates) return null;

  return (
    <div className="plates-section">
      <h3 className="plates-section-title">
        plates ({plateImages.length}
        {isLoadingPlates && platesTotalCount > 0
          ? ` of ${platesTotalCount}`
          : ""}
        )
      </h3>

      {isMerging && (
        <div className="merge-progress-overlay">
          <div className="merge-spinner" />
          <span>merging plates...</span>
        </div>
      )}

      {/* Skeleton loading */}
      {isLoadingPlates && plateImages.length === 0 && (
        <div className="plates-grid">
          {Array.from({ length: skeletonCount }).map((_, i) => (
            <div key={i} className="plate-card plate-skeleton">
              <div className="plate-card-image plate-skeleton-img" />
              <div className="plate-card-info">
                <div className="plate-skeleton-swatch" />
                <div className="plate-skeleton-text" />
                <div className="plate-skeleton-text short" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Plate cards */}
      {(plateImages.length > 0 || !isLoadingPlates) && (
        <div className="plates-grid">
          {plateImages.map((plate, i) => (
            <div
              className={`plate-card ${mergeMode && selectedForMerge.includes(i) ? "plate-selected" : ""}`}
              key={i}
              onClick={() => {
                if (mergeMode) {
                  onToggleMergeSelect(i);
                } else {
                  onZoomPlate(i);
                }
              }}
              style={{ cursor: "pointer" }}
            >
              <div
                className="plate-card-image"
                style={{ borderColor: rgbToHex(plate.color) }}
              >
                <img src={plate.url} alt={plate.name} />
                <div
                  className="plate-card-color-overlay"
                  style={{ backgroundColor: rgbToHex(plate.color) }}
                />
              </div>
              <div className="plate-card-info">
                <span
                  className="plate-card-swatch"
                  style={{ backgroundColor: rgbToHex(plate.color) }}
                />
                <span className="plate-card-hex">
                  {rgbToHex(plate.color).toUpperCase()}
                </span>
                <span className="plate-card-coverage">
                  {plate.coverage.toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default PlatesGrid;
