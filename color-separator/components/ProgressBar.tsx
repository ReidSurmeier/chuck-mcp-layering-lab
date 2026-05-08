interface ProgressBarProps {
  isLoading: boolean;
  progressStage: string | null;
  progressPct: number;
}

export default function ProgressBar({
  isLoading,
  progressStage,
  progressPct,
}: ProgressBarProps) {
  if (!isLoading || !progressStage) return null;

  return (
    <div className="progress-bar-container">
      <div className="progress-bar" style={{ width: `${progressPct}%` }} />
      <span className="progress-label">
        {progressStage} — {progressPct}%
      </span>
    </div>
  );
}
