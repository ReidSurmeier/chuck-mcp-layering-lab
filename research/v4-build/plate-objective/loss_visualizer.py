"""Per-term loss curve plotter for debugging the plate objective.

Call `record_step()` from your solver inner loop, then `plot()` at the end
to see which term is dominant / decreasing / diverging.

Headless-safe: uses the Agg matplotlib backend so it runs in solver
batch jobs without a display.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Names match composite_loss term keys so the visualiser keys 1:1 to the
# weight knobs in LossWeights.
TERM_NAMES = (
    "final_image",
    "checkpoint_proof",
    "plate_not_composite",
    "cell_exclusivity",
    "role_coverage_caps",
    "role_frequency_permission",
    "load_bearing_singleton",
    "load_bearing_pair",
    "printability",
)


@dataclass
class LossHistory:
    """Append-only per-step record of each term's contribution.

    Solver loop:
        history = LossHistory()
        for step in range(N):
            terms = {... compute each ...}
            history.record_step(step, terms, total=sum(terms.values()))
        history.plot("/tmp/loss-curves.png")
    """

    steps: list[int] = field(default_factory=list)
    terms: dict[str, list[float]] = field(default_factory=lambda: {n: [] for n in TERM_NAMES})
    total: list[float] = field(default_factory=list)

    def record_step(self, step: int, terms: dict[str, float], total: float) -> None:
        self.steps.append(int(step))
        for name in TERM_NAMES:
            self.terms[name].append(float(terms.get(name, 0.0)))
        self.total.append(float(total))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"steps": self.steps, "terms": self.terms, "total": self.total},
                indent=2,
            )
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "LossHistory":
        data = json.loads(Path(path).read_text())
        h = cls(steps=data["steps"], total=data["total"])
        for k, v in data["terms"].items():
            h.terms[k] = v
        return h

    def plot(self, output_path: str | Path, *, log_y: bool = False) -> Path:
        """Render a multi-line chart: one line per term + total.

        Returns the output Path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        for name in TERM_NAMES:
            ys = self.terms[name]
            if any(y > 0 for y in ys):  # only plot terms that fired
                ax.plot(self.steps, ys, label=name, alpha=0.85)
        ax.plot(self.steps, self.total, label="TOTAL", color="black", linewidth=2.0)
        ax.set_xlabel("solver step")
        ax.set_ylabel("loss")
        ax.set_title("chuck-mcp v4 plate-objective term decomposition")
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def dominant_terms(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k term names ranked by final value."""
        finals = [(name, self.terms[name][-1] if self.terms[name] else 0.0) for name in TERM_NAMES]
        finals.sort(key=lambda x: x[1], reverse=True)
        return finals[:top_k]


def quick_plot_from_steps(
    output_path: str | Path,
    step_records: Iterable[tuple[int, dict[str, float], float]],
) -> Path:
    """Convenience: pass an iterable of (step, terms_dict, total) tuples."""
    h = LossHistory()
    for step, terms, total in step_records:
        h.record_step(step, terms, total)
    return h.plot(output_path)
