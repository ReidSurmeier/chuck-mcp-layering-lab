"""
jaccard_evaluator.py — compare Opus predictions vs MediaPipe ground truth.

Metric definitions:
    Jaccard  = |intersection| / |union|
    Precision = |intersection| / |predicted|
    Recall    = |intersection| / |ground_truth|
    F1        = 2 * precision * recall / (precision + recall)

Both sides are SETS of cell IDs per region. A region is scored only when
at least one side has IDs — if both sides are empty for that region (e.g.
no hair visible AND Opus correctly returns []) the region is "trivial
match" with Jaccard = 1.0 by convention, but is flagged so it can be
excluded from aggregates if needed.

Aggregates are computed at three levels:
    per-image    -> mean/median/min over regions present in that image
    per-region   -> mean/median/min over images that contain that region
    overall      -> mean of per-image means across the dataset
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class RegionScore:
    region: str
    jaccard: float
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_truth: int
    n_intersect: int
    is_trivial: bool  # both sets empty -> conventionally jaccard = 1.0


@dataclass
class ImageScore:
    image_id: str
    region_scores: list[RegionScore] = field(default_factory=list)

    def mean_jaccard(self, *, include_trivial: bool = False) -> float:
        rs = self._filtered(include_trivial)
        if not rs:
            return 0.0
        return statistics.fmean(r.jaccard for r in rs)

    def median_jaccard(self, *, include_trivial: bool = False) -> float:
        rs = self._filtered(include_trivial)
        if not rs:
            return 0.0
        return statistics.median(r.jaccard for r in rs)

    def min_jaccard(self, *, include_trivial: bool = False) -> float:
        rs = self._filtered(include_trivial)
        if not rs:
            return 0.0
        return min(r.jaccard for r in rs)

    def mean_f1(self, *, include_trivial: bool = False) -> float:
        rs = self._filtered(include_trivial)
        if not rs:
            return 0.0
        return statistics.fmean(r.f1 for r in rs)

    def _filtered(self, include_trivial: bool) -> list[RegionScore]:
        return [r for r in self.region_scores
                if include_trivial or not r.is_trivial]


@dataclass
class BenchResult:
    image_scores: list[ImageScore] = field(default_factory=list)

    # ---------- per-image aggregates ----------
    def per_image_mean_jaccard(self) -> dict[str, float]:
        return {im.image_id: im.mean_jaccard() for im in self.image_scores}

    # ---------- per-region aggregates ----------
    def per_region_jaccard(self) -> dict[str, list[float]]:
        bucket: dict[str, list[float]] = {}
        for im in self.image_scores:
            for r in im.region_scores:
                if r.is_trivial:
                    continue
                bucket.setdefault(r.region, []).append(r.jaccard)
        return bucket

    def per_region_summary(self) -> dict[str, dict[str, float]]:
        bucket = self.per_region_jaccard()
        return {
            region: {
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
            for region, vals in bucket.items()
        }

    # ---------- overall aggregates ----------
    def overall_mean_jaccard(self) -> float:
        per_image = self.per_image_mean_jaccard()
        if not per_image:
            return 0.0
        return statistics.fmean(per_image.values())

    def overall_median_jaccard(self) -> float:
        per_image = self.per_image_mean_jaccard()
        if not per_image:
            return 0.0
        return statistics.median(per_image.values())

    def overall_min_jaccard(self) -> float:
        per_image = self.per_image_mean_jaccard()
        if not per_image:
            return 0.0
        return min(per_image.values())

    def overall_mean_f1(self) -> float:
        if not self.image_scores:
            return 0.0
        return statistics.fmean(im.mean_f1() for im in self.image_scores)

    def to_dict(self) -> dict:
        return {
            "n_images": len(self.image_scores),
            "overall_mean_jaccard": self.overall_mean_jaccard(),
            "overall_median_jaccard": self.overall_median_jaccard(),
            "overall_min_jaccard": self.overall_min_jaccard(),
            "overall_mean_f1": self.overall_mean_f1(),
            "per_image_mean_jaccard": self.per_image_mean_jaccard(),
            "per_region_summary": self.per_region_summary(),
            "per_image_detail": [
                {
                    "image_id": im.image_id,
                    "mean_jaccard": im.mean_jaccard(),
                    "median_jaccard": im.median_jaccard(),
                    "min_jaccard": im.min_jaccard(),
                    "mean_f1": im.mean_f1(),
                    "regions": [
                        {
                            "region": r.region,
                            "jaccard": r.jaccard,
                            "precision": r.precision,
                            "recall": r.recall,
                            "f1": r.f1,
                            "n_pred": r.n_pred,
                            "n_truth": r.n_truth,
                            "n_intersect": r.n_intersect,
                            "is_trivial": r.is_trivial,
                        }
                        for r in im.region_scores
                    ],
                }
                for im in self.image_scores
            ],
        }


# ---------------------------------------------------------------------------
# Core comparators
# ---------------------------------------------------------------------------


def _score_pair(predicted: Iterable[int], truth: Iterable[int]
                ) -> tuple[float, float, float, float, int, int, int, bool]:
    pset = set(int(x) for x in predicted)
    tset = set(int(x) for x in truth)
    inter = pset & tset
    union = pset | tset

    n_pred = len(pset)
    n_truth = len(tset)
    n_inter = len(inter)

    if not union:  # trivial match — both empty
        return 1.0, 1.0, 1.0, 1.0, 0, 0, 0, True

    jacc = n_inter / len(union)
    precision = n_inter / n_pred if n_pred else 0.0
    recall = n_inter / n_truth if n_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return jacc, precision, recall, f1, n_pred, n_truth, n_inter, False


def compare(
    opus_predictions: dict[str, list[int]],
    ground_truth: dict[str, list[int]],
    *,
    image_id: str = "",
    regions: list[str] | None = None,
) -> ImageScore:
    """Score one image's worth of predictions against ground truth.

    Args:
        opus_predictions: {region_name: [cell_ids]} from Opus
        ground_truth: {region_name: [cell_ids]} from MediaPipe pipeline
        image_id: label used in output (default empty)
        regions: explicit region list to score. Defaults to union of keys
                 from both inputs. Missing regions on either side become [].
    """
    if regions is None:
        regions = sorted(set(opus_predictions) | set(ground_truth))

    region_scores: list[RegionScore] = []
    for r in regions:
        pred = opus_predictions.get(r, [])
        truth = ground_truth.get(r, [])
        (jacc, prec, rec, f1, n_pred, n_truth,
         n_inter, trivial) = _score_pair(pred, truth)
        region_scores.append(
            RegionScore(
                region=r,
                jaccard=jacc,
                precision=prec,
                recall=rec,
                f1=f1,
                n_pred=n_pred,
                n_truth=n_truth,
                n_intersect=n_inter,
                is_trivial=trivial,
            )
        )

    return ImageScore(image_id=image_id, region_scores=region_scores)


def aggregate(image_scores: list[ImageScore]) -> BenchResult:
    return BenchResult(image_scores=list(image_scores))
