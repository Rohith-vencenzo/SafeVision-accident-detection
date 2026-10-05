"""Detection metrics, with the sample-count honesty attached.

Two levels, because they answer different questions:

**Event level** - "did the system raise an incident for this accident, and did
it stay quiet otherwise?"  This is what an operator experiences.  Yields a
confusion matrix, precision, recall, F1, false-positive rate and
false-negative rate.

**Score level** - "as the confirmation threshold moves, what happens to the
detection/false-alarm trade-off?"  Yields a PR curve and PR-AUC, used to *choose*
a threshold on validation data.

Every rate is reported with a **Wilson confidence interval**, because a 0/2
false-alarm rate on two negatives is not a measurement - it is an anecdote with a
decimal point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ClassificationMetrics",
    "ScorePoint",
    "WilsonInterval",
    "confusion_from_events",
    "precision_recall_curve",
    "pr_auc",
    "summarise_split_sufficiency",
    "wilson_interval",
]


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score interval for a binomial proportion."""

    low: float
    high: float
    point: float
    n: int
    z: float = 1.959963985  # 95%

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point": round(self.point, 4),
            "ci95_low": round(self.low, 4),
            "ci95_high": round(self.high, 4),
            "n": int(self.n),
        }

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> WilsonInterval:
    """Wilson interval - behaves sanely at 0/n and n/n, unlike normal-approx."""
    if n <= 0:
        return WilsonInterval(0.0, 1.0, 0.0, 0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return WilsonInterval(
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        point=p,
        n=n,
        z=z,
    )


# --------------------------------------------------------------------------- #
# event-level
# --------------------------------------------------------------------------- #
@dataclass
class ClassificationMetrics:
    """Event-level confusion matrix and the rates derived from it."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    negatives_evaluated: int = 0
    positives_evaluated: int = 0
    notes: List[str] = field(default_factory=list)

    # -- derived --------------------------------------------------------- #
    @property
    def precision(self) -> Optional[float]:
        d = self.true_positive + self.false_positive
        return None if d == 0 else self.true_positive / d

    @property
    def recall(self) -> Optional[float]:
        d = self.true_positive + self.false_negative
        return None if d == 0 else self.true_positive / d

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Of the clips that were quiet, how many produced an incident."""
        d = self.false_positive + self.true_negative
        return None if d == 0 else self.false_positive / d

    @property
    def false_negative_rate(self) -> Optional[float]:
        d = self.false_negative + self.true_positive
        return None if d == 0 else self.false_negative / d

    def confusion(self) -> Dict[str, int]:
        return {
            "TP": self.true_positive,
            "FP": self.false_positive,
            "FN": self.false_negative,
            "TN": self.true_negative,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confusion_matrix": self.confusion(),
            "positives_evaluated": self.positives_evaluated,
            "negatives_evaluated": self.negatives_evaluated,
            "precision": _r(self.precision),
            "recall": _r(self.recall),
            "f1": _r(self.f1),
            "false_positive_rate": _r(self.false_positive_rate),
            "false_negative_rate": _r(self.false_negative_rate),
            "precision_ci95": wilson_interval(self.true_positive,
                                              self.true_positive + self.false_positive).to_dict(),
            "recall_ci95": wilson_interval(self.true_positive,
                                           self.true_positive + self.false_negative).to_dict(),
            "false_positive_rate_ci95": wilson_interval(
                self.false_positive, self.false_positive + self.true_negative).to_dict(),
            "false_negative_rate_ci95": wilson_interval(
                self.false_negative, self.false_negative + self.true_positive).to_dict(),
            "notes": list(self.notes),
        }


def confusion_from_events(
    predictions: Sequence[Tuple[bool, float, float]],
    labels: Sequence[Tuple[bool, Sequence[Tuple[float, float]]]],
    tolerance_seconds: float = 0.0,
) -> ClassificationMetrics:
    """Build the confusion matrix from ``(is_positive_clip, peak_score, event_start)``.

    ``predictions[i]`` and ``labels[i]`` describe the same clip.  A predicted
    incident counts as a true positive when it overlaps a labelled accident
    window; on a negative clip any incident is a false positive.
    """
    metrics = ClassificationMetrics()
    for (clip_is_positive, _score, start), (label_positive, windows) in zip(predictions, labels):
        if label_positive:
            metrics.positives_evaluated += 1
            if windows:
                hit = any(
                    (start - tolerance_seconds) <= end and start >= (begin - tolerance_seconds)
                    for begin, end in windows
                )
            else:
                hit = False
            if hit:
                metrics.true_positive += 1
            else:
                metrics.false_negative += 1
        else:
            metrics.negatives_evaluated += 1
            if clip_is_positive:
                metrics.false_positive += 1
            else:
                metrics.true_negative += 1
    return metrics


# --------------------------------------------------------------------------- #
# score level
# --------------------------------------------------------------------------- #
@dataclass
class ScorePoint:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    false_positive_rate: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "TP": self.true_positives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "TN": self.true_negatives,
            "precision": _r(self.precision),
            "recall": _r(self.recall),
            "f1": _r(self.f1),
            "false_positive_rate": _r(self.false_positive_rate),
        }


def precision_recall_curve(
    positives: Sequence[float],
    negatives: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> List[ScorePoint]:
    """Sweep a confirmation threshold over clip-level peak scores.

    ``positives`` / ``negatives`` are *clip* peak scores, because the system's
    output is an incident per clip, not per frame.
    """
    if thresholds is None:
        candidates = sorted({round(v, 4) for v in list(positives) + list(negatives)} | {0.0, 1.0})
        # sweep low -> high; "predicted positive" means peak >= threshold
        thresholds = list(reversed(candidates))

    points: List[ScorePoint] = []
    for threshold in thresholds:
        tp = len([s for s in positives if s >= threshold])
        fn = len(positives) - tp
        fp = len([s for s in negatives if s >= threshold])
        tn = len(negatives) - fp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else None)
        points.append(
            ScorePoint(
                threshold=float(threshold),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=precision,
                recall=recall,
                f1=f1,
                false_positive_rate=(fp / (fp + tn)) if (fp + tn) else None,
            )
        )
    return points


def pr_auc(points: Sequence[ScorePoint]) -> Optional[float]:
    """Average precision (step interpolation).

    ``AP = sum_i (R_i - R_{i-1}) * P_i`` over thresholds walked from high to
    low.  The result is already a proportion of the positive class, so it must
    **not** be divided by the positive count again.

    Returns ``None`` when there are no positives - PR-AUC is undefined then, and
    inventing a number is worse than saying so.
    """
    if not points:
        return None
    total_pos = points[0].true_positives + points[0].false_negatives
    if total_pos == 0:
        return None
    ap = 0.0
    previous_recall = 0.0
    for point in points:
        if point.recall is None:
            continue
        ap += (point.recall - previous_recall) * (point.precision or 0.0)
        previous_recall = point.recall
    return ap


# --------------------------------------------------------------------------- #
# honesty helpers
# --------------------------------------------------------------------------- #
def summarise_split_sufficiency(splits: Dict[str, Any]) -> Dict[str, Any]:
    """State plainly whether a split can support any claim at all."""
    parts = {}
    degenerate = []
    for name, split in splits.items():
        parts[name] = split.describe() if hasattr(split, "describe") else dict(split)
        if getattr(split, "n_clips", 0) < 5:
            degenerate.append(name)
    test = splits.get("test")
    test_clips = getattr(test, "n_clips", 0) if test else 0
    test_pos = getattr(test, "n_positive", 0) if test else 0
    usable = test_clips >= 20 and test_pos >= 5 and not degenerate
    reasons: List[str] = []
    if degenerate:
        reasons.append(f"partition(s) with <5 clips: {degenerate}")
    if test_clips < 20:
        reasons.append(f"test partition has {test_clips} clip(s); >=20 needed for a stable rate")
    if test_pos < 5:
        reasons.append(f"test partition has {test_pos} positive clip(s); >=5 needed")
    return {
        "parts": parts,
        "sufficient_for_quality_claim": usable,
        "reasons": reasons or ["sufficient"],
    }


def class_imbalance(positives: int, negatives: int) -> Dict[str, Any]:
    total = positives + negatives
    ratio = (negatives / positives) if positives else float("inf")
    return {
        "positives": positives,
        "negatives": negatives,
        "total": total,
        "negative_to_positive_ratio": None if ratio == float("inf") else round(ratio, 2),
        "majority_class_pct": None if not total else round(100 * max(positives, negatives) / total, 1),
    }


def _r(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)