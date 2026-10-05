"""Evaluation harness: manifest, leakage-safe splits, honest metrics.

    from ai_engine.evaluation import load_manifest, evaluate_manifest

Nothing here may ever place a call or send a message: ``evaluate_manifest``
forcibly disables the notification subsystem before touching the pipeline.
"""

from __future__ import annotations

from ai_engine.evaluation.evaluate import (
    ClipPrediction,
    EvaluationReport,
    evaluate_manifest,
    render_report,
    threshold_hashes,
)
from ai_engine.evaluation.manifest import (
    ClipLabel,
    EvalManifest,
    EventLabel,
    ManifestError,
    load_manifest,
)
from ai_engine.evaluation.metrics import (
    ClassificationMetrics,
    class_imbalance,
    confusion_from_events,
    pr_auc,
    precision_recall_curve,
    summarise_split_sufficiency,
    wilson_interval,
)
from ai_engine.evaluation.splits import (
    Split,
    assert_no_leakage,
    group_fingerprint,
    group_split,
)

__all__ = [
    "ClassificationMetrics",
    "ClipLabel",
    "ClipPrediction",
    "EvalManifest",
    "EvaluationReport",
    "EventLabel",
    "ManifestError",
    "Split",
    "assert_no_leakage",
    "class_imbalance",
    "confusion_from_events",
    "evaluate_manifest",
    "group_fingerprint",
    "group_split",
    "load_manifest",
    "pr_auc",
    "precision_recall_curve",
    "render_report",
    "summarise_split_sufficiency",
    "threshold_hashes",
    "wilson_interval",
]