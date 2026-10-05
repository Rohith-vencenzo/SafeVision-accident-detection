"""Run an evaluation: manifest -> predictions -> metrics -> a written report.

Design constraints that shape this module:

* **Reproducible.**  Everything is seeded and hashed; the report records the
  model weights hash, the threshold set hash and the data fingerprint so a later
  run can prove it evaluated the same thing.
* **Honest by construction.**  The report leads with whether the set is even
  allowed to support a quality claim, and prints the reasons when it is not.
* **Never edits thresholds automatically.**  It proposes a threshold; a human
  applies it.  Silently retuning a safety gate from a possibly-unrepresentative
  set is worse than leaving it alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_engine.evaluation.manifest import ClipLabel, EvalManifest, load_manifest
from ai_engine.evaluation.metrics import (
    ClassificationMetrics,
    class_imbalance,
    confusion_from_events,
    pr_auc,
    precision_recall_curve,
    summarise_split_sufficiency,
)
from ai_engine.evaluation.splits import assert_no_leakage, group_fingerprint, group_split
from ai_engine.utils.jsonio import atomic_write_json
from ai_engine.utils.logging_utils import get_logger
from ai_engine.utils.timeutil import now_iso

__all__ = ["ClipPrediction", "EvaluationReport", "evaluate_manifest", "threshold_hashes"]

_LOGGER = get_logger("evaluation")

#: Below this many positives anywhere, a rate is reported but explicitly marked
#: as not meaningful.
MIN_POSITIVES_FOR_CLAIM = 5


# --------------------------------------------------------------------------- #
@dataclass
class ClipPrediction:
    """What the pipeline produced for one clip."""

    clip: str
    source_id: str
    label: str
    quality: str
    peak_score: float = 0.0
    peak_physical: float = 0.0
    peak_temporal: float = 0.0
    incident_count: int = 0
    incident_times: List[float] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip": self.clip,
            "source_id": self.source_id,
            "label": self.label,
            "quality": self.quality,
            "peak_score": round(self.peak_score, 4),
            "peak_physical": round(self.peak_physical, 4),
            "peak_temporal": round(self.peak_temporal, 4),
            "incident_count": self.incident_count,
            "incident_times": [round(t, 2) for t in self.incident_times],
            "frames": self.frames,
            "seconds": round(self.seconds, 2),
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Everything an evaluator needs to judge - and to distrust - a result."""

    generated_at: str
    manifest: Dict[str, Any]
    splits: Dict[str, Any]
    leakage_check: str
    provenance: Dict[str, Any]
    event_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    score_curve: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    auc: Dict[str, Optional[float]] = field(default_factory=dict)
    imbalance: Dict[str, Any] = field(default_factory=dict)
    suggested_threshold: Dict[str, Any] = field(default_factory=dict)
    sufficiency: Dict[str, Any] = field(default_factory=dict)
    claims_allowed: List[str] = field(default_factory=list)
    claims_withheld: List[str] = field(default_factory=list)
    per_clip: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "manifest": self.manifest,
            "splits": self.splits,
            "leakage_check": self.leakage_check,
            "provenance": self.provenance,
            "class_imbalance": self.imbalance,
            "event_metrics": self.event_metrics,
            "score_curve": self.score_curve,
            "pr_auc": self.auc,
            "threshold_suggestion": self.suggested_threshold,
            "split_sufficiency": self.sufficiency,
            "claims_allowed": self.claims_allowed,
            "claims_withheld": self.claims_withheld,
            "per_clip": self.per_clip,
            "note": (
                "Raw detector scores are NOT calibrated probabilities unless a "
                "calibration report says so. 'score' is an uncalibrated evidence index."
            ),
        }


# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def threshold_hashes(config: Any) -> Dict[str, str]:
    """Hash the thresholds that decide an incident.

    Two runs with different hashes are not comparable, so the hash goes in the
    report rather than in a footnote.
    """
    payload = json.dumps(
        {
            "verification": config.verification.__dict__,
            "collision": config.collision.__dict__,
            "scoring": config.scoring.__dict__,
            "temporal": config.temporal.__dict__,
        },
        sort_keys=True,
        default=str,
    )
    return {
        "threshold_set": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "confirm_score": str(config.verification.confirm_score),
        "temporal_confirm_score": str(config.verification.temporal_confirm_score),
        "verify_entry_score": str(config.verification.verify_entry_score),
        "suspicious_score": str(config.verification.suspicious_score),
    }


def _predict_clip(clip: ClipLabel, config: Any) -> ClipPrediction:
    """Run the real pipeline over one clip (detector injected for testability)."""
    from ai_engine.detection.yolo_detector import YoloDetector
    from ai_engine.pipeline import AccidentPipeline
    from ai_engine.video import VideoReader, VideoSourceError

    prediction = ClipPrediction(
        clip=Path(clip.clip).name,
        source_id=clip.source_id,
        label=clip.label,
        quality=clip.quality,
    )
    local = config
    local.evidence.enabled = False
    local.runtime.write_incidents_file = False
    local.emergency.enabled = False      # evaluation must never notify anybody
    local.emergency.demo_mode = False

    detector = None
    try:
        detector = YoloDetector(local.model)
        pipeline = AccidentPipeline(local, detector=detector, camera_id=clip.camera_id or None)
        reader = VideoReader(clip.clip, local.video)
        reader.open()
        fps = reader.fps or 15.0
        base = time.time()
        seen: List[Tuple[float, float]] = []
        for packet in reader.frames():
            result = pipeline.process_frame(packet.frame, packet.index, packet.timestamp)
            prediction.frames += 1
            prediction.peak_score = max(prediction.peak_score, float(result.accident_score))
            prediction.peak_physical = max(prediction.peak_physical,
                                           float(result.verification.physical_score))
            prediction.peak_temporal = max(prediction.peak_temporal,
                                           float(result.temporal.temporal_score))
            if result.incident:
                prediction.incident_count += 1
                prediction.incident_times.append(packet.index / max(fps, 1e-3))
            if len(seen) < 1:
                seen.append((packet.index / max(fps, 1e-3), packet.index / max(fps, 1e-3)))
        reader.close()
        pipeline.close()
        prediction.seconds = prediction.frames / max(fps, 1e-3)
    except (VideoSourceError, Exception) as exc:  # noqa: BLE001 - report, do not crash
        prediction.error = f"{type(exc).__name__}: {exc}"
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:  # noqa: BLE001
                pass
    return prediction


def _metrics_for(clips: Sequence[ClipLabel], preds: Sequence[ClipPrediction]) -> ClassificationMetrics:
    predictions = [(p.incident_count > 0, p.peak_score, (t if t else 0.0))
                   for p in preds for t in (p.incident_times[:1] or [0.0])]
    labels = [(c.label == "accident", c.positive_windows()) for c in clips]
    return confusion_from_events(predictions, labels)


def evaluate_manifest(
    manifest_path: str | Path,
    config: Any = None,
    seed: int = 20261002,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    out_path: Optional[str | Path] = None,
    predict=None,
) -> EvaluationReport:
    """Evaluate a manifest.  ``predict`` is injectable so tests need no weights."""
    from ai_engine.config import load_config

    if config is None:
        config = load_config()
    manifest = load_manifest(manifest_path)

    splits = group_split(manifest, ratios=ratios, seed=seed)
    try:
        assert_no_leakage(splits)
        leakage = "PASS - no source_id appears in more than one partition"
    except AssertionError as exc:
        leakage = f"FAIL - {exc}"

    run_predict = predict or (lambda clip, cfg: _predict_clip(clip, cfg))
    predictions: Dict[str, ClipPrediction] = {}
    for clip in manifest.clips:
        predictions[clip.clip] = run_predict(clip, config)

    event_metrics: Dict[str, Dict[str, Any]] = {}
    score_curve: Dict[str, List[Dict[str, Any]]] = {}
    auc: Dict[str, Optional[float]] = {}
    for name, split in splits.items():
        if not split.clips:
            continue
        preds = [predictions[c.clip] for c in split.clips]
        metrics = _metrics_for(split.clips, preds)
        event_metrics[name] = metrics.to_dict()
        positives = [p.peak_score for p in preds if p.label == "accident"]
        negatives = [p.peak_score for p in preds if p.label != "accident"]
        points = precision_recall_curve(positives, negatives)
        score_curve[name] = [p.to_dict() for p in points]
        auc[name] = pr_auc(points)

    # Threshold may be chosen on VALIDATION data ONLY.  If validation is empty we
    # refuse to suggest anything: silently falling back to the training split
    # would tune the threshold on data the detector was effectively fitted to
    # and produce a flattering, meaningless number.
    suggestion = _suggest_threshold(score_curve.get("validation") or [])

    sufficiency = summarise_split_sufficiency(splits)
    imbalance = class_imbalance(
        len([c for c in manifest.clips if c.label == "accident"]),
        len([c for c in manifest.clips if c.label != "accident"]),
    )
    blockers = manifest.quality_blockers()
    total_pos = sum(len(c.positive_windows()) for c in manifest.clips)

    allowed: List[str] = []
    withheld: List[str] = [
        "accuracy / false-alarm rate claims",
        "any statement about detector accuracy on real-world footage",
        "any claim that the thresholds are optimal",
        "any calibrated-probability interpretation of 'accident_score'",
    ]
    if not blockers:
        allowed.append("reporting event-level precision/recall/F1 with confidence intervals")
    allowed.append("reporting the pipeline runs end to end and is reproducible")

    provenance = {
        "model_weights": str(getattr(config.model, "weights", "")),
        "model_weights_sha256": _sha256_file(Path(config.model.weights))
        if Path(config.model.weights).is_file() else None,
        "imgsz": config.model.imgsz,
        "device": config.model.device,
        "thresholds": threshold_hashes(config),
        "data_fingerprint": group_fingerprint(manifest),
        "manifest_version": manifest.version,
        "split_seed": seed,
        "split_ratios": list(ratios),
        "positive_events": total_pos,
    }

    report = EvaluationReport(
        generated_at=now_iso(),
        manifest=manifest.describe(),
        splits={k: v.describe() for k, v in splits.items()},
        leakage_check=leakage,
        provenance=provenance,
        event_metrics=event_metrics,
        score_curve=score_curve,
        auc=auc,
        imbalance=imbalance,
        suggested_threshold=suggestion,
        sufficiency=sufficiency,
        claims_allowed=allowed,
        claims_withheld=withheld,
        per_clip=[predictions[c.clip].to_dict() for c in manifest.clips],
    )
    if out_path:
        atomic_write_json(out_path, report.to_dict())
    return report


def _suggest_threshold(points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Propose a threshold from validation data.  Never applied automatically."""
    usable = [p for p in points if p.get("f1") is not None]
    if not usable:
        return {
            "threshold": None,
            "reason": "the validation split is empty or has no positive example, so no "
                      "threshold can be selected without tuning on data the detector "
                      "was fitted to. This is reported as 'no suggestion', not guessed.",
            "source": "validation split only",
            "applied_automatically": False,
        }
    best = max(usable, key=lambda p: (p["f1"], -(p.get("false_positive_rate") or 0.0)))
    return {
        "threshold": best["threshold"],
        "source": "validation split only",
        "expected_precision": best["precision"],
        "expected_recall": best["recall"],
        "expected_f1": best["f1"],
        "expected_false_positive_rate": best["false_positive_rate"],
        "applied_automatically": False,
        "note": "A suggestion only. config.yaml is NOT modified. A false negative "
                "(a missed crash) is far more costly than a false alarm, so "
                "re-weight the objective if the trade-off looks wrong.",
    }


# --------------------------------------------------------------------------- #
def render_report(report: EvaluationReport) -> str:
    """Human-readable summary - the thing a reviewer actually reads."""
    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add(" SafeVision evaluation report")
    add("=" * 78)
    add(f"generated      : {report.generated_at}")
    m = report.manifest
    add(f"manifest       : {m['name']} v{m['version']}")
    add(f"clips          : {m['clips']} "
        f"({m['real_clips']} real, {m['synthetic_clips']} synthetic, "
        f"{m['verified_clips']} verified)")
    add(f"events         : {m['positive_events']} labelled accident event(s) "
        f"across {m['source_groups']} source group(s)")
    add(f"leakage check  : {report.leakage_check}")
    add("")
    add("-- provenance -------------------------------------------------")
    for key in ("model_weights_sha256", "data_fingerprint", "split_seed"):
        add(f"  {key:<22} {report.provenance.get(key)}")
    add(f"  {'threshold_set':<22} {report.provenance['thresholds']['threshold_set']}")
    add("")
    add("-- class imbalance --------------------------------------------")
    imb = report.imbalance
    add(f"  positives {imb['positives']} / negatives {imb['negatives']} "
        f"(majority {imb['majority_class_pct']}%)")
    add("")
    add("-- split sufficiency -----------------------------------------")
    suff = report.sufficiency
    add(f"  sufficient for a quality claim: {suff['sufficient_for_quality_claim']}")
    for reason in suff["reasons"]:
        add(f"    - {reason}")
    add("")
    add("-- event-level metrics ----------------------------------------")
    for name, metrics in report.event_metrics.items():
        cm = metrics["confusion_matrix"]
        add(f"  [{name}]  TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")
        add(f"      precision {_fmt_ci(metrics['precision'], metrics['precision_ci95'])}")
        add(f"      recall    {_fmt_ci(metrics['recall'], metrics['recall_ci95'])}")
        add(f"      f1        {_fmt(metrics['f1'])}")
        add(f"      FPR       {_fmt_ci(metrics['false_positive_rate'], metrics['false_positive_rate_ci95'])}")
        add(f"      FNR       {_fmt_ci(metrics['false_negative_rate'], metrics['false_negative_rate_ci95'])}")
    add("")
    add("-- PR-AUC ------------------------------------------------------")
    for name, value in report.auc.items():
        add(f"  {name:<12} {_fmt(value) if value is not None else 'undefined (no positives)'}")
    add("")
    add("-- threshold suggestion (validation only, NOT applied) ---------")
    s = report.suggested_threshold
    if s.get("threshold") is None:
        add(f"  none: {s.get('reason')}")
    else:
        add(f"  suggested threshold {s['threshold']}  "
            f"(P={_fmt(s.get('expected_precision'))} R={_fmt(s.get('expected_recall'))})")
    add("")
    add("-- CLAIMS ------------------------------------------------------")
    add("  allowed:")
    for item in report.claims_allowed:
        add(f"    + {item}")
    add("  withheld:")
    for item in report.claims_withheld:
        add(f"    - {item}")
    add("")
    add("-- why the withheld claims are withheld ------------------------")
    for blocker in m["quality_blockers"]:
        add(f"  * {blocker}")
    add("")
    add("  NOTE: the raw 'accident_score' is an uncalibrated evidence index, NOT a")
    add("        probability. Do not write '44% chance of an accident' without a")
    add("        calibration report on a real labelled set.")
    add("=" * 78)
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_ci(value: Optional[float], ci: Dict[str, Any]) -> str:
    if value is None:
        return "n/a"
    return (f"{value:.3f}  [95% CI {ci['ci95_low']:.3f}-{ci['ci95_high']:.3f}]  n={ci['n']}")