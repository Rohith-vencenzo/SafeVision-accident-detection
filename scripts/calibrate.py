"""Calibration helper: measure the score distribution on labelled videos.

The point of this script is honesty.  The shipped thresholds were chosen on
*scripted* scenarios (see ``scripts/validate_thresholds.py``) and are explicitly
**not** claimed to be optimal.  This script lets you measure the real
distributions on your own footage and see how far the current thresholds sit
from your data.

Workflow
--------
1. Collect labelled clips and put them in two folders::

       calibration/accident/*.mp4
       calibration/normal/*.mp4

2. Run the sweep::

       python scripts/calibrate.py --accident-dir calibration/accident \
                                   --normal-dir   calibration/normal

3. Read the report.  It prints, for both classes, the peak score, the peak
   *physical* evidence and the peak temporal score, plus the threshold that best
   separates them, and a suggested config diff.

4. Apply the suggestion (or edit ``config.yaml`` by hand), then re-run
   ``scripts/validate_thresholds.py`` to confirm the scripted traps still behave.

Nothing is written to ``config.yaml`` automatically unless you pass
``--write``, and even then it only appends the changed keys as a commented block
so you can review the diff.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.config import load_config  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.utils.jsonio import atomic_write_json  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402
from ai_engine.video import VideoReader, VideoSourceError  # noqa: E402

_LOGGER = get_logger("scripts.calibrate")

CLIPS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class ClipStats:
    """What one clip produced."""

    path: str
    frames: int = 0
    peak_score: float = 0.0
    peak_physical: float = 0.0
    peak_temporal: float = 0.0
    peak_collision: float = 0.0
    peak_signals: int = 0
    confirmed: bool = False
    incident_score: Optional[float] = None
    incident_severity: Optional[str] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "frames": self.frames,
            "peak_score": round(self.peak_score, 4),
            "peak_physical": round(self.peak_physical, 4),
            "peak_temporal": round(self.peak_temporal, 4),
            "peak_collision": round(self.peak_collision, 4),
            "peak_signals": int(self.peak_signals),
            "confirmed": bool(self.confirmed),
            "incident_score": self.incident_score,
            "incident_severity": self.incident_severity,
            "error": self.error,
        }


def find_clips(directory: Path) -> List[Path]:
    if not directory.is_dir():
        _LOGGER.warning("not a directory: %s", directory)
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in CLIPS)


def run_clip(path: Path, overrides: Sequence[str], max_frames: int, evidence_root: str) -> ClipStats:
    config = load_config(overrides=list(overrides))
    config.evidence.enabled = bool(evidence_root)
    if evidence_root:
        config.evidence.root = evidence_root
    config.runtime.write_incidents_file = False
    config.visualization.enabled = False
    config.location.default_camera_id = "CALIBRATION"

    pipeline = AccidentPipeline(config, camera_id="CALIBRATION", source_label=str(path))
    stats = ClipStats(path=str(path))
    try:
        reader = VideoReader(str(path), config.video)
        reader.open()
    except VideoSourceError as exc:
        pipeline.close()
        stats.error = str(exc)
        return stats

    try:
        for packet in reader.frames(max_frames=max_frames):
            result = pipeline.process_frame(packet.frame, packet.index, packet.timestamp)
            stats.frames += 1
            stats.peak_score = max(stats.peak_score, result.accident_score)
            stats.peak_physical = max(stats.peak_physical, result.verification.physical_score)
            stats.peak_temporal = max(stats.peak_temporal, result.temporal.temporal_score)
            if result.collision is not None:
                stats.peak_collision = max(stats.peak_collision, result.collision.score)
                stats.peak_signals = max(stats.peak_signals, result.collision.agreement)
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the sweep
        stats.error = f"{type(exc).__name__}: {exc}"
    finally:
        reader.close()

    stats.confirmed = bool(pipeline.incidents)
    if pipeline.incidents:
        incident = pipeline.incidents[0]
        stats.incident_score = round(float(incident.accident.score), 4)
        stats.incident_severity = str(incident.accident.severity)
    pipeline.close()
    return stats


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
        "max": round(ordered[-1], 4),
    }


def best_threshold(negatives: List[float], positives: List[float]) -> Dict[str, Any]:
    """Threshold that maximises (TPR - FPR) over the observed values."""
    if not negatives or not positives:
        return {"threshold": None, "note": "need both classes to suggest a threshold"}
    candidates = sorted({round(v, 3) for v in negatives + positives})
    best = {"threshold": None, "tpr": 0.0, "fpr": 1.0, "accuracy": 0.0}
    for threshold in candidates:
        tpr = sum(1 for v in positives if v >= threshold) / len(positives)
        fpr = sum(1 for v in negatives if v >= threshold) / len(negatives)
        accuracy = (tpr + (1.0 - fpr)) / 2.0
        if (tpr - fpr) > (best["tpr"] - best["fpr"]):
            best = {"threshold": threshold, "tpr": round(tpr, 3), "fpr": round(fpr, 3), "accuracy": round(accuracy, 3)}
    return best


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate SafeVision thresholds on labelled clips")
    parser.add_argument("--accident-dir", default="calibration/accident")
    parser.add_argument("--normal-dir", default="calibration/normal")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    parser.add_argument("--evidence-root", default="", help="write evidence (default: off, faster)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--json", default="", help="write the full report to this file")
    parser.add_argument("--write", default="", help="append the suggested thresholds to this YAML file")
    args = parser.parse_args(argv)

    configure_logging("INFO")

    accident_clips = find_clips(Path(args.accident_dir))
    normal_clips = find_clips(Path(args.normal_dir))
    if not accident_clips and not normal_clips:
        print("No clips found.")
        print(f"  looked in: {args.accident_dir} and {args.normal_dir}")
        print("  supported extensions: " + ", ".join(CLIPS))
        print("\nUntil you have labelled footage you can still validate the decision")
        print("logic on the scripted scenarios:  python scripts/validate_thresholds.py")
        return 1

    print(f"accident clips: {len(accident_clips)}   normal clips: {len(normal_clips)}\n")
    rows: List[Dict[str, Any]] = []
    for label, clips in (("accident", accident_clips), ("normal", normal_clips)):
        for clip in clips:
            stats = run_clip(clip, args.overrides, args.max_frames, args.evidence_root)
            rows.append({"label": label, **stats.to_dict()})
            flag = "ALARM" if stats.confirmed else "quiet"
            print(
                f"  {label:<9} {clip.name:<38} {flag:<6}"
                f" peak={stats.peak_score:.3f} physical={stats.peak_physical:.3f}"
                f" temporal={stats.peak_temporal:.3f}"
                + (f"  [{stats.error}]" if stats.error else "")
            )

    accident_scores = [r["peak_score"] for r in rows if r["label"] == "accident"]
    normal_scores = [r["peak_score"] for r in rows if r["label"] == "normal"]

    print("\n" + "=" * 70)
    print(" SCORE DISTRIBUTION (peak accident_score per clip)")
    print("=" * 70)
    print(f"  accident : {_stats(accident_scores)}")
    print(f"  normal   : {_stats(normal_scores)}")

    suggestion = best_threshold(normal_scores, accident_scores)
    print("\n suggested confirmation threshold (best TPR - FPR on this data):")
    print(f"  verification.confirm_score = {suggestion.get('threshold')}"
          f"  (TPR={suggestion.get('tpr')}, FPR={suggestion.get('fpr')})")
    print("\n NOTE: this is a *measurement on the clips you supplied*, not a")
    print(" validated operating point. Add more clips (especially hard negatives:")
    print(" parked cars, night footage, rain, close vehicles in adjacent lanes)")
    print(" before trusting it.")

    confirmed_normal = [r for r in rows if r["label"] == "normal" and r["confirmed"]]
    if confirmed_normal:
        print(f"\n !! {len(confirmed_normal)} normal clip(s) produced a FALSE ALARM:")
        for row in confirmed_normal:
            print(f"    {row['path']}  peak={row['peak_score']:.3f}")

    report = {
        "accident_dir": args.accident_dir,
        "normal_dir": args.normal_dir,
        "rows": rows,
        "accident_scores": _stats(accident_scores),
        "normal_scores": _stats(normal_scores),
        "suggestion": suggestion,
        "false_alarms_on_normal": len(confirmed_normal),
    }
    if args.json:
        path = atomic_write_json(args.json, report)
        print(f"\nreport written to {path}")

    if args.write and suggestion.get("threshold") is not None:
        try:
            with open(args.write, "a", encoding="utf-8") as handle:
                handle.write(
                    "\n# --- added by scripts/calibrate.py ---\n"
                    f"verification:\n  confirm_score: {suggestion['threshold']}\n"
                )
            print(f"\nsuggestion appended to {args.write} (review it!)")
        except OSError as exc:
            _LOGGER.error("could not write %s: %s", args.write, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
