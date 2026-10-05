"""Scenario driven regression / threshold validation for the whole pipeline.

Runs every scenario from :mod:`ai_engine.detection.scenarios` through the real
pipeline (with the scripted detector) and prints a table of the outcome.  Use it

* after changing any threshold or weight, and
* as the honest sanity check that the false-alarm traps stay quiet.

Usage::

    python scripts/validate_thresholds.py
    python scripts/validate_thresholds.py --fps 15 --frames 70 --json report.json
    python scripts/validate_thresholds.py --only head_on_crash,parked_cars

This measures the *decision logic* on synthetic inputs.  It is not an accuracy
benchmark - see ``docs/scoring_and_tuning.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ai_engine.config import load_config  # noqa: E402
from ai_engine.detection.scenarios import SCENARIOS, expectation_for, list_scenarios  # noqa: E402
from ai_engine.detection.scripted import ScriptedDetector  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.utils.jsonio import atomic_write_json  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger("scripts.validate")

_FRAME_W, _FRAME_H = 640, 360


def run_scenario(
    name: str,
    fps: float = 15.0,
    frames: int = 70,
    config_overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one scenario and return a summary row."""
    config = load_config(overrides=config_overrides)
    config.evidence.enabled = False          # no disk writes for a dry validation run
    config.visualization.enabled = False     # skip drawing (nothing to see)
    config.runtime.write_incidents_file = False

    pipeline = AccidentPipeline(config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-VALIDATION")
    frame = np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)

    peak_score = 0.0
    peak_state = "NORMAL"
    peak_event = 0.0
    states: List[str] = []
    for index in range(int(frames)):
        result = pipeline.process_frame(frame, index, index / max(1e-3, fps))
        states.append(result.state)
        if result.accident_score > peak_score:
            peak_score = result.accident_score
            peak_state = result.state
        peak_event = max(peak_event, float(result.verification.event_score))
    pipeline.close()

    confirmed = len(pipeline.incidents) > 0
    expected = expectation_for(name)
    passed = confirmed == expected
    incident = pipeline.incidents[0] if pipeline.incidents else None

    row = {
        "scenario": name,
        "expected_alarm": expected,
        "confirmed": confirmed,
        "pass": passed,
        "peak_score": round(peak_score, 4),
        "peak_event_score": round(peak_event, 4),
        "peak_state": peak_state,
        "final_state": states[-1] if states else "NORMAL",
        "false_alarms": pipeline.verification.false_alarm_count,
        "severity": str(incident.accident.severity) if incident else None,
        "severity_score": round(incident.accident.severity_score, 4) if incident else None,
        "vehicles_involved": incident.objects.vehicles_involved if incident else 0,
        "agreement_signals": incident.evidence.agreement_signals if incident else 0,
        "reasons": incident.evidence.reasons if incident else [],
    }
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SafeVision thresholds on synthetic scenarios")
    parser.add_argument("--fps", type=float, default=15.0, help="simulated processing FPS")
    parser.add_argument("--frames", type=int, default=70, help="frames per scenario")
    parser.add_argument("--only", default="", help="comma separated scenario names")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="config override, e.g. --set verification.confirm_score=0.6")
    parser.add_argument("--json", default="", help="write the full result table to this JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging("WARNING")

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list_scenarios()
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s): {unknown}")

    rows = [
        run_scenario(name, fps=args.fps, frames=args.frames, config_overrides=args.overrides, verbose=args.verbose)
        for name in names
    ]

    header = f"{'scenario':<20}{'expect':>7}{'got':>7}{'peak':>8}{'event':>8}{'severity':>10}{'veh':>5}{'sig':>5}  result"
    print(header)
    print("-" * len(header))
    for row in rows:
        verdict = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['scenario']:<20}"
            f"{('ALARM' if row['expected_alarm'] else 'quiet'):>7}"
            f"{('ALARM' if row['confirmed'] else 'quiet'):>7}"
            f"{row['peak_score']:>8.3f}"
            f"{row['peak_event_score']:>8.3f}"
            f"{str(row['severity'] or '-'):>10}"
            f"{row['vehicles_involved']:>5}"
            f"{row['agreement_signals']:>5}"
            f"  {verdict}"
        )

    passed = sum(1 for row in rows if row["pass"])
    print("-" * len(header))
    print(f"{passed}/{len(rows)} scenarios behave as configured")
    if passed != len(rows):
        print("\nFailing scenarios:")
        for row in rows:
            if not row["pass"]:
                print(
                    f"  - {row['scenario']}: expected {'ALARM' if row['expected_alarm'] else 'quiet'}, "
                    f"got {'ALARM' if row['confirmed'] else 'quiet'} "
                    f"(peak {row['peak_score']:.3f}, peak event {row['peak_event_score']:.3f}, "
                    f"final state {row['final_state']})"
                )

    if args.json:
        path = atomic_write_json(args.json, {"fps": args.fps, "frames": args.frames, "rows": rows})
        print(f"\nwrote {path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
