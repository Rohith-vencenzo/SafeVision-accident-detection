"""Manual test script - watch the decision process frame by frame (Phase 23).

Unlike ``main.py`` this prints a readable trace of *why* the system decided what
it decided: every state change, the per-component scores around the event, and
the final incident JSON.  It is the script to use when you want to understand
or debug a specific clip.

    python scripts/manual_test.py test_videos/accident.mp4
    python scripts/manual_test.py clip.mp4 --camera-id CAM-002 --set collision.min_signals=3
    python scripts/manual_test.py --list-scenarios          # no video needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.config import load_config  # noqa: E402
from ai_engine.detection.scripted import ScriptedDetector  # noqa: E402
from ai_engine.detection.scenarios import SCENARIOS, list_scenarios  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.schemas import validate_incident  # noqa: E402
from ai_engine.utils.jsonio import dumps  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging  # noqa: E402
from ai_engine.video import VideoReader, VideoSourceError  # noqa: E402

SEPARATOR = "=" * 100


def trace_video(args: argparse.Namespace) -> int:
    config = load_config(overrides=args.overrides)
    if args.camera_id:
        config.location.default_camera_id = args.camera_id
    if args.max_frames:
        config.runtime.max_frames = args.max_frames
    if args.width:
        config.video.target_width = args.width
    if not args.evidence:
        config.evidence.enabled = False

    print(SEPARATOR)
    print(f" SafeVision manual test")
    print(f" source   : {args.source}")
    print(f" model    : {config.model.weights}  conf={config.model.conf}  imgsz={config.model.imgsz}")
    print(f" camera   : {config.location.default_camera_id or '(unregistered)'}")
    print(f" confirm  : score>={config.verification.confirm_score}  "
          f"temporal>={config.verification.temporal_confirm_score}  "
          f"signals>={config.verification.min_signals}  vehicles>={config.verification.min_vehicles}")
    print(SEPARATOR)

    try:
        reader = VideoReader(args.source, config.video)
        reader.open()
    except VideoSourceError as exc:
        print(f"cannot open the source: {exc}")
        return 2

    pipeline = AccidentPipeline(config, camera_id=config.location.default_camera_id, source_label=args.source)
    print(f"\n{'frame':>6} {'state':<19}{'score':>7}{'phys':>7}{'temporal':>9}  "
          f"{'collision':>9}{'motion':>7}{'traj':>6}  tracks  notes")
    print("-" * 100)

    incidents = []
    try:
        for packet in reader.frames(max_frames=config.runtime.max_frames):
            result = pipeline.process_frame(packet.frame, packet.index, packet.timestamp)
            verification = result.verification

            interesting = (
                verification.changed
                or result.frame_index % max(1, config.demo.print_every) == 0
            )
            if not interesting:
                continue
            components = result.component_scores
            note = ""
            if verification.changed:
                note = ">> " + verification.transitions[-1].reason
            elif verification.blocking_reasons:
                note = "holding: " + verification.blocking_reasons[0]
            print(
                f"{result.frame_index:>6} {result.state:<19}"
                f"{result.accident_score:>7.3f}{verification.physical_score:>7.3f}"
                f"{result.temporal.temporal_score:>9.3f}  "
                f"{components.get('collision', 0.0):>9.3f}"
                f"{components.get('motion', 0.0):>7.3f}"
                f"{components.get('trajectory', 0.0):>6.3f}  "
                f"{len(result.tracks):>6}  {note[:42]}"
            )
            if result.incident:
                incidents.append(result.incident)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        reader.close()

    print("\n" + SEPARATOR)
    print(" RESULT")
    print(SEPARATOR)
    print(f" frames processed : {pipeline._frame_index}")
    print(f" confirmed        : {pipeline.verification.confirmed_count}")
    print(f" false alarms     : {pipeline.verification.false_alarm_count}")
    print(f" final state      : {pipeline.state}")
    print(f" video stats      : {reader.stats()}")

    for incident in incidents:
        problems = validate_incident(incident)
        print(f"\n incident {incident['incident_id']}  "
              f"(schema {'OK' if not problems else problems})")
        print(f"   score    : {incident['accident']['score']} "
              f"({incident['accident']['confidence_percent']}%)")
        print(f"   severity : {incident['accident']['severity']} "
              f"(score {incident['accident']['severity_score']})")
        print(f"   location : {incident['location']['name']} "
              f"({incident['location']['latitude']}, {incident['location']['longitude']})")
        print(f"   objects  : {incident['objects']['vehicles_involved']} vehicles, "
              f"{incident['objects']['persons_detected']} persons")
        print(f"   evidence : {dumps(incident['evidence']['reasons'])}")
        print(f"   media    : {incident['media']['evidence_dir']}")
        if args.full:
            print()
            print(dumps(incident))
    pipeline.close()
    return 0


def trace_scenario(name: str, overrides: List[str], frames: int) -> int:
    """Run a scripted scenario - no video, no YOLO weights required."""
    import numpy as np

    config = load_config(overrides=overrides)
    config.evidence.enabled = False
    config.visualization.enabled = False
    config.runtime.write_incidents_file = False
    pipeline = AccidentPipeline(config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-001")

    print(SEPARATOR)
    print(f" SafeVision manual test - scripted scenario '{name}'")
    print(SEPARATOR)
    print(f"\n{'frame':>6} {'state':<19}{'score':>7}{'phys':>7}{'temporal':>9}  notes")
    print("-" * 80)

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for index in range(frames):
        result = pipeline.process_frame(frame, index, index / 15.0)
        if result.verification.changed or index % 10 == 0:
            note = ">> " + result.verification.transitions[-1].reason if result.verification.changed else ""
            print(
                f"{index:>6} {result.state:<19}{result.accident_score:>7.3f}"
                f"{result.verification.physical_score:>7.3f}"
                f"{result.verification.temporal_score:>9.3f}  {note[:40]}"
            )
    incident = pipeline.get_incident_result()
    print(f"\n confirmed: {bool(incident)}   false alarms: {pipeline.verification.false_alarm_count}")
    if incident:
        print(dumps(incident["accident"]))
    pipeline.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision manual test / trace")
    parser.add_argument("source", nargs="?", default="", help="video file | rtsp://... | (omit with --scenario)")
    parser.add_argument("--scenario", default="", choices=list_scenarios(), help="scripted scenario instead of a video")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--evidence", action="store_true", help="write evidence files")
    parser.add_argument("--full", action="store_true", help="print the complete incident JSON")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    configure_logging("WARNING")

    if args.list_scenarios:
        for name in list_scenarios():
            print(f"  {name}")
        return 0
    if args.scenario:
        return trace_scenario(args.scenario, args.overrides, 70)
    if not args.source:
        parser.error("give a video source, or --scenario NAME, or --list-scenarios")
    return trace_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
