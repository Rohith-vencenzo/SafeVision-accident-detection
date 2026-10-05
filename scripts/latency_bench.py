"""Latency benchmark: measure, then report distributions - never a single number.

    python scripts\\latency_bench.py                      # recorded file (offline)
    python scripts\\latency_bench.py --source 0           # live webcam
    python scripts\\latency_bench.py --max-frames 200
    python scripts\\latency_bench.py --json reports/latency.json

Reports stage p50/p95/max with the sample count, an explicit
RECORDED / LIVE badge, and the throughput comparison against the source's own
frame rate.  That comparison is the honest way to say "this is offline
throughput" rather than a live response time.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from ai_engine.config import load_config  # noqa: E402
from ai_engine.detection.yolo_detector import YoloDetector  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.telemetry.latency import LatencyLedger  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging  # noqa: E402
from ai_engine.video import VideoReader  # noqa: E402


def hardware() -> dict:
    import os

    return {
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision latency benchmark")
    parser.add_argument("--source", default=str(ROOT / "test_videos" / "accident_demo.mp4"))
    parser.add_argument("--camera-id", default="CAM-001")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=0, help="downscale before inference")
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--json", default="", help="write the report here")
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    config = load_config()
    config.evidence.enabled = False
    config.runtime.write_incidents_file = False
    config.emergency.enabled = False      # a benchmark must never notify anybody
    config.emergency.demo_mode = False
    config.visualization.enabled = False
    if args.width:
        config.video.target_width = args.width
    if args.stride:
        config.video.frame_stride = args.stride

    source = args.source
    if source not in ("0", "-1") and not source.isdigit():
        source = str((ROOT / source).resolve()) if not Path(source).is_absolute() else source

    ledger = LatencyLedger()
    detector = YoloDetector(config.model)
    print("loading detector ...", flush=True)
    detector.load()

    pipeline = AccidentPipeline(config, detector=detector, camera_id=args.camera_id,
                                ledger=ledger, source_kind=kind)
    reader = VideoReader(source, config.video)
    reader.open()
    fps = float(reader.fps or 0.0)
    is_live = bool(reader.is_live)
    kind = str(getattr(reader, "kind", "UNKNOWN"))
    # the reader is the authority on whether this is a live stream
    ledger.source_kind = kind
    ledger.mode = "LIVE" if is_live else "RECORDED"

    print(f"source={source}")
    print(f"kind={kind}  live={is_live}  fps={fps:.2f}  "
          f"model={config.model.weights} imgsz={config.model.imgsz}")
    print(f"frame_stride={config.video.frame_stride} target_width={config.video.target_width}")
    print("-" * 78)

    frames = 0
    incidents: list = []
    started = time.perf_counter()
    last_report = started
    for packet in reader.frames(max_frames=args.max_frames or config.runtime.max_frames or 0):
        result = pipeline.process_frame(packet.frame, packet.index, packet.timestamp)
        frames += 1
        if result.incident:
            incidents.append(result.incident["incident_id"])
        now = time.perf_counter()
        if now - last_report >= 5.0:
            done = now - started
            print(f"  {frames} frames in {done:6.1f}s  "
                  f"{frames / max(done, 1e-6):5.2f} fps analysed  "
                  f"incidents={len(incidents)}")
            last_report = now
    wall = time.perf_counter() - started

    reader.close()
    pipeline.close()
    detector.close()

    analysed_fps = frames / max(wall, 1e-6)
    footage_seconds = frames / fps if fps > 0 else 0.0
    report = ledger.report()
    report["throughput"] = {
        "frames": frames,
        "wall_seconds": round(wall, 3),
        "analysed_fps": round(analysed_fps, 2),
        "source_fps": round(fps, 2),
        "footage_seconds": round(footage_seconds, 2),
        "slower_than_realtime_by": (
            None if analysed_fps >= fps else round(fps / max(analysed_fps, 1e-6), 2)
        ),
        "processing_mode": "LIVE" if is_live else "RECORDED",
    }
    report["hardware"] = hardware()
    report["incidents"] = incidents
    report["video_config"] = {
        "weights": config.model.weights,
        "imgsz": config.model.imgsz,
        "device": config.model.device,
        "frame_stride": config.video.frame_stride,
        "target_width": config.video.target_width,
        "process_fps": config.video.process_fps,
    }
    report["interpretation"] = _interpret(report)

    print()
    print(ledger.text())
    print()
    tp = report["throughput"]
    print(f"throughput: {tp['analysed_fps']} fps analysed vs {tp['source_fps']} fps source "
          f"-> {tp['slower_than_realtime_by']}x slower than real time"
          if tp["slower_than_realtime_by"] else
          f"throughput: {tp['analysed_fps']} fps analysed vs {tp['source_fps']} fps source "
          f"-> keeps up with real time")
    print(f"mode      : {tp['processing_mode']}")
    print(f"meaning   : {report['interpretation']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {out}")
    return 0


def _interpret(report: dict) -> str:
    tp = report["throughput"]
    if tp["processing_mode"] == "LIVE":
        return ("LIVE source: these stage times describe a real stream. A 30-60s "
                "incident-to-notification budget would still have to be demonstrated "
                "end to end, including the provider round trip.")
    factor = tp.get("slower_than_realtime_by")
    if factor:
        return (f"RECORDED file analysed at {factor}x slower than real time. This is "
                "OFFLINE THROUGHPUT, not a live response time, and must never be "
                "quoted as one.")
    return ("RECORDED file processed at or above real time. Still offline throughput: "
            "the stream was already on disk, so this says nothing about reacting to a "
            "live camera.")


if __name__ == "__main__":
    raise SystemExit(main())