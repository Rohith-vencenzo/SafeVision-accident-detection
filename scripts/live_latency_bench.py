"""Live-stream latency measurement.

Two modes, and the distinction is the whole point:

``--live-real``
    A genuinely live source (``--source 0`` for a webcam, ``rtsp://...``).
    Measures capture wait, decode and per-stage latency on a real stream.
    **BLOCKED on this machine**: no webcam and no reachable RTSP camera exist
    here, so this mode has never been run. The report says so.

``--live-replay`` (default)
    Replays a recorded clip at a *fixed wall-clock pace* through the live code
    path, so frames arrive like a camera would deliver them and the reader's
    bounded queue / drop behaviour is exercised.

    This measures the **pipeline's** ability to keep up and the
    incident-to-notification-enqueue latency. It is **NOT** a camera test and
    **NOT** a live response time: there is no capture, no network, no encoder
    and no real source clock. A result here says "the engine can process N fps
    of 640x360 on CPU", not "the system will respond in N seconds".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

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


class PacedReplay:
    """Feeds recorded frames at a fixed wall-clock rate, like a live source.

    When the pipeline is slower than the target rate the frames are *dropped*
    rather than queued, which is what a bounded live pipeline must do: an
    unbounded queue turns a slow detector into unbounded latency and an
    ever-growing memory.
    """

    def __init__(self, clip: Path, target_fps: float, video_config):
        self.reader = VideoReader(str(clip), video_config)
        self.target_fps = max(0.1, float(target_fps))
        self.is_live = True          # deliberately: it exercises the live path
        self.kind = "replay"
        self.fps = self.target_fps
        self.total_frames = 0

    def open(self) -> "PacedReplay":
        self.reader.open()
        return self

    def packets(self, max_frames: int = 0):
        limit = max_frames or 10 ** 9
        interval = 1.0 / self.target_fps
        start = time.perf_counter()
        emitted = 0
        dropped = 0
        for packet in self.reader.frames(max_frames=limit):
            due = start + emitted * interval
            now = time.perf_counter()
            if now < due - interval * 0.5:
                # more than half a frame late behind schedule -> behind budget
                dropped += 1
                continue
            if now > due:
                remaining = due - now
                if remaining > 0:
                    time.sleep(remaining)
            self.total_frames += 1
            emitted += 1
            yield packet
        self.dropped = dropped

    def close(self) -> None:
        self.reader.close()

    def stats(self) -> dict:
        return {"target_fps": self.target_fps, "emitted": self.total_frames,
                "dropped_behind_budget": getattr(self, "dropped", 0)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision live-stream latency")
    parser.add_argument("--clip", default=str(ROOT / "test_videos" / "accident_demo.mp4"))
    parser.add_argument("--source", default="0", help="for --live-real")
    parser.add_argument("--live-real", action="store_true",
                        help="measure a genuine live source (webcam / RTSP)")
    parser.add_argument("--live-replay", action="store_true",
                        help="paced replay through the live code path (default)")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--camera-id", default="CAM-001")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    config = load_config()
    config.evidence.enabled = False
    config.runtime.write_incidents_file = False
    config.emergency.enabled = False       # never notify during a measurement
    config.emergency.demo_mode = False
    config.visualization.enabled = False

    mode = "LIVE_REAL" if args.live_real else "LIVE_REPLAY_PACED"
    print("=" * 74)
    print(f" live latency measurement - mode = {mode}")
    print("=" * 74)

    if args.live_real:
        source = args.source
    else:
        source = str(Path(args.clip))

    ledger = LatencyLedger(mode="LIVE")
    detector = YoloDetector(config.model)
    print("loading detector ...", flush=True)
    detector.load()

    if args.live_real:
        stream = VideoReader(source, config.video)
        stream.open()
        ledger.source_kind = str(getattr(stream, "kind", "UNKNOWN"))
        target_fps = float(stream.fps or args.target_fps)
    else:
        stream = PacedReplay(Path(args.clip), args.target_fps, config.video).open()
        ledger.source_kind = "replay_paced"
        target_fps = args.target_fps

    pipeline = AccidentPipeline(config, detector=detector, camera_id=args.camera_id,
                                ledger=ledger, source_kind="LIVE")

    frames = 0
    incidents: List[str] = []
    latencies: List[float] = []
    started = time.perf_counter()
    for packet in stream.packets(max_frames=args.max_frames or config.runtime.max_frames or 0):
        result = pipeline.process_frame(packet.frame, packet.index, packet.timestamp)
        frames += 1
        latencies.append((time.perf_counter() - started) / frames)
        if result.incident:
            incidents.append(result.incident["incident_id"])
    wall = time.perf_counter() - started

    stats = stream.stats() if hasattr(stream, "stats") else {}
    stream.close()
    pipeline.close()
    detector.close()

    achieved = frames / max(wall, 1e-6)
    report = ledger.report()
    report["live"] = {
        "mode": mode,
        "target_fps": target_fps,
        "achieved_fps": round(achieved, 2),
        "keeps_up": achieved >= target_fps * 0.95,
        "frames": frames,
        "wall_seconds": round(wall, 3),
        "stream_stats": stats,
        "incidents": incidents,
        "incident_count": len(incidents),
    }
    if mode == "LIVE_REPLAY_PACED":
        report["live"]["honesty"] = (
            "SIMULATED LIVE PACING over a recorded clip. No capture, no network, "
            "no encoder, no source clock. This measures how fast the ENGINE can "
            "process 640x360 frames on CPU under a fixed arrival rate. It is NOT "
            "a live response time and must not be quoted as one."
        )
    else:
        report["live"]["honesty"] = (
            "Measured against a real live source. Still does not bound the "
            "provider's or the network's contribution to a call being answered."
        )

    print()
    print(ledger.text())
    print()
    live = report["live"]
    print(f"target {live['target_fps']} fps -> achieved {live['achieved_fps']} fps "
          f"({'keeps up' if live['keeps_up'] else 'DOES NOT keep up'})")
    if stats:
        print(f"frames dropped behind budget: {stats.get('dropped_behind_budget')}")
    print(f"incidents: {live['incident_count']} {incidents}")
    print()
    print("MEANING:")
    print(" ", live["honesty"])

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())