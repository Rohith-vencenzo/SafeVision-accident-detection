"""Generate the synthetic demo clips used by ``python main.py --demo``.

IMPORTANT - read this before you demo with these files
------------------------------------------------------
These clips are **synthetic drawings**, not real road footage.  They exist so
that the demo mode, the README walkthrough and the automated end-to-end test all
have deterministic input on a machine with no sample video.  YOLO is very likely
to detect *nothing* in them, because they are not photographs of vehicles.

What they are good for:

* proving the video I/O, tracking, analysis, state machine and JSON output work
  end to end (``scripts/make_test_video.py --verify`` runs the scripted detector
  on them and prints the state trace)
* demoing the *interface* if you have no footage of your own

What they are NOT good for:

* claiming any detection accuracy - use real accident footage for that

    python scripts/make_test_video.py                  # write all three clips
    python scripts/make_test_video.py --verify         # + run the pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger("scripts.make_test_video")

WIDTH, HEIGHT = 640, 360
FPS = 15.0


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def background(frame_index: int, total: int) -> np.ndarray:
    """A simple scrolling road: dark asphalt, lane dashes, verges."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = (48, 48, 52)                                  # asphalt
    frame[: int(HEIGHT * 0.32), :] = (60, 90, 70)                # sky / far verge
    frame[int(HEIGHT * 0.78) :, :] = (55, 85, 65)                # near verge
    offset = (frame_index * 3) % 60
    for x in range(-60 + offset, WIDTH, 60):                      # centre dashes
        cv2 = __import__("cv2")
        cv2.rectangle(frame, (x, HEIGHT // 2 - 3), (x + 30, HEIGHT // 2 + 3), (200, 200, 200), -1)
    for y in (int(HEIGHT * 0.38), int(HEIGHT * 0.68)):           # lane edges
        cv2 = __import__("cv2")
        cv2.line(frame, (0, y), (WIDTH, y), (160, 160, 160), 2)
    return frame


def draw_vehicle(
    frame: np.ndarray,
    x: float,
    y: float,
    width: float = 40.0,
    height: float = 24.0,
    colour: Tuple[int, int, int] = (60, 90, 200),
) -> Tuple[int, int, int, int]:
    """Draw a simple car-like box with a windscreen; returns its bounds."""
    import cv2

    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 2)
    cv2.rectangle(frame, (x1 + int(width * 0.28), y1 + 3), (x2 - int(width * 0.28), y2 - 5), (180, 190, 200), -1)
    cv2.circle(frame, (x1 + 7, y2), 3, (30, 30, 30), -1)
    cv2.circle(frame, (x2 - 7, y2), 3, (30, 30, 30), -1)
    return (x1, y1, x2, y2)


def write_clip(path: Path, frames: List[np.ndarray]) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    size_kb = path.stat().st_size / 1024
    _LOGGER.info("wrote %s (%d frames, %.0f KB)", path, len(frames), size_kb)


# --------------------------------------------------------------------------- #
# the three clips
# --------------------------------------------------------------------------- #
def clip_normal_traffic(frames: int = 90) -> List[np.ndarray]:
    """Steady traffic, nothing unusual.  The engine must stay NORMAL."""
    import cv2

    out = []
    for index in range(frames):
        frame = background(index, frames)
        draw_vehicle(frame, 60 + 6 * index, 120, colour=(60, 90, 200))
        draw_vehicle(frame, 520 - 5 * index, 210, colour=(80, 120, 90))
        draw_vehicle(frame, 300 - 2 * index, 280, width=56, height=30, colour=(120, 90, 60))
        cv2.putText(frame, "NORMAL TRAFFIC", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        out.append(frame)
    return out


def clip_accident(frames: int = 90) -> List[np.ndarray]:
    """Two vehicles approach, collide at frame 40 and stop."""
    import cv2

    out = []
    for index in range(frames):
        frame = background(index, frames)
        a = 40 + 5.5 * min(index, 40)
        b = 560 - 5.5 * min(index, 40)
        if index > 40:                      # both come to rest
            a = min(560 - 5.5 * 40 + 2 * min(20, index - 40), 380)
            b = max(560 - 5.5 * 40 - 2 * min(20, index - 40), 260)
        box_a = draw_vehicle(frame, a, 165, colour=(60, 90, 200))
        box_b = draw_vehicle(frame, b, 170, colour=(70, 70, 190))
        if 38 <= index <= 48:              # impact marker
            mid = int((a + b) / 2 + 20)
            cv2.circle(frame, (mid, 178), 18 + (index - 38), (60, 60, 240), 2)
        cv2.putText(frame, "ACCIDENT CLIP", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        cv2.putText(frame, f"frame {index}", (WIDTH - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        out.append(frame)
    return out


def clip_suspicious(frames: int = 90) -> List[np.ndarray]:
    """Hard braking with no other vehicle - must stay NORMAL (false-alarm trap)."""
    import cv2

    out = []
    for index in range(frames):
        frame = background(index, frames)
        x = 80 + 7.0 * min(index, 30)          # approaches, then brakes to a stop
        draw_vehicle(frame, x, 150, colour=(90, 90, 210))
        draw_vehicle(frame, 540, 260, width=60, height=32, colour=(100, 130, 90))
        if 26 <= index <= 40:
            cv2.putText(frame, "HARD BRAKE", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2)
        cv2.putText(frame, "SUSPICIOUS / FALSE-ALARM CLIP", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
        out.append(frame)
    return out


# --------------------------------------------------------------------------- #
def verify(paths: List[Path]) -> None:
    """Run the scripted scenarios through the real pipeline for comparison."""
    from ai_engine.config import load_config
    from ai_engine.detection.scenarios import SCENARIOS
    from ai_engine.detection.scripted import ScriptedDetector
    from ai_engine.pipeline import AccidentPipeline

    print("\nNote: the clips above are synthetic drawings, so a real YOLO model")
    print("will normally find nothing in them.  The verification below therefore")
    print("uses the scripted detector on the *scenario* equivalents, which is how")
    print("the decision logic is regression-tested.\n")

    config = load_config()
    config.evidence.enabled = False
    config.visualization.enabled = False
    config.runtime.write_incidents_file = False

    for name in ("head_on_crash", "parked_cars", "hard_brake"):
        pipeline = AccidentPipeline(config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-001")
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for index in range(70):
            pipeline.process_frame(frame, index, index / FPS)
        confirmed = len(pipeline.incidents)
        print(
            f"  scenario {name:<20} incident={'yes' if confirmed else 'no ':<4}"
            f" peak physical evidence={pipeline.verification.event_peak:.3f}"
        )
        pipeline.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic demo clips")
    parser.add_argument("--out", default="test_videos", help="output directory")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--verify", action="store_true", help="also run the pipeline regression check")
    args = parser.parse_args(argv)

    configure_logging("INFO")
    out = Path(args.out)

    write_clip(out / "normal_traffic.mp4", clip_normal_traffic(args.frames))
    write_clip(out / "accident.mp4", clip_accident(args.frames))
    write_clip(out / "suspicious.mp4", clip_suspicious(args.frames))
    print(f"\nwrote three synthetic clips to {out}/")
    print("They are drawings, not road footage - see the module docstring.")
    if args.verify:
        verify([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
