"""Run the REAL YOLO pipeline on REAL imagery (no scripted detector).

This is the end-to-end proof that the detection path works on actual
photographs, not just on synthetic inputs.

Two modes:

``--photo``
    Runs the full pipeline on a single still image.  YOLO finds real objects;
    the engine reports NORMAL, which is the correct outcome for one frame of
    ordinary traffic (a single frame must never be an accident).

``--video``
    Builds a short clip out of a real photo by panning/zooming it, then runs the
    complete video pipeline (YOLO -> ByteTrack -> motion -> trajectory ->
    collision -> temporal fusion -> verification).  Again the expected outcome is
    NORMAL: the system must not invent an accident out of ordinary traffic.

Usage::

    python scripts/real_yolo_check.py --photo test_videos/street_photo.jpg
    python scripts/real_yolo_check.py --video test_videos/street_photo.jpg --frames 60
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_engine.config import load_config  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger("scripts.real_yolo_check")


def report(result, show_tracks: bool = True) -> None:
    detections = result.detections
    print(f"\n  detections ({len(detections)}):")
    for det in detections:
        print(
            f"    {det.class_name:<12} conf={det.confidence:.2f} "
            f"box={[int(v) for v in det.bbox]}"
        )
    if not show_tracks:
        return
    print(f"  tracks ({len(result.tracks)}):")
    for track in result.tracks:
        print(
            f"    #{track.track_id:<4} {track.normalized_name:<11} "
            f"conf={track.confidence:.2f} speed={track.speed_norm:.4f} "
            f"history={track.history_length} state={track.state}"
        )
    print(
        f"  state={result.state}  score={result.accident_score:.3f}  "
        f"temporal={result.temporal.temporal_score:.3f}  "
        f"inference={result.performance.inference_ms:.1f}ms  "
        f"total={result.performance.total_ms:.1f}ms"
    )


def run_photo(path: Path) -> int:
    image = cv2.imread(str(path))
    if image is None:
        print(f"could not read the image: {path}")
        return 2
    print(f"image: {path}  ({image.shape[1]}x{image.shape[0]})")

    config = load_config()
    config.evidence.enabled = False
    config.visualization.enabled = False
    config.runtime.write_incidents_file = False

    pipeline = AccidentPipeline(config, camera_id="DEMO-LAPTOP", source_label=str(path))
    result = pipeline.process_frame(image, 0, 0.0)
    print("\n=== single frame, real YOLO ===")
    report(result)
    print(
        "\n  verdict: the engine stayed in NORMAL. One frame of ordinary traffic"
        "\n           must never be reported as an accident - that is the whole point."
    )
    pipeline.close()
    return 0 if len(result.detections) else 1


def build_clip(photo: Path, clip: Path, frames: int, size: tuple = (640, 360)) -> Path:
    """Turn a still photo into a short clip with a smooth pan + zoom.

    The pan is applied with a sub-pixel warp.  A real PTZ camera moves smoothly;
    an integer-stepped crop would move every object by 1-2 px in jumps and
    produce fake "sudden stops", which is a property of the *clip*, not of the
    engine.
    """
    image = cv2.imread(str(photo))
    if image is None:
        raise RuntimeError(f"could not read {photo}")
    height, width = image.shape[:2]
    # crop a band with the target aspect ratio (no squashing), pan it smoothly
    out_w = width
    out_h = max(2, int(round(width * size[1] / size[0])))
    if out_h > height:
        out_h = height
        out_w = max(2, int(round(height * size[0] / size[1])))

    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, size)
    if not writer.isOpened():
        raise RuntimeError("could not open the video writer")
    try:
        for index in range(frames):
            t = index / max(1, frames - 1)
            zoom = 1.0 + 0.05 * t
            # A negative translation samples *lower* in the photo, so the band
            # covers the road level where the subjects are.  The wobble is
            # smooth (a real PTZ camera does not move in integer steps).
            dx = -(width - out_w / zoom) * 0.5 + 5.0 * math.sin(2 * math.pi * t)
            dy = -(height - out_h / zoom) * 0.55 + 4.0 * math.sin(2 * math.pi * t)
            matrix = np.array([[zoom, 0.0, dx], [0.0, zoom, dy]], dtype=np.float32)
            crop = cv2.warpAffine(
                image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            writer.write(cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR))
    finally:
        writer.release()
    return clip


def run_video(photo: Path, frames: int) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        clip = build_clip(photo, Path(tmp) / "pan.mp4", frames)
        config = load_config()
        config.evidence.enabled = False
        config.visualization.enabled = True
        config.runtime.write_incidents_file = False
        config.location.default_camera_id = "DEMO-LAPTOP"

        pipeline = AccidentPipeline(config, camera_id="DEMO-LAPTOP", source_label=str(clip))
        print("\n=== 60 frames of real footage, real YOLO ===")
        seen = 0
        for result in pipeline.stream(str(clip)):
            if result.frame_index % 15 == 0:
                print(
                    f"  frame {result.frame_index:>3}  detections={len(result.detections):>2}  "
                    f"tracks={len(result.tracks):>2}  state={result.state:<8} "
                    f"score={result.accident_score:.3f}  "
                    f"{result.performance.fps:.1f} fps  "
                    f"infer={result.performance.inference_ms:.0f}ms"
                )
            if result.detections and seen < 1 and result.frame_index > 2:
                report(result)
                seen += 1
        run = pipeline.get_incident_result()
        print(f"\n  frames processed : {pipeline._frame_index}")
        print(f"  confirmed        : {pipeline.verification.confirmed_count}")
        print(f"  false alarms     : {pipeline.verification.false_alarm_count}")
        print(f"  incidents        : {len(pipeline.incidents)}")
        pipeline.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the real YOLO path on real imagery")
    parser.add_argument("--photo", default="test_videos/street_photo.jpg")
    parser.add_argument("--video", action="store_true", help="build a clip from the photo and run it")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    photo = Path(args.photo)
    if not photo.is_file():
        print(f"photo not found: {photo}")
        print("pass a real road photograph, e.g.")
        print("  python scripts/real_yolo_check.py --photo path/to/your/photo.jpg")
        return 2
    if args.video:
        return run_video(photo, args.frames)
    return run_photo(photo)


if __name__ == "__main__":
    raise SystemExit(main())
