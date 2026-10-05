"""Download the YOLO weights into ``models/`` (Phase 8 setup helper).

    python scripts/download_model.py                # yolo11n.pt (~5 MB)
    python scripts/download_model.py --model yolo11s.pt
    python scripts/download_model.py --list         # show recommended sizes

The weights come from the official Ultralytics release assets, so no account,
API key or paid service is involved.  Once the file is in ``models/`` the engine
runs completely offline.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.utils.logging_utils import configure_logging, get_logger  # noqa: E402

_LOGGER = get_logger("scripts.download_model")

#: (name, size on disk, note) - sizes are approximate for the standard build
KNOWN_MODELS = [
    ("yolo11n.pt", "~5 MB", "fastest, CPU friendly - the default"),
    ("yolo11s.pt", "~19 MB", "better accuracy, still CPU friendly"),
    ("yolo11m.pt", "~40 MB", "good accuracy, needs a decent CPU"),
    ("yolo11l.pt", "~50 MB", "accurate, slow on a laptop CPU"),
    ("yolo11x.pt", "~110 MB", "most accurate, GPU recommended"),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download YOLO weights for SafeVision")
    parser.add_argument("--model", default="yolo11n.pt", help="checkpoint name or path")
    parser.add_argument("--dest", default="models", help="destination directory")
    parser.add_argument("--list", action="store_true", help="list the known checkpoints and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    configure_logging("INFO")

    if args.list:
        print(f"{'checkpoint':<16}{'size':<10}note")
        print("-" * 64)
        for name, size, note in KNOWN_MODELS:
            print(f"{name:<16}{size:<10}{note}")
        return 0

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / Path(args.model).name

    if target.is_file() and not args.force:
        size_mb = target.stat().st_size / (1024 * 1024)
        _LOGGER.info("%s is already present (%.1f MB) - use --force to re-download", target, size_mb)
        print(f"already downloaded: {target}")
        return 0

    _LOGGER.info("downloading %s -> %s", args.model, target)
    try:
        from ultralytics import YOLO
    except ImportError:
        _LOGGER.error("ultralytics is not installed: pip install -r requirements.txt")
        return 1

    try:
        model = YOLO(args.model)          # ultralytics caches the file
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("download failed: %s", exc)
        _LOGGER.error("If this machine is offline, download the file manually and")
        _LOGGER.error("place it in %s, then run: main.py --model models/<name>.pt", destination)
        return 1

    cached = Path(getattr(model, "ckpt_path", "") or "")
    if cached.is_file() and cached.resolve() != target.resolve():
        shutil.copy2(cached, target)
        _LOGGER.info("copied %s -> %s", cached, target)
    elif not target.is_file() and cached.is_file():
        shutil.copy2(cached, target)

    print(f"\nweights ready: {target}")
    print("run it with:  python main.py --source <video> --model "
          f"{target.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
