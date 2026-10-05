"""Verify that every command advertised in the README actually runs.

Each step is executed exactly as documented, so the "run commands" section of
the README cannot drift away from reality.  Nothing here dials anybody: the
demo-call step is checked with ``--check-emergency`` (which places no call).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

#: (label, argv-after-python, expected substrings that must appear in stdout)
STEPS = [
    ("deps importable",
     ["-c", "import cv2, numpy, ultralytics, torch; print('deps OK')"],
     ["deps OK"]),
    ("list cameras",
     ["main.py", "--list-cameras"],
     ["CAM-001"]),
    ("print config",
     ["main.py", "--print-config"],
     ['"verification"', '"emergency"']),
    ("print schema",
     ["main.py", "--print-schema"],
     ["incident_id", "demo_emergency_call"]),
    ("download_model --list",
     [str(Path("scripts") / "download_model.py"), "--list"],
     ["yolo11n"]),
    ("check model",
     ["main.py", "--check-model"],
     ["loaded successfully"]),
    ("check source",
     ["main.py", "--check-source", str(Path("test_videos") / "street_photo.jpg")],
     ["usable"]),
    ("make test video",
     [str(Path("scripts") / "make_test_video.py")],
     []),
    ("check emergency",
     ["main.py", "--check-emergency"],
     ["DEMO EMERGENCY CALL", "EMERGENCY_CONTACT_NUMBER"]),
    ("integration example (contract)",
     [str(Path("examples") / "integration_example.py"), "--print-schema"],
     ["schema_version"]),
    ("integration example (demo call)",
     [str(Path("examples") / "integration_example.py"), "--demo-call"],
     ["DEMO EMERGENCY CALL", "COMPLETED"]),
    ("demo (scripted state machine)",
     ["main.py", "--demo", "--log-level", "ERROR"],
     ["SCENARIO 1/5", "CONFIRMED_ACCIDENT"]),
    # the run summary is logged at INFO, so leave the level at its default
    ("run on a clip",
     ["main.py", "--source", str(Path("test_videos") / "accident.mp4"), "--max-frames", "40"],
     ["processed 40 frame(s)"]),
    ("no demo call for one run",
     ["main.py", "--source", str(Path("test_videos") / "accident.mp4"), "--max-frames", "20",
      "--no-demo-call"],
     ["processed 20 frame(s)"]),
]


def run(label: str, argv: list) -> tuple:
    started = time.perf_counter()
    proc = subprocess.run(
        [PY, *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - started
    # the run summary is logged to stderr by design (stdout must stay clean
    # for the --json contract), so both streams are searched for the markers
    combined = proc.stdout + proc.stderr
    return label, proc.returncode, combined, proc.stderr, elapsed


def main() -> int:
    print("=" * 78)
    print(" README run-commands check")
    print("=" * 78)
    failures = []
    for label, argv, expected in STEPS:
        label, code, out, err, elapsed = run(label, argv)
        ok = code == 0 and all(token in out for token in expected)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} exit={code} {elapsed:5.1f}s")
        if not ok:
            failures.append(label)
            if code != 0:
                print(f"         stderr: {err.strip()[:400]}")
            missing = [t for t in expected if t not in out]
            if missing:
                print(f"         expected but not in the output: {missing}")
    print("-" * 78)
    print(f"  {len(STEPS) - len(failures)}/{len(STEPS)} documented commands verified")
    if failures:
        print("  failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())