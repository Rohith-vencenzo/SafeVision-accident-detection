"""Show the exact SMS body produced for a confirmed incident.

Used to inspect (and eyeball) the alert text during development.  Writes to a
file as well, because the body contains an emoji that a default Windows console
cannot print.

    python scripts\show_sms_body.py
    python scripts\show_sms_body.py --out sms_preview.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ai_engine.config import load_config  # noqa: E402
from ai_engine.detection.scenarios import SCENARIOS  # noqa: E402
from ai_engine.detection.scripted import ScriptedDetector  # noqa: E402
from ai_engine.emergency import load_env_file  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview the demo SMS body")
    parser.add_argument("--scenario", default="head_on_crash", choices=sorted(SCENARIOS))
    parser.add_argument("--out", default="", help="also write the body to this file")
    args = parser.parse_args()

    configure_logging("ERROR")
    load_env_file()

    config = load_config()
    config.evidence.enabled = False
    config.visualization.enabled = False
    config.runtime.write_incidents_file = False
    config.emergency.enabled = True
    config.emergency.demo_mode = True
    config.emergency.async_calls = False
    config.emergency.write_call_log = False
    config.emergency.expose_recipient = False

    pipeline = AccidentPipeline(
        config,
        detector=ScriptedDetector(SCENARIOS[args.scenario]),
        camera_id="CAM-001",
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    base = time.time()
    for index in range(80):
        pipeline.process_frame(frame, index, base + index / 15.0)
    pipeline.close()

    incident = pipeline.get_incident_result()
    if not incident:
        print(f"scenario {args.scenario!r} produced no incident")
        return 1
    sms = incident["demo_emergency_sms"]
    body = sms.get("body") or "(empty)"
    # print through the replacement codec so the emoji survives the console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 60)
    print(f"SMS status : {sms['status']}  (sent={sms['sent']})")
    print(f"provider   : {sms['provider']}  recipient {sms['recipient']}")
    print("=" * 60)
    print(body)
    print("=" * 60)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())