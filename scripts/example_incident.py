"""Print one real, complete incident JSON (for docs / for your teammate).

    python scripts/example_incident.py
    python scripts/example_incident.py --out sample_incident.json
    python scripts/example_incident.py --markdown

The output is produced by actually running the pipeline on a scripted crash
scenario - nothing here is hand-written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ai_engine.config import load_config  # noqa: E402
from ai_engine.detection.scenarios import SCENARIOS  # noqa: E402
from ai_engine.detection.scripted import ScriptedDetector  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.utils.jsonio import atomic_write_json, dumps  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print a complete SafeVision incident")
    parser.add_argument("--scenario", default="head_on_crash", choices=sorted(SCENARIOS))
    parser.add_argument("--out", default="", help="also write the incident to this file")
    parser.add_argument("--markdown", action="store_true", help="print the AI report instead")
    parser.add_argument(
        "--demo-call",
        action="store_true",
        help="also demonstrate the automatic DEMO EMERGENCY CALL (simulation provider)",
    )
    parser.add_argument(
        "--recipient",
        default="+15005550306",
        help="recipient used by --demo-call (the default is Twilio's documented "
             "test endpoint, which does not ring a real phone)",
    )
    args = parser.parse_args(argv)

    configure_logging("WARNING")
    config = load_config()
    config.evidence.enabled = True
    config.evidence.root = "evidence"
    config.visualization.enabled = True
    # never dial from a documentation script
    config.emergency.enabled = False
    if args.demo_call:
        from ai_engine.emergency.dispatcher import DemoCallDispatcher

        config.emergency.enabled = True
        config.emergency.demo_mode = True
        config.emergency.provider = "simulation"
        config.emergency.write_call_log = False
        config.emergency.async_calls = False
        config.emergency.expose_recipient = False  # keep the number out of the sample
        emergency = DemoCallDispatcher(
            config.emergency, environ={"EMERGENCY_CONTACT_NUMBER": args.recipient}
        )
    else:
        emergency = None

    pipeline = AccidentPipeline(
        config,
        detector=ScriptedDetector(SCENARIOS[args.scenario]),
        camera_id="CAM-001",
        emergency=emergency,
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

    if args.markdown:
        print(incident["report"]["markdown"])
    else:
        print(dumps(incident))
    if args.out:
        path = atomic_write_json(args.out, incident)
        print(f"\nwritten to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
