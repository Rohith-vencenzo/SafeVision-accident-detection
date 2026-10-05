"""Integration example - how a backend consumes the SafeVision AI result.

This file is the "handoff document" for the teammate implementing the other 40%
of the project (backend, database, dashboard, notifications).  It shows the
three ways you can connect, and none of them require changing the AI engine.

    python examples/integration_example.py                    # runs the demo set
    python examples/integration_example.py --source clip.mp4   # your own clip
    python examples/integration_example.py --print-schema       # just the contract

The three integration patterns
------------------------------
1. **In-process callback** - the AI engine calls your function as soon as an
   incident is confirmed (lowest latency, single process).
2. **Batch file** - the engine writes ``incidents.json``; your job polls or
   watches that file (simplest to deploy, survives restarts).
3. **Reader** - your code owns the loop and calls
   ``pipeline.process_frame(frame)`` for frames it already has (for example
   frames coming out of an RTSP client you manage yourself).

Plus one opt-in extra: ``--demo-call`` shows the automatic **demonstration**
emergency call. It is off by default, fires only after an incident is CONFIRMED,
places exactly one call per incident, and needs no button.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.config import load_config  # noqa: E402
from ai_engine.pipeline import AccidentPipeline  # noqa: E402
from ai_engine.schemas import (  # noqa: E402
    INCIDENT_JSON_SCHEMA,
    SCHEMA_VERSION,
    generate_emergency_incident,
    validate_incident,
)
from ai_engine.utils.jsonio import dumps  # noqa: E402
from ai_engine.utils.logging_utils import configure_logging

# The demo SMS body contains an emoji; a default cp1252 Windows console would
# raise UnicodeEncodeError while merely *printing* the result.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. in-process callback
# --------------------------------------------------------------------------- #
def on_incident(incident: Dict[str, Any]) -> None:
    """Called by the AI engine the moment an accident is confirmed.

    Replace the body with a queue write / HTTP POST / DB insert.  Keep it fast
    and non-blocking: it runs on the detection thread.
    """
    # `minimum_score` lets a subscriber receive *every* incident and decide here
    # which ones deserve an alert.  0.0 means "everything the engine confirmed".
    alert = generate_emergency_incident(incident, minimum_score=0.0)
    if not alert:
        return
    print("\n>>> incident callback  (this is the payload your backend would store)")
    print(json.dumps(alert, indent=2))
    # Your code would do something like:
    #   requests.post("http://backend/api/incidents", json=alert, timeout=2)


# --------------------------------------------------------------------------- #
# 2. reader pattern
# --------------------------------------------------------------------------- #
def run_reader(source: str) -> List[Dict[str, Any]]:
    """You own the frame loop; SafeVision owns the analysis."""
    config = load_config()
    config.evidence.enabled = True

    pipeline = AccidentPipeline(config, camera_id="CAM-001", on_incident=on_incident)
    incidents: List[Dict[str, Any]] = []

    try:
        with pipeline:  # releases the model + evidence handles deterministically
            for result in pipeline.process_video(source):
                # per-frame telemetry, e.g. for a live dashboard
                if result.frame_index % 30 == 0:
                    print(
                        f"    frame {result.frame_index:>4}  {result.state:<18}"
                        f" score={result.accident_score:.2f}"
                        f" fps={result.performance.fps:.1f}"
                    )
                if result.incident:
                    incidents.append(result.incident)
    finally:
        pass

    return incidents


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def show_dashboard_fields(pipeline: AccidentPipeline, incident: Dict[str, Any]) -> None:
    """The live/recorded, latency and location-provenance fields a dashboard needs.

    Added in Phase 3.  A dashboard must be able to tell an operator, without
    interpretation, whether this is a live or a recorded source, how long the
    engine took, and whether the position is a camera coordinate or a real fix.
    """
    ledger = pipeline.ledger.report()
    tp = pipeline.ledger.report().get("stages", {})
    detect = tp.get("detect", {})
    print("\nDashboard fields (Phase 3):")
    print(json.dumps(
        {
            "processing_mode": ledger["mode"],
            "source_kind": ledger["source_kind"],
            "frames_processed": ledger["frames_processed"],
            "stage_latency_p95_ms": {k: v.get("p95_ms") for k, v in tp.items()},
            "stage_latency_samples": {k: v.get("count") for k, v in tp.items()},
            "incident_timelines_ms": ledger.get("incident_timelines", {}),
            "latency_methodology": ledger["methodology"],
            "location_provenance": incident.get("location_provenance"),
            "dropped_frames": pipeline.dropped_frames,
        },
        indent=2,
    ))
    note = ledger["methodology"]["caveat"]
    print(f"  caveat: {note}")


def show_contract() -> None:
    print("=" * 78)
    print(f" SafeVision incident contract - schema_version {SCHEMA_VERSION}")
    print("=" * 78)
    print("\nRequired fields of an incident object:")
    for key in INCIDENT_JSON_SCHEMA["required"]:
        print(f"  - {key}")
    print("\nStatuses you will see:")
    for value in INCIDENT_JSON_SCHEMA["properties"]["status"]["enum"]:
        print(f"  - {value}")
    print("\nThe full JSON Schema is available as")
    print("    from ai_engine.schemas import INCIDENT_JSON_SCHEMA")
    print("and as `python main.py --print-schema`")
    print("\nVersioning rule: reject an unknown MAJOR version, ignore unknown")
    print("fields (a new MINOR only ever adds fields).")


def show_minimal_consumer(payload: Dict[str, Any]) -> None:
    """What a backend minimally needs to display an alert."""
    print("\nMinimal consumer view:")
    print(json.dumps(
        {
            "incident_id": payload["incident_id"],
            "status": payload["status"],
            "timestamp": payload["timestamp"],
            "camera_id": payload["camera"]["camera_id"],
            "location": payload["location"]["name"],
            "latitude": payload["location"]["latitude"],
            "longitude": payload["location"]["longitude"],
            "accident_score": payload["accident"]["score"],
            "severity": payload["accident"]["severity"],
            "vehicles": payload["objects"]["vehicles_involved"],
            "annotated_frame": payload["media"]["annotated_frame"],
        },
        indent=2,
    ))


# --------------------------------------------------------------------------- #
# 4. the demonstration emergency call (demo mode only)
# --------------------------------------------------------------------------- #
def on_call_status(status: Dict[str, Any]) -> None:
    """Live status feed for the demonstration notification (call, then SMS).

    Fires on every transition: INITIATED -> RINGING -> IN_PROGRESS -> COMPLETED
    for the call, then SENT for the SMS.  Or SKIPPED / BLOCKED when a safety
    interlock refused.

    Your dashboard renders ``label`` + ``status`` verbatim - that is the whole
    point of the label, so a demonstration notification can never be mistaken
    for a real emergency alert.  There is no button: it is automatic once an
    incident is CONFIRMED, exactly one call and one SMS per incident.
    """
    icon = {
        "COMPLETED": "OK",
        "SENT": "OK",
        "FAILED": "!!",
        "BLOCKED": "!!",
        "SKIPPED": "--",
    }.get(str(status.get("status")), "..")
    print(f"    [{icon}] {status.get('label')} - {status.get('status')} "
          f"| {status.get('recipient') or '(unconfigured)'} "
          f"| {status.get('provider')}")


def show_demo_call(pipeline: AccidentPipeline, incident: Dict[str, Any]) -> None:
    """What the dashboard needs, straight out of the incident payload."""
    call = incident.get("demo_emergency_call", {})
    sms = incident.get("demo_emergency_sms", {})
    print("\nDemonstration emergency notification "
          "(from demo_emergency_call + demo_emergency_sms):")
    print(json.dumps({"call": call, "sms": sms}, indent=2))
    if not call.get("attempted"):
        print(f"    -> no call was placed: {call.get('reason')}")
    else:
        print(f"    -> snapshot taken when the incident was emitted "
              f"(call {call.get('status')}, sms {sms.get('status')}).")
        print("       The notification keeps running after that, so for a LIVE view")
        print("       use the on_call_status feed above, or pipeline.demo_calls below.")

    if sms.get("body"):
        print("\n    --- the SMS the recipient receives ---")
        for line in str(sms["body"]).splitlines():
            print(f"    | {line}")

    # the full audit trail for this run
    if pipeline.demo_calls:
        final = pipeline.demo_calls[-1]
        print(f"    -> {len(pipeline.demo_calls)} notification record(s); final "
              f"call {final.get('status')}, sms {(final.get('sms') or {}).get('status')}")


def run_with_demo_call(scenario: str = "head_on_crash") -> Dict[str, Any]:
    """Run a scripted scenario with the demonstration call enabled.

    Uses the **simulation** provider explicitly, so this example can never place
    a real call no matter what is configured in your ``.env``.  A real call
    needs ``emergency.provider: twilio`` (or ``auto`` plus Twilio credentials in
    ``.env``) and ``DEMO_MODE=true``.
    """
    import time

    import numpy as np

    from ai_engine.detection.scenarios import SCENARIOS
    from ai_engine.detection.scripted import ScriptedDetector
    from ai_engine.emergency import DemoCallDispatcher, load_env_file

    # DEMO_MODE / EMERGENCY_CONTACT_NUMBER come from .env, never from code.
    # (Real environment variables always win over the file.)
    load_env_file()

    config = load_config()
    config.emergency.enabled = True
    config.emergency.demo_mode = True
    config.emergency.provider = "simulation"   # <- the safe choice for a demo
    # The dispatcher reads DEMO_MODE from the environment on every access, so a
    # `.env` saying `DEMO_MODE=false` would otherwise make this example - which
    # explicitly asked for a demo - place nothing at all. Set both.
    os.environ["DEMO_MODE"] = "true"

    dispatcher = DemoCallDispatcher(config.emergency, on_status=on_call_status)
    pipeline = AccidentPipeline(
        config,
        detector=ScriptedDetector(SCENARIOS[scenario]),
        camera_id="CAM-001",
        on_incident=on_incident,
        on_call_status=on_call_status,
        emergency=dispatcher,
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    base = time.time()
    for index in range(80):
        result = pipeline.process_frame(frame, index, base + index / 15.0)
        # a live dashboard can also just read the per-frame state
        if result.demo_call and index == 79:
            print(f"    final per-frame call state: {result.demo_call['label']} - "
                  f"{result.demo_call['status']}")
    pipeline.close()   # waits for the in-flight call to finish

    incident = pipeline.get_incident_result()
    if incident:
        show_demo_call(pipeline, incident)
        # Report the state the provider actually reached. The incident payload is
        # a snapshot from the moment the incident was built (still INITIATED /
        # PENDING), so it must not be presented as the outcome.
        final = dispatcher.final_states().get(incident["incident_id"])
        if final:
            print(f"    FINAL call state : {final['call']}")
            print(f"    FINAL sms  state : {final['sms']}")
            print(f"    provider         : {final['provider']} "
                  f"({final['provider_status'] or 'n/a'}, ref {final['provider_ref'] or 'n/a'})")
    else:
        print("    -> no incident for this scenario, so no call")
    return incident


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision integration example")
    parser.add_argument("--source", default="", help="video to analyse (default: scripted demo set)")
    parser.add_argument("--print-schema", action="store_true", help="print the contract and exit")
    parser.add_argument("--full", action="store_true", help="print the complete incident JSON")
    parser.add_argument(
        "--demo-call",
        action="store_true",
        help="demonstrate the automatic DEMO EMERGENCY CALL (simulation provider, no real call)",
    )
    args = parser.parse_args(argv)

    configure_logging("WARNING")

    if args.print_schema:
        show_contract()
        return 0

    if args.demo_call:
        print("=" * 78)
        print(" SafeVision - demonstration emergency call")
        print("=" * 78)
        print("\nThe notification is automatic: no button, exactly one call and one")
        print("SMS per CONFIRMED incident (call first, SMS second).")
        print("This example pins the simulation provider, so nothing is dialled and")
        print("no SMS is sent.\n")
        run_with_demo_call()
        return 0

    if args.source:
        incidents = run_reader(args.source)
    else:
        print("=" * 78)
        print(" SafeVision integration example")
        print("=" * 78)
        print("\nNo --source given, so the scripted demo scenarios are used")
        print("(they exercise the same code path as a real video).\n")
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        import numpy as np

        config = load_config()
        config.evidence.enabled = True
        import time

        base = time.time()
        for name in ("head_on_crash", "parked_cars", "hard_brake"):
            print(f"--- scenario: {name} " + "-" * (40 - len(name)))
            pipeline = AccidentPipeline(
                config, detector=ScriptedDetector(SCENARIOS[name]), camera_id="CAM-001",
                on_incident=on_incident,
            )
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            for index in range(70):
                pipeline.process_frame(frame, index, base + index / 15.0)
            pipeline.close()
            incident = pipeline.get_incident_result()
            if incident:
                print(f"    -> incident produced: {incident['incident_id']}")
            else:
                print(f"    -> no incident (correct outcome for this scenario)")
        show_contract()
        return 0

    print()
    if not incidents:
        print("No incident was produced for this source.")
        print("That is a valid result - the engine only reports confirmed, "
              "multi-frame evidence.")
        return 0

    incident = incidents[0]
    problems = validate_incident(incident)
    if problems:
        print("SCHEMA PROBLEMS:", problems)
    else:
        print("Incident validates against the SafeVision contract.")

    show_minimal_consumer(incident)
    if args.full:
        print("\nFull incident JSON:")
        print(dumps(incident))
    else:
        print("\n( pass --full to print the complete incident object )")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
