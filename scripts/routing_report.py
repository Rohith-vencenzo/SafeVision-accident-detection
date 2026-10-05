"""Run the routing decision trace against the shipped directory.

    python scripts\\routing_report.py
    python scripts\\routing_report.py --mode live --json reports/routing.json

Prints what SafeVision *would* do for a confirmed incident and why - the
decision trace, not a phone number.  It sends nothing in any mode: routing and
dispatching are separate concerns, and this exercises only the first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from ai_engine.dispatch import (  # noqa: E402
    DispatchMode,
    RecipientDirectory,
    Router,
    RoutingPolicy,
)  # noqa: E402


def sample_incident() -> dict:
    return {
        "incident_id": "INC-DEMO-0001",
        "status": "CONFIRMED_ACCIDENT",
        "timestamp": "2026-10-03T18:31:35+00:00",
        "camera": {"camera_id": "CAM-001"},
        "location": {"name": "Chennai Main Road", "latitude": 13.0827,
                     "longitude": 80.2707},
        "accident": {"score": 0.44, "severity": "HIGH"},
        "extra": {"location_provenance": {
            "source": "CAMERA_REGISTERED",
            "latitude": 13.0827, "longitude": 80.2707,
            "confidence": "APPROXIMATE_CAMERA_POSITION",
        }},
    }


def render(router: Router, incident: dict) -> str:
    decision = router.route(incident)
    lines = [
        "=" * 74,
        f" routing decision  mode={decision.mode}",
        "=" * 74,
        f"  incident      : {decision.incident_id}",
        f"  outcome       : {'WOULD DISPATCH' if decision.is_dispatchable else 'REFUSED'}",
        f"  rule applied  : {decision.rule}",
        f"  reason        : {decision.reason}",
        f"  recipient     : {decision.recipient.recipient_id if decision.recipient else '(none)'}"
        f"  [{decision.recipient.masked if decision.recipient else '-'}]",
        f"  would_dispatch: {decision.would_dispatch}",
        "",
        "  checks in order:",
    ]
    for step in decision.evaluated:
        mark = "pass" if step.get("passed") else "FAIL"
        extra = {k: v for k, v in step.items() if k not in ("rule", "passed")}
        lines.append(f"    [{mark}] {step['rule']:<26} {extra if extra else ''}")
    lines.append("")
    lines.append(f"  summary: {router.trace.summary()}")
    lines.append("  NOTE: routing only. Nothing is sent by this script, in any mode.")
    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SafeVision routing decision trace")
    parser.add_argument("--directory", default=str(ROOT / "dispatch" / "recipients.json"))
    parser.add_argument("--mode", choices=("simulation", "live"), default="simulation")
    parser.add_argument("--role", default="safety-officer")
    parser.add_argument("--jurisdiction", default="IN-TN")
    parser.add_argument("--require-location", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    policy = RoutingPolicy(
        mode=DispatchMode.from_config(args.mode),
        required_role=args.role,
        required_jurisdiction=args.jurisdiction,
        require_location=args.require_location,
    )
    book = RecipientDirectory.from_file(args.directory)
    router = Router(policy=policy, directory=book)
    print(render(router, sample_incident()))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(router.trace.to_json(), encoding="utf-8")
        print(f"\ndecision log written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())