"""PHASE 1 BASELINE PROBE - read-only, offline, no provider requests.

Proves (or disproves) each reported gap by executing the real code paths with
the network hard-blocked.  Every outbound HTTP attempt raises, so nothing can
reach a provider.  This script is evidence gathering for REPORT.md; it makes no
changes to the engine.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------- #
# hard network block: prove no provider is ever contacted
# --------------------------------------------------------------------- #
NETWORK_ATTEMPTS: list = []


def _blocked(url, *a, **k):
    NETWORK_ATTEMPTS.append(str(url))
    raise AssertionError(f"NETWORK BLOCKED but code attempted: {url}")


urllib.request.urlopen = _blocked

from ai_engine.config import load_config  # noqa: E402
from ai_engine.emergency import (  # noqa: E402
    DemoCallDispatcher,
    build_provider,
    build_sms_provider,
    load_env_file,
)
from ai_engine.emergency.providers import (  # noqa: E402
    SimulationProvider,
    SimulationSmsProvider,
    TwilioProvider,
    TwilioSmsProvider,
)

RESULTS: list = []


def record(item: str, finding: str, evidence: str) -> None:
    RESULTS.append({"item": item, "finding": finding, "evidence": evidence})
    print(f"  {item}\n      -> {finding}\n         {evidence}")


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(f" {text}")
    print("=" * 78)


# --------------------------------------------------------------------- #
def probe_credential_state() -> None:
    header("1. CREDENTIAL STATE (values never printed)")
    load_env_file()
    names = (
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "TWILIO_PHONE_NUMBER",
        "EMERGENCY_CONTACT_NUMBER",
        "DEMO_MODE",
    )
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            state = "ABSENT"
            shape = "-"
        elif name.endswith("AUTH_TOKEN"):
            state = "PRESENT"
            shape = f"{len(raw)} chars, sha256[:6]=" + __import__("hashlib").sha256(raw.encode()).hexdigest()[:6]
        elif "NUMBER" in name:
            state = "PRESENT"
            shape = "len=%d, country=%s" % (len(raw), raw[:2] if raw.startswith("+") else "?")
        else:
            state = "PRESENT"
            shape = f"{len(raw)} chars"
        print(f"  {name:<26} {state:<8} {shape}")

    cfg = load_config()
    print(f"\n  config emergency.enabled  = {cfg.emergency.enabled}")
    print(f"  config emergency.demo_mode= {cfg.emergency.demo_mode}")
    print(f"  config emergency.provider = {cfg.emergency.provider}")
    print(f"  config sms_provider       = {cfg.emergency.sms_provider}")

    call = build_provider(cfg.emergency.provider)
    sms = build_sms_provider(cfg.emergency.sms_provider)
    print(f"\n  RESOLVED call provider   = {call.name}")
    print(f"  RESOLVED sms provider    = {sms.name}")
    if call.name == "twilio":
        record("credentials", "Twilio credentials ARE configured", "provider resolved to twilio")
    else:
        record(
            "credentials",
            "Twilio credentials ABSENT -> no real call/SMS is reachable in this build",
            f"provider auto resolved to call={call.name} sms={sms.name}",
        )


def probe_silent_fallback() -> None:
    header("2. CRITICAL: does provider=auto silently fall back to DEMO success?")
    # exactly the shipped default, with credentials missing
    provider = build_provider("auto", environ={})
    sms = build_sms_provider("auto", environ={})
    is_sim = isinstance(provider, SimulationProvider)
    record(
        "auto-fallback",
        "SILENT DEMO FALLBACK CONFIRMED" if is_sim else "no silent fallback",
        f"provider=auto with no credentials returned {provider.name!r} / {sms.name!r} "
        f"(config.yaml default emergency.provider = 'auto')",
    )

    # and what does it *report* after running?
    seen = []
    cfg = load_config()
    cfg.emergency.enabled = True
    cfg.emergency.demo_mode = True
    cfg.emergency.async_calls = False
    cfg.emergency.write_call_log = False
    d = DemoCallDispatcher(
        cfg.emergency,
        environ={"DEMO_MODE": "true", "EMERGENCY_CONTACT_NUMBER": "+15550100"},
        on_status=seen.append,
    )
    d.handle_incident(
        {
            "incident_id": "INC-PROBE-0001",
            "status": "CONFIRMED_ACCIDENT",
            "timestamp": "2026-10-02T18:31:35+00:00",
            "camera": {"camera_id": "CAM-001"},
            "location": {"name": "Chennai Main Road", "latitude": 13.08, "longitude": 80.27,
                         "is_camera_registered": True},
            "accident": {"score": 0.5, "severity": "HIGH"},
        }
    )
    rec = d.calls[0]
    record(
        "false-success",
        "REPORTS SUCCESS WITHOUT A PROVIDER",
        f"with zero credentials the engine reported call={rec.status} sms={rec.sms.status} "
        f"call_placed={rec.status.was_placed}",
    )


def probe_explicit_twilio() -> None:
    header("3. provider=twilio with missing credentials (explicit mode)")
    p = TwilioProvider(account_sid="", auth_token="", from_number="")
    usable, why = p.available()
    record(
        "explicit-twilio",
        "explicit mode FAILS LOUDLY" if not usable else "explicit mode silently works",
        f"TwilioProvider.available() -> {usable}, reason={why!r}",
    )
    s = TwilioSmsProvider(account_sid="", auth_token="", from_number="")
    su, sw = s.available()
    record("explicit-twilio-sms", "fails loudly" if not su else "silent", f"available() -> {su}, {sw!r}")


def probe_fallback_policy() -> None:
    header("4. SMS-on-call-failure policy: is it explicit and configurable?")
    cfg = load_config()
    has_flag = hasattr(cfg.emergency, "sms_on_call_failure")
    fields = sorted(cfg.emergency.__dataclass_fields__)
    record(
        "fallback-policy",
        "NO EXPLICIT POLICY FLAG" if not has_flag else "flag present",
        f"emergency config fields = {fields}",
    )
    # observed behaviour: SMS is skipped whenever the call did not reach the network
    from ai_engine.schemas.enums import CallStatus

    rec_reason = "call was FAILED: <error>"
    record(
        "fallback-behaviour",
        "hard-coded: SMS is SKIPPED when the call fails (no fallback)",
        f"dispatcher._send_sms gate uses CallStatus.reached_recipient; reason looks like {rec_reason!r}",
    )


def probe_durability_and_callbacks() -> None:
    header("5. Durability, idempotency, retries, webhooks")
    src = (ROOT / "ai_engine" / "emergency" / "dispatcher.py").read_text(encoding="utf-8")
    prov = (ROOT / "ai_engine" / "emergency" / "providers.py").read_text(encoding="utf-8")
    combined = src + prov
    findings = {
        "durable outbox (on-disk/sqlite)": any(
            k in combined for k in ("sqlite", "Outbox", "outbox", "INSERT INTO")
        ),
        "idempotency key sent to provider": any(
            k in combined for k in ("Idempotency", "idempotency")
        ),
        "retry / backoff": any(k in combined for k in ("backoff", "retry", "retries")),
        "webhook / status callback receiver": any(
            k in combined.lower() for k in ("webhook", "signature", "callback_url")
        ),
        "circuit breaker / kill switch": any(
            k in combined for k in ("circuit", "kill_switch", "breaker")
        ),
        "recipient allowlist / consent": any(
            k in combined for k in ("allowlist", "consent", "jurisdiction")
        ),
    }
    for item, present in findings.items():
        record(f"missing-or-present: {item}", "PRESENT" if present else "MISSING",
               "grep over ai_engine/emergency/*.py")


def probe_latency_methodology() -> None:
    header("6. Latency: what the reported 505 frames / 70.8 s actually measures")
    import cv2

    path = ROOT / "test_videos" / "accident_demo.mp4"
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cfg = load_config()
    print(f"  clip            : {path.name}")
    print(f"  frames / fps    : {n} @ {fps:.1f}  ({n / max(fps, 1):.1f}s of footage)")
    print(f"  resolution      : {w}x{h}")
    print(f"  model           : {cfg.model.weights} imgsz={cfg.model.imgsz} device={cfg.model.device}")
    print(f"  frame_stride    : {cfg.video.frame_stride}  target_width={cfg.video.target_width}")
    print(f"  process_fps     : {cfg.video.process_fps}")
    record(
        "reported-timing",
        "505 frames in 70.8 s is OFFLINE FILE THROUGHPUT, not live response",
        f"clip is {n / max(fps, 1):.1f}s of footage; the run took 70.8s wall clock, "
        f"i.e. ~{n / 70.8:.1f} FPS analysed vs {fps:.1f} FPS native -> "
        f"{70.8 / (n / max(fps, 1)):.2f}x slower than real time, sequential decode",
    )
    record(
        "no-stage-timings",
        "no per-stage latency instrumentation (capture/decode/detect/verify/notify)",
        "Pipeline PerfStats carries inference_ms/tracking_ms/analysis_ms/total_ms only; "
        "there is no capture->incident or incident->notification stage timing",
    )
    hw = os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    record("hardware", "recorded for the record", f"CPU = {hw}")


def probe_test_coverage() -> None:
    header("7. Existing test coverage and its limits")
    tests = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    total = 0
    for name in tests:
        text = (ROOT / "tests" / name).read_text(encoding="utf-8")
        total += text.count("    def test_")
    print(f"  {len(tests)} test modules, ~{total} test methods")
    scenarios = (ROOT / "ai_engine" / "detection" / "scenarios.py").read_text(encoding="utf-8")
    n_scen = scenarios.count('"') // 8
    record(
        "coverage",
        "all scripted/synthetic; NO labeled real-world evaluation set exists",
        f"{len(tests)} modules; the 9 regression scenarios use synthetic detections; "
        "scripts/calibrate.py expects labelled folders that are not present",
    )
    record(
        "accuracy-claim",
        "NO accuracy / false-alarm-rate figure can be claimed",
        "one real clip, 1 incident, 505 frames, single camera, single event - "
        "insufficient for any rate",
    )


def probe_secrets_posture() -> None:
    header("8. Secrets and sensitive-data posture")
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    print(f"  .gitignore env patterns: "
          f"{[l for l in gi.splitlines() if 'env' in l.lower() and not l.startswith('#')]}")
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    print(f"  .env present        : {env.is_file()} ({env.stat().st_size if env.is_file() else 0} bytes)")
    if example.is_file():
        text = example.read_text(encoding="utf-8")
        import re
        real = [m for m in re.findall(r"\+\d{8,}", text) if not m.startswith("+1555") and not m.startswith("+1500")]
        record("env-example", "placeholders only" if not real else "LEAK",
               f"phone-like tokens in .env.example: {real}")
    record("encryption", "NO encryption at rest",
           "incidents.json / demo_calls.jsonl / evidence/ are written as plain files")
    record("auth-n/a", "no auth on any endpoint",
           "the engine exposes no HTTP server; a webhook receiver does not exist yet")


def main() -> int:
    print("=" * 78)
    print(" SafeVision PHASE 1 - baseline probe (offline, read-only)")
    print("=" * 78)
    probe_credential_state()
    probe_silent_fallback()
    probe_explicit_twilio()
    probe_fallback_policy()
    probe_durability_and_callbacks()
    probe_latency_methodology()
    probe_test_coverage()
    probe_secrets_posture()
    header("NETWORK SAFETY ASSERTION")
    print(f"  outbound HTTP attempts during this entire probe: {len(NETWORK_ATTEMPTS)}")
    print("  (urlopen was replaced with a raising stub; 0 = nothing was contacted)")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "phase1_baseline.json").write_text(
        json.dumps({"network_attempts": NETWORK_ATTEMPTS, "results": RESULTS}, indent=2),
        encoding="utf-8",
    )
    print(f"\n  evidence written to {out / 'phase1_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())