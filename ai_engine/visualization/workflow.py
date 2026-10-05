"""The end-to-end workflow display.

    VIDEO -> DETECTION -> TRACKING -> VERIFICATION -> ANALYSIS -> CONFIRMED
          -> INCIDENT -> CALL -> CALL COMPLETED -> SMS -> SMS SENT -> REPORT

Every stage is printed with the state that produced it, not a bare "success".
The reporter is fed by three callbacks - the pipeline's live state-change hook,
the incident callback, and the dispatcher's status feed - so a stage appears when
it actually happens rather than being reconstructed at the end.

Formatting is deliberately plain ASCII with a fixed width, so it is readable on a
Windows console (cp1252) without depending on box-drawing characters.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

__all__ = ["WorkflowReporter"]

WIDTH = 74

#: The ordered pipeline the user is meant to *see*.  Kept as data so the run
#: summary can mark each stage as reached, skipped or refused.
STAGES: tuple = (
    ("LIVE DETECTION", "YOLO detections + object tracking on each frame"),
    ("ANALYZING ACCIDENT", "multi-signal evidence collected and scored"),
    ("VERIFICATION", "temporal confirmation / false-alarm prevention"),
    ("ACCIDENT CONFIRMED", "state machine reached CONFIRMED_ACCIDENT"),
    ("INCIDENT CREATED", "incident record written with id, severity, location"),
    ("EMERGENCY CALL", "call created for the confirmed incident"),
    ("CALL COMPLETED", "provider reported a terminal call state"),
    ("SMS GENERATED", "alert text composed from the incident"),
    ("SMS SENT", "provider accepted the message"),
    ("INCIDENT REPORT", "incident JSON + call log + evidence saved"),
)


class WorkflowReporter:
    """Prints the workflow as it happens.

    ``quiet`` silences it entirely (used for ``--json``, where stdout must stay
    a clean one-object-per-incident stream).
    """

    def __init__(self, quiet: bool = False, stream: Any = None) -> None:
        self.quiet = quiet
        self.stream = stream
        self.stage_status: Dict[str, str] = {name: "pending" for name, _ in STAGES}
        self.call_events: List[Dict[str, Any]] = []
        self.incidents: List[str] = []
        #: True once a simulated transition has been reported, so the final
        #: summary can say so instead of implying a real alert went out.
        self.simulated = False
        self._progress_active = False

    # ------------------------------------------------------------------ #
    # output helpers
    # ------------------------------------------------------------------ #
    def _write(self, text: str = "") -> None:
        if self.quiet:
            return
        print(text, flush=True)

    def clear_progress(self) -> None:
        """Finish an in-place progress line so block output is not corrupted."""
        if self._progress_active:
            print("", flush=True)
            self._progress_active = False

    def banner(self, title: str) -> None:
        self.clear_progress()
        self._write()
        self._write("=" * WIDTH)
        self._write(f"  {title}")
        self._write("=" * WIDTH)

    def rule(self, title: str) -> None:
        self.clear_progress()
        self._write()
        self._write(f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4))

    def kv(self, key: str, value: Any, width: int = 22) -> None:
        self.clear_progress()
        self._write(f"   {key:<{width}}: {value}")

    def mark(self, stage: str, status: str) -> None:
        if stage in self.stage_status:
            self.stage_status[stage] = status

    def progress(self, text: str) -> None:
        """An in-place status line; auto-cleared before any block output."""
        if self.quiet:
            return
        print(f"\r   {text}"[:WIDTH + 20], end="", flush=True)
        self._progress_active = True

    # ------------------------------------------------------------------ #
    # stage 1-4: live detection -> verification
    # ------------------------------------------------------------------ #
    def on_transition(self, event: Dict[str, Any]) -> None:
        """A verification state changed: render the ANALYSIS stage in full."""
        analysis = event.get("analysis") or {}
        state = str(event.get("to") or "")
        previous = str(event.get("from") or "")

        if state == "SUSPICIOUS":
            self.mark("LIVE DETECTION", "done")
            self.mark("ANALYZING ACCIDENT", "active")
            self.rule(f"ANALYZING ACCIDENT  ({previous} -> {state})")
            self.kv("detection status", f"anomalous motion detected, {analysis.get('detections', 0)} object(s) in frame")
            self.kv("accident confidence", f"{_pct(analysis.get('score'))}")
            self.kv("detected vehicles", analysis.get("vehicles_involved", 0))
            self.kv("people involved", analysis.get("persons_involved", 0))
            self.kv("physical evidence", _pct(analysis.get("physical_score")))
            self.kv("why", event.get("reason") or "-")
            return

        if state == "VERIFYING":
            self.mark("LIVE DETECTION", "done")
            self.mark("ANALYZING ACCIDENT", "done")
            self.mark("VERIFICATION", "active")
            self.rule(f"VERIFICATION  ({previous} -> {state})")
            self.kv("temporal verification", f"active, {analysis.get('frames_in_state', 0)} frame(s) held "
                                             f"({analysis.get('seconds_in_state', 0)}s)")
            self.kv("evidence so far", _signals(analysis))
            self.kv("accident confidence", _pct(analysis.get("score")))
            self.kv("why", event.get("reason") or "-")
            return

        if state == "CONFIRMED_ACCIDENT":
            self.mark("LIVE DETECTION", "done")
            self.mark("ANALYZING ACCIDENT", "done")
            self.mark("VERIFICATION", "done")
            self.mark("ACCIDENT CONFIRMED", "done")
            self.rule("ACCIDENT CONFIRMED")
            self.kv("result", "CONFIRMED_ACCIDENT - false-alarm prevention passed")
            self.kv("accident confidence", _pct(analysis.get("score")))
            self.kv("physical evidence", _pct(analysis.get("physical_score")))
            self.kv("temporal evidence", _pct(analysis.get("temporal_score")))
            held = int(analysis.get("frames_in_state") or 0)
            if held:
                # Only shown when the counter is meaningful. At the instant of
                # the transition it has just been reset, so printing "0 frames
                # held" would contradict the duration quoted in the reason below.
                self.kv("verification window", f"{held} frame(s) held over "
                                               f"{analysis.get('seconds_in_state', 0)}s "
                                               f"(multi-frame evidence required)")
            self.kv("evidence / signals used", _signals(analysis))
            self.kv("detected vehicles", analysis.get("vehicles_involved", 0))
            self.kv("collision candidate", _collision(analysis.get("collision")))
            negative = analysis.get("negative_evidence") or []
            self.kv("negative evidence", ", ".join(negative) if negative else "none")
            self.kv("why confirmed", event.get("reason") or "-")
            return

        if state == "FALSE_ALARM":
            self.mark("ANALYZING ACCIDENT", "done")
            self.mark("VERIFICATION", "done")
            self.mark("ACCIDENT CONFIRMED", "refused")
            self.rule("FALSE ALARM - no notification")
            self.kv("result", "FALSE_ALARM - evidence did not hold")
            self.kv("accident confidence", _pct(analysis.get("score")))
            self.kv("blocking reasons", ", ".join(analysis.get("blocking_reasons") or []) or "-")
            self.kv("why rejected", event.get("reason") or "-")
            self.kv("notification", "SUPPRESSED (correct: not CONFIRMED_ACCIDENT)")
            return

        if state == "NORMAL":
            self.clear_progress()
            self._write(f"   [state] {previous} -> NORMAL : {event.get('reason') or '-'}")

    # ------------------------------------------------------------------ #
    # stage 5: incident created
    # ------------------------------------------------------------------ #
    def on_incident(self, incident: Dict[str, Any]) -> None:
        incident_id = str(incident.get("incident_id") or "")
        self.incidents.append(incident_id)
        self.mark("INCIDENT CREATED", "done")
        self.rule("INCIDENT CREATED")
        accident = incident.get("accident") or {}
        camera = incident.get("camera") or {}
        location = incident.get("location") or {}
        media = incident.get("media") or {}
        provenance = ((incident.get("extra") or {}).get("location_provenance") or {})

        self.kv("incident id", incident_id)
        self.kv("camera id", camera.get("camera_id") or "(unregistered)")
        self.kv("accident confidence", _pct(accident.get("score")))
        self.kv("severity", f"{accident.get('severity') or '-'} "
                            f"(confidence {_pct(accident.get('score'))})")
        self.kv("vehicles involved", (incident.get("objects") or {}).get("vehicles_involved", "-"))
        self.kv("detected at", media.get("detected_at") or incident.get("timestamp") or "-")
        self.kv("location", f"{location.get('name') or '(none)'}"
                            + (f" ({location.get('latitude')}, {location.get('longitude')})"
                               if location.get("latitude") is not None else ""))
        self.kv("location source", f"{provenance.get('source') or 'CAMERA_REGISTERED'} "
                                   f"- not a live GPS fix")
        self.kv("evidence dir", media.get("evidence_dir") or "(disabled)")

    # ------------------------------------------------------------------ #
    # stage 6-9: notification
    # ------------------------------------------------------------------ #
    def on_call_status(self, status: Dict[str, Any]) -> None:
        """Every call and SMS transition, printed as it is reported.

        A SIMULATED transition is tagged ``[SIM]`` and spelled out, because
        ``COMPLETED`` printed for a simulated call reads exactly like a delivered
        call. That ambiguity is what made a working demo look like a broken
        notification path.
        """
        if self.quiet:
            return
        self.clear_progress()
        state = str(status.get("status") or "")
        label = str(status.get("label") or "")
        recipient = status.get("recipient") or "(unconfigured)"
        is_sms = "SMS" in label.upper()
        simulated = self._is_simulated(status, label)
        tag = "[SIM]" if simulated else "[CALL]" if not is_sms else "[SMS]"

        if simulated:
            self.simulated = True

        if is_sms:
            if state == "SENT":
                # The body is composed before the provider is called, so by the
                # time SENT is reported both stages are genuinely done. Marking
                # only SENT left "SMS GENERATED" permanently pending on the fast
                # path, which never emits an intermediate PENDING.
                self.mark("SMS GENERATED", "done")
                self.mark("SMS SENT", "done")
                self._write(f"   {tag}   GENERATED + {state:<14} -> {recipient}")
                if simulated:
                    self._write("            ^^ SIMULATED - no SMS was sent to any real number")
            elif state == "PENDING":
                self.mark("SMS GENERATED", "done")
                self._write(f"   {tag}   {state:<18} -> {recipient} (waiting for the call)")
            else:
                self.mark("SMS SENT", "refused")
                reason = status.get("reason") or status.get("error") or ""
                self._write(f"   {tag}   {state:<18} -> {recipient}"
                            + (f"  ({reason})" if reason else ""))
        else:
            if state in ("INITIATED", "QUEUED"):
                self.mark("EMERGENCY CALL", "done")
                self._write(f"   {tag}  {state:<18} -> {recipient}")
            elif state in ("RINGING", "IN_PROGRESS"):
                self._write(f"   {tag}  {state:<18} -> {recipient}")
            elif state == "COMPLETED":
                self.mark("CALL COMPLETED", "done")
                self._write(f"   {tag}  {state:<18} -> {recipient}")
                if simulated:
                    self._write("            ^^ SIMULATED - no phone call was placed")
            elif state in ("FAILED", "BUSY", "NO_ANSWER"):
                self.mark("CALL COMPLETED", "refused")
                reason = status.get("reason") or status.get("error") or ""
                self._write(f"   {tag}  {state:<18} -> {recipient}"
                            + (f"  ({reason})" if reason else ""))
            elif state in ("SKIPPED", "BLOCKED", "NOT_CONFIGURED"):
                self.mark("EMERGENCY CALL", "refused")
                reason = status.get("reason") or status.get("error") or ""
                self._write(f"   {tag}  {state:<18} -> {recipient}"
                            + (f"  ({reason})" if reason else ""))

        detail = status.get("reason") or status.get("error") or ""
        if detail and state not in ("INITIATED", "RINGING", "IN_PROGRESS"):
            self._write(f"          reason: {detail}")

    @staticmethod
    def _is_simulated(status: Dict[str, Any], label: str) -> bool:
        """Was this transition simulated rather than really dialled?"""
        if "DEMO" in label.upper():
            return True
        provider = str(status.get("provider") or "").lower()
        return provider in ("simulation", "none", "null", "")

    def simulated_banner(self) -> None:
        """Say up front that nothing will actually be sent."""
        self.clear_progress()
        self._write()
        self._write("!" * WIDTH)
        self._write("!!  SIMULATION MODE - DEMO_MODE is true")
        self._write("!!  The call and SMS below are SIMULATED. No phone call is placed")
        self._write("!!  and no SMS is sent to any real number. Nothing leaves this PC.")
        self._write("!!  For a REAL call/SMS set DEMO_MODE=false and fill in the")
        self._write("!!  TWILIO_* values in .env, then run with --no-demo-call.")
        self._write("!" * WIDTH)

    def show_sms(self, body: str, header: str = "SMS CONTENT") -> None:
        """Print the exact alert text that was generated."""
        self.clear_progress()
        self.rule(header)
        for line in str(body).splitlines():
            self._write(f"   | {line}")

    # ------------------------------------------------------------------ #
    # stage 10 + final summary
    # ------------------------------------------------------------------ #
    def on_report_saved(self, incidents_file: Optional[str],
                        call_log: Optional[str]) -> None:
        self.mark("INCIDENT REPORT", "done")
        self.rule("INCIDENT REPORT / LOG SAVED")
        self.kv("incidents json", incidents_file or "(not written)")
        self.kv("call log", call_log or "(not written)")

    def summary(self) -> None:
        """The ordered workflow with each stage marked - the whole point."""
        self.clear_progress()
        self.rule("COMPLETE WORKFLOW")
        for index, (name, description) in enumerate(STAGES, start=1):
            status = self.stage_status.get(name, "pending")
            if status == "done":
                mark = "[x]"
            elif status == "refused":
                mark = "[!]"
            elif status == "active":
                mark = "[>]"
            else:
                mark = "[ ]"
            self._write(f"   {index:2d}. {mark} {name:<22} {description}")
        if self.simulated:
            self._write()
            self._write("   REMINDER: the call and SMS above were SIMULATED. Nothing was")
            self._write("   sent to a real phone number. Set DEMO_MODE=false in .env and")
            self._write("   add the TWILIO_* credentials for real notification.")


# --------------------------------------------------------------------------- #
def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _signals(analysis: Dict[str, Any]) -> str:
    parts: List[str] = []
    count = analysis.get("agreement_signals")
    if count is not None:
        parts.append(f"{count} agreeing signal(s)")
    if analysis.get("evidence_sufficient"):
        parts.append("evidence sufficient")
    components = analysis.get("components") or {}
    for key, value in sorted(components.items()):
        parts.append(f"{key} {_pct(value)}")
    return ", ".join(parts) if parts else "-"


def _collision(collision: Any) -> str:
    if not collision:
        return "none"
    classes = [c for c in (collision.get("classes") or []) if c]
    label = " + ".join(classes) if classes else "objects"
    return (f"tracks {collision.get('tracks')} ({label}) "
            f"score {_pct(collision.get('score'))}")