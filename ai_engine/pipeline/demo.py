"""Scripted presentation mode (Phase 20).

Runs a fixed set of clips through the real pipeline and prints the state
transitions as they happen, so the false-alarm prevention mechanism can be
*shown* rather than described::

    ==========================================================================
     SCENARIO 2/3  accident clip           source: test_videos/accident.mp4
    ==========================================================================
      frame  0  STATE  NORMAL               score 0.00
      frame 34  STATE  SUSPICIOUS           score 0.31   [physical evidence 0.37]
      frame 46  STATE  VERIFYING            score 0.45   [evidence accumulating]
      frame 61  STATE  CONFIRMED_ACCIDENT   score 0.49   [severity MEDIUM]
      incident INC-20260928-0001 written to incidents.json
    ==========================================================================

Every number printed here comes from the pipeline; nothing is scripted.  If a
clip does not produce an incident, the demo says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_engine.config.classes import Config
from ai_engine.pipeline.accident_pipeline import AccidentPipeline
from ai_engine.utils.jsonio import atomic_write_json, to_jsonable
from ai_engine.utils.logging_utils import get_logger

__all__ = ["DemoRunner", "DemoResult"]

_LOGGER = get_logger("demo")

_BANNER_WIDTH = 78


@dataclass
class DemoResult:
    """Outcome of one demo scenario."""

    label: str
    source: str
    available: bool = True
    frames: int = 0
    peak_score: float = 0.0
    final_state: str = "NORMAL"
    states_seen: List[str] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    false_alarms: int = 0
    elapsed: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "label": self.label,
                "source": self.source,
                "available": self.available,
                "frames": self.frames,
                "peak_score": round(self.peak_score, 4),
                "final_state": self.final_state,
                "states_seen": self.states_seen,
                "transitions": self.transitions,
                "incident_count": len(self.incidents),
                "incidents": self.incidents,
                "false_alarms": self.false_alarms,
                "elapsed_seconds": round(self.elapsed, 2),
                "note": self.note,
            }
        )


class DemoRunner:
    """Runs the demo scenarios and prints an annotated transcript."""

    def __init__(self, config: Config, camera_id: Optional[str] = None) -> None:
        self.config = config
        self.camera_id = camera_id or config.demo.camera_id or config.location.default_camera_id
        self.results: List[DemoResult] = []

    # ------------------------------------------------------------------ #
    def scenarios(self) -> List[Tuple[str, str]]:
        """``[(label, source), ...]``.

        A source prefixed with ``scenario:`` runs a **scripted** scenario from
        :mod:`ai_engine.detection.scenarios` instead of a video file.  Those
        feed pre-computed detections into the pipeline, so they demonstrate the
        *decision logic* end to end on a machine with no accident footage - and
        they are labelled as such wherever they are printed.  Missing video
        files are reported, not fatal.
        """
        demo = self.config.demo
        raw: List[Tuple[str, str]] = [
            ("scripted crash (synthetic detections)", "scenario:head_on_crash"),
            ("scripted false-alarm trap (synthetic detections)", "scenario:hard_brake"),
            ("normal traffic", demo.scenarios.get("normal", demo.normal_source)),
            ("accident clip", demo.scenarios.get("accident", demo.accident_source)),
        ]
        if demo.suspicious_source:
            raw.append(("suspicious / false-alarm clip", demo.scenarios.get("suspicious", demo.suspicious_source)))
        for label, name in demo.scenarios.items():
            if label not in ("normal", "accident", "suspicious"):
                raw.append((f"{label} (scripted)", f"scenario:{name}"))
        return raw

    def run(self, max_frames: int = 0) -> List[DemoResult]:
        self.results = []
        scenarios = self.scenarios()
        for index, (label, source) in enumerate(scenarios, start=1):
            self._header(index, len(scenarios), label, source)
            if source.startswith("scenario:"):
                self.results.append(self._run_scenario(label, source.split(":", 1)[1], max_frames))
            else:
                self.results.append(self._run_one(label, source, max_frames))
        self._summary()
        return self.results

    # ------------------------------------------------------------------ #
    def _run_scenario(self, label: str, name: str, max_frames: int) -> DemoResult:
        """Run a scripted scenario (synthetic detections, no video needed)."""
        import numpy as np

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector

        result = DemoResult(label=label, source=f"scenario:{name}")
        if name not in SCENARIOS:
            result.available = False
            result.note = f"unknown scripted scenario {name!r}"
            print(f"  [ skipped ] {result.note}")
            return result

        config = self.config
        pipeline = AccidentPipeline(
            config,
            detector=ScriptedDetector(SCENARIOS[name]),
            camera_id=self.camera_id,
            source_label=f"scenario:{name}",
            on_call_status=self._print_call_status,
        )
        started = time.perf_counter()
        every = max(1, int(config.demo.print_every))
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # a wall-clock base, so the incident timestamps in the JSON are real
        base = time.time()
        for index in range(max_frames or 90):
            result_ = pipeline.process_frame(frame, index, base + index / 15.0)
            result.frames += 1
            state = str(result_.state)
            if state not in result.states_seen:
                result.states_seen.append(state)
            if index % every == 0:
                print(
                    f"    frame {index:>4}  STATE {state:<18} score {result_.accident_score:.2f}"
                    f"   [physical {result_.verification.physical_score:.2f}, "
                    f"temporal {result_.temporal.temporal_score:.2f}]"
                )
            if result_.verification.changed:
                transition = result_.verification.transitions[-1]
                result.transitions.append(transition.to_dict())
                print(f"    frame {index:>4}  >> {transition.from_state} -> {transition.to_state}   ({transition.reason})")
            if result_.incident:
                result.incidents.append(result_.incident)

        result.elapsed = time.perf_counter() - started
        result.peak_score = max((float(i["accident"]["score"]) for i in result.incidents), default=0.0)
        result.final_state = str(pipeline.state)
        result.false_alarms = pipeline.verification.false_alarm_count
        pipeline.close()

        if result.incidents:
            incident = result.incidents[0]
            print(
                f"    incident {incident['incident_id']}  severity {incident['accident']['severity']}  "
                f"score {incident['accident']['score']:.2f}  "
                f"vehicles {incident['objects']['vehicles_involved']}  "
                f"signals {incident['evidence']['agreement_signals']}"
            )
            call = incident.get("demo_emergency_call") or {}
            if call.get("attempted"):
                print(
                    f"    {call.get('label')} {call.get('status')} -> "
                    f"{call.get('recipient')} ({call.get('trigger')}, "
                    f"{call.get('provider')} provider)"
                )
            elif call.get("demo_mode") or call.get("reason"):
                print(f"    no demo call: {call.get('reason')}")
        else:
            print("    no confirmed incident (this is the expected result for this scenario)")
        print(f"    {result.frames} frames in {result.elapsed:.1f}s | false alarms: {result.false_alarms}")
        return result

    # ------------------------------------------------------------------ #
    def _run_one(self, label: str, source: str, max_frames: int) -> DemoResult:
        path = Path(source)
        result = DemoResult(label=label, source=source)

        if not path.is_absolute() and not path.exists():
            pass  # resolved relative to the project root below
        resolved = path if path.exists() else (Path.cwd() / path)
        if not resolved.exists():
            result.available = False
            result.note = (
                f"clip not found: {source}. Drop a video at that path (or set "
                f"demo.scenarios in config.yaml) to include it in the demo."
            )
            print(f"  [ skipped ] {result.note}")
            return result

        config = self.config
        pipeline = AccidentPipeline(
            config,
            camera_id=self.camera_id,
            source_label=str(resolved),
            on_call_status=self._print_call_status,
        )
        started = time.perf_counter()
        every = max(1, int(config.demo.print_every))

        def on_frame(frame_result: Any) -> None:
            state = str(frame_result.state)
            if state not in result.states_seen:
                result.states_seen.append(state)
            if frame_result.frame_index % every == 0:
                print(
                    f"    frame {frame_result.frame_index:>4}  "
                    f"STATE {state:<18} score {frame_result.accident_score:.2f}"
                    f"   [physical {frame_result.verification.physical_score:.2f}, "
                    f"temporal {frame_result.temporal.temporal_score:.2f}]"
                )
            for transition in frame_result.verification.transitions[-1:] if frame_result.verification.changed else []:
                entry = transition.to_dict()
                result.transitions.append(entry)
                print(
                    f"    frame {frame_result.frame_index:>4}  >> {entry['from']} -> {entry['to']}"
                    f"   ({entry['reason']})"
                )

        run = pipeline.process_video(str(resolved), max_frames=max_frames, on_frame=on_frame)
        elapsed = time.perf_counter() - started

        result.frames = run.frames_processed
        result.peak_score = max((float(i["accident"]["score"]) for i in run.incidents), default=0.0)
        result.final_state = str(pipeline.state)
        result.incidents = run.incidents
        result.false_alarms = run.false_alarms
        result.elapsed = elapsed
        pipeline.close()

        if run.incidents:
            incident = run.incidents[0]
            print(
                f"    incident {incident['incident_id']}  "
                f"severity {incident['accident']['severity']}  "
                f"score {incident['accident']['score']:.2f}  "
                f"vehicles {incident['objects']['vehicles_involved']}  "
                f"location {incident['location'].get('name')}"
            )
            call = incident.get("demo_emergency_call") or {}
            if call.get("attempted"):
                print(
                    f"    {call.get('label')} {call.get('status')} -> "
                    f"{call.get('recipient')} ({call.get('provider')} provider)"
                )
            elif call.get("reason"):
                print(f"    no demo call: {call.get('reason')}")
        else:
            print("    no confirmed incident in this clip (this is a valid result)")
        print(f"    {result.frames} frames in {elapsed:.1f}s | false alarms: {result.false_alarms}")
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_call_status(status: Dict[str, Any]) -> None:
        """Show the DEMO EMERGENCY CALL state machine during the demo.

        Automatic and one-call-per-incident; there is no button in this code.
        """
        label = status.get("label", "DEMO EMERGENCY CALL")
        recipient = status.get("recipient") or "(unconfigured)"
        detail = status.get("reason") or status.get("provider") or ""
        print(f"    >> {label} - {status.get('status')} | {recipient} | {detail}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _header(index: int, total: int, label: str, source: str) -> None:
        print("=" * _BANNER_WIDTH)
        print(f" SCENARIO {index}/{total}  {label:<40} source: {source}")
        if source.startswith("scenario:"):
            print(
                " NOTE: scripted detections (not YOLO output). This demonstrates the"
            )
            print(
                "       decision logic - tracking, motion, collision, temporal fusion,"
            )
            print("       verification and the incident JSON - not detector accuracy.")
        print("=" * _BANNER_WIDTH)

    def _summary(self) -> None:
        print()
        print("=" * _BANNER_WIDTH)
        print(" DEMO SUMMARY")
        print("=" * _BANNER_WIDTH)
        for result in self.results:
            if not result.available:
                print(f"  {result.label:<32} skipped (clip not available)")
                continue
            states = " -> ".join(result.states_seen) or "NORMAL"
            print(
                f"  {result.label:<32} incidents: {len(result.incidents)}  "
                f"false alarms: {result.false_alarms}  states: {states}"
            )
        print()
        print("  NORMAL -> SUSPICIOUS -> VERIFYING -> CONFIRMED_ACCIDENT")
        print("                          |")
        print("                          +-> FALSE_ALARM -> NORMAL")
        print("  (only multi-frame, multi-signal evidence reaches CONFIRMED_ACCIDENT)")
        if any((r.incidents[0].get("demo_emergency_call", {}).get("attempted") if r.incidents else False)
               for r in self.results):
            print()
            print("  DEMO EMERGENCY CALL: automatic, one call per CONFIRMED incident,")
            print("  shown above as INITIATED -> RINGING -> IN_PROGRESS -> COMPLETED.")
            print("  It is a demonstration, not an emergency-service notification.")

    # ------------------------------------------------------------------ #
    def write_report(self) -> Optional[str]:
        """Persist the transcript summary as JSON (config.demo.report_file)."""
        if not self.config.demo.write_report:
            return None
        try:
            payload = {
                "note": "Demo transcript produced by SafeVision; values are measured, not scripted.",
                "scenarios": [r.to_dict() for r in self.results],
            }
            return atomic_write_json(self.config.demo.report_file, payload)
        except OSError as exc:
            _LOGGER.error("could not write the demo report: %s", exc)
            return None
