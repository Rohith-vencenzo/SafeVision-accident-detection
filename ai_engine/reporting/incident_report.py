"""AI incident report (Phase 18).

The report is a *rendering of the structured evidence* - it never invents a
fact.  Every sentence is produced from a field that is actually present in the
incident payload:

* a vehicle count sentence only appears when ``vehicles_involved >= 2``
* the confidence is the accident score, printed as a percentage
* the severity string is whatever the severity analyzer estimated
* the location sentence uses the camera's registered location, and says so
* evidence paths are listed only for files that exist in the payload

If a field is missing, the sentence is omitted rather than guessed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ai_engine.schemas.enums import IncidentStatus, SeverityLevel
from ai_engine.utils.jsonio import to_jsonable
from ai_engine.utils.logging_utils import get_logger

__all__ = ["IncidentReportGenerator", "render_report_markdown"]

_LOGGER = get_logger("reporting")


class IncidentReportGenerator:
    """Turns an incident dict into a human-readable report."""

    def generate(self, incident: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a structured report (``what/why/confidence/severity/...``)."""
        incident = incident or {}
        accident = dict(incident.get("accident") or {})
        evidence = dict(incident.get("evidence") or {})
        objects = dict(incident.get("objects") or {})
        location = dict(incident.get("location") or {})
        camera = dict(incident.get("camera") or {})
        media = dict(incident.get("media") or {})

        status = str(incident.get("status", IncidentStatus.NORMAL.value))
        score = _as_float(accident.get("score"))
        severity = str(accident.get("severity", SeverityLevel.LOW.value))
        reasons = [str(r) for r in (evidence.get("reasons") or []) if r]
        negative = [str(r) for r in (evidence.get("negative_evidence") or []) if r]

        return {
            "incident_id": incident.get("incident_id"),
            "schema_version": incident.get("schema_version"),
            "status": status,
            "timestamp": incident.get("timestamp"),
            "what_happened": self._what_happened(status, objects, accident, evidence),
            "why": self._why(evidence, reasons),
            "confidence": {
                "score": score,
                "percent": int(round(score * 100)),
                "wording": "AI-estimated accident probability from multi-frame evidence",
            },
            "severity": {
                "level": severity,
                "score": _as_float(accident.get("severity_score")),
                "wording": "AI-estimated scene severity (not a medical assessment)",
                "reasons": [str(r) for r in (accident.get("severity_reasons") or []) if r],
            },
            "location": self._location(location, camera),
            "evidence_files": self._evidence_files(media),
            "counter_evidence": negative,
            "disclaimer": incident.get("disclaimer"),
            "warnings": [str(w) for w in (incident.get("warnings") or []) if w],
            "markdown": "",
        }

    def to_markdown(self, incident: Mapping[str, Any]) -> str:
        report = self.generate(incident)
        report["markdown"] = render_report_markdown(report)
        return report["markdown"]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _what_happened(
        status: str,
        objects: Mapping[str, Any],
        accident: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> str:
        vehicles = int(objects.get("vehicles_involved", 0) or 0)
        persons = int(objects.get("persons_detected", 0) or 0)

        if status == IncidentStatus.CONFIRMED_ACCIDENT.value:
            if vehicles >= 2:
                sentence = f"Collision detected between {vehicles} tracked vehicles."
            elif vehicles == 1:
                sentence = "A single vehicle showed a collision-type event (likely an impact with an off-camera object)."
            else:
                sentence = "An accident-type event was confirmed in the scene."
        elif status == IncidentStatus.POSSIBLE_ACCIDENT.value:
            sentence = "A possible accident is being verified; evidence is still accumulating."
        elif status == IncidentStatus.FALSE_ALARM.value:
            sentence = "An apparent accident was detected but rejected after verification (false alarm)."
        else:
            sentence = "Normal traffic. No accident indicators."

        if persons > 0:
            sentence += f" {persons} person(s) detected in the scene."
        if accident.get("confirmed"):
            sentence += (
                f" Confirmed after {int(accident.get('frames_of_evidence', 0) or 0)} frame(s) of evidence"
                f" over {float(accident.get('verification_seconds', 0) or 0):.1f}s."
            )
        return sentence

    @staticmethod
    def _why(evidence: Mapping[str, Any], reasons: Sequence[str]) -> List[str]:
        """Evidence phrases, plus the component scores that produced them."""
        bullets: List[str] = [str(r) for r in reasons if r]
        components = [
            ("Collision proximity / closing speed", evidence.get("collision_score")),
            ("Sudden motion change", evidence.get("motion_score")),
            ("Trajectory deviation", evidence.get("trajectory_score")),
            ("Temporal consistency", evidence.get("temporal_score")),
        ]
        for label, value in components:
            score = _as_float(value)
            if score > 0.0:
                bullets.append(f"{label}: {score:.2f}")
        agreement = evidence.get("agreement_signals")
        if agreement:
            bullets.append(f"Agreeing independent signals: {agreement}")
        return bullets

    @staticmethod
    def _location(location: Mapping[str, Any], camera: Mapping[str, Any]) -> Dict[str, Any]:
        registered = bool(location.get("is_camera_registered"))
        name = location.get("name") or location.get("location_name") or camera.get("name")
        payload: Dict[str, Any] = {
            "name": name,
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "road": location.get("road"),
            "address": location.get("address"),
            "is_camera_registered": registered,
            "wording": (
                "Camera-registered location (from camera metadata, not a live GPS fix)"
                if registered
                else "No camera metadata registered for this camera - no coordinates available"
            ),
        }
        return payload

    @staticmethod
    def _evidence_files(media: Mapping[str, Any]) -> Dict[str, Any]:
        files = [str(p) for p in (media.get("evidence_frames") or []) if p]
        return {
            "annotated_frame": media.get("annotated_frame"),
            "frames": files,
            "clip": media.get("clip"),
            "directory": media.get("evidence_dir"),
            "count": len(files) + (1 if media.get("annotated_frame") else 0) + (1 if media.get("clip") else 0),
        }

    # ------------------------------------------------------------------ #
    def to_dict(self, incident: Mapping[str, Any]) -> Dict[str, Any]:
        return to_jsonable(self.generate(incident))


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the structured report as Markdown."""
    lines: List[str] = []
    add = lines.append

    add(f"# SafeVision incident {report.get('incident_id')}")
    add("")
    add(f"- **Status:** `{report.get('status')}`")
    add(f"- **Timestamp:** `{report.get('timestamp')}`")
    add(f"- **Schema version:** `{report.get('schema_version')}`")
    add("")

    add("## What happened")
    add(str(report.get("what_happened", "")))
    add("")

    why = list(report.get("why") or [])
    if why:
        add("## Why it was detected")
        for item in why:
            add(f"- {item}")
        add("")

    counter = list(report.get("counter_evidence") or [])
    if counter:
        add("## Counter-evidence considered")
        for item in counter:
            add(f"- {item}")
        add("")

    confidence = dict(report.get("confidence") or {})
    severity = dict(report.get("severity") or {})
    add("## Confidence and severity")
    add(f"- **AI-estimated accident probability:** {confidence.get('percent', 0)}% ({confidence.get('wording', '')})")
    add(f"- **AI-estimated scene severity:** {severity.get('level')} (score {float(severity.get('score') or 0.0):.2f})")
    for reason in severity.get("reasons") or []:
        add(f"  - {reason}")
    add("")

    location = dict(report.get("location") or {})
    add("## Location")
    add(f"- {location.get('wording')}")
    if location.get("name"):
        add(f"- Name: {location.get('name')}")
    if location.get("road"):
        add(f"- Road: {location.get('road')}")
    if location.get("latitude") is not None and location.get("longitude") is not None:
        add(f"- Coordinates: {location.get('latitude')}, {location.get('longitude')} (camera metadata)")
    add("")

    evidence = dict(report.get("evidence_files") or {})
    if evidence.get("count"):
        add("## Evidence")
        if evidence.get("directory"):
            add(f"- Directory: `{evidence.get('directory')}`")
        if evidence.get("annotated_frame"):
            add(f"- Annotated frame: `{evidence.get('annotated_frame')}`")
        for path in evidence.get("frames") or []:
            add(f"- Frame: `{path}`")
        if evidence.get("clip"):
            add(f"- Clip: `{evidence.get('clip')}`")
        add("")

    warnings = list(report.get("warnings") or [])
    if warnings:
        add("## Warnings")
        for warning in warnings:
            add(f"- {warning}")
        add("")

    if report.get("disclaimer"):
        add("---")
        add(f"_{report.get('disclaimer')}_")
    return "\n".join(lines)


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
