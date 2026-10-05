"""Visual debug overlay (Phase 19).

Pure OpenCV drawing on a copy of the frame - no GUI toolkit, no window
management, works headless when only ``--save-output`` is used::

    CAM-001                              STATE: VERIFYING
    ACCIDENT SCORE 0.84  (84%)

    Vehicle #12  car     0.91
    Vehicle #17  car     0.88
    Person  #21  person  0.74

    Collision 0.88   Motion 0.81   Trajectory 0.79
    Temporal 0.92    Objects 0.70  Scene  0.45

    [#####-----] verifying 0.62   evidence 5/8 frames

    12.4 FPS | infer 41ms | track 3ms | analysis 2ms | 640x360
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_engine.config.classes import VisualizationConfig
from ai_engine.schemas.enums import DetectionState, SeverityLevel
from ai_engine.utils.logging_utils import get_logger

__all__ = ["STATE_COLORS", "SEVERITY_COLORS", "OverlayRenderer"]

_LOGGER = get_logger("visualization")

#: BGR colours per verification state
STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    DetectionState.NORMAL.value: (80, 200, 80),
    DetectionState.SUSPICIOUS.value: (0, 170, 255),
    DetectionState.VERIFYING.value: (0, 210, 255),
    DetectionState.CONFIRMED_ACCIDENT.value: (40, 40, 235),
    DetectionState.FALSE_ALARM.value: (150, 150, 150),
}

SEVERITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    SeverityLevel.LOW.value: (120, 200, 120),
    SeverityLevel.MEDIUM.value: (0, 200, 220),
    SeverityLevel.HIGH.value: (0, 140, 255),
    SeverityLevel.CRITICAL.value: (30, 30, 220),
}

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


class OverlayRenderer:
    """Draws boxes, trails, collision links and the HUD onto a frame."""

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()

    # ------------------------------------------------------------------ #
    def render(self, frame: np.ndarray, frame_result: Any) -> np.ndarray:
        """Return an annotated copy of ``frame`` (the input is never modified)."""
        cfg = self.config
        if frame is None:
            return frame
        canvas = frame.copy()
        if not cfg.enabled:
            return canvas

        state = str(getattr(getattr(frame_result, "verification", None), "state", DetectionState.NORMAL))
        color = STATE_COLORS.get(state, (200, 200, 200))

        if cfg.show_trails:
            self._draw_trails(canvas, frame_result, color)
        if cfg.show_boxes:
            self._draw_boxes(canvas, frame_result, color)
        if cfg.show_collision:
            self._draw_collision(canvas, frame_result)
        if cfg.show_hud:
            self._draw_hud(canvas, frame_result, state, color)
        if cfg.state_banner:
            self._draw_state_banner(canvas, frame_result, state, color)
        self._draw_demo_call_banner(canvas, frame_result)
        return canvas

    # ------------------------------------------------------------------ #
    def _draw_boxes(self, canvas: np.ndarray, result: Any, color: Tuple[int, int, int]) -> None:
        import cv2

        cfg = self.config
        for track in getattr(result, "tracks", []) or []:
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            p1 = (max(0, x1), max(0, y1))
            p2 = (min(canvas.shape[1], x2), min(canvas.shape[0], y2))
            if p2[0] <= p1[0] or p2[1] <= p1[1]:
                continue
            track_color = color
            if track.is_person:
                track_color = (220, 120, 60)
            elif not track.is_vehicle:
                track_color = (180, 180, 180)
            cv2.rectangle(canvas, p1, p2, track_color, cfg.line_thickness)

            if cfg.show_labels:
                speed = f" v{track.speed_norm:.2f}"
                label = f"{_display_class(track)} #{track.track_id} {track.confidence:.2f}{speed}"
                _label(canvas, p1, label, track_color, cfg.font_scale)

    def _draw_trails(self, canvas: np.ndarray, result: Any, color: Tuple[int, int, int]) -> None:
        import cv2

        length = max(2, int(self.config.trail_length))
        for track in getattr(result, "tracks", []) or []:
            points = track.recent_points(length)
            if len(points) < 2:
                continue
            trail_color = (60, 120, 240) if track.is_person else color
            polyline = [(int(p.center[0]), int(p.center[1])) for p in points]
            for i in range(1, len(polyline)):
                # fade the tail so the direction of travel is obvious
                alpha = i / max(1, len(polyline) - 1)
                thickness = 1 if alpha < 0.5 else 2
                cv2.line(canvas, polyline[i - 1], polyline[i], trail_color, thickness, cv2.LINE_AA)
            head = polyline[-1]
            cv2.circle(canvas, head, 3, trail_color, -1, cv2.LINE_AA)

    def _draw_collision(self, canvas: np.ndarray, result: Any) -> None:
        import cv2

        event = getattr(result, "collision", None)
        if event is None:
            return
        a = event_point = None
        track_a = _find_track(result, event.track_a)
        track_b = _find_track(result, event.track_b)
        if track_a is not None:
            a = (int(track_a.center[0]), int(track_a.center[1]))
        if track_b is not None:
            event_point = (int(track_b.center[0]), int(track_b.center[1]))
        if a and event_point:
            cv2.line(canvas, a, event_point, (0, 0, 255), 2, cv2.LINE_AA)
            mid = ((a[0] + event_point[0]) // 2, (a[1] + event_point[1]) // 2)
            cv2.circle(canvas, mid, 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"COLLISION {event.score:.2f}",
                (mid[0] + 12, mid[1] - 6),
                _FONT,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            label = f"#{event.track_a} <-> #{event.track_b}  {event.agreement} signals"
            _label(canvas, (min(mid[0], canvas.shape[1] - 190), mid[1] + 16), label, (0, 0, 255), 0.4)

    # ------------------------------------------------------------------ #
    def _draw_hud(self, canvas: np.ndarray, result: Any, state: str, color: Tuple[int, int, int]) -> None:
        import cv2

        cfg = self.config
        width = canvas.shape[1]
        lines = self._hud_lines(result, state)
        if cfg.show_perf:
            lines.extend(self._perf_lines(result))
        if not lines:
            return

        line_height = 18
        panel_height = line_height * len(lines) + 16
        panel_width = min(width - 8, 320)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (4, 4), (4 + panel_width, 4 + panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, cfg.hud_alpha, canvas, 1.0 - cfg.hud_alpha, 0, canvas)
        cv2.rectangle(canvas, (4, 4), (4 + panel_width, 4 + panel_height), color, 1)

        y = 22
        for text, text_color, bold in lines:
            cv2.putText(
                canvas,
                text,
                (12, y),
                _FONT,
                cfg.font_scale * (1.0 if bold else 0.85),
                text_color,
                2 if bold else 1,
                cv2.LINE_AA,
            )
            y += line_height

    def _draw_state_banner(self, canvas: np.ndarray, result: Any, state: str, color: Tuple[int, int, int]) -> None:
        import cv2

        height, width = canvas.shape[:2]
        text = state.replace("_", " ")
        scale = 0.7
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 2)
        x = max(4, width - tw - 14)
        y = 34
        cv2.rectangle(canvas, (x - 8, y - th - 8), (width - 4, y + 8), color, -1)
        cv2.putText(canvas, text, (x, y), _FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)

        severity = str(getattr(getattr(result, "severity", None), "level", "")) if getattr(result, "severity", None) else ""
        if severity and state == DetectionState.CONFIRMED_ACCIDENT.value:
            sev_color = SEVERITY_COLORS.get(severity, (200, 200, 200))
            (sw, sh), _ = cv2.getTextSize(f"SEVERITY {severity}", _FONT, 0.5, 2)
            sx = max(4, width - sw - 14)
            sy = 34 + 26
            cv2.rectangle(canvas, (sx - 8, sy - sh - 8), (width - 4, sy + 8), sev_color, -1)
            cv2.putText(canvas, f"SEVERITY {severity}", (sx, sy), _FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_demo_call_banner(self, canvas: np.ndarray, result: Any) -> None:
        """Show ``DEMO EMERGENCY CALL`` and its status (demo mode only).

        The banner is deliberately loud and permanent for the whole call: a
        demonstration that rings a phone must never be mistakable for a real
        emergency notification.
        """
        import cv2

        call = getattr(result, "demo_call", None) or {}
        if not call.get("demo_mode"):
            return
        sms = call.get("sms") or {}
        status = str(call.get("status") or "IDLE")
        recipient = call.get("recipient") or "(unconfigured)"
        sms_status = str(sms.get("status") or "")
        if status in ("SKIPPED", "BLOCKED"):
            headline = f"DEMO EMERGENCY CALL - {status}"
            detail = str(call.get("reason") or "")[:64]
            color = (0, 165, 255)  # amber: a safety interlock refused
        else:
            headline = f"DEMO EMERGENCY CALL - {status}"
            detail = f"{recipient} | {call.get('provider', '?')}"
            if sms_status and sms_status != "PENDING":
                detail += f" | SMS {sms_status}"
            color = (0, 40, 235) if status != "COMPLETED" else (40, 40, 235)

        height, width = canvas.shape[:2]
        scale = 0.62
        (tw, th), _ = cv2.getTextSize(headline, _FONT, scale, 2)
        box_w = min(width - 8, max(tw + 24, 260))
        box_h = th + 34 if detail else th + 22
        x, y = 4, height - box_h - 4
        if y < 40:  # tiny frame - keep it on screen
            y = 40
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), color, -1)
        cv2.putText(canvas, headline, (x + 12, y + th + 8), _FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
        if detail:
            cv2.putText(
                canvas, detail, (x + 12, y + th + 26), _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hud_lines(result: Any, state: str) -> List[Tuple[str, Tuple[int, int, int], bool]]:
        import cv2

        color = STATE_COLORS.get(state, (220, 220, 220))
        score = float(getattr(result, "accident_score", 0.0))
        temporal = float(getattr(result, "temporal_score", 0.0))
        camera_id = getattr(result, "camera_id", None) or "UNREGISTERED"
        lines: List[Tuple[str, Tuple[int, int, int], bool]] = [
            (f"CAMERA {camera_id}", (235, 235, 235), False),
            (f"STATE: {state}", color, True),
            (f"ACCIDENT SCORE {score:.2f}  ({int(round(score * 100))}%)", (255, 255, 255), True),
        ]

        components: Dict[str, float] = dict(getattr(result, "component_scores", {}) or {})
        if components:
            row1 = "  ".join(
                f"{_short(name)} {components[name]:.2f}"
                for name in ("collision", "motion", "trajectory", "temporal")
                if name in components
            )
            lines.append((row1, (200, 220, 255), False))
            row2 = "  ".join(
                f"{_short(name)} {components[name]:.2f}"
                for name in ("object_evidence", "scene_evidence")
                if name in components
            )
            if row2:
                lines.append((row2, (200, 220, 255), False))

        verification = getattr(result, "verification", None)
        if verification is not None:
            progress = float(getattr(verification, "progress", 0.0))
            consecutive = int(getattr(verification, "consecutive_above", 0))
            bar = _progress_bar(progress, 12)
            if state == DetectionState.VERIFYING.value:
                lines.append((f"[{bar}] verifying {progress:.2f}", (0, 230, 255), False))
            elif progress > 0.0:
                lines.append((f"[{bar}] evidence {progress:.2f}", (0, 230, 255), False))
            lines.append((f"consecutive supporting frames: {consecutive}", (190, 190, 190), False))
            blocking = list(getattr(verification, "blocking_reasons", []) or [])
            for reason in blocking[:2]:
                lines.append((f"  holding: {reason[:44]}", (120, 190, 255), False))
            negative = list(getattr(verification, "negative_evidence", []) or [])
            if negative:
                lines.append((f"  negative: {negative[0][:44]}", (150, 150, 200), False))

        collision = getattr(result, "collision", None)
        if collision is not None:
            lines.append((f"collision: #{collision.track_a} <-> #{collision.track_b} ({collision.agreement} signals)", (0, 120, 255), False))

        call = getattr(result, "demo_call", None) or {}
        if call.get("demo_mode"):
            status = str(call.get("status") or "IDLE")
            recipient = call.get("recipient") or "unconfigured"
            if status in ("SKIPPED", "BLOCKED"):
                lines.append((f"DEMO EMERGENCY CALL {status} ({recipient})", (0, 165, 255), False))
            else:
                lines.append((f"DEMO EMERGENCY CALL {status} -> {recipient}", (60, 60, 255), True))
            sms_status = str((call.get("sms") or {}).get("status") or "")
            if sms_status and sms_status != "PENDING":
                lines.append((f"DEMO SMS {sms_status}", (60, 60, 255), True))

        return lines

    @staticmethod
    def _perf_lines(result: Any) -> List[Tuple[str, Tuple[int, int, int], bool]]:
        perf = getattr(result, "performance", None)
        if perf is None:
            return []
        payload = perf.to_dict() if hasattr(perf, "to_dict") else {}
        fps = payload.get("fps", 0.0)
        line = (
            f"{fps:5.1f} FPS | infer {payload.get('inference_ms', 0):.0f}ms | "
            f"track {payload.get('tracking_ms', 0):.0f}ms | analysis {payload.get('analysis_ms', 0):.0f}ms"
        )
        size = payload.get("frame_size") or ""
        return [(line, (170, 230, 170), False)] + ([(str(size), (140, 180, 140), False)] if size else [])


# --------------------------------------------------------------------------- #
def _display_class(track: Any) -> str:
    return "Vehicle" if getattr(track, "is_vehicle", False) else _title(getattr(track, "normalized_name", ""))


def _title(text: str) -> str:
    return str(text).replace("_", " ").title() if text else "Object"


def _short(name: str) -> str:
    return {
        "collision": "Col",
        "motion": "Mot",
        "trajectory": "Traj",
        "temporal": "Temp",
        "object_evidence": "Obj",
        "scene_evidence": "Scene",
    }.get(name, name[:4])


def _progress_bar(value: float, width: int = 12) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "-" * (width - filled)


def _find_track(result: Any, track_id: int) -> Optional[Any]:
    for track in getattr(result, "tracks", []) or []:
        if track.track_id == track_id:
            return track
    return None


def _label(canvas: np.ndarray, anchor: Tuple[int, int], text: str, color: Tuple[int, int, int], scale: float) -> None:
    """Small filled label above a box."""
    import cv2

    (tw, th), baseline = cv2.getTextSize(text, _FONT, max(0.3, scale * 0.8), 1)
    x = max(0, min(anchor[0], canvas.shape[1] - tw - 4))
    y = max(th + 2, anchor[1] - 4)
    cv2.rectangle(canvas, (x, y - th - 2), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(canvas, text, (x + 2, y), _FONT, max(0.3, scale * 0.8), (255, 255, 255), 1, cv2.LINE_AA)
