"""YOLO object detection (Phase 4).

Wraps an Ultralytics YOLO checkpoint and converts its output into SafeVision's
:class:`DetectionResult`.  The model is loaded **once** and kept resident
(Phase 21); per-frame work is a single ``predict`` call.

Error handling (Phase 24): a missing/broken model raises
:class:`ModelNotAvailableError` at load time (so the CLI can print a helpful
message), while a failing *inference* raises :class:`DetectionError` which the
pipeline catches per frame - one bad frame must not kill a live stream.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ai_engine.config.classes import ModelConfig
from ai_engine.detection.detection_types import (
    ACCIDENT_CLASS_ALIASES,
    ROAD_CLASS_NAMES,
    BBox,
    Detection,
    DetectionResult,
    normalize_class_name,
)
from ai_engine.utils.logging_utils import get_logger

__all__ = ["BaseDetector", "DetectionError", "ModelNotAvailableError", "YoloDetector", "resolve_device"]

_LOGGER = get_logger("detection.yolo")


class DetectionError(RuntimeError):
    """A single inference call failed."""


class ModelNotAvailableError(RuntimeError):
    """The detector weights could not be loaded (missing file / no network)."""


def resolve_device(requested: str) -> str:
    """Translate ``"auto"`` into a concrete torch device string."""
    wanted = (requested or "auto").strip().lower()
    if wanted != "auto":
        return wanted
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseDetector:
    """Minimal interface the pipeline depends on (also the test seam)."""

    model_name: str = "base"

    def detect(self, frame: Any, frame_index: int = 0, timestamp: float = 0.0) -> DetectionResult:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    def __enter__(self) -> "BaseDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class YoloDetector(BaseDetector):
    """Ultralytics YOLO wrapper producing :class:`DetectionResult` objects.

    Parameters
    ----------
    config:
        :class:`~ai_engine.config.classes.ModelConfig`.
    weights_override:
        CLI value for ``--model``; wins over ``config.weights``.
    """

    def __init__(self, config: Optional[ModelConfig] = None, weights_override: Optional[str] = None) -> None:
        self.config = config or ModelConfig()
        self._weights = weights_override or self.config.weights
        self._model: Any = None
        self._names: dict = {}
        self._class_ids: Optional[List[int]] = None
        self._accident_ids: Tuple[int, ...] = ()
        self._lock = threading.Lock()
        self.model_name = Path(self._weights).stem or str(self._weights)
        self._loaded = False

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> "YoloDetector":
        """Load the checkpoint (idempotent, thread-safe)."""
        if self._loaded:
            return self
        with self._lock:
            if self._loaded:
                return self
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ModelNotAvailableError(
                    "ultralytics is not installed. Run: pip install -r requirements.txt"
                ) from exc

            device = resolve_device(self.config.device)
            if self.config.torch_threads:
                try:
                    import torch

                    torch.set_num_threads(int(self.config.torch_threads))
                except Exception as exc:  # pragma: no cover
                    _LOGGER.warning("Could not set torch threads: %s", exc)

            if self.config.half and device in ("cpu", "mps"):
                _LOGGER.info("half precision disabled: %s does not support fp16 well", device)
                self.config.half = False

            try:
                model = YOLO(self._weights)
            except Exception as exc:
                # offline-first: a configured local path that has not been
                # downloaded yet falls back to the well-known checkpoint name
                # (ultralytics can resolve / download it), and only then fails.
                fallback = self._fallback_name()
                if fallback:
                    try:
                        _LOGGER.info("'%s' unavailable (%s); trying '%s'", self._weights, exc, fallback)
                        model = YOLO(fallback)
                        self._weights = fallback
                    except Exception:  # noqa: BLE001
                        raise ModelNotAvailableError(self._load_error_message(exc)) from exc
                else:
                    raise ModelNotAvailableError(self._load_error_message(exc)) from exc

            self._model = model
            self.model_name = Path(str(self._weights)).stem or str(self._weights)
            self._extract_vocabulary()
            self._loaded = True
            _LOGGER.info(
                "Loaded detector '%s' (device=%s, imgsz=%d, conf=%.2f, classes=%s)",
                self.model_name,
                device,
                self.config.imgsz,
                self.config.conf,
                self._class_ids if self._class_ids is not None else "road-scene filter",
            )
            return self

    def _fallback_name(self) -> Optional[str]:
        """Checkpoint name to try when the configured path is missing.

        ``models/yolo11n.pt`` -> ``yolo11n.pt`` (a name ultralytics knows how to
        download or find in its cache).  ``None`` when the configured value is
        already a bare name.
        """
        text = str(self._weights)
        if "/" not in text and "\\" not in text:
            return None
        name = Path(text).name
        return name if name.lower().endswith(".pt") and name != text else None

    def _load_error_message(self, exc: Exception) -> str:
        weights = str(self._weights)
        looks_like_path = ("/" in weights or "\\" in weights or weights.lower().endswith(".pt"))
        hint = ""
        if looks_like_path and not Path(weights).is_file():
            hint = (
                f"\nThe weights file '{weights}' does not exist. Either:\n"
                f"  * download it (e.g. yolo11n.pt) into models/, or\n"
                f"  * use a known ultralytics model name, e.g. --model yolo11n.pt"
            )
        else:
            hint = (
                "\nCheckpoints are downloaded automatically on first run; if this machine is "
                "offline, place the .pt file in models/ and pass --model models/<name>.pt"
            )
        return f"Could not load YOLO weights '{weights}': {exc}{hint}"

    def _extract_vocabulary(self) -> None:
        """Map class ids -> internal names and pick the id allow-list."""
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        self._names = {int(k): str(v) for k, v in names.items()}

        accident_ids = tuple(
            cid for cid, raw in self._names.items() if str(raw).strip().lower() in ACCIDENT_CLASS_ALIASES
        )
        self._accident_ids = accident_ids

        if self.config.classes:
            resolved: List[int] = []
            for item in self.config.classes:
                if isinstance(item, str) and not item.isdigit():
                    matches = [cid for cid, raw in self._names.items() if raw.lower() == item.lower()]
                    if not matches:
                        _LOGGER.warning("Configured class '%s' is not present in the model", item)
                        continue
                    resolved.extend(matches)
                else:
                    cid = int(item)
                    if self._names and cid not in self._names:
                        _LOGGER.warning("Configured class id %d is not present in the model", cid)
                        continue
                    resolved.append(cid)
            self._class_ids = sorted(set(resolved))
        elif self._names:
            road_ids = [
                cid
                for cid, raw in self._names.items()
                if normalize_class_name(raw) in ROAD_CLASS_NAMES
                or str(raw).strip().lower() in ACCIDENT_CLASS_ALIASES
            ]
            self._class_ids = sorted(road_ids) if road_ids else None
        else:
            self._class_ids = None

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def detect(self, frame: Any, frame_index: int = 0, timestamp: float = 0.0) -> DetectionResult:
        """Run one inference pass and return structured detections."""
        if frame is None:
            raise DetectionError("cannot run detection on a None frame")
        self.load()
        assert self._model is not None

        device = resolve_device(self.config.device)
        started = time.perf_counter()
        try:
            predictions = self._model.predict(
                source=frame,
                imgsz=int(self.config.imgsz),
                conf=float(self.config.conf),
                iou=float(self.config.iou),
                classes=self._class_ids,
                device=device,
                half=bool(self.config.half),
                max_det=int(self.config.max_det),
                verbose=bool(self.config.verbose),
            )
        except Exception as exc:  # noqa: BLE001 - any inference failure
            raise DetectionError(f"YOLO inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not predictions:
            raise DetectionError("YOLO returned no results object")

        height, width = (frame.shape[0], frame.shape[1]) if getattr(frame, "shape", None) and len(frame.shape) >= 2 else (0, 0)
        return self._parse(predictions[0], frame_index, timestamp, (width, height), elapsed_ms)

    def _parse(
        self,
        result: Any,
        frame_index: int,
        timestamp: float,
        frame_size: Tuple[int, int],
        elapsed_ms: float,
    ) -> DetectionResult:
        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        raw_total = 0

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy
            confs = boxes.conf if getattr(boxes, "conf", None) is not None else None
            clss = boxes.cls if getattr(boxes, "cls", None) is not None else None
            raw_total = len(xyxy)

            for i in range(raw_total):
                box = xyxy[i]
                coords = [float(v) for v in (box[0], box[1], box[2], box[3])]
                confidence = float(confs[i]) if confs is not None else 0.0
                class_id = int(clss[i]) if clss is not None else -1
                raw_name = self._names.get(class_id, f"class_{class_id}")
                normalized = normalize_class_name(raw_name)
                is_accident = str(raw_name).strip().lower() in ACCIDENT_CLASS_ALIASES or class_id in self._accident_ids
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=raw_name,
                        confidence=confidence,
                        bbox=BBox(*coords),
                        frame_index=frame_index,
                        timestamp=timestamp,
                        normalized_name=normalized,
                        is_accident_class=is_accident,
                    )
                )

        return DetectionResult(
            detections=detections,
            frame_index=frame_index,
            timestamp=timestamp,
            inference_ms=elapsed_ms,
            model_name=self.model_name,
            frame_size=frame_size,
            raw_total=raw_total,
        )

    def warmup(self, frame_size: Tuple[int, int] = (640, 384)) -> None:
        """Run one throwaway inference so the first real frame is not slow."""
        import numpy as np

        if not self.config.warmup:
            return
        width, height = frame_size
        dummy = np.zeros((max(32, height), max(32, width), 3), dtype="uint8")
        try:
            self.detect(dummy, 0, 0.0)
            _LOGGER.debug("Detector warm-up complete (%dx%d)", width, height)
        except DetectionError as exc:  # warm-up is best effort only
            _LOGGER.debug("Warm-up skipped: %s", exc)

    # ------------------------------------------------------------------ #
    # optional: delegate tracking to ultralytics' built-in ByteTrack
    # ------------------------------------------------------------------ #
    def track(
        self,
        frame: Any,
        frame_index: int = 0,
        timestamp: float = 0.0,
        tracker: str = "bytetrack.yaml",
    ) -> DetectionResult:
        """Detect *and* track in one pass using Ultralytics ByteTrack.

        Used when ``tracking.backend = ultralytics``.  The returned detections
        carry ``track_id``; SafeVision still owns the history/velocity
        bookkeeping (see :class:`~ai_engine.tracking.byte_tracker.ByteTracker`).

        ``persist=True`` is what keeps ids stable across calls on a stream.
        """
        if frame is None:
            raise DetectionError("cannot run detection on a None frame")
        self.load()
        assert self._model is not None

        device = resolve_device(self.config.device)
        started = time.perf_counter()
        try:
            predictions = self._model.track(
                source=frame,
                imgsz=int(self.config.imgsz),
                conf=float(self.config.conf),
                iou=float(self.config.iou),
                classes=self._class_ids,
                device=device,
                half=bool(self.config.half),
                max_det=int(self.config.max_det),
                verbose=bool(self.config.verbose),
                persist=True,
                tracker=tracker,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"YOLO tracking failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not predictions:
            raise DetectionError("YOLO tracking returned no results object")

        height, width = (frame.shape[0], frame.shape[1]) if getattr(frame, "shape", None) and len(frame.shape) >= 2 else (0, 0)
        result = self._parse(predictions[0], frame_index, timestamp, (width, height), elapsed_ms)
        self._attach_track_ids(result, predictions[0])
        return result

    @staticmethod
    def _attach_track_ids(result: DetectionResult, prediction: Any) -> None:
        """Copy ultralytics' ``boxes.id`` onto our detections (order is kept)."""
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or not len(boxes):
            return
        ids = getattr(boxes, "id", None)
        if ids is None:
            return
        for i, detection in enumerate(result.detections):
            if i < len(ids):
                value = ids[i]
                if value is not None:
                    try:
                        detection.track_id = int(value)
                    except (TypeError, ValueError):  # pragma: no cover
                        detection.track_id = None

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._loaded = False
        _LOGGER.debug("Detector released")

    def __repr__(self) -> str:  # pragma: no cover
        return f"YoloDetector(weights={self._weights!r}, loaded={self._loaded})"
