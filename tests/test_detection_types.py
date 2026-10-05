"""Detection data structures (Phase 4)."""

from __future__ import annotations

import unittest

from tests.helpers import make_detection

from ai_engine.detection import (
    BBox,
    Detection,
    DetectionResult,
    VEHICLE_CLASS_NAMES,
    is_vehicle_class,
    normalize_class_name,
)


class TestBBox(unittest.TestCase):
    def test_properties(self) -> None:
        box = BBox(10.0, 20.0, 40.0, 60.0)
        self.assertEqual(box.width, 30.0)
        self.assertEqual(box.height, 40.0)
        self.assertEqual(box.area, 1200.0)
        self.assertEqual(box.center, (25.0, 40.0))
        self.assertAlmostEqual(box.diagonal, 50.0, places=6)
        self.assertEqual(box.as_int(), (10, 20, 40, 60))

    def test_clamped_keeps_ordering(self) -> None:
        clamped = BBox(-10.0, -10.0, 700.0, 400.0).clamped(640, 360)
        self.assertGreaterEqual(clamped.x1, 0.0)
        self.assertLessEqual(clamped.x2, 640.0)
        self.assertGreaterEqual(clamped.y1, 0.0)
        self.assertLessEqual(clamped.y2, 360.0)

    def test_zero_sized_box_has_no_area(self) -> None:
        self.assertEqual(BBox(10.0, 10.0, 10.0, 10.0).area, 0.0)


class TestClassVocabulary(unittest.TestCase):
    def test_aliases_normalise(self) -> None:
        self.assertEqual(normalize_class_name("Auto"), "car")
        self.assertEqual(normalize_class_name("LORRY"), "truck")
        self.assertEqual(normalize_class_name("Pedestrian"), "person")
        self.assertEqual(normalize_class_name("motorbike"), "motorcycle")

    def test_accident_class_is_recognised(self) -> None:
        self.assertEqual(normalize_class_name("accident_scene"), "accident")

    def test_unknown_class_passes_through(self) -> None:
        self.assertEqual(normalize_class_name("hydrant"), "hydrant")

    def test_is_vehicle_class(self) -> None:
        self.assertTrue(is_vehicle_class("car"))
        self.assertTrue(is_vehicle_class("bus"))
        self.assertFalse(is_vehicle_class("person"))
        self.assertFalse(is_vehicle_class("accident"))


class TestDetection(unittest.TestCase):
    def test_fields_and_json(self) -> None:
        det = make_detection("car", 0.87, (10, 20, 40, 42), frame_index=7, timestamp=1.5)
        self.assertEqual(det.class_id, -1)
        self.assertEqual(det.normalized_name, "car")
        self.assertTrue(det.is_vehicle)
        self.assertFalse(det.is_person)
        payload = det.to_dict()
        for key in ("class_id", "class_name", "class", "confidence", "bbox", "center", "frame_index", "timestamp"):
            self.assertIn(key, payload)
        self.assertEqual(payload["bbox"], [10.0, 20.0, 40.0, 42.0])
        self.assertEqual(payload["center"], [25.0, 31.0])

    def test_person_detection(self) -> None:
        det = make_detection("person", 0.5, (0, 0, 10, 40))
        self.assertTrue(det.is_person)
        self.assertFalse(det.is_vehicle)

    def test_accident_class_flagged(self) -> None:
        det = Detection(
            class_id=7,
            class_name="accident",
            confidence=0.8,
            bbox=BBox(0, 0, 10, 10),
            is_accident_class=True,
        )
        self.assertTrue(det.is_accident_class)
        self.assertFalse(det.is_vehicle)


class TestDetectionResult(unittest.TestCase):
    def _result(self) -> DetectionResult:
        return DetectionResult(
            detections=[
                make_detection("car", 0.9, (0, 0, 30, 22)),
                make_detection("truck", 0.6, (50, 0, 96, 28)),
                make_detection("person", 0.8, (200, 0, 216, 44)),
            ],
            frame_index=3,
            timestamp=0.5,
            inference_ms=12.5,
            model_name="yolo11n",
            frame_size=(640, 360),
            raw_total=9,
        )

    def test_filters(self) -> None:
        result = self._result()
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result.vehicles), 2)
        self.assertEqual(len(result.persons), 1)
        self.assertEqual(result.count_by_class(), {"car": 1, "truck": 1, "person": 1})
        self.assertTrue(bool(result))

    def test_confidence_helpers(self) -> None:
        result = self._result()
        self.assertAlmostEqual(result.max_confidence, 0.9, places=6)
        self.assertAlmostEqual(result.mean_confidence, (0.9 + 0.6 + 0.8) / 3, places=6)

    def test_empty_result(self) -> None:
        empty = DetectionResult()
        self.assertFalse(bool(empty))
        self.assertEqual(empty.mean_confidence, 0.0)
        self.assertEqual(empty.accident_class_score, 0.0)
        self.assertEqual(empty.filter_classes(VEHICLE_CLASS_NAMES), [])

    def test_accident_class_score(self) -> None:
        result = DetectionResult(
            detections=[
                make_detection("car", 0.7, (0, 0, 10, 10)),
                Detection(
                    class_id=3,
                    class_name="accident",
                    confidence=0.81,
                    bbox=BBox(0, 0, 5, 5),
                    is_accident_class=True,
                ),
            ]
        )
        self.assertAlmostEqual(result.accident_class_score, 0.81, places=6)

    def test_to_dict_is_serialisable(self) -> None:
        import json

        payload = self._result().to_dict()
        json.dumps(payload)  # must not raise
        self.assertEqual(payload["model"], "yolo11n")
        self.assertEqual(payload["frame_size"], [640, 360])


if __name__ == "__main__":
    unittest.main()
