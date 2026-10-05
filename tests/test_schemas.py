"""Incident schema, incident ids, location metadata and evidence paths (Phases 14-17)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_engine.evidence.incident_id import INCIDENT_ID_PATTERN, IncidentIdGenerator
from ai_engine.location import CameraLocation, LocationError, LocationManager
from ai_engine.schemas import (
    SCHEMA_VERSION,
    AccidentInfo,
    CameraRef,
    EvidenceInfo,
    IncidentResult,
    IncidentStatus,
    LocationInfo,
    MediaInfo,
    ObjectsInfo,
    SeverityLevel,
    generate_emergency_incident,
    get_incident_result,
    validate_incident,
)


def sample_incident(**overrides) -> IncidentResult:
    incident = IncidentResult(
        incident_id="INC-20260928-0001",
        status=IncidentStatus.CONFIRMED_ACCIDENT.value,
        detected_at_unix=1759040100.0,
        camera=CameraRef(camera_id="CAM-001", name="Main Road Camera"),
        location=LocationInfo(
            name="Chennai Main Road", latitude=13.0827, longitude=80.2707, is_camera_registered=True
        ),
        accident=AccidentInfo(
            score=0.89,
            severity=SeverityLevel.HIGH.value,
            severity_score=0.72,
            confirmed=True,
            frames_of_evidence=24,
            verification_seconds=1.6,
        ),
        evidence=EvidenceInfo(
            collision_score=0.93,
            motion_score=0.88,
            trajectory_score=0.9,
            temporal_score=0.95,
            reasons=["Sudden vehicle velocity change", "Trajectory convergence"],
        ),
        objects=ObjectsInfo(vehicles_involved=2, persons_detected=1, involved_track_ids=[12, 17]),
        media=MediaInfo(annotated_frame="evidence/INC-1/annotated.jpg", evidence_frames=["evidence/INC-1/before.jpg"]),
    )
    for key, value in overrides.items():
        setattr(incident, key, value)
    return incident


class TestIncidentResult(unittest.TestCase):
    def test_required_fields_present(self) -> None:
        payload = sample_incident().to_dict()
        for key in (
            "schema_version", "incident_id", "status", "timestamp",
            "camera", "location", "accident", "evidence", "objects", "media",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)

    def test_payload_is_json_serialisable(self) -> None:
        json.dumps(sample_incident().to_dict())

    def test_contains_no_nan(self) -> None:
        incident = sample_incident()
        incident.accident.score = float("nan")
        text = json.dumps(incident.to_dict())
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(incident.to_dict()["accident"]["score"], 0.0)

    def test_alert_payload_is_compact(self) -> None:
        alert = sample_incident().to_alert_dict()
        self.assertEqual(alert["incident_id"], "INC-20260928-0001")
        self.assertEqual(alert["confidence_percent"], 89)
        self.assertIn("disclaimer", alert)
        self.assertNotIn("performance", alert)

    def test_alert_payload_can_be_filtered(self) -> None:
        self.assertEqual(sample_incident().to_alert_dict(minimum_score=0.95), {})

    def test_validate_catches_problems(self) -> None:
        self.assertEqual(sample_incident().validate(), [])

        bad = sample_incident()
        bad.accident.score = 1.5
        self.assertTrue(any("outside [0, 1]" in p for p in bad.validate()))

        bad = sample_incident()
        bad.status = "MAYBE"
        self.assertTrue(any("not a valid IncidentStatus" in p for p in bad.validate()))

        bad = sample_incident()
        bad.location.latitude = 130.0
        self.assertTrue(any("latitude" in p for p in bad.validate()))

        bad = sample_incident()
        bad.incident_id = ""
        self.assertIn("incident_id is empty", bad.validate())

        bad = sample_incident()
        bad.evidence.reasons = []
        self.assertTrue(any("without any evidence reason" in p for p in bad.validate()))

    def test_round_trip(self) -> None:
        original = sample_incident()
        payload = original.to_dict()
        restored = IncidentResult.from_dict(payload)
        self.assertEqual(restored.incident_id, original.incident_id)
        self.assertEqual(restored.accident.score, original.accident.score)
        self.assertEqual(restored.objects.involved_track_ids, original.objects.involved_track_ids)
        self.assertEqual(restored.to_dict()["location"], payload["location"])

    def test_json_schema_declares_the_contract(self) -> None:
        from ai_engine.schemas import INCIDENT_JSON_SCHEMA

        for key in ("schema_version", "incident_id", "status", "accident", "evidence", "objects", "media"):
            self.assertIn(key, INCIDENT_JSON_SCHEMA["properties"])
        self.assertIn("CONFIRMED_ACCIDENT", INCIDENT_JSON_SCHEMA["properties"]["status"]["enum"])
        self.assertIn("disclaimer", INCIDENT_JSON_SCHEMA["properties"])
        self.assertIn("not a verified human-injury assessment", INCIDENT_JSON_SCHEMA["description"].lower())

    def test_validate_incident_helper(self) -> None:
        self.assertEqual(validate_incident(sample_incident().to_dict()), [])

    def test_wording_is_honest(self) -> None:
        text = json.dumps(sample_incident().to_dict()).lower()
        self.assertIn("not a verified human-injury assessment", text)
        for forbidden in ("100% accurate", "zero false alarm", "guaranteed"):
            self.assertNotIn(forbidden, text)


class TestEmergencyInterface(unittest.TestCase):
    def test_generate_emergency_incident_from_object(self) -> None:
        alert = generate_emergency_incident(sample_incident())
        self.assertEqual(alert["status"], "CONFIRMED_ACCIDENT")

    def test_generate_emergency_incident_from_dict(self) -> None:
        alert = generate_emergency_incident(sample_incident().to_dict())
        self.assertEqual(alert["incident_id"], "INC-20260928-0001")

    def test_can_strip_media_paths(self) -> None:
        alert = generate_emergency_incident(sample_incident(), include_evidence_paths=False)
        self.assertNotIn("evidence_dir", alert)

    def test_minimum_score_filter(self) -> None:
        self.assertEqual(generate_emergency_incident(sample_incident(), minimum_score=0.99), {})

    def test_wrong_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            generate_emergency_incident(42)

    def test_get_incident_result_from_sequence(self) -> None:
        payload = get_incident_result([{"incident_id": "A"}, sample_incident()])
        self.assertEqual(payload["incident_id"], "INC-20260928-0001")
        self.assertEqual(get_incident_result([]), {})


class TestIncidentIds(unittest.TestCase):
    def test_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generator = IncidentIdGenerator(state_file=Path(tmp) / "counter.json")
            incident_id = generator.next_id()
        self.assertRegex(incident_id, INCIDENT_ID_PATTERN)
        self.assertTrue(incident_id.startswith("INC-"))
        self.assertEqual(len(incident_id.split("-")[-1]), 4)

    def test_sequential_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generator = IncidentIdGenerator(state_file=Path(tmp) / "counter.json")
            ids = [generator.next_id() for _ in range(5)]
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual([int(i.split("-")[-1]) for i in ids], [1, 2, 3, 4, 5])

    def test_counter_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "counter.json"
            first = IncidentIdGenerator(state_file=state).next_id()
            second = IncidentIdGenerator(state_file=state).next_id()
        self.assertNotEqual(first, second)
        self.assertEqual(int(second.split("-")[-1]), int(first.split("-")[-1]) + 1)

    def test_existing_evidence_directories_are_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            today = IncidentIdGenerator(state_file=None)._ensure_loaded.__self__._date
            (root / f"INC-{today}-0007").mkdir()
            generator = IncidentIdGenerator(state_file=root / "counter.json", evidence_root=root)
            self.assertEqual(int(generator.next_id().split("-")[-1]), 8)

    def test_in_memory_only(self) -> None:
        generator = IncidentIdGenerator(state_file=None)
        self.assertEqual(generator.next_id(), generator.next_id().replace("0002", "0001"))


class TestLocationManager(unittest.TestCase):
    def test_camera_validation(self) -> None:
        camera = CameraLocation(camera_id="CAM-1", name="Main", latitude=13.0, longitude=80.0)
        self.assertTrue(camera.has_coordinates)
        with self.assertRaises(LocationError):
            CameraLocation(camera_id="CAM-2", latitude=95.0, longitude=0.0)
        with self.assertRaises(LocationError):
            CameraLocation(camera_id="CAM-3", latitude=13.0)
        with self.assertRaises(LocationError):
            CameraLocation(camera_id="")

    def test_lookup_and_default(self) -> None:
        manager = LocationManager(
            cameras={"CAM-001": CameraLocation(camera_id="CAM-001", name="Main")},
            default_camera_id="CAM-001",
        )
        self.assertIsNotNone(manager.get())
        self.assertIsNotNone(manager.get("CAM-001"))
        self.assertIsNone(manager.get("CAM-999"))
        self.assertIn("CAM-001", manager)

    def test_unregistered_camera_returns_none(self) -> None:
        manager = LocationManager()
        self.assertIsNone(manager.get("CAM-404"))
        self.assertIsNone(manager.require("CAM-404"))

    def test_register_at_runtime(self) -> None:
        manager = LocationManager()
        manager.register(CameraLocation(camera_id="CAM-NEW", name="New", latitude=1.0, longitude=2.0))
        self.assertEqual(len(manager), 1)
        self.assertTrue(manager.get("CAM-NEW").has_coordinates)

    def test_load_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            path.write_text(
                "default_camera_id: CAM-001\n"
                "cameras:\n"
                "  CAM-001:\n"
                "    name: Main Road Camera\n"
                "    latitude: 13.0827\n"
                "    longitude: 80.2707\n"
                "    location_name: Chennai Main Road\n",
                encoding="utf-8",
            )
            manager = LocationManager.from_file(path)
        self.assertEqual(len(manager), 1)
        camera = manager.get("CAM-001")
        self.assertEqual(camera.name, "Main Road Camera")
        self.assertAlmostEqual(camera.latitude, 13.0827, places=4)
        self.assertEqual(manager.default_camera_id, "CAM-001")

    def test_missing_file_is_not_fatal(self) -> None:
        manager = LocationManager.from_file(Path("definitely_missing_cameras.yaml"))
        self.assertEqual(len(manager), 0)

    def test_invalid_entry_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            path.write_text(
                "cameras:\n  BAD:\n    latitude: 999\n  GOOD:\n    name: Fine\n",
                encoding="utf-8",
            )
            manager = LocationManager.from_file(path)
        self.assertIn("GOOD", manager)
        self.assertNotIn("BAD", manager)

    def test_payload_is_json_safe(self) -> None:
        manager = LocationManager(cameras={"C": CameraLocation(camera_id="C", name="x")})
        json.dumps(manager.to_payload())


if __name__ == "__main__":
    unittest.main()
