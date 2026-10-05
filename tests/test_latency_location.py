"""Phase 3: latency instrumentation and location provenance.

Two themes:

**Latency** - the ledger must measure on a monotonic clock, report
distributions with a sample count, keep a live stream and a recorded file
distinguishable, and never let "queued for notification" read as "notified".

**Provenance** - a camera's registered coordinates must never be presented as a
live GPS fix, a missing or stale fix must fall back to
``LOCATION_UNAVAILABLE`` rather than to a guess, and IP geolocation must be
impossible.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from ai_engine.location.location_provider import (
    LOCATION_CAMERA_REGISTERED,
    LOCATION_LIVE_GPS,
    LOCATION_UNAVAILABLE,
    CameraRegisteredProvider,
    FixPolicy,
    GpsDeviceProvider,
    GpsFix,
    LocationProviderRegistry,
)
from ai_engine.location.location_manager import LocationManager
from ai_engine.telemetry.latency import (
    LATENCY_STAGES,
    LatencyLedger,
    StageTimer,
    percentiles,
)
from tests.helpers import test_config


def _manager():
    return LocationManager.from_config(test_config().location)


class TestPercentiles(unittest.TestCase):
    def test_empty_is_none_not_zero(self) -> None:
        stats = percentiles([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["p50_ms"])
        self.assertIsNone(stats["max_ms"])

    def test_sample_count_is_reported(self) -> None:
        self.assertEqual(percentiles([1.0, 2.0, 3.0])["count"], 3)

    def test_single_sample(self) -> None:
        stats = percentiles([7.5])
        self.assertEqual(stats["p50_ms"], 7.5)
        self.assertEqual(stats["p95_ms"], 7.5)

    def test_p95_is_a_real_sample(self) -> None:
        values = [float(v) for v in range(1, 101)]
        stats = percentiles(values)
        # median of 1..100 is the mean of the two central samples
        self.assertEqual(stats["p50_ms"], 50.5)
        self.assertGreaterEqual(stats["p95_ms"], 95.0)
        self.assertEqual(stats["max_ms"], 100.0)

    def test_p95_is_not_interpolated(self) -> None:
        # 10 samples: nearest-rank p95 is the 10th, not 9.55
        stats = percentiles([float(v) for v in range(1, 11)])
        self.assertIn(stats["p95_ms"], [float(v) for v in range(1, 11)])


class TestLatencyLedger(unittest.TestCase):
    def test_stage_records_a_duration(self) -> None:
        ledger = LatencyLedger()
        with ledger.stage("detect"):
            time.sleep(0.01)
        stats = ledger.report()["stages"]["detect"]
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["p50_ms"], 5.0)

    def test_a_failing_stage_is_not_counted_as_zero(self) -> None:
        ledger = LatencyLedger()
        with self.assertRaises(RuntimeError):
            with ledger.stage("detect"):
                raise RuntimeError("boom")
        report = ledger.report()
        self.assertNotIn("detect", report["stages"])
        self.assertEqual(report.get("stages", {}).get("detect", {}).get("errors"), None)

    def test_samples_are_bounded(self) -> None:
        ledger = LatencyLedger(max_samples_per_stage=5)
        for _ in range(50):
            ledger.record("detect", 1.0)
        self.assertEqual(ledger.report()["stages"]["detect"]["count"], 5)

    def test_marks_are_monotonic_and_ordered(self) -> None:
        ledger = LatencyLedger()
        ledger.mark("INC-1", "incident_created")
        ledger.mark("INC-1", "provider_request")
        ledger.mark("INC-1", "call_terminal")
        timeline = ledger.incident_timeline("INC-1")
        self.assertLessEqual(timeline["incident_created"], timeline["provider_request"])
        self.assertLessEqual(timeline["provider_request"], timeline["call_terminal"])

    def test_unknown_incident_has_no_timeline(self) -> None:
        self.assertEqual(LatencyLedger().incident_timeline("nope"), {})

    def test_mode_defaults_to_unknown(self) -> None:
        self.assertEqual(LatencyLedger().report()["mode"], "UNKNOWN")

    def test_report_declares_monotonic_clock_and_caveat(self) -> None:
        ledger = LatencyLedger(mode="RECORDED")
        method = ledger.report()["methodology"]
        self.assertIn("monotonic", method["clock"])
        self.assertIn("do not bound", method["caveat"])
        self.assertIn("not", method["caveat"].lower())

    def test_text_report_marks_the_mode(self) -> None:
        ledger = LatencyLedger(mode="LIVE", source_kind="rtsp")
        ledger.record("detect", 1.0)
        text = ledger.text()
        self.assertIn("LIVE", text)
        self.assertIn("p50", text)

    def test_stage_names_cover_the_notification_chain(self) -> None:
        for stage in ("detect", "verify", "incident_build", "notify_enqueue",
                      "provider_request", "call_terminal", "sms_status"):
            self.assertIn(stage, LATENCY_STAGES)


class TestPipelineInstrumentation(unittest.TestCase):
    def test_pipeline_fills_stage_timings(self) -> None:
        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline
        import numpy as np

        ledger = LatencyLedger()
        pipeline = AccidentPipeline(
            test_config(), detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001", ledger=ledger, source_kind="RECORDED",
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for index in range(50):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()
        stages = ledger.report()["stages"]
        for stage in ("detect", "track", "analysis", "verify", "incident_build"):
            self.assertIn(stage, stages, f"{stage} was not measured")
        self.assertGreater(stages["detect"]["p50_ms"], 0.0)

    def test_live_and_recorded_modes_are_distinguished(self) -> None:
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        recorded = AccidentPipeline(test_config(),
                                    detector=ScriptedDetector([]), source_kind="RECORDED")
        self.assertEqual(recorded.ledger.report()["mode"], "RECORDED")
        live = AccidentPipeline(test_config(),
                                detector=ScriptedDetector([]), source_kind="rtsp")
        self.assertEqual(live.ledger.report()["mode"], "LIVE")
        webcam = AccidentPipeline(test_config(),
                                  detector=ScriptedDetector([]), source_kind="webcam")
        self.assertEqual(webcam.ledger.report()["mode"], "LIVE")

    def test_benchmark_cannot_notify_anybody(self) -> None:
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        pipeline = AccidentPipeline(test_config(),
                                    detector=ScriptedDetector([]), source_kind="RECORDED")
        self.assertIsNone(pipeline.emergency,
                          "a benchmark run must not build a dispatcher")


class TestDashboardContract(unittest.TestCase):
    """Phase 3 dashboard fields: mode, latency, dropped frames, provenance."""

    def _pipeline(self, **kw):
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        return AccidentPipeline(test_config(), detector=ScriptedDetector([]), **kw)

    def test_dashboard_fields_carry_the_processing_mode(self) -> None:
        fields = self._pipeline(source_kind="RECORDED").dashboard_fields()
        self.assertEqual(fields["processing_mode"], "RECORDED")
        self.assertIn("stage_latency_p95_ms", fields)
        self.assertIn("latency_methodology", fields)
        self.assertIn("dropped_frames", fields)

    def test_live_mode_is_labelled_live(self) -> None:
        self.assertEqual(self._pipeline(source_kind="rtsp").dashboard_fields()["processing_mode"],
                         "LIVE")

    def test_dashboard_never_claims_a_response_time(self) -> None:
        fields = self._pipeline(source_kind="RECORDED").dashboard_fields()
        text = str(fields["latency_methodology"]).lower()
        self.assertIn("do not bound", text)

    def test_dropped_frames_counter_starts_at_zero(self) -> None:
        self.assertEqual(self._pipeline().dropped_frames, 0)

    def test_incident_carries_location_provenance(self) -> None:
        import numpy as np

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        pipeline = AccidentPipeline(
            test_config(), detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-001", source_kind="RECORDED")
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for index in range(80):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()
        payload = pipeline.incidents[0].to_dict()
        provenance = payload["extra"]["location_provenance"]
        self.assertEqual(provenance["source"], LOCATION_CAMERA_REGISTERED)
        self.assertIn("not a live GPS fix", provenance["note"])
        self.assertIsNone(provenance["accuracy_m"],
                          "a registered position has no accuracy estimate")

    def test_provenance_is_missing_not_invented_for_an_unknown_camera(self) -> None:
        import numpy as np

        from ai_engine.detection.scenarios import SCENARIOS
        from ai_engine.detection.scripted import ScriptedDetector
        from ai_engine.pipeline import AccidentPipeline

        pipeline = AccidentPipeline(
            test_config(), detector=ScriptedDetector(SCENARIOS["head_on_crash"]),
            camera_id="CAM-NOT-REGISTERED")
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for index in range(80):
            pipeline.process_frame(frame, index, index / 15.0)
        pipeline.close()
        provenance = pipeline.incidents[0].to_dict()["extra"]["location_provenance"]
        self.assertEqual(provenance["source"], LOCATION_UNAVAILABLE)
        self.assertIsNone(provenance["latitude"])


class TestFixPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = FixPolicy(max_fix_age_seconds=30, max_accuracy_m=100,
                                max_radius_m=200)
        self.now = time.time()

    def _fix(self, **kw) -> GpsFix:
        base = dict(latitude=13.08, longitude=80.27, accuracy_m=8.0,
                    radius_m=20.0, fix_timestamp=self.now, provider="test",
                    raw_status="ok")
        base.update(kw)
        return GpsFix(**base)

    def test_a_good_fix_is_accepted(self) -> None:
        ok, why, age = self.policy.check(self._fix(), self.now)
        self.assertTrue(ok)
        self.assertEqual(why, "accepted")

    def test_a_stale_fix_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(
            self._fix(fix_timestamp=self.now - 600), self.now)
        self.assertFalse(ok)
        self.assertIn("old", why)

    def test_an_imprecise_fix_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(self._fix(accuracy_m=500.0), self.now)
        self.assertFalse(ok)
        self.assertIn("accuracy", why)

    def test_a_wide_radius_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(self._fix(radius_m=900.0), self.now)
        self.assertFalse(ok)
        self.assertIn("radius", why)

    def test_out_of_bounds_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(self._fix(latitude=999.0), self.now)
        self.assertFalse(ok)
        self.assertIn("latitude", why)

    def test_a_future_timestamp_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(
            self._fix(fix_timestamp=self.now + 3600), self.now)
        self.assertFalse(ok)
        self.assertIn("future", why)

    def test_no_fix_status_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(
            GpsFix(raw_status="permission_denied"), self.now)
        self.assertFalse(ok)
        self.assertIn("permission_denied", why)

    def test_missing_position_is_rejected(self) -> None:
        ok, why, _ = self.policy.check(GpsFix(raw_status="ok"), self.now)
        self.assertFalse(ok)


class TestLocationProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()))

    def test_no_gps_configured_reports_unavailable_not_a_guess(self) -> None:
        fix = self.registry.resolve("CAM-001", time.time())
        self.assertEqual(fix.source, LOCATION_CAMERA_REGISTERED)
        self.assertIn("not a live GPS fix", fix.summary_line())

    def test_a_registered_camera_without_coordinates_is_unavailable(self) -> None:
        fix = self.registry.resolve("CAM-003", time.time())
        self.assertEqual(fix.source, LOCATION_UNAVAILABLE)
        self.assertIsNone(fix.latitude)
        self.assertEqual(fix.confidence, "NONE")

    def test_an_unregistered_camera_is_unavailable(self) -> None:
        fix = self.registry.resolve("CAM-DOES-NOT-EXIST", time.time())
        self.assertEqual(fix.source, LOCATION_UNAVAILABLE)
        self.assertIsNone(fix.latitude)

    def test_a_live_gps_fix_wins_and_carries_quality(self) -> None:
        now = time.time()
        registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()),
            gps_provider=GpsDeviceProvider(
                lambda: GpsFix(latitude=13.083, longitude=80.271, accuracy_m=8.0,
                                fix_timestamp=now, provider="tracker", raw_status="ok")))
        fix = registry.resolve("CAM-001", now)
        self.assertEqual(fix.source, LOCATION_LIVE_GPS)
        self.assertTrue(fix.is_live_gps)
        self.assertEqual(fix.accuracy_m, 8.0)
        self.assertIsNotNone(fix.fix_age_seconds)
        self.assertEqual(fix.confidence, "DEVICE_REPORTED")

    def test_a_rejected_gps_fix_falls_back_and_says_why(self) -> None:
        now = time.time()
        registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()),
            gps_provider=GpsDeviceProvider(
                lambda: GpsFix(latitude=13.083, longitude=80.271, accuracy_m=8.0,
                                fix_timestamp=now - 999, provider="tracker",
                                raw_status="ok")))
        fix = registry.resolve("CAM-001", now)
        self.assertEqual(fix.source, LOCATION_CAMERA_REGISTERED)
        self.assertIn("no usable live GPS fix", fix.reason)
        self.assertTrue(fix.rejected, "the refused fix must be recorded")

    def test_a_raising_gps_reader_never_breaks_resolution(self) -> None:
        def boom():
            raise RuntimeError("device disconnected")

        registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()),
            gps_provider=GpsDeviceProvider(boom))
        fix = registry.resolve("CAM-001", time.time())
        self.assertEqual(fix.source, LOCATION_CAMERA_REGISTERED)

    def test_no_gps_device_means_unavailable_when_there_is_no_camera_either(self) -> None:
        registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()),
            gps_provider=GpsDeviceProvider(None))
        fix = registry.resolve("CAM-NOT-REGISTERED", time.time())
        self.assertEqual(fix.source, LOCATION_UNAVAILABLE)

    def test_camera_fallback_can_be_switched_off(self) -> None:
        registry = LocationProviderRegistry(
            camera_provider=CameraRegisteredProvider(_manager()),
            gps_provider=GpsDeviceProvider(None),
            allow_camera_fallback=False)
        fix = registry.resolve("CAM-001", time.time())
        self.assertEqual(fix.source, LOCATION_UNAVAILABLE)

    def test_the_payload_never_calls_a_camera_position_a_fix(self) -> None:
        fix = self.registry.resolve("CAM-001", time.time())
        payload = fix.to_dict()
        self.assertEqual(payload["source"], LOCATION_CAMERA_REGISTERED)
        self.assertIn("APPROXIMATE", payload["confidence"])
        self.assertIn("not a live GPS fix", payload["note"])
        self.assertIsNone(payload["accuracy_m"])

    def test_ip_geolocation_is_never_produced(self) -> None:
        described = self.registry.describe()
        self.assertIn("ip_geolocation", described)
        self.assertIn("never used", described["ip_geolocation"])
        from ai_engine.location.location_provider import LOCATION_IP_GEOLOCATION
        fix = self.registry.resolve("CAM-001", time.time())
        self.assertNotEqual(fix.source, LOCATION_IP_GEOLOCATION)


if __name__ == "__main__":
    unittest.main()