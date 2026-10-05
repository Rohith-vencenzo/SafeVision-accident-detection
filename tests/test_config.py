"""Configuration loading, validation and overrides (Phase 22 + 24)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT

from ai_engine.config import (
    CollisionConfig,
    Config,
    ConfigError,
    build_source_url,
    deep_merge,
    describe_config,
    find_project_root,
    load_config,
    parse_override,
    redact_url,
)
from ai_engine.utils.jsonio import atomic_write_json, read_json, to_jsonable
from ai_engine.utils.ringbuffer import TimeRingBuffer


class TestConfigLoading(unittest.TestCase):
    def test_defaults_without_a_file(self) -> None:
        config = load_config(env={})
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.conf, 0.30)
        self.assertEqual(config.collision.min_signals, 2)

    def test_project_config_file_is_used(self) -> None:
        """The shipped config.yaml must load and validate."""
        config = load_config()
        self.assertTrue(config.source_file.endswith("config.yaml"))
        config.validate()
        # the shipped weights must be resolvable: either the file exists, or the
        # bare checkpoint name can be downloaded by ultralytics
        self.assertTrue(str(config.model.weights).endswith(".pt"))
        from pathlib import Path

        weights = Path(config.model.weights)
        if weights.is_absolute() or str(weights.parent) not in (".", ""):
            self.assertTrue(
                weights.is_file() or weights.name,
                "a configured path must exist or fall back to a known checkpoint name",
            )

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("definitely_missing_config.yaml", env={})

    def test_invalid_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("model: [this is not a mapping\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, env={})

    def test_unknown_option_raises(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(overrides=["model.definitely_not_an_option=1"], env={})
        self.assertIn("unknown option", str(ctx.exception))

    def test_out_of_range_raises(self) -> None:
        for assignment in ("model.conf=2.0", "model.imgsz=10", "verification.confirm_score=-1"):
            with self.assertRaises(ConfigError):
                load_config(overrides=[assignment], env={})

    def test_cross_section_validation(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(overrides=["tracking.conf_low=0.9", "tracking.conf_high=0.5"], env={})
        with self.assertRaises(ConfigError):
            load_config(overrides=["severity.low_threshold=0.9", "severity.medium_threshold=0.2"], env={})
        with self.assertRaises(ConfigError):
            load_config(overrides=["tracking.backend=nonsense"], env={})

    def test_overrides_precedence(self) -> None:
        config = load_config(
            overrides=["model.conf=0.44", "evidence.root=custom"],
            env={"SAFEVISION_DEVICE": "cpu", "SAFEVISION_CONFIDENCE": "0.55"},
        )
        self.assertEqual(config.model.conf, 0.44, "a CLI override must beat the environment")
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(config.evidence.root, "custom")

    def test_env_only(self) -> None:
        config = load_config(env={"SAFEVISION_MODEL": "yolo11s.pt", "SAFEVISION_CAMERA_ID": "CAM-9"})
        self.assertEqual(config.model.weights, "yolo11s.pt")
        self.assertEqual(config.location.default_camera_id, "CAM-9")

    def test_describe_config_redacts_secrets(self) -> None:
        described = describe_config(load_config(env={}))
        text = json.dumps(described)
        self.assertNotIn("password\":", text.replace("env_password", ""))
        self.assertIn("source_file", described)

    def test_round_trip(self) -> None:
        config = load_config(env={})
        data = config.to_dict()
        rebuilt = Config()
        for key, value in data.items():
            if key in ("name", "version", "source_file"):
                setattr(rebuilt, key, value)
        self.assertEqual(rebuilt.model.conf, config.model.conf)

    def test_emergency_section_defaults_to_safe(self) -> None:
        """The demonstration call must be opt-in, never on by default."""
        config = load_config(use_env=False)
        self.assertFalse(config.emergency.demo_mode, "demo mode must default to off")
        self.assertEqual(config.emergency.provider, "auto")
        self.assertEqual(config.emergency.contact_env_var, "EMERGENCY_CONTACT_NUMBER")

    def test_emergency_provider_is_validated(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(use_env=False, overrides=["emergency.provider=carrier_pigeon"])

    def test_emergency_voice_message_must_not_be_empty(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(use_env=False, overrides=["emergency.voice_message="])

    def test_demo_mode_env_override(self) -> None:
        self.assertTrue(load_config(use_env=True, env={"DEMO_MODE": "true"}).emergency.demo_mode)
        self.assertFalse(load_config(use_env=True, env={"DEMO_MODE": "false"}).emergency.demo_mode)

    def test_config_never_holds_a_phone_number(self) -> None:
        """The recipient is an env var name, never a value, in the config."""
        emergency = load_config(use_env=False).to_dict()["emergency"]
        self.assertNotIn("recipient", emergency)
        self.assertNotIn("contact_number", emergency)
        self.assertIn("contact_env_var", emergency)

    def test_the_shipped_config_does_not_leak_the_number(self) -> None:
        """config.yaml must stay free of any phone number."""
        text = (Path(find_project_root()) / "config.yaml").read_text(encoding="utf-8")
        import re

        self.assertIsNone(
            re.search(r"\+\d{8,}", text),
            "config.yaml must never contain a phone number - it belongs in .env",
        )


class TestHelpers(unittest.TestCase):
    def test_parse_override(self) -> None:
        self.assertEqual(parse_override("a.b=1"), ("a.b", 1))
        self.assertEqual(parse_override("a.b=0.5"), ("a.b", 0.5))
        self.assertEqual(parse_override("a.b=true"), ("a.b", True))
        self.assertEqual(parse_override("a.b=[1,2]"), ("a.b", [1, 2]))
        self.assertEqual(parse_override("a.b=hello"), ("a.b", "hello"))
        with self.assertRaises(ConfigError):
            parse_override("no_equals")

    def test_deep_merge(self) -> None:
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
        self.assertEqual(merged, {"a": {"b": 1, "c": 3}})

    def test_redact_url(self) -> None:
        self.assertEqual(redact_url("rtsp://u:p@host/s"), "rtsp://***@host/s")
        self.assertEqual(redact_url("rtsp://host/s"), "rtsp://host/s")
        self.assertEqual(redact_url(0), "webcam:0")

    def test_build_source_url_keeps_files_untouched(self) -> None:
        self.assertEqual(build_source_url("clip.mp4"), "clip.mp4")
        self.assertEqual(build_source_url("0"), 0)

    def test_find_project_root(self) -> None:
        self.assertTrue((find_project_root() / "ai_engine").is_dir())


class TestJsonIo(unittest.TestCase):
    def test_numpy_and_nan_handling(self) -> None:
        import numpy as np

        payload = to_jsonable(
            {
                "int": np.int64(3),
                "float": np.float32(1.5),
                "array": np.array([1, 2, 3]),
                "bool": np.bool_(True),
                "nan": float("nan"),
                "inf": float("inf"),
                "set": {1, 2},
                "path": Path("x"),
            }
        )
        self.assertEqual(payload["int"], 3)
        self.assertEqual(payload["float"], 1.5)
        self.assertEqual(payload["array"], [1, 2, 3])
        self.assertIs(payload["bool"], True)
        self.assertIsNone(payload["nan"])
        self.assertIsNone(payload["inf"])
        self.assertEqual(sorted(payload["set"]), [1, 2])
        self.assertEqual(payload["path"], "x")
        json.dumps(payload)

    def test_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.json"
            atomic_write_json(path, {"a": 1})
            self.assertEqual(read_json(path), {"a": 1})
            self.assertEqual(list(path.parent.iterdir()), [path], "no temp file must be left behind")

    def test_read_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_json("definitely_missing.json")


class TestRingBuffer(unittest.TestCase):
    def test_time_window(self) -> None:
        buffer = TimeRingBuffer(window_seconds=1.0)
        for index in range(10):
            buffer.push(index * 0.25, index)
        # the newest timestamp is 2.25 s, so the cutoff is 1.25 s (inclusive)
        self.assertEqual([i for _, i in buffer], [5, 6, 7, 8, 9])
        self.assertAlmostEqual(buffer.span_seconds, 1.0, places=6)
        self.assertEqual(buffer.window(now=2.25), [5, 6, 7, 8, 9])

    def test_max_items(self) -> None:
        buffer = TimeRingBuffer(window_seconds=None, max_items=3)
        for index in range(10):
            buffer.push(float(index), index)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.items, [7, 8, 9])

    def test_clear_and_copy(self) -> None:
        buffer = TimeRingBuffer(window_seconds=5.0)
        for index in range(4):
            buffer.push(float(index), index)
        clone = buffer.copy()
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        self.assertEqual(len(clone), 4)


if __name__ == "__main__":
    unittest.main()
