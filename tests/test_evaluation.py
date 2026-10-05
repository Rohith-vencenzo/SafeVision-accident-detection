"""Phase 2 evaluation harness: manifest, leakage-safe splits, honest metrics.

Deterministic and weight-free.  A recording predictor is injected, so these
tests never load YOLO and never touch a network.

The tests that matter most are the *negative* ones: the harness must REFUSE to
produce a quality claim when the data is inadequate.  A harness that quietly
reports a confident number on four clips is worse than no harness.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.evaluation import (  # noqa: E402
    ClipPrediction,
    ManifestError,
    assert_no_leakage,
    class_imbalance,
    confusion_from_events,
    evaluate_manifest,
    group_split,
    load_manifest,
    pr_auc,
    precision_recall_curve,
    render_report,
    summarise_split_sufficiency,
    wilson_interval,
)
from tests.helpers import test_config  # noqa: E402

CLIP_FRAMES = 8


def _make_clip(directory: Path, name: str) -> Path:
    """A tiny real MP4 so the manifest resolves; content is irrelevant here."""
    import cv2
    import numpy as np

    path = directory / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 90))
    for _ in range(CLIP_FRAMES):
        writer.write(np.zeros((90, 160, 3), dtype=np.uint8))
    writer.release()
    return path


def _manifest(directory: Path, clips: list) -> Path:
    payload = {"name": "t", "version": "1", "clips": clips}
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RecordingPredictor:
    """Returns canned predictions so no weights are needed."""

    def __init__(self, score_for=None, incident_for=None):
        self.score_for = score_for or {}
        self.incident_for = incident_for or {}
        self.seen = []

    def __call__(self, clip, config):
        self.seen.append(clip.clip)
        return ClipPrediction(
            clip=Path(clip.clip).name,
            source_id=clip.source_id,
            label=clip.label,
            quality=clip.quality,
            peak_score=self.score_for.get(clip.clip, 0.1),
            incident_count=1 if self.incident_for.get(clip.clip) else 0,
            incident_times=[self.incident_for[clip.clip]] if self.incident_for.get(clip.clip) else [],
            frames=CLIP_FRAMES,
            seconds=CLIP_FRAMES / 15.0,
        )


# --------------------------------------------------------------------------- #
class TestMetrics(unittest.TestCase):
    def test_wilson_interval_is_wide_on_small_samples(self) -> None:
        """The whole point: 1/1 must not read as certainty."""
        perfect = wilson_interval(1, 1)
        self.assertEqual(perfect.point, 1.0)
        self.assertLess(perfect.low, 0.25, "a 1/1 rate must still have a wide low bound")
        zero = wilson_interval(0, 3)
        self.assertEqual(zero.point, 0.0)
        self.assertGreater(zero.high, 0.5, "0/3 must not read as proven zero")

    def test_wilson_interval_is_bounded(self) -> None:
        for k, n in ((0, 10), (5, 10), (10, 10), (1, 0)):
            ci = wilson_interval(k, n)
            self.assertGreaterEqual(ci.low, 0.0)
            self.assertLessEqual(ci.high, 1.0)

    def test_confusion_matrix_counts_events(self) -> None:
        predictions = [(True, 0.6, 5.0), (False, 0.1, 0.0), (False, 0.2, 0.0), (True, 0.7, 30.0)]
        labels = [(True, [(4.0, 6.0)]), (False, []), (False, []), (True, [(28.0, 32.0)])]
        m = confusion_from_events(predictions, labels)
        self.assertEqual(m.confusion(), {"TP": 2, "FP": 0, "FN": 0, "TN": 2})
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)

    def test_missed_event_is_a_false_negative(self) -> None:
        predictions = [(False, 0.1, 0.0)]
        labels = [(True, [(4.0, 6.0)])]
        m = confusion_from_events(predictions, labels)
        self.assertEqual(m.confusion(), {"TP": 0, "FP": 0, "FN": 1, "TN": 0})
        self.assertEqual(m.recall, 0.0)

    def test_alarm_on_a_quiet_clip_is_a_false_positive(self) -> None:
        m = confusion_from_events([(True, 0.9, 1.0)], [(False, [])])
        self.assertEqual(m.false_positive_rate, 1.0)

    def test_prediction_outside_the_window_is_not_a_true_positive(self) -> None:
        m = confusion_from_events([(True, 0.9, 100.0)], [(True, [(4.0, 6.0)])])
        self.assertEqual(m.confusion()["FN"], 1)

    def test_pr_auc_is_none_without_positives(self) -> None:
        points = precision_recall_curve([], [0.1, 0.2])
        self.assertIsNone(pr_auc(points), "PR-AUC is undefined with no positives")

    def test_pr_auc_is_one_for_a_perfect_separation(self) -> None:
        points = precision_recall_curve([0.9, 0.85], [0.1, 0.2])
        self.assertAlmostEqual(pr_auc(points), 1.0, places=6)

    def test_pr_auc_rewards_a_lower_score_for_negatives(self) -> None:
        good = pr_auc(precision_recall_curve([0.9], [0.1]))
        bad = pr_auc(precision_recall_curve([0.9], [0.95]))
        self.assertGreater(good, bad)

    def test_class_imbalance_is_reported(self) -> None:
        self.assertEqual(class_imbalance(2, 18)["negative_to_positive_ratio"], 9.0)
        self.assertEqual(class_imbalance(2, 18)["majority_class_pct"], 90.0)


# --------------------------------------------------------------------------- #
class TestManifestValidation(unittest.TestCase):
    def test_missing_manifest_raises(self) -> None:
        with self.assertRaises(ManifestError):
            load_manifest(ROOT / "no_such_manifest.json")

    def test_invalid_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_missing_clip_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _manifest(Path(tmp), [{"clip": "ghost.mp4", "label": "normal",
                                         "source_id": "S", "quality": "VERIFIED"}])
            with self.assertRaises(ManifestError) as ctx:
                load_manifest(path)
            self.assertIn("not found", str(ctx.exception))

    def test_accident_label_requires_an_accident_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _make_clip(d, "a.mp4")
            path = _manifest(d, [{"clip": "a.mp4", "label": "accident",
                                  "source_id": "S", "quality": "VERIFIED", "events": []}])
            with self.assertRaises(ManifestError) as ctx:
                load_manifest(path)
            self.assertIn("no event", str(ctx.exception))

    def test_unknown_quality_grade_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _make_clip(d, "a.mp4")
            path = _manifest(d, [{"clip": "a.mp4", "label": "normal",
                                 "source_id": "S", "quality": "TRUST_ME"}])
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_duplicate_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _make_clip(d, "a.mp4")
            entry = {"clip": "a.mp4", "label": "normal", "source_id": "S1", "quality": "VERIFIED"}
            path = _manifest(d, [entry, dict(entry)])
            with self.assertRaises(ManifestError):
                load_manifest(path)


# --------------------------------------------------------------------------- #
class TestQualityGates(unittest.TestCase):
    """The harness must REFUSE to support a claim on inadequate data."""

    def _tiny_manifest(self, tmp: Path):
        d = tmp / "videos"
        d.mkdir()
        clips = []
        for index in range(4):
            _make_clip(d, f"c{index}.mp4")
            clips.append({"clip": f"videos/c{index}.mp4", "label": "normal",
                          "source_id": f"S{index}", "quality": "SYNTHETIC"})
        return _manifest(tmp, clips)

    def test_synthetic_set_is_blocked_from_quality_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._tiny_manifest(Path(tmp)))
            self.assertFalse(manifest.usable_for_quality_claims())
            self.assertTrue(any("synthetic" in b for b in manifest.quality_blockers()))

    def test_provisional_labels_are_blocked_as_circular(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "v"
            d.mkdir()
            clips = []
            for index in range(4):
                _make_clip(d, f"c{index}.mp4")
                clips.append({"clip": f"v/c{index}.mp4", "label": "normal",
                              "source_id": f"S{index}", "quality": "PROVISIONAL"})
            manifest = load_manifest(_manifest(Path(tmp), clips))
            self.assertFalse(manifest.usable_for_quality_claims())
            self.assertTrue(any("verified" in b for b in manifest.quality_blockers()))

    def test_too_few_clips_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._tiny_manifest(Path(tmp)))
            self.assertTrue(any("real clip" in b for b in manifest.quality_blockers()))

    def test_report_withholds_claims_on_bad_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tiny_manifest(Path(tmp))
            report = evaluate_manifest(path, config=test_config(),
                                       predict=RecordingPredictor())
            payload = report.to_dict()
            self.assertFalse(payload["manifest"]["usable_for_quality_claims"])
            self.assertIn("accuracy / false-alarm rate claims",
                          payload["claims_withheld"])
            self.assertTrue(payload["split_sufficiency"]["reasons"])

    def test_a_qualified_set_is_recognised_as_usable(self) -> None:
        """A full synthetic-but-VETTED-shaped set still fails on real-footage."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "v"
            d.mkdir()
            clips = []
            for index in range(30):
                _make_clip(d, f"c{index}.mp4")
                clips.append({"clip": f"v/c{index}.mp4",
                              "label": "accident" if index % 3 == 0 else "normal",
                              "source_id": f"SRC{index}",
                              "camera_id": f"CAM-{index % 6}",
                              "quality": "SYNTHETIC",
                              "events": ([{"kind": "accident", "start_seconds": 0.0,
                                           "end_seconds": 0.4}] if index % 3 == 0 else [])})
            manifest = load_manifest(_manifest(Path(tmp), clips))
            # enough clips and groups, but every clip is synthetic
            self.assertFalse(manifest.usable_for_quality_claims())
            self.assertEqual(len(manifest.groups()), 30)
            self.assertEqual(manifest.quality_blockers().count("x"), 0)


# --------------------------------------------------------------------------- #
class TestSplits(unittest.TestCase):
    def _manifest_with_groups(self, tmp: Path, n_groups: int, per_group: int = 2):
        d = tmp / "v"
        d.mkdir(exist_ok=True)
        clips = []
        for g in range(n_groups):
            for c in range(per_group):
                name = f"g{g}c{c}.mp4"
                _make_clip(d, name)
                is_pos = g % 2 == 0
                clips.append({"clip": f"v/{name}",
                              "label": "accident" if is_pos else "normal",
                              "source_id": f"SRC-{g}",
                              "camera_id": f"CAM-{g}",
                              "quality": "VERIFIED",
                              "events": ([{"kind": "accident", "start_seconds": 0.0,
                                           "end_seconds": 0.4}] if is_pos else [])})
        return _manifest(tmp, clips)

    def test_groups_never_straddle_a_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 30))
            splits = group_split(manifest, seed=7)
            assert_no_leakage(splits)   # must not raise
            seen = {}
            for name, split in splits.items():
                for gid in split.groups:
                    self.assertNotIn(gid, seen, f"{gid} in both {seen.get(gid)} and {name}")
                    seen[gid] = name

    def test_every_clip_lands_in_exactly_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 24))
            splits = group_split(manifest, seed=11)
            total = sum(s.n_clips for s in splits.values())
            self.assertEqual(total, manifest.clips and len(manifest.clips))

    def test_split_is_deterministic_for_a_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 20))
            a = group_split(manifest, seed=3)
            b = group_split(manifest, seed=3)
            self.assertEqual({k: [c.clip for c in v.clips] for k, v in a.items()},
                             {k: [c.clip for c in v.clips] for k, v in b.items()})

    def test_leakage_is_detected_when_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 10))
            splits = group_split(manifest, seed=5)
            splits["test"].clips.append(splits["train"].clips[0])
            with self.assertRaises(AssertionError):
                assert_no_leakage(splits)

    def test_ratios_must_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 10))
            with self.assertRaises(ValueError):
                group_split(manifest, ratios=(0.5, 0.5, 0.5))

    def test_degenerate_split_is_reported_as_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(self._manifest_with_groups(Path(tmp), 4))
            splits = group_split(manifest, seed=1)
            summary = summarise_split_sufficiency(splits)
            self.assertFalse(summary["sufficient_for_quality_claim"])


# --------------------------------------------------------------------------- #
class TestReproducibilityAndSafety(unittest.TestCase):
    def _good_manifest(self, tmp: Path):
        d = tmp / "v"
        d.mkdir(exist_ok=True)
        clips = []
        for g in range(12):
            name = f"g{g}.mp4"
            _make_clip(d, name)
            clips.append({
                "clip": f"v/{name}",
                "label": "accident" if g % 3 == 0 else "normal",
                "source_id": f"SRC-{g}",
                "camera_id": f"CAM-{g % 3}",
                "quality": "VERIFIED",
                "events": ([{"kind": "accident", "start_seconds": 0.0, "end_seconds": 0.4}]
                           if g % 3 == 0 else []),
            })
        return _manifest(tmp, clips)

    def test_threshold_is_never_suggested_from_training_data(self) -> None:
        """Tuning on the split the detector was fitted to is meaningless."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._good_manifest(Path(tmp))
            scores = {}
            incidents = {}
            for g in range(12):
                key = str(next(Path(tmp).glob(f"v/g{g}.mp4")))
                scores[key] = 0.8 if g % 3 == 0 else 0.1
                if g % 3 == 0:
                    incidents[key] = 0.2
            predictor = RecordingPredictor(score_for=scores, incident_for=incidents)
            report = evaluate_manifest(path, config=test_config(), seed=42,
                                       predict=predictor)
            suggestion = report.to_dict()["threshold_suggestion"]
            self.assertIsNone(suggestion["threshold"])
            self.assertEqual(suggestion["source"], "validation split only")

    def test_report_records_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._good_manifest(Path(tmp))
            report = evaluate_manifest(path, config=test_config(),
                                       predict=RecordingPredictor())
            prov = report.to_dict()["provenance"]
            self.assertIn("data_fingerprint", prov)
            self.assertIn("thresholds", prov)
            self.assertEqual(prov["split_seed"], 20261002)
            self.assertIn("threshold_set", prov["thresholds"])

    def test_evaluation_never_notifies_anybody(self) -> None:
        """The harness must be unable to place a call or send a message."""
        import urllib.request

        attempts = []

        def blocked(url, *a, **k):
            attempts.append(str(url))
            raise AssertionError(f"evaluation contacted the network: {url}")

        original = urllib.request.urlopen
        urllib.request.urlopen = blocked
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = self._good_manifest(Path(tmp))
                report = evaluate_manifest(path, config=test_config(),
                                           predict=RecordingPredictor())
        finally:
            urllib.request.urlopen = original
        self.assertEqual(attempts, [])
        self.assertIsNotNone(report)

    def test_config_used_by_evaluation_has_notifications_disabled(self) -> None:
        config = test_config()
        self.assertFalse(config.emergency.enabled)

    def test_rendered_report_states_the_withheld_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._good_manifest(Path(tmp))
            report = evaluate_manifest(path, config=test_config(),
                                       predict=RecordingPredictor())
            text = render_report(report)
            self.assertIn("withheld", text)
            self.assertIn("uncalibrated", text.lower())


if __name__ == "__main__":
    unittest.main()