"""CPU timestamp regressions; no SDK/CameraFeed imports or device instances."""
import ast
import contextlib
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from observation_timing import validate_observation_timing


class ObservationTimingTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.cameras = {"main": self.now - 30_000_000,
                        "wrist2": self.now - 20_000_000,
                        "wrist": self.now - 10_000_000}
        self.state = self.now - 5_000_000

    def test_preserves_sources_and_uses_oldest_acquisition(self):
        result = validate_observation_timing(self.cameras, self.state, self.now)
        self.assertEqual(result["timestamp_ns"], self.now)
        self.assertEqual(result["source_timestamp_ns"], self.cameras["main"])
        self.assertEqual(result["oldest_capture_ns"], self.cameras["main"])
        self.assertEqual(result["camera_timestamps_ns"], self.cameras)
        self.assertEqual(result["state_timestamp_ns"], self.state)
        self.assertEqual(result["modality_skew_ms"], 25.0)

    def test_new_packaging_time_cannot_refresh_stale_images(self):
        cameras = {key: self.now - 251_000_000 for key in self.cameras}
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_observation_timing(cameras, self.now, self.now)

    def test_delayed_repackaging_rechecks_original_acquisition(self):
        first = validate_observation_timing(self.cameras, self.state, self.now)
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_observation_timing(first["camera_timestamps_ns"],
                                        first["state_timestamp_ns"], self.now + 221_000_000)

    def test_old_state_also_makes_snapshot_stale(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_observation_timing(self.cameras, self.now - 250_000_001, self.now)

    def test_exact_age_and_skew_boundaries_are_allowed(self):
        cameras = {"main": self.now - 250_000_000,
                   "wrist2": self.now - 225_000_000,
                   "wrist": self.now - 200_000_000}
        result = validate_observation_timing(cameras, self.now - 200_000_000, self.now)
        self.assertEqual(result["modality_skew_ms"], 50.0)

    def test_one_nanosecond_over_skew_is_rejected(self):
        cameras = dict(self.cameras, main=self.now - 50_000_001)
        with self.assertRaisesRegex(ValueError, "synchronized"):
            validate_observation_timing(cameras, self.now, self.now)

    def test_any_camera_or_state_in_future_is_rejected(self):
        for role in self.cameras:
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "future"):
                validate_observation_timing(dict(self.cameras, **{role: self.now + 1}),
                                            self.state, self.now)
        with self.assertRaisesRegex(ValueError, "future"):
            validate_observation_timing(self.cameras, self.now + 1, self.now)

    def test_non_integer_non_finite_boolean_and_negative_stamps_rejected(self):
        for value in (None, True, False, 1.5, 1.0, float("nan"), float("inf"),
                      float("-inf"), "1000000000", -1):
            for field in ("camera", "state", "now"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    cameras = dict(self.cameras, main=value) if field == "camera" else self.cameras
                    state = value if field == "state" else self.state
                    now = value if field == "now" else self.now
                    # None for now intentionally requests the clock, so test it as
                    # a malformed returned clock value rather than the sentinel.
                    if field == "now" and value is None:
                        with patch("observation_timing.time.monotonic_ns", return_value=None):
                            validate_observation_timing(cameras, state)
                    else:
                        validate_observation_timing(cameras, state, now)

    def test_missing_and_unexpected_camera_entries_rejected(self):
        for role in self.cameras:
            cameras = {k: v for k, v in self.cameras.items() if k != role}
            with self.subTest(missing=role), self.assertRaisesRegex(ValueError, "camera timestamps"):
                validate_observation_timing(cameras, self.state, self.now)
        for cameras in (None, [], {}, {**self.cameras, "extra": self.now}):
            with self.subTest(cameras=cameras), self.assertRaises(ValueError):
                validate_observation_timing(cameras, self.state, self.now)

    def test_out_of_order_mapping_does_not_mask_oldest_camera(self):
        cameras = {"wrist": self.cameras["wrist"], "main": self.cameras["main"],
                   "wrist2": self.cameras["wrist2"]}
        result = validate_observation_timing(cameras, self.state, self.now)
        self.assertEqual(result["source_timestamp_ns"], self.cameras["main"])
        self.assertEqual(result["camera_timestamps_ns"], self.cameras)

    def test_numpy_integer_input_is_returned_as_plain_integer(self):
        result = validate_observation_timing({k: np.int64(v) for k, v in self.cameras.items()},
                                            np.int64(self.state), np.int64(self.now))
        self.assertTrue(all(type(v) is int for v in result["camera_timestamps_ns"].values()))
        self.assertIs(type(result["state_timestamp_ns"]), int)
        self.assertIs(type(result["source_timestamp_ns"]), int)

    def test_default_clock_is_monotonic_and_read_once(self):
        with patch("observation_timing.time.monotonic_ns", return_value=self.now) as clock:
            result = validate_observation_timing(self.cameras, self.state)
        clock.assert_called_once_with()
        self.assertEqual(result["timestamp_ns"], self.now)

    def test_input_mapping_and_returned_mapping_are_independent(self):
        before = self.cameras.copy()
        result = validate_observation_timing(self.cameras, self.state, self.now)
        self.assertEqual(self.cameras, before)
        result["camera_timestamps_ns"]["main"] = 0
        self.assertEqual(self.cameras, before)


class CameraReceiptTimestampTests(unittest.TestCase):
    def test_conversion_delay_does_not_relabel_frame_as_fresh(self):
        # Execute only the existing capture method with fakes. Do not import
        # live_observation_source, CameraFeed or pyrealsense2.
        source = ROOT / "vendor/bridge/live_observation_source.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CameraFeed")
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_read"))
        clock = SimpleNamespace(now=1_000_000_000)
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        def get_data():
            clock.now += 300_000_000
            return image
        calls = iter((False, True))
        fake = SimpleNamespace(
            stop=SimpleNamespace(is_set=lambda: next(calls)),
            pipeline=SimpleNamespace(wait_for_frames=lambda timeout: SimpleNamespace(
                get_color_frame=lambda: SimpleNamespace(get_data=get_data))),
            rotate=90, lock=contextlib.nullcontext(), latest=None, error=None)
        namespace = {"np": np, "time": SimpleNamespace(monotonic_ns=lambda: clock.now)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(source), "exec"), namespace)
        namespace["_read"](fake)
        self.assertIsNone(fake.error)
        np.testing.assert_array_equal(fake.latest[0], np.rot90(image))
        self.assertEqual(fake.latest[1], 1_000_000_000)
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_observation_timing({r: fake.latest[1] for r in ("main", "wrist2", "wrist")},
                                        clock.now, clock.now)


if __name__ == "__main__":
    unittest.main(verbosity=2)
