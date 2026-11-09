"""Offline recovery tests: real client functions, fake clocks/devices only."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from observation_timing import (
    acquire_synchronized, validate_observation_timing,
    ObservationTimingError, ObservationSkewError,
)
import test_client_fixes as client_tests
from test_client_fixes import Clock, CLIENT_MODES


class ReacquisitionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.events = []
        self.calls = 0

    def capture(self, age_ms):
        self.calls += 1
        now = self.clock.monotonic_ns()
        stamps = {"main": now - int(age_ms * 1e6), "wrist2": now, "wrist": now}
        timing = validate_observation_timing(stamps, now, now)
        return ("whole fresh snapshot", timing)

    def test_transient_skew_recaptures_and_preserves_actual_timestamps(self):
        ages = iter([66, 60, 25])
        result = acquire_synchronized(lambda: self.capture(next(ages)), self.events, clock=self.clock)
        event = self.events[0]
        self.assertEqual(event["status"], "recovered")
        self.assertEqual(event["attempts"], 3)
        self.assertEqual(event["waited_ms"], 10)
        self.assertEqual(event["first_failure"]["oldest_sources"], ["main"])
        self.assertEqual(event["first_failure"]["modality_skew_ms"], 66)
        self.assertEqual(result[-1]["camera_timestamps_ns"]["main"], self.clock.ns - 25_000_000)

    def test_persistent_skew_aborts_at_budget_without_accepting_bad_input(self):
        with self.assertRaises(ObservationSkewError):
            acquire_synchronized(lambda: self.capture(70), self.events, clock=self.clock)
        self.assertEqual(self.events[0]["status"], "aborted")
        self.assertEqual(self.events[0]["waited_ms"], 100)
        self.assertEqual(self.calls, 21)

    def test_stale_future_and_hardware_errors_are_not_retried(self):
        for age, reason in ((251, "stale"), (-1, "future")):
            with self.subTest(reason=reason):
                self.events.clear()
                before = self.calls
                with self.assertRaises(ObservationTimingError):
                    acquire_synchronized(lambda: self.capture(age), self.events, clock=self.clock)
                self.assertEqual(self.calls, before + 1)
                self.assertEqual(self.events[0]["reason"], reason)
                self.assertEqual(self.events[0]["waited_ms"], 0)
        with self.assertRaisesRegex(RuntimeError, "SDK"):
            acquire_synchronized(lambda: (_ for _ in ()).throw(RuntimeError("SDK")), [], clock=self.clock)

    def test_capture_that_runs_past_retry_deadline_is_not_accepted(self):
        def capture():
            if not self.calls:
                return self.capture(60)
            self.clock.sleep(0.101)
            return self.capture(0)
        with self.assertRaisesRegex(ObservationTimingError, "100 ms"):
            acquire_synchronized(capture, self.events, clock=self.clock)
        self.assertEqual(self.events[0]["reason"], "reacquisition_timeout")

    def test_non_skew_failure_and_interrupt_during_retry_propagate(self):
        for error in (RuntimeError("camera disconnected"), KeyboardInterrupt()):
            self.calls = 0
            events = []
            def capture():
                if not self.calls:
                    return self.capture(60)
                raise error
            with self.assertRaises(type(error)):
                acquire_synchronized(capture, events, clock=self.clock)
            self.assertEqual(events[0]["status"], "aborted")


class ClientRecoveryTests(unittest.TestCase):
    # Reuse the real-control-flow harness without rerunning its whole test suite.
    setUp = client_tests.ClientFixIntegrationTests.setUp
    harness = client_tests.ClientFixIntegrationTests.harness
    assert_all_closed = client_tests.ClientFixIntegrationTests.assert_all_closed
    def install_skew(self, h, persistent=False):
        original = h.make_camera
        def camera(serial, rotation):
            c = original(serial, rotation)
            snapshot = c.snapshot
            if serial == "241122305532":
                calls = 0
                def skewed():
                    nonlocal calls
                    calls += 1
                    image, stamp = snapshot()
                    # Initial model observation is valid. Fail only on a control tick.
                    if calls >= 2 and (persistent or calls <= 3):
                        self.assertEqual(len(h.arm_commands), 1)
                        self.assertEqual(h.hand_commands, {"left": [], "right": []})
                        return image, h.clock.monotonic_ns() - 66_000_000
                    return image, stamp
                c.snapshot = skewed
            return c
        h.make_camera = camera

    def test_transient_control_tick_skew_sends_no_targets_until_recovery(self):
        for client, sync in CLIENT_MODES:
            with self.subTest(client=client, sync=sync):
                h = self.harness(client, sync)
                self.install_skew(h)
                rc, report = h.run()
                self.assertEqual(rc, 0, report)
                self.assertEqual(len(report["samples"]), 1)
                self.assertEqual(report["observation_timing_events"][0]["status"], "recovered")
                self.assertEqual(report["observation_timing_events"][0]["waited_ms"], 10)
                self.assertEqual(len(h.arm_commands), 2)
                self.assert_all_closed(h, report)

    def test_persistent_control_tick_skew_stops_and_closes_every_resource(self):
        for client, sync in CLIENT_MODES:
            with self.subTest(client=client, sync=sync):
                h = self.harness(client, sync)
                self.install_skew(h, persistent=True)
                rc, report = h.run()
                self.assertEqual(rc, 1)
                self.assertEqual(report["samples"], [])
                self.assertEqual(report["observation_timing_events"][0]["status"], "aborted")
                self.assertEqual(len(h.arm_commands), 1)
                self.assertEqual(len(h.model_requests), 1)
                self.assert_all_closed(h, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
