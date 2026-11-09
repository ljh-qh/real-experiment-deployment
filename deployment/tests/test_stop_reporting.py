"""CPU-only stop-report merge and wrapper exit-status integration tests."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import launcher


class StopReportTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "test_tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="stop_report_", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.arms = {"status": "idle_confirmed", "scope": "arms", "after": [0, 0]}

    def merge(self, payload, name="result.json", arms=None):
        (self.output / name).write_text(json.dumps(payload))
        return launcher.merge_motion_shutdown(self.output, self.arms if arms is None else arms, name)

    def assert_saved(self, result):
        saved = json.loads((self.output / "stop_confirmation.json").read_text())
        self.assertEqual(saved, result)

    def test_arms_idle_without_child_report_does_not_confirm_hands(self):
        result = launcher.merge_motion_shutdown(self.output, self.arms, "result.json")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["arm_status"], "idle_confirmed")
        self.assertEqual(result["hands_status"], "unconfirmed")
        self.assert_saved(result)

    def test_missing_or_partial_hand_reports_never_count_as_closed(self):
        for hands in ({}, {"left": {"status": "closed"}}, {"left": {"status": "closed"}, "right": {"status": "error", "error": "disable failed"}}):
            with self.subTest(hands=hands):
                result = self.merge({"status": "completed", "hand_shutdown": hands})
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["hands_status"], "unconfirmed")
                self.assert_saved(result)

    def test_hand_cleanup_failure_remains_visible_when_arms_are_idle(self):
        result = self.merge({
            "status": "aborted", "shutdown_error": "hand/right: disable failed",
            "hand_shutdown": {"left": {"status": "closed"}, "right": {"status": "error", "error": "disable failed"}}})
        self.assertEqual(result["arm_status"], "idle_confirmed")
        self.assertEqual(result["status"], "error")
        self.assertIn("disable failed", result["shutdown_error"])
        self.assertEqual(result["hand_shutdown"]["right"]["status"], "error")
        self.assert_saved(result)

    def test_both_hand_reports_are_required_and_input_arm_result_is_not_mutated(self):
        original = dict(self.arms)
        result = self.merge({"status": "completed", "hand_shutdown": {
            "left": {"status": "closed"}, "right": {"status": "closed"}}})
        self.assertEqual(result["status"], "idle_confirmed")
        self.assertEqual(result["hands_status"], "closed")
        self.assertEqual(result["arm_status"], "idle_confirmed")
        self.assertEqual(self.arms, original)
        self.assert_saved(result)

    def test_hand_success_cannot_override_arm_confirmation_failure(self):
        result = self.merge({"status": "completed", "hand_shutdown": {
            "left": {"status": "closed"}, "right": {"status": "closed"}}},
            arms={"status": "error", "scope": "arms", "error": "idle not acknowledged"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["arm_status"], "error")
        self.assertEqual(result["hands_status"], "closed")
        self.assertEqual(result["error"], "idle not acknowledged")

    def test_return_backend_and_manager_errors_are_not_lost(self):
        for key, value, expected in (
            ("backend_close_error", "release failed", "release failed"),
            ("manager_disconnect_errors", {"left-sn": "disconnect failed"}, "disconnect failed"),
            ("hand_shutdown_error", "disable failed", "disable failed"),
        ):
            with self.subTest(field=key):
                result = self.merge({"status": "completed", "hands_shutdown": "disabled; SDK source",
                                     key: value}, name="return.json")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["arm_status"], "idle_confirmed")
                self.assertIn(expected, result["shutdown_error"])
                self.assert_saved(result)

    def test_return_requires_explicit_successful_hand_shutdown(self):
        for payload in ({"status": "aborted"}, {"status": "completed", "hands_shutdown": "unknown"}):
            with self.subTest(payload=payload):
                result = self.merge(payload, name="return.json")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["hands_status"], "unconfirmed")
        result = self.merge({"status": "completed", "hands_shutdown": "disabled; SDK source"}, name="return.json")
        self.assertEqual(result["status"], "idle_confirmed")
        self.assertEqual(result["hands_status"], "closed")

    def test_invalid_json_and_valid_json_with_invalid_shape_fail_closed(self):
        texts = (
            "{", "null", "[]",
            json.dumps({"hand_shutdown": None}),
            json.dumps({"hand_shutdown": ["left", "right"]}),
            json.dumps({"hand_shutdown": {"left": "closed", "right": "closed"}}),
            json.dumps({"hand_shutdown": {"left": {}, "right": {}}}),
        )
        for text in texts:
            with self.subTest(text=text):
                (self.output / "result.json").write_text(text)
                result = launcher.merge_motion_shutdown(self.output, self.arms, "result.json")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["arm_status"], "idle_confirmed")
                self.assertEqual(result["hands_status"], "unconfirmed")
                self.assert_saved(result)


class WrapperExitStatusTests(unittest.TestCase):
    def run_wrapper(self, wrapper, expected_source, status):
        scratch = ROOT / "test_tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wrapper_mock_", dir=scratch) as folder:
            output = Path(folder)
            cfg = {
                "hardware": {"vel_ratio": 20, "acc_ratio": 20},
                "client": {"hz": 30, "sync": True, "duration": 0, "max_joint_step_deg": 0,
                           "max_hand_step_deg": 2, "feedback_confirm_ms": 100},
            }
            (output / "model_config.yaml").write_text(yaml.safe_dump(cfg))
            main = mock.Mock(return_value=status)
            fake_globals = {"ARM_CFG": SimpleNamespace(vel_ratio=50, acc_ratio=50), "main": main}
            path = ROOT / "src" / wrapper
            with mock.patch.object(runpy, "run_path", return_value=fake_globals) as load, \
                    mock.patch.object(signal, "signal"), \
                    mock.patch.object(sys, "path", list(sys.path)), \
                    mock.patch.object(sys, "argv", [wrapper, "--output", str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_info:
                    exec(compile(path.read_text(), str(path), "exec"),
                         {"__name__": "__main__", "__file__": str(path)})
            self.assertEqual(exit_info.exception.code, status)
            main.assert_called_once_with()
            self.assertEqual(Path(load.call_args.args[0]).name, expected_source)
            effective = json.loads((output / "effective_config.json").read_text())
            self.assertEqual(effective["vel_ratio"], 20)
            self.assertEqual(effective["acc_ratio"], 20)

    def test_accepted_wrapper_preserves_success_and_failure_exit_codes(self):
        for status in (0, 1):
            with self.subTest(status=status):
                self.run_wrapper("accepted_client.py", "infra_state_client.py", status)

    def test_confirmed_wrapper_preserves_success_and_failure_exit_codes(self):
        for status in (0, 1):
            with self.subTest(status=status):
                self.run_wrapper("confirmed_client.py", "rtc_state_client.py", status)


class RTCRecoveryTimestampTests(unittest.TestCase):
    def test_recovery_recaptures_after_slow_prime_before_building_next_request(self):
        import ast
        import copy
        import numpy as np
        from observation_timing import validate_observation_timing

        path = ROOT / "src/rtc_state_client.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in ast.walk(main)
                    if isinstance(n, ast.While) and "args.duration" in ast.unparse(n.test))
        feedback = next(n for n in loop.body
                        if isinstance(n, ast.If) and ast.unparse(n.test) == "prev_cmd is not None")
        request = next(n for n in loop.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "request" for t in n.targets))
        msg = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == "msg")
        elapsed = [1.0]
        captures = []
        primes = []

        def validate(stamps, state):
            return validate_observation_timing(stamps, state, now_ns=int(elapsed[0] * 1e9))

        def caps():
            stamp = int(elapsed[0] * 1e9)
            captures.append(stamp)
            timing = validate({role: stamp for role in ("main", "wrist2", "wrist")}, stamp)
            return {"A": np.zeros(7), "B": np.zeros(7)}, np.zeros(44), np.zeros(80), {}, timing

        def reset():
            elapsed[0] += .1

        def prime(message):
            primes.append(message)
            elapsed[0] += .4  # More than the observation freshness limit.

        namespace = {
            "np": np, "args": SimpleNamespace(arm_error_deg=8, hand_error_deg=15,
                feedback_confirm_ms=100, sync=False, task="badminton"), "prev_cmd": np.zeros(14),
            "prev_hand": np.zeros(44), "q": {"A": np.full(7, 9.), "B": np.zeros(7)},
            "fingers": np.zeros(44), "report": {"feedback_max_arm_deg": 0,
                "feedback_max_hand_deg": 0, "feedback_confirmations": []},
            "i": 7, "t0": 0, "seq": 1,
            "time": SimpleNamespace(monotonic=lambda: elapsed[0]),
            "confirm_tracking": lambda *args: {"recovered": True, "readings": []},
            "read_feedback": lambda: self.fail("native feedback should not be called by the mock confirmation"),
            "rtc_queue": SimpleNamespace(reset=reset, prime=prime),
            "caps": caps, "validate_observation_timing": validate,
            "REPRESENTATION": "openwam_canonical80_mount",
        }
        code = compile(ast.fix_missing_locations(ast.Module(
            body=copy.deepcopy([msg, feedback, request]), type_ignores=[])), str(path), "exec")
        exec(code, namespace)
        self.assertEqual(len(primes), 1)
        self.assertEqual(len(captures), 2)
        self.assertGreater(namespace["request"]["source_timestamp_ns"], primes[0]["source_timestamp_ns"])
        self.assertEqual(namespace["request"]["source_timestamp_ns"], captures[-1])
        self.assertEqual(captures[-1], int(elapsed[0] * 1e9))


if __name__ == "__main__":
    unittest.main()
