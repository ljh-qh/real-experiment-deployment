"""CPU-only regressions for actual return helpers and main failure cleanup."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/bridge")]
import return_full_origin as returning


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSDK:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.mode = 0
        self.pending_mode = 0
        self.sends = 0

    def answer(self, key, record):
        self.calls.append(record)
        return key != self.fail

    def clear_set(self):
        return self.answer("clear", ("clear",))

    def set_joint_cmd_pose(self, arm, joints):
        return self.answer("pose_" + arm, ("pose", arm, list(joints)))

    def set_vel_acc(self, arm, velRatio, AccRatio):
        return self.answer("velocity_" + arm, ("velocity", arm, velRatio, AccRatio))

    def set_state(self, arm, state):
        ok = self.answer("state_" + arm, ("state", arm, state))
        if ok:
            self.pending_mode = state
        return ok

    def send_cmd(self):
        ok = self.answer("send", ("send",))
        if ok:
            self.sends += 1
            self.mode = self.pending_mode
        return ok


class ReturnTransactionTests(unittest.TestCase):
    def test_success_keeps_both_targets_and_mode_parameters(self):
        sdk = FakeSDK()
        left = np.linspace(-3, 3, 7)
        right = left + 10
        original_left, original_right = left.copy(), right.copy()
        returning.send_arm_targets(sdk, left, right, position_mode=True)
        self.assertEqual(sdk.calls, [
            ("clear",), ("pose", "A", left.tolist()), ("velocity", "A", 20, 20), ("state", "A", 1),
            ("pose", "B", right.tolist()), ("velocity", "B", 20, 20), ("state", "B", 1), ("send",)])
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)

    def test_plain_target_transaction_does_not_change_modes_or_speed(self):
        sdk = FakeSDK()
        returning.send_arm_targets(sdk, np.zeros(7), np.ones(7))
        self.assertEqual([c[0] for c in sdk.calls], ["clear", "pose", "pose", "send"])

    def test_any_failed_transaction_stage_prevents_later_stages(self):
        order = ["clear", "pose_A", "velocity_A", "state_A", "pose_B", "velocity_B", "state_B", "send"]
        for index, failure in enumerate(order):
            with self.subTest(failure=failure):
                sdk = FakeSDK(failure)
                with self.assertRaises(RuntimeError):
                    returning.send_arm_targets(sdk, np.zeros(7), np.zeros(7), position_mode=True)
                self.assertEqual(len(sdk.calls), index + 1)

    def test_both_targets_are_validated_before_opening_transaction(self):
        for invalid in (np.zeros(6), np.full(7, np.nan)):
            with self.subTest(shape=invalid.shape):
                sdk = FakeSDK()
                with self.assertRaises(ValueError):
                    returning.send_arm_targets(sdk, np.zeros(7), invalid)
                self.assertEqual(sdk.calls, [])

    def test_idle_retries_busy_and_failed_idle_transactions_without_pose_targets(self):
        for failure in ("clear", "state_A", "state_B", "send"):
            with self.subTest(failure=failure):
                sdk = FakeSDK()
                sdk.mode = 1
                clock = Clock()
                first = True
                original = sdk.answer

                def fail_once(key, record):
                    nonlocal first
                    if first and key == failure:
                        first = False
                        sdk.calls.append(record)
                        return False
                    return original(key, record)

                sdk.answer = fail_once
                backend = SimpleNamespace(robot=sdk, _subscribe=lambda: {"states": [
                    {"cur_state": sdk.mode}, {"cur_state": sdk.mode}]})
                with mock.patch.object(returning, "time", clock):
                    attempts = returning.confirm_idle(backend, timeout=1.1)
                self.assertEqual(attempts, 2)
                self.assertGreaterEqual(clock.now, .5)
                self.assertLess(clock.now, .55)
                self.assertFalse(any(c[0] in ("pose", "velocity") for c in sdk.calls))
                self.assertEqual(sdk.calls.count(("send",)), 2 if failure == "send" else 1)

    def test_idle_busy_timeout_never_sends_partial_transaction(self):
        sdk = FakeSDK("clear")
        sdk.mode = 1
        clock = Clock()
        backend = SimpleNamespace(robot=sdk, _subscribe=lambda: {"states": [
            {"cur_state": 1}, {"cur_state": 1}]})
        with mock.patch.object(returning, "time", clock):
            with self.assertRaisesRegex(RuntimeError, "last transaction failure: clear_set"):
                returning.confirm_idle(backend, timeout=1.1)
        self.assertEqual(sdk.calls, [("clear",), ("clear",), ("clear",)])

    def test_idle_already_confirmed_can_succeed_despite_busy_request(self):
        sdk = FakeSDK("clear")
        backend = SimpleNamespace(robot=sdk, _subscribe=lambda: {"states": [
            {"cur_state": 0}, {"cur_state": 0}]})
        with mock.patch.object(returning, "time", Clock()):
            self.assertEqual(returning.confirm_idle(backend), 1)
        self.assertEqual(sdk.calls, [("clear",)])


class ReturnMainTests(unittest.TestCase):
    def run_return(self, *, fail_trajectory=False, fail_cleanup=False):
        scratch = ROOT / "test_tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="return_mock_", dir=scratch) as folder:
            root = Path(folder)
            urdf = root / "assets/robot/marvin_dual_arm.urdf"
            urdf.parent.mkdir(parents=True)
            urdf.write_text("<robot name='mock'/>")
            output = root / "return.json"
            sdk = FakeSDK()
            original_pose = sdk.set_joint_cmd_pose

            def pose(arm, joints):
                if fail_trajectory and sdk.mode == 1:
                    sdk.calls.append(("failed_trajectory_pose", arm))
                    return False
                return original_pose(arm, joints)

            sdk.set_joint_cmd_pose = pose
            cleanup_calls = []

            def backend_close():
                cleanup_calls.append("backend")
                if fail_cleanup:
                    raise RuntimeError("mock backend release failed")

            backend = SimpleNamespace(
                robot=sdk, _verify_connection=lambda: True,
                _subscribe=lambda: {"states": [{"cur_state": sdk.mode, "err_code": 0}] * 2},
                read_joints_deg=lambda: (np.zeros(7), np.zeros(7)), close=backend_close)
            hand_writes = []

            def control_close():
                cleanup_calls.append("hands")
                if fail_cleanup:
                    raise RuntimeError("mock hand disable failed")

            control = SimpleNamespace(acquire=lambda: None, read=lambda: np.zeros(44),
                                      check_ownership=lambda: None, write=lambda q: hand_writes.append(q.copy()),
                                      close=control_close)
            serials = ("C65E9038C65F", "CA56933FCA57")

            def disconnect(serial):
                cleanup_calls.append(serial)
                if fail_cleanup and serial == serials[0]:
                    raise RuntimeError("mock left disconnect failed")
                return None

            manager = SimpleNamespace(connect=lambda *a, **k: object(), disconnect=disconnect)
            fake_sdk = SimpleNamespace(HandSide=SimpleNamespace(LEFT="left", RIGHT="right"))
            devices = [SimpleNamespace(sn=sn, hand_side=side) for sn, side in zip(serials, ("left", "right"))]
            arm_module = ModuleType("mujoco_marvin.tianji_backend")
            arm_module.TianjiJointBackend = lambda *a, **k: backend
            arm_module.DEFAULT_JOINT_LIMITS_DEG = {s: np.tile([-180., 180.], (7, 1)) for s in ("left", "right")}
            hand_module = ModuleType("mujoco_marvin.sharpa_backend")
            hand_module.load_sharpa = lambda: fake_sdk
            hand_module.discover = lambda sdk: (manager, devices)
            hand_module.LOWER_RAD = np.full(22, -np.pi)
            hand_module.UPPER_RAD = np.full(22, np.pi)
            policy_module = ModuleType("policy_hands")
            policy_module.PolicyHands = lambda *a: control
            package = ModuleType("mujoco_marvin")
            package.__path__ = []
            fake_modules = {"mujoco_marvin": package, arm_module.__name__: arm_module,
                            hand_module.__name__: hand_module, "policy_hands": policy_module}
            profile = {"urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}
            argv = ["return_full_origin", "--root", str(root), "--reference", str(root / "reference.json"),
                    "--output", str(output), "--execute"]
            with mock.patch.dict(sys.modules, fake_modules), \
                    mock.patch.object(returning, "load_reference", return_value=(np.zeros(58), profile)), \
                    mock.patch.object(returning, "time", Clock()), \
                    mock.patch.object(returning.signal, "signal"), \
                    mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = 0
                try:
                    returning.main()
                except SystemExit as exc:
                    code = exc.code
            self.assertTrue(output.is_file(), "cleanup failures must not prevent saving the return report")
            return code, json.loads(output.read_text()), hand_writes, cleanup_calls

    def test_failed_arm_transaction_does_not_dispatch_subsequent_hand_target(self):
        code, report, writes, cleanup = self.run_return(fail_trajectory=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "aborted")
        self.assertIn("set_joint_cmd_pose", report["error"])
        self.assertEqual(writes, [])
        self.assertEqual(cleanup, ["hands", "backend", "C65E9038C65F", "CA56933FCA57"])

    def test_cleanup_failures_are_all_saved_and_both_serials_are_disconnected(self):
        code, report, writes, cleanup = self.run_return(fail_cleanup=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(writes), 91)  # Original two-second return plus 31 held samples.
        self.assertEqual(report["hand_shutdown_error"], "mock hand disable failed")
        self.assertEqual(report["backend_close_error"], "mock backend release failed")
        self.assertEqual(report["manager_disconnect_errors"], {"C65E9038C65F": "mock left disconnect failed"})
        self.assertEqual(cleanup, ["hands", "backend", "C65E9038C65F", "CA56933FCA57"])

    def test_success_keeps_original_return_cadence_and_target(self):
        code, report, writes, cleanup = self.run_return()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["duration_s"], 2)
        self.assertEqual(len(report["samples"]), 91)
        self.assertTrue(all(np.array_equal(q, np.zeros(44)) for q in writes))


if __name__ == "__main__":
    unittest.main()
