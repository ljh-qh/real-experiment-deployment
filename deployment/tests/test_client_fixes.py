"""Execute real client control flow with CPU-only device/transport fakes.

Only function ASTs are compiled from clients: their module-level hardware
imports and hardware constructors are never executed. Every import inside
main resolves to a stub module. The existing measured-position startup hold
is allowed; stale input must block inference and every model-generated target.
"""
import argparse
import ast
import contextlib
import copy
from concurrent.futures import Future
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from control_clock import advance_deadline
from execution_cleanup import close_execution_resources
from observation_timing import validate_observation_timing, acquire_synchronized, ObservationTimingError
from rtc_queue import RTCQueue

CLIENT_MODES = (("infra_state_client.py", True),
                ("rtc_state_client.py", True),
                ("rtc_state_client.py", False))
CAMERAS = {"head_camera": "main", "left_wrist_camera": "wrist2", "right_wrist_camera": "wrist"}
SERIAL_ROLES = {"241122305532": "main", "262422271713": "wrist2", "260522271432": "wrist"}


def source_functions(client):
    tree = ast.parse((ROOT / "src" / client).read_text())
    return [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)]


def main_node(client):
    return next(n for n in source_functions(client) if n.name == "main")


def compile_nodes(nodes, client, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(ROOT / "src" / client), "exec"), namespace)


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def monotonic_ns(self):
        return self.ns

    def monotonic(self):
        return self.ns / 1e9

    def sleep(self, seconds):
        if seconds < 0:
            raise AssertionError("negative fake sleep")
        self.ns += int(round(seconds * 1e9))


class ClientHarness:
    def __init__(self, root, client, sync=True, close_failure=None,
                 camera_problem=None, camera_constructor_failure=None,
                 transient_feedback=False, inference_delays=()):
        self.root, self.client_source, self.sync = root, client, sync
        self.clock = Clock()
        self.closed = []
        self.model_requests = []
        self.hand_commands = {"left": [], "right": []}
        self.arm_commands = []
        self.initial_hands = {"left": np.full(22, 0.2), "right": np.full(22, -0.15)}
        self.close_failure = close_failure
        self.camera_problem = camera_problem
        self.camera_constructor_failure = camera_constructor_failure
        self.transient_feedback = transient_feedback
        self.inference_delays = list(inference_delays)
        self.cameras = {}
        self.hands = {}
        self.camera_offsets_ns = {"main": 30_000_000, "wrist2": 20_000_000, "wrist": 10_000_000}
        self.output = root / "result"
        (root / "assets/references").mkdir(parents=True)
        (root / "assets/references/badminton_demo_origin.json").write_text(json.dumps({"target_deg": [0.0] * 58}))
        self.ready = root / "worker_ready.json"
        self.ready.write_text(json.dumps({"observation_endpoint": "mock://observation", "engine": "openwam_rtc", "rtc_mode": "identity_vjp"}))
        self.arm = self.make_arm()
        self.namespace = self.make_namespace()
        compile_nodes(source_functions(client), client, self.namespace)

    def make_arm(self):
        harness = self
        class FakeArm:
            connected = False
            mode = 0
            read_round = 0
            def __init__(self):
                self._robot = self
            def connect(self):
                self.connected = True
            def read_joint_position_deg(self, side):
                if side == "A":
                    self.read_round += 1
                value = 9.0 if harness.transient_feedback and self.read_round == 3 else 0.0
                return np.full(7, value)
            def enter_position_modes(self, sides):
                self.mode = 1
            def send_joint_positions_deg(self, commands, blocking=False):
                harness.arm_commands.append({side: tuple(values) for side, values in commands.items()})
                if len(harness.arm_commands) > 1:
                    harness.clock.sleep(0.02)
            def read_arm_status(self, side):
                return SimpleNamespace(cur_state=self.mode, err_code=0)
            def clear_set(self):
                return True
            def set_state(self, arm, state):
                self.mode = state
                return True
            def send_cmd(self):
                return True
            def close(self):
                harness.closed.append("arm")
                self.connected = False
        return FakeArm()

    def make_hand(self, _cfg, side, sdk_paths):
        harness = self
        class FakeHand:
            def __init__(self):
                self._hand = SimpleNamespace(get_fault_code=lambda: ())
            def connect(self):
                pass
            def read_state(self):
                return SimpleNamespace(joint_pos=harness.initial_hands[side].copy())
            def send_joint_position_rad(self, values):
                harness.hand_commands[side].append(np.asarray(values).copy())
            def close(self):
                harness.closed.append("hand/" + side)
                if harness.close_failure == side:
                    raise RuntimeError("simulated " + side + " hand close failure")
        result = FakeHand()
        self.hands[side] = result
        return result

    def make_camera(self, serial, rotation):
        role = SERIAL_ROLES[serial]
        if self.camera_constructor_failure == role:
            raise RuntimeError("simulated camera construction failure: " + role)
        harness = self
        class FakeCamera:
            def snapshot(self):
                if harness.camera_problem == (role, "missing"):
                    return None
                age = 251_000_000 if harness.camera_problem == (role, "stale") else harness.camera_offsets_ns[role]
                return np.zeros((2, 3, 3), np.uint8), harness.clock.monotonic_ns() - age
            def close(self):
                harness.closed.append("camera/" + role)
        result = FakeCamera()
        self.cameras[role] = result
        return result

    def make_transport(self, codec, observation_endpoint):
        harness = self
        class FakeTransport:
            def connect(self, metadata, session):
                pass
            def infer(self, request):
                harness.model_requests.append(copy.deepcopy(request))
                if harness.inference_delays:
                    harness.clock.sleep(harness.inference_delays.pop(0))
                action = np.zeros((32, 80), np.float32)
                action[:, 10:32] = 1.2
                action[:, 44:66] = 1.2
                return {"action": action, "roundtrip_seconds": 0.0}
            def close(self):
                harness.closed.append("transport")
        return FakeTransport()

    @staticmethod
    def thread_client(factory):
        transport = factory()
        class FakeOwnedClient:
            def submit(self, message):
                result = Future()
                try:
                    result.set_result(transport.infer(message))
                except BaseException as error:
                    result.set_exception(error)
                return result
            def close(self):
                transport.close()
        return FakeOwnedClient()

    @staticmethod
    def canonical(joints, robot):
        joints = np.asarray(joints)
        result = np.zeros((joints.shape[0], 80), np.float32)
        result[:, 10:32], result[:, 44:66] = joints[:, 14:36], joints[:, 36:58]
        return result

    def timing(self, cameras, state, now_ns=None):
        return validate_observation_timing(cameras, state,
                                          self.clock.monotonic_ns() if now_ns is None else now_ns)

    def confirm(self, read_errors, arm_limit, hand_limit, window):
        self.clock.sleep(0.02)
        arm_error, hand_error = read_errors()
        return {"recovered": arm_error <= arm_limit and hand_error <= hand_limit,
                "waited_s": 0.02, "readings": [{"arm_deg": arm_error, "hand_deg": hand_error}]}

    def make_namespace(self):
        return {
            "__name__": "cpu_client_functions", "argparse": argparse,
            "json": json, "sys": sys, "threading": threading,
            "time": self.clock, "Path": Path, "np": np,
            "ROOT": self.root, "C": ROOT / "vendor/bridge",
            "ARM_CFG": SimpleNamespace(),
            "EE_CFG": SimpleNamespace(sdk=SimpleNamespace(vendor_root=str(self.root / "sdk"))),
            "TianjiRobot": lambda cfg: self.arm, "SharpaHand": self.make_hand,
            "sharpa_manus_sdk_paths": lambda path: path,
            "configure_sharpa_runtime": lambda paths: None,
            "robot": object(), "canonical80": self.canonical, "CAMERAS": CAMERAS,
            "decode_action": lambda action: ({"left": None, "right": None}, np.r_[action[10:32], action[44:66]]),
            "PLANNERS": {side: SimpleNamespace(solve=lambda pose, seed: np.asarray(seed).copy()) for side in ("left", "right")},
            "LIMITS": np.tile([-170.0, 170.0], (7, 1)),
            "HAND_LIMITS": {side: np.tile([-2.0, 2.0], (22, 1)) for side in ("left", "right")},
            "advance_deadline": advance_deadline, "validate_observation_timing": self.timing,
            "acquire_synchronized": acquire_synchronized, "ObservationTimingError": ObservationTimingError,
            "close_execution_resources": close_execution_resources,
            "RTCQueue": RTCQueue, "ThreadOwnedClient": self.thread_client,
            "confirm_tracking": self.confirm,
        }

    def stub_modules(self):
        class FaultChecker:
            def _faults_benign(self, faults):
                return not faults
        parent = module("mujoco_marvin")
        parent.__path__ = []
        return {
            "live_observation_source": module("live_observation_source", CameraFeed=self.make_camera),
            "zmq_chunk_client": module("zmq_chunk_client", ChunkClient=self.make_transport),
            "zmq_offline_bridge": module("zmq_offline_bridge", REPRESENTATION="openwam_canonical80_mount", load_codec=lambda path: object()),
            "mujoco_marvin": parent,
            "mujoco_marvin.sharpa_backend": module("mujoco_marvin.sharpa_backend", SharpaReplayBackend=FaultChecker, load_sharpa=lambda: object()),
        }

    def run(self, hand_step=2.0, task="badminton", origin_reference=None):
        argv = [self.client_source, "--worker-ready", str(self.ready), "--output", str(self.output),
                "--duration", "0.01", "--hz", "30", "--max-joint-step-deg", "0",
                "--max-hand-step-deg", str(hand_step)]
        if self.client_source == "infra_state_client.py":
            argv.extend(["--task", task])
            if origin_reference is not None:
                argv.extend(["--origin-reference", str(origin_reference)])
        if self.sync:
            argv.append("--sync")
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, self.stub_modules()), contextlib.redirect_stdout(io.StringIO()):
            rc = self.namespace["main"]()
        return rc, json.loads((self.output / "result.json").read_text())

    def nested_capture_functions(self):
        # Bind real nested caps/msg functions in their equivalent closure values.
        for role, serial in ((role, serial) for serial, role in SERIAL_ROLES.items()):
            self.make_camera(serial, 0)
        for side in ("left", "right"):
            self.make_hand(None, side, None)
        ns = dict(self.namespace, arm=self.arm, hands=self.hands, cameras=self.cameras, report={}, args=SimpleNamespace(task="badminton"),
                  REPRESENTATION="openwam_canonical80_mount")
        ns["read_feedback"] = lambda: ({"A": np.zeros(7), "B": np.zeros(7)},
                                       np.r_[self.initial_hands["left"], self.initial_hands["right"]], np.zeros(80))
        nodes = [copy.deepcopy(n) for n in ast.walk(main_node(self.client_source))
                 if isinstance(n, ast.FunctionDef) and n.name in ("caps", "msg")]
        compile_nodes(nodes, self.client_source, ns)
        return ns


class ClientFixIntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "offline_checks/fixes_20261108/client_mock_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case_number = 0

    def harness(self, client, sync=True, **kwargs):
        self.case_number += 1
        return ClientHarness(self.root / str(self.case_number), client, sync, **kwargs)

    def assert_all_closed(self, h, report):
        self.assertEqual(set(h.closed), {"hand/left", "hand/right", "camera/main", "camera/wrist2", "camera/wrist", "transport", "arm"})
        self.assertEqual(len(h.closed), 7)
        self.assertEqual(report["transport_shutdown"]["status"], "closed")
        self.assertEqual(report["arm_connection_shutdown"]["status"], "closed")
        self.assertTrue(all(x["status"] == "closed" for x in report["camera_shutdown"].values()))

    def test_first_model_command_limits_both_hands_and_normal_shutdown_succeeds(self):
        for client, sync in CLIENT_MODES:
            for step in (2.0, 3.0):
                with self.subTest(client=client, sync=sync, hand_step=step):
                    h = self.harness(client, sync)
                    rc, report = h.run(step)
                    self.assertEqual(rc, 0, report)
                    self.assertEqual(report["status"], "completed")
                    self.assertEqual(len(report["samples"]), 1)
                    self.assertEqual(len(h.arm_commands), 2)  # original measured hold + one model command
                    np.testing.assert_array_equal(report["initial_hand_rad"], np.r_[h.initial_hands["left"], h.initial_hands["right"]])
                    for side in ("left", "right"):
                        self.assertEqual(len(h.hand_commands[side]), 1)
                        delta = h.hand_commands[side][0] - h.initial_hands[side]
                        np.testing.assert_allclose(np.rad2deg(delta), step, rtol=0, atol=1e-9)
                        self.assertEqual(report["hand_shutdown"][side]["status"], "closed")
                    self.assertNotIn("shutdown_error", report)
                    self.assert_all_closed(h, report)

    def test_independent_arm_guard_rejects_error_without_previous_hand(self):
        for client in ("infra_state_client.py", "rtc_state_client.py"):
            with self.subTest(client=client):
                h = self.harness(client)
                loop = next(n for n in ast.walk(main_node(client)) if isinstance(n, ast.While) and "args.duration" in ast.unparse(n.test))
                guard = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "prev_cmd is not None")
                q = {"A": np.full(7, 9.0), "B": np.full(7, 9.0)}
                report = {"feedback_max_arm_deg": 0.0, "feedback_max_hand_deg": 0.0, "feedback_confirmations": []}
                ns = dict(h.namespace, prev_cmd=np.zeros(14), prev_hand=None, q=q,
                          fingers=np.zeros(44), report=report, t0=0.0, i=0,
                          args=SimpleNamespace(arm_error_deg=8, hand_error_deg=15, feedback_confirm_ms=100, sync=True),
                          read_feedback=lambda: (q, np.zeros(44), np.zeros(80)))
                with self.assertRaises(RuntimeError):
                    compile_nodes([copy.deepcopy(guard)], client, ns)
                self.assertEqual(report["feedback_max_arm_deg"], 9.0)
                self.assertEqual(report["feedback_max_hand_deg"], 0.0)
                self.assertEqual(h.arm_commands, [])

    def test_capture_timestamps_reach_real_message_function_unchanged(self):
        for client in ("infra_state_client.py", "rtc_state_client.py"):
            with self.subTest(client=client):
                h = self.harness(client)
                ns = h.nested_capture_functions()
                q, fingers, state, images, timing = ns["caps"]()
                expected = {role: h.clock.monotonic_ns() - age for role, age in h.camera_offsets_ns.items()}
                h.clock.sleep(0.01)
                packet = ns["msg"](7, state, images, timing)
                self.assertEqual(packet["camera_timestamps_ns"], expected)
                self.assertEqual(packet["state_timestamp_ns"], timing["state_timestamp_ns"])
                self.assertEqual(packet["source_timestamp_ns"], min(expected.values()))
                self.assertGreater(packet["timestamp_ns"], packet["source_timestamp_ns"])
                self.assertEqual(packet["sequence"], 7)

    def test_delayed_repackaging_rejects_old_capture_before_any_request(self):
        for client in ("infra_state_client.py", "rtc_state_client.py"):
            with self.subTest(client=client):
                h = self.harness(client)
                ns = h.nested_capture_functions()
                _, _, state, images, timing = ns["caps"]()
                h.clock.sleep(0.251)
                with self.assertRaisesRegex(ValueError, "stale"):
                    ns["msg"](1, state, images, timing)
                self.assertEqual(h.model_requests, [])
                self.assertEqual(h.arm_commands, [])

    def test_stale_or_missing_camera_blocks_models_after_only_measured_startup_hold(self):
        for client, sync in CLIENT_MODES:
            for problem in ("stale", "missing"):
                with self.subTest(client=client, sync=sync, problem=problem):
                    h = self.harness(client, sync, camera_problem=("main", problem))
                    rc, report = h.run()
                    self.assertNotEqual(rc, 0)
                    self.assertEqual(report["status"], "aborted")
                    self.assertIn(problem, report["error"])
                    self.assertEqual(h.model_requests, [])
                    self.assertEqual(len(h.arm_commands), 1)
                    for values in h.arm_commands[0].values():
                        np.testing.assert_array_equal(values, np.zeros(7))
                    self.assertEqual(h.hand_commands, {"left": [], "right": []})
                    self.assert_all_closed(h, report)

    def test_either_hand_close_failure_preserves_other_cleanup_and_nonzero_exit(self):
        for client, sync in CLIENT_MODES:
            for side in ("left", "right"):
                with self.subTest(client=client, sync=sync, failed_side=side):
                    h = self.harness(client, sync, close_failure=side)
                    rc, report = h.run()
                    self.assertNotEqual(rc, 0)
                    self.assertEqual(report["status"], "aborted")
                    self.assertEqual(report["execution_status"], "completed")
                    self.assertIn("hand/" + side, report["shutdown_error"])
                    self.assertEqual(report["hand_shutdown"][side]["status"], "error")
                    other = "right" if side == "left" else "left"
                    self.assertEqual(report["hand_shutdown"][other]["status"], "closed")
                    self.assert_all_closed(h, report)

    def test_partial_camera_constructor_failure_closes_prior_camera_and_hands(self):
        for client in ("infra_state_client.py", "rtc_state_client.py"):
            with self.subTest(client=client):
                h = self.harness(client, camera_constructor_failure="wrist2")
                rc, report = h.run()
                self.assertNotEqual(rc, 0)
                self.assertEqual(report["status"], "aborted")
                self.assertEqual(h.model_requests, [])
                self.assertEqual(h.arm_commands, [])
                self.assertEqual(set(h.closed), {"hand/left", "hand/right", "camera/main", "arm"})
                self.assertEqual(report["camera_shutdown"]["main"]["status"], "closed")

    def test_rtc_recovery_recaptures_after_blocking_prime(self):
        h = self.harness("rtc_state_client.py", False, transient_feedback=True, inference_delays=(0.0, 0.4))
        rc, report = h.run()
        self.assertEqual(rc, 0, report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(h.model_requests), 2)
        self.assertTrue(report["feedback_confirmations"][0]["recovered"])
        self.assertEqual(len(report["samples"]), 1)
        self.assert_all_closed(h, report)

    def test_wrapper_preserves_main_return_code(self):
        for wrapper in ("accepted_client.py", "confirmed_client.py"):
            for return_code in (0, 1):
                with self.subTest(wrapper=wrapper, return_code=return_code):
                    out = self.root / f"{wrapper}_{return_code}"
                    out.mkdir()
                    cfg = {"hardware": {"vel_ratio": 20, "acc_ratio": 20},
                           "client": {"hz": 30, "sync": True, "duration": 0,
                                      "max_joint_step_deg": 0, "max_hand_step_deg": 2,
                                      "feedback_confirm_ms": 100}}
                    (out / "model_config.yaml").write_text(yaml.safe_dump(cfg))
                    arm = SimpleNamespace(vel_ratio=50, acc_ratio=50)
                    calls = []
                    def main():
                        calls.append(True)
                        return return_code
                    with mock.patch.object(sys, "argv", [wrapper, "--output", str(out)]), \
                         mock.patch("runpy.run_path", return_value={"ARM_CFG": arm, "main": main}), \
                         mock.patch.object(signal, "signal"), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            exec(compile((ROOT / "src" / wrapper).read_text(), str(ROOT / "src" / wrapper), "exec"),
                                 {"__name__": "__main__", "__file__": str(ROOT / "src" / wrapper)})
                    self.assertEqual(raised.exception.code, return_code)
                    self.assertEqual(calls, [True])
                    self.assertEqual((arm.vel_ratio, arm.acc_ratio), (20, 20))


if __name__ == "__main__":
    unittest.main(verbosity=2)
