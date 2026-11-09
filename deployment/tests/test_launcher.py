"""CPU-only launcher lifecycle regressions; every process and hardware call is mocked."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT / "src"))
import launcher


class FakeProcess:
    """A child with explicit wait outcomes; never starts an OS process."""

    def __init__(self, pid, waits=()):
        self.pid = pid
        self.returncode = None
        self.waits = list(waits)
        self.signals = []
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        result = self.waits.pop(0) if self.waits else (self.returncode or 0)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            result = result()
        self.returncode = result
        return result

    def send_signal(self, signum):
        self.signals.append(signum)

    def terminate(self):
        self.terminated = True
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL


class LauncherLifecycleTests(unittest.TestCase):
    def setUp(self):
        scratch = DEPLOYMENT / "test_tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="launcher_cpu_", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "runs" / "trial"
        for name in ("runs", "src", "configs/profiles", "assets/references", "envs/robot/bin"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        for name in ("src/fake_worker.py", "src/fake_client.py", "envs/robot/bin/python",
                     "assets/references/badminton_demo_origin.json"):
            (self.root / name).write_text("{}")
        self.cfg = {
            "model": "openwam",
            "worker": {
                "python": str(self.root / "envs/robot/bin/python"),
                "script": str(self.root / "src/fake_worker.py"),
                "cwd": str(self.root),
                "env": {"PYTHONPATH": str(self.root / "src")},
                "args": {"ready-file": "$OUTPUT/worker_ready.json"},
            },
            "client_entrypoint": "fake_client.py",
            "client": {"sync": True, "duration": 0, "hz": 30},
        }
        (self.root / "configs/profiles/openwam_sync.yaml").write_text(yaml.safe_dump(self.cfg))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(launcher, "ROOT", self.root))
        self.stack.enter_context(mock.patch.dict(os.environ))
        self.stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.gpu = self.stack.enter_context(mock.patch.object(
            launcher.subprocess, "run", return_value=subprocess.CompletedProcess(["mock-nvidia-smi"], 0, "")))
        self.popen = self.stack.enter_context(mock.patch.object(
            launcher.subprocess, "Popen", side_effect=AssertionError("unexpected child process")))
        # Guard against future process-group cleanup accidentally signaling a real PID.
        self.killpg = self.stack.enter_context(mock.patch.object(launcher.os, "killpg"))
        self.confirm = self.stack.enter_context(mock.patch.object(
            launcher, "confirm_stop", return_value={"status": "idle_confirmed", "after": [0, 0]}))
        self.robot_environment = self.stack.enter_context(mock.patch.object(
            launcher, "robot_env", return_value={
                "PYTHONPATH": str(self.root / "src"),
                "MARVIN_SDK_DIR": str(self.root / "sdk/tianji"),
                "SHARPA_SDK_PY": str(self.root / "sdk/sharpa/python"),
            }))
        self.handlers = {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
        self.signal_calls = []

        def set_handler(signum, handler):
            previous = self.handlers.get(signum, signal.SIG_DFL)
            self.handlers[signum] = handler
            self.signal_calls.append((signum, handler))
            return previous

        self.stack.enter_context(mock.patch.object(launcher.signal, "signal", side_effect=set_handler))
        self.stack.enter_context(mock.patch.object(
            launcher.signal, "getsignal", side_effect=lambda signum: self.handlers[signum]))

    def main(self, *options):
        with mock.patch.object(sys, "argv", ["launcher", "--output", str(self.output), *options]):
            return launcher.main()

    def start_config(self):
        self.output.mkdir()
        return launcher.read_profile(self.root / "configs/profiles/openwam_sync.yaml", self.output)

    def test_default_and_explicit_check_are_offline(self):
        before_modules = set(sys.modules)
        with mock.patch.object(launcher, "start_worker") as start:
            for options in ((), ("--check",)):
                with self.subTest(options=options):
                    self.assertEqual(self.main(*options), 0)
                    start.assert_not_called()
                    self.popen.assert_not_called()
                    self.gpu.assert_not_called()
                    self.confirm.assert_not_called()
                    self.robot_environment.assert_not_called()
                    self.assertFalse(self.output.exists())
        new_modules = set(sys.modules) - before_modules
        self.assertFalse({"sharpa", "fx_robot", "mujoco_marvin.tianji_backend"} & new_modules)

    def test_worker_only_never_confirms_hardware_even_when_interrupted(self):
        worker = FakeProcess(51001, waits=[KeyboardInterrupt()])
        with mock.patch.object(launcher, "start_worker", return_value=(worker, self.output / "worker_ready.json")):
            self.assertEqual(self.main("--worker-only"), 130)
        self.popen.assert_not_called()
        self.confirm.assert_not_called()
        self.assertTrue(worker.terminated or self.killpg.called)
        stage = json.loads((self.output / "stage.json").read_text())
        self.assertEqual(stage["stage"], "stopped")
        self.assertFalse(stage["stop_confirmation"]["hardware_connected"])

    def test_worker_only_normal_exit_never_confirms_hardware(self):
        worker = FakeProcess(51002, waits=[0])
        with mock.patch.object(launcher, "start_worker", return_value=(worker, self.output / "worker_ready.json")):
            self.assertEqual(self.main("--worker-only"), 0)
        self.popen.assert_not_called()
        self.confirm.assert_not_called()

    def test_start_worker_reaps_process_if_pid_recording_fails(self):
        cfg = self.start_config()
        worker = FakeProcess(51003)
        self.popen.side_effect = None
        self.popen.return_value = worker
        original_write = Path.write_text

        def write_or_fail(path, content, *args, **kwargs):
            if path.name == "worker.pid":
                raise OSError("simulated PID record failure")
            return original_write(path, content, *args, **kwargs)

        with mock.patch.object(Path, "write_text", new=write_or_fail):
            with self.assertRaisesRegex(OSError, "PID record"):
                launcher.start_worker(cfg)
        self.assertTrue(worker.terminated or self.killpg.called)
        self.assertTrue(worker.wait_calls)
        self.confirm.assert_not_called()

    def test_start_worker_reaps_process_if_readiness_wait_is_interrupted(self):
        cfg = self.start_config()
        worker = FakeProcess(51004)
        self.popen.side_effect = None
        self.popen.return_value = worker
        with mock.patch.object(launcher.time, "sleep", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                launcher.start_worker(cfg)
        self.assertTrue(worker.terminated or self.killpg.called)
        self.assertTrue(worker.wait_calls)
        self.confirm.assert_not_called()

    def test_aborted_client_report_is_failure_despite_zero_process_exit(self):
        worker = FakeProcess(51005)
        returned = FakeProcess(51006, waits=[0])

        def report_aborted():
            (self.output / "result.json").write_text(json.dumps({
                "status": "aborted", "error": "feedback watchdog", "idle_states": [0, 0]}))
            return 0

        client = FakeProcess(51007, waits=[report_aborted])
        self.popen.side_effect = [returned, client]
        with mock.patch.object(launcher, "start_worker", return_value=(worker, self.output / "worker_ready.json")):
            self.assertEqual(self.main("--run"), 1)
        self.confirm.assert_called_once_with(self.output)
        self.assertTrue(worker.terminated or self.killpg.called)

    def test_teardown_ignores_second_interrupts_until_motion_child_is_reaped(self):
        events = []
        worker = FakeProcess(51008)

        def reaped_after_interrupt():
            self.assertEqual(self.handlers[signal.SIGINT], signal.SIG_IGN)
            self.assertEqual(self.handlers[signal.SIGTERM], signal.SIG_IGN)
            events.append("child_reaped")
            return 0

        child = FakeProcess(51009, waits=[KeyboardInterrupt(), reaped_after_interrupt])
        self.popen.side_effect = None
        self.popen.return_value = child

        def confirm_only_after_reaping(output):
            self.assertEqual(events, ["child_reaped"])
            self.assertEqual(child.poll(), 0)
            events.append("stop_confirmed")
            return {"status": "idle_confirmed", "after": [0, 0]}

        self.confirm.side_effect = confirm_only_after_reaping
        with mock.patch.object(launcher, "start_worker", return_value=(worker, self.output / "worker_ready.json")):
            self.assertEqual(self.main("--run"), 130)
        self.assertEqual(events, ["child_reaped", "stop_confirmed"])
        self.assertEqual(child.signals, [signal.SIGINT])
        self.assertTrue(worker.terminated or self.killpg.called)


if __name__ == "__main__":
    unittest.main()
