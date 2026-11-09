"""CPU-only execution regression tests; no SDK loading or hardware connections."""
import ast
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "src"
UPSTREAM = ROOT / "tests/fixtures/legacy_infra_state_client.py"
sys.path[:0] = [str(ROOT / "deps/infra"), str(HERE), str(ROOT / "vendor/universal_robot_infra/src")]
from control_clock import advance_deadline
from tianji_command_adapter import ReliableTianjiRobot

SIM_RESULTS = []


def simulate(source, latency, stalls=None):
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.While) and "args.duration" in ast.unparse(n.test))
    loop.test = ast.parse("i < 96", mode="eval").body
    body = []
    for n in loop.body:
        if ast.unparse(n).startswith("if prev_cmd is not None"):
            continue
        body.append(n)
        if ast.unparse(n) == "action = buffer.pop(0)":
            break
    # Retain the real deadline advancement code, replace only hardware dispatch.
    tail = []
    collecting = False
    for n in loop.body:
        if ast.unparse(n) == "seed_deg = q_deg_cmd":
            collecting = True
            continue
        if collecting:
            tail.append(n)
    loop.body = body + ast.parse("record(action)\nfinished_at = time.monotonic()").body + tail
    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(source), "exec")

    class Clock:
        t = 0.0
        def monotonic(self):
            return self.t
        def sleep(self, duration):
            assert duration >= 0
            self.t += duration

    clock = Clock()
    events = []
    generated = 1

    class Client:
        def infer(self, message):
            nonlocal generated
            clock.sleep(latency)
            chunk = [[generated, k] for k in range(32)]
            generated += 1
            return {"action": chunk}

    def record(action):
        clock.sleep(0.0015 + (stalls or {}).get(len(events), 0))
        events.append((clock.t, tuple(action)))

    capture_count = next(len(n.targets[0].elts) for n in ast.walk(loop)
                         if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name) and n.value.func.id == "caps")
    env = dict(args=SimpleNamespace(duration=0, sync=True, prefetch_delay=16),
               t0=0.0, next_tick=0.0, i=0, period=1/30, time=clock,
               caps=lambda: (None,) * capture_count, client=Client(),
               msg=lambda *a: None, seq=0, pending=None,
               np=SimpleNamespace(asarray=lambda a, dtype: a),
               buffer=[[0, k] for k in range(32)], report={"clock_resets": []},
               record=record, advance_deadline=advance_deadline)
    exec(code, env)
    dt = [b[0]-a[0] for a, b in zip(events, events[1:])]
    return events, dt, env["report"]["clock_resets"]


class ClockTests(unittest.TestCase):
    def test_stalls_no_burst_and_same_actions(self):
        for latency in (0.0, 0.18, 0.70):
            for stalls in (None, {47: 0.25, 70: 0.04}):
                with self.subTest(latency=latency, stalls=stalls):
                    old, old_dt, _ = simulate(UPSTREAM, latency, stalls)
                    new, new_dt, resets = simulate(HERE/"infra_state_client.py", latency, stalls)
                    self.assertEqual([e[1] for e in old], [e[1] for e in new])
                    self.assertEqual(len(new), 96)
                    self.assertGreaterEqual(min(new_dt), 1/30-1e-10)
                    if latency or stalls:
                        self.assertLess(min(old_dt), 0.01)
                        self.assertTrue(resets)
                    SIM_RESULTS.append({
                        "inference_seconds": latency, "send_stalls": stalls,
                        "old_min_interval_ms": min(old_dt)*1000,
                        "new_min_interval_ms": min(new_dt)*1000,
                        "clock_resets": len(resets), "identical_action_sequence": True})

    def test_normal_deadline_and_overrun(self):
        self.assertEqual(advance_deadline(1.0, 1.001, 0.1), (1.1, False))
        self.assertEqual(advance_deadline(1.0, 1.7, 0.1), (1.8, True))

    def test_action_transforms_and_thresholds_unchanged(self):
        def action_body(path):
            t = ast.parse(path.read_text())
            m = next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="main")
            loop = next(n for n in ast.walk(m) if isinstance(n,ast.While) and "args.duration" in ast.unparse(n.test))
            lines = []; include = False
            for n in loop.body:
                s = ast.unparse(n)
                if s == "action = buffer.pop(0)": include = True
                if s.startswith('report[\'samples\'].append') or s == "finished_at = time.monotonic()": break
                if include: lines.append(ast.dump(n))
            return lines
        self.assertEqual(action_body(UPSTREAM), action_body(HERE/"infra_state_client.py"))
        new = (HERE/"infra_state_client.py").read_text()
        self.assertIn("if arm_err > 8.0 or hand_err > 15.0:", new)
        self.assertIn("while args.duration <= 0 or", new)


class SDK:
    def __init__(self, clears=(True,), sends=(True,), fail_arm=None):
        self.clears = iter(clears); self.sends = iter(sends)
        self.fail_arm = fail_arm; self.calls = []
    def clear_set(self):
        value = next(self.clears)
        self.calls.append(("clear", value))
        return value
    def set_joint_cmd_pose(self, arm, joints):
        self.calls.append(("set", arm, tuple(joints)))
        return arm != self.fail_arm
    def send_cmd(self):
        value = next(self.sends)
        self.calls.append(("send", value))
        return value


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.commands = {"A": tuple(range(7)), "B": tuple(-x for x in range(7))}
        self.robot = ReliableTianjiRobot(SimpleNamespace(send_cmd_retries=2, send_cmd_retry_sleep_s=0.02))
        self.waits = []
        self.patch = patch("tianji_command_adapter.time.sleep", side_effect=self.waits.append)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_busy_then_ready_preserves_both_targets(self):
        sdk = SDK(clears=(False, True)); self.robot._robot = sdk
        self.assertTrue(self.robot._send_joint_positions_deg_with_retry(self.commands))
        self.assertEqual(sdk.calls, [("clear",False),("clear",True),
                         ("set","A",self.commands["A"]),("set","B",self.commands["B"]),("send",True)])
        self.assertEqual(self.waits, [0.02])

    def test_persistent_busy_bounded_no_write(self):
        sdk = SDK(clears=(False,False,False)); self.robot._robot = sdk
        with self.assertRaisesRegex(RuntimeError, "clear_set after 3 attempts"):
            self.robot._send_joint_positions_deg_with_retry(self.commands)
        self.assertEqual(sdk.calls, [("clear",False)]*3)
        self.assertEqual(self.waits, [0.02,0.02])

    def test_failed_setter_aborts_without_send_or_retry(self):
        for side in ("A","B"):
            with self.subTest(side=side):
                sdk = SDK(fail_arm=side); self.robot._robot = sdk
                with self.assertRaisesRegex(RuntimeError, "set_joint_cmd_pose failed for arm "+side):
                    self.robot._send_joint_positions_deg_with_retry(self.commands)
                self.assertFalse(any(c[0]=="send" for c in sdk.calls))
                self.assertEqual(sum(c[0]=="clear" for c in sdk.calls),1)

    def test_send_retry_rebuilds_whole_transaction(self):
        sdk=SDK(clears=(True,True),sends=(False,True));self.robot._robot=sdk
        self.assertTrue(self.robot._send_joint_positions_deg_with_retry(self.commands))
        self.assertEqual([c for c in sdk.calls if c[0]=="set"],
                         [("set","A",self.commands["A"]),("set","B",self.commands["B"])]*2)

    def test_persistent_send_failure_has_correct_phase(self):
        sdk=SDK(clears=(True,)*3,sends=(False,)*3);self.robot._robot=sdk
        with self.assertRaisesRegex(RuntimeError,"send_cmd after 3 attempts"):
            self.robot._send_joint_positions_deg_with_retry(self.commands)

    def test_invalid_targets_rejected_before_sdk(self):
        sdk=SDK();self.robot._robot=sdk
        for commands in ({"A": (0,)*6}, {"A": (math.nan,)*7}, {"C": (0,)*7}):
            with self.assertRaises(ValueError):
                self.robot._send_joint_positions_deg_with_retry(commands)
        self.assertEqual(sdk.calls,[])

    def test_interrupt_is_not_swallowed(self):
        self.robot._robot=SimpleNamespace(clear_set=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        with self.assertRaises(KeyboardInterrupt):
            self.robot._send_joint_positions_deg_with_retry(self.commands)


class EntrypointTests(unittest.TestCase):
    def test_wrapper_uses_fixed_client_and_accepted_config(self):
        import runpy
        import yaml
        scratch=HERE/"test_tmp";scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            out=Path(directory)
            cfg=yaml.safe_load((ROOT/"configs/profiles/openwam_sync.yaml").read_text())
            (out/"model_config.yaml").write_text(yaml.safe_dump(cfg))
            arm=SimpleNamespace(vel_ratio=50,acc_ratio=50)
            called=[]
            def fake_run(path,run_name):
                self.assertEqual(Path(path),HERE/"infra_state_client.py")
                return {"ARM_CFG":arm,"main":lambda: called.append(True)}
            argv=["accepted_client.py","--output",str(out),"--duration","0","--sync","--max-hand-step-deg","2"]
            with patch.object(sys,"argv",argv),patch("runpy.run_path",side_effect=fake_run):
                with self.assertRaises(SystemExit) as stopped:
                    exec(compile((HERE/"accepted_client.py").read_text(),str(HERE/"accepted_client.py"),"exec"),
                         {"__name__":"__main__","__file__":str(HERE/"accepted_client.py")})
                self.assertIn(stopped.exception.code, (None, 0))
            self.assertEqual((arm.vel_ratio,arm.acc_ratio),(20,20))
            self.assertEqual(called,[True])
            self.assertEqual(json.loads((out/"effective_config.json").read_text())["duration"],0)


if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "simulation":SIM_RESULTS,"hardware_connected":False,"gpu_used":False}
    (ROOT/"offline_checks/test_execution_report.json").write_text(json.dumps(report,indent=2))
    raise SystemExit(not result.wasSuccessful())
