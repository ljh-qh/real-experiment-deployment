"""Task selection regressions; recorded inputs and fake hardware only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P / "src"), str(P / "vendor/bridge"), str(P / "assets/robot")]
import launcher
from zmq_offline_bridge import to_openwam, REPRESENTATION
from demo_origin import load_reference
import test_client_fixes as client_tests


class TaskRoutingTests(unittest.TestCase):
    def message(self, task):
        from PIL import Image
        folder = P / "assets/references/top_observations/recorded_start"
        return {"type": "observation", "representation": REPRESENTATION, "task": task,
                "observation": {"state": np.load(folder / "state80.npy"),
                                "images": {role: np.asarray(Image.open(folder / (role + ".png")))
                                           for role in ("main", "wrist2", "wrist")}}}

    def test_exact_training_prompt_and_unchanged_badminton_prompt(self):
        for task, prompt in (("top", "pull the string to spin the top on the table"), ("bottle", "unscrew the bottle cap and pour water into the cup"), ("badminton", "badminton")):
            message = self.message(task)
            result = to_openwam(message, encode_images=False)
            self.assertEqual(result["prompt"], prompt)
            np.testing.assert_array_equal(result["state"], message["observation"]["state"])

    def test_unknown_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            to_openwam(self.message("wrong"), encode_images=False)

    def test_top_profile_origin_and_worker_agree_and_control_is_unchanged(self):
        top = launcher.read_profile(P / "configs/profiles/openwam_top_sync.yaml", P / "runs/unused_test")
        bad = launcher.read_profile(P / "configs/profiles/openwam_sync.yaml", P / "runs/unused_test")
        self.assertTrue(launcher.check_profile(top)["profile_valid"])
        self.assertEqual(top["client"], bad["client"])
        self.assertEqual(top["hardware"], bad["hardware"])
        self.assertEqual(top["worker"]["args"]["config"], bad["worker"]["args"]["config"])
        q, ref = load_reference(launcher.origin_reference(top))
        self.assertEqual(ref["task"], "top")
        self.assertEqual(ref["arm_row"], 0)
        self.assertEqual(q.shape, (58,))
        for key in ("task",):
            broken = {**top, key: "badminton"}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                launcher.check_profile(broken)
        broken = {**top, "origin_reference": str(P / "assets/references/badminton_demo_origin.json")}
        with self.assertRaises((KeyError, ValueError)):
            launcher.check_profile(broken)


    def test_bottle_profile_has_its_own_assets_and_origin(self):
        cfg = launcher.read_profile(P / "configs/profiles/openwam_bottle_sync.yaml", P / "runs/unused_test")
        top = launcher.read_profile(P / "configs/profiles/openwam_top_sync.yaml", P / "runs/unused_test")
        self.assertTrue(launcher.check_profile(cfg)["profile_valid"])
        self.assertEqual(cfg["client"], top["client"])
        self.assertEqual(cfg["hardware"], top["hardware"])
        q, ref = load_reference(launcher.origin_reference(cfg))
        self.assertEqual(ref["task"], "bottle")
        self.assertTrue(ref["stable_start_audit"]["found"])
        self.assertEqual(cfg["worker"]["args"]["checkpoint-name"], "checkpoint_step_65640.safetensors")
        bad = {**cfg, "origin_reference": str(P / "assets/references/top_demo_origin.json")}
        with self.assertRaises(ValueError):
            launcher.check_profile(bad)


class TopClientTests(unittest.TestCase):
    setUp = client_tests.ClientFixIntegrationTests.setUp
    harness = client_tests.ClientFixIntegrationTests.harness
    assert_all_closed = client_tests.ClientFixIntegrationTests.assert_all_closed

    def test_top_client_uses_selected_origin_and_sends_top_task(self):
        h = self.harness("infra_state_client.py")
        origin = h.root / "assets/references/top_demo_origin.json"
        origin.write_text(json.dumps({"target_deg": [0.0] * 58}))
        # If the client accidentally uses badminton, this test must fail.
        (h.root / "assets/references/badminton_demo_origin.json").write_text(json.dumps({"target_deg": [50.0] * 58}))
        rc, report = h.run(task="top", origin_reference=origin)
        self.assertEqual(rc, 0, report)
        self.assertEqual(h.model_requests[0]["task"], "top")
        self.assertEqual(len(report["samples"]), 1)
        self.assert_all_closed(h, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
