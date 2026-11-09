"""CPU-only regressions for the real SharpaHand.close method."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("vendor/universal_robot_infra/src", "deps/infra", "deps/observer")]
from unirobot_infra.robots.end_effector.sharpa import SharpaHand


class CleanupTests(unittest.TestCase):
    def make_hand(self, *, disable=None, stop=True, disconnect=None, enabled=True, started=True):
        calls = []
        device = SimpleNamespace()
        manager = SimpleNamespace()

        def invoke(name, value, *args):
            calls.append((name, *args))
            if isinstance(value, Exception):
                raise value
            return value

        device.set_enable_state = lambda value: invoke(
            "disable", disable if disable is not None else SimpleNamespace(code=0, message=""), value)
        device.stop = lambda: invoke("stop", stop)
        manager.disconnect = lambda serial: invoke("disconnect", disconnect, serial)
        # Bypass construction/configuration; no native SDK is imported or connected.
        hand = object.__new__(SharpaHand)
        hand.side = "left"
        hand._sn = "mock-left"
        hand._hand = device
        hand._manager = manager
        hand._enabled = enabled
        hand._started = started
        hand.connected = True
        return hand, calls

    def assert_detached(self, hand):
        self.assertFalse(hand.connected)
        self.assertFalse(hand._enabled)
        self.assertFalse(hand._started)
        self.assertIsNone(hand._hand)
        self.assertIsNone(hand._manager)
        self.assertEqual(hand.sn, "")

    def test_success_preserves_disable_stop_disconnect_order_and_is_idempotent(self):
        hand, calls = self.make_hand()
        self.assertIsNone(hand.close())
        self.assertEqual(calls, [("disable", False), ("stop",), ("disconnect", "mock-left")])
        self.assert_detached(hand)
        hand.close()
        self.assertEqual(len(calls), 3)

    def test_disable_sdk_error_still_stops_and_disconnects_then_raises(self):
        hand, calls = self.make_hand(disable=SimpleNamespace(code=7, message="disable rejected"))
        with self.assertRaisesRegex(RuntimeError, "left hand mock-left.*disable.*disable rejected"):
            hand.close()
        self.assertEqual([x[0] for x in calls], ["disable", "stop", "disconnect"])
        self.assert_detached(hand)

    def test_stop_false_still_disconnects_and_error_is_visible(self):
        hand, calls = self.make_hand(stop=False)
        with self.assertRaisesRegex(RuntimeError, "stop: Sharpa stop failed"):
            hand.close()
        self.assertEqual([x[0] for x in calls], ["disable", "stop", "disconnect"])
        self.assert_detached(hand)

    def test_all_step_exceptions_are_aggregated(self):
        hand, calls = self.make_hand(
            disable=RuntimeError("disable transport"), stop=RuntimeError("stop transport"),
            disconnect=RuntimeError("disconnect transport"))
        with self.assertRaises(RuntimeError) as raised:
            hand.close()
        for text in ("disable transport", "stop transport", "disconnect transport"):
            self.assertIn(text, str(raised.exception))
        self.assertEqual([x[0] for x in calls], ["disable", "stop", "disconnect"])
        self.assert_detached(hand)

    def test_disconnect_false_is_not_silently_accepted(self):
        hand, calls = self.make_hand(disconnect=False)
        with self.assertRaisesRegex(RuntimeError, "disconnect: Sharpa disconnect failed"):
            hand.close()
        self.assertEqual(len(calls), 3)
        self.assert_detached(hand)

    def test_partial_connect_cleanup_only_attempts_owned_steps(self):
        hand, calls = self.make_hand(started=False)
        hand.close()
        self.assertEqual(calls, [("disable", False), ("disconnect", "mock-left")])
        self.assert_detached(hand)

    def test_no_sdk_modules_are_loaded(self):
        self.assertNotIn("sharpa", sys.modules)
        self.assertNotIn("fx_robot", sys.modules)


if __name__ == "__main__":
    unittest.main()
