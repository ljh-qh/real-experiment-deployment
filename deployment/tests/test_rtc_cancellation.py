"""CPU-only regression coverage for RTC Future cancellation and callback races."""
from concurrent.futures import CancelledError, Future
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rtc_queue import RTCQueue


class ObservedFuture(Future):
    """Expose entry into result() without sleeps or executor scheduling guesses."""

    def __init__(self):
        super().__init__()
        self.result_entered = threading.Event()

    def result(self, timeout=None):
        self.result_entered.set()
        return super().result(timeout)


class GatedPlanFuture(Future):
    """Pause a callback before it claims ownership of its derived Future."""

    def __init__(self):
        super().__init__()
        self.claim_entered = threading.Event()
        self.release_claim = threading.Event()

    def set_running_or_notify_cancel(self):
        self.claim_entered.set()
        if not self.release_claim.wait(3):
            raise AssertionError("Test did not release callback claim")
        return super().set_running_or_notify_cancel()


class RTCQueueCancellationTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.repeat(np.arange(32, dtype=np.float32)[:, None], 80, axis=1)
        self.requests = []
        self.sources = []
        self.anchors = []

    @staticmethod
    def packets(actions):
        return [{"physical": row.copy(), "command": row.copy()} for row in actions]

    def prepare(self, actions, anchor):
        self.anchors.append(anchor["command"].copy())
        return self.packets(actions)

    def submit(self, request):
        self.requests.append(request)
        source = ObservedFuture()
        if "rtc" not in request:
            source.set_result({"action": self.initial.copy(), "roundtrip_seconds": 0.2})
        else:
            self.sources.append(source)
        return source

    def make_queue(self, prepare=True):
        queue = RTCQueue(
            self.submit,
            self.packets,
            plan_from_anchor=self.prepare if prepare else None,
        )
        queue.prime({"sequence": 0})
        return queue

    def request_pending(self, queue):
        for step in range(13):
            packet = queue.next(step, step / 30, {"sequence": step})
            self.assertEqual(packet["command"][0], step)
        self.assertIsNotNone(queue.pending)
        self.assertEqual(queue.pending["adopt_step"], 26)
        return self.sources[-1], queue.pending["future"]

    def reach_adoption(self, queue):
        for step in range(13, 26):
            queue.next(step, step / 30, {"sequence": step})

    def reply(self, offset=100):
        return {"action": self.initial + offset, "roundtrip_seconds": 0.4}

    def launch(self, function):
        errors = []
        done = threading.Event()

        def run():
            try:
                function()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, done, errors

    def finish_thread(self, thread, done, errors):
        self.assertTrue(done.wait(3), "Thread did not finish")
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def assert_reset(self, queue):
        self.assertIsNone(queue.pending)
        self.assertEqual(len(queue.queue), 0)
        self.assertEqual(queue.events[-1], {"event": "reset"})

    def test_reset_cancels_pending_source_and_derived_future(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        with self.assertNoLogs("concurrent.futures", level="ERROR"):
            queue.reset()
        self.assertTrue(source.cancelled())
        self.assertTrue(prepared.cancelled())
        self.assertEqual(self.anchors, [])
        self.assert_reset(queue)

    def test_reset_cancels_pending_source_without_planning_callback(self):
        queue = self.make_queue(prepare=False)
        source, prepared = self.request_pending(queue)
        self.assertIs(source, prepared)
        queue.reset()
        self.assertTrue(source.cancelled())
        self.assert_reset(queue)

    def test_upstream_cancellation_stays_cancellation(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        self.assertTrue(source.cancel())
        self.assertTrue(prepared.cancelled())
        queue.reset()
        self.assert_reset(queue)
        self.assertEqual(self.anchors, [])

    def test_reset_waits_for_running_source_then_discards_result(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        self.assertTrue(source.set_running_or_notify_cancel())
        thread, done, errors = self.launch(queue.reset)
        try:
            self.assertTrue(source.result_entered.wait(3))
            self.assertFalse(done.is_set())
            source.set_result(self.reply())
            self.finish_thread(thread, done, errors)
        finally:
            if not source.done():
                source.set_result(self.reply())
            thread.join(3)
        self.assertTrue(prepared.done())
        self.assert_reset(queue)
        self.assertFalse(any(e["event"] == "adopt" for e in queue.events))

    def test_reset_waits_for_running_plan_then_discards_result(self):
        queue = self.make_queue()
        entered = threading.Event()
        release = threading.Event()

        def prepare(actions, anchor):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Test did not release planner")
            return self.prepare(actions, anchor)

        queue.plan_from_anchor = prepare
        source, prepared = self.request_pending(queue)
        source.set_running_or_notify_cancel()
        producer, produced, producer_errors = self.launch(lambda: source.set_result(self.reply()))
        reset_thread = None
        try:
            self.assertTrue(entered.wait(3))
            self.assertTrue(prepared.running())
            source.result_entered.clear()
            reset_thread, reset_done, reset_errors = self.launch(queue.reset)
            self.assertTrue(source.result_entered.wait(3))
            self.assertFalse(reset_done.is_set())
            release.set()
            self.finish_thread(producer, produced, producer_errors)
            self.finish_thread(reset_thread, reset_done, reset_errors)
        finally:
            release.set()
            producer.join(3)
            if reset_thread is not None:
                reset_thread.join(3)
        self.assert_reset(queue)
        self.assertEqual(self.anchors[0][0], 25)
        self.assertFalse(any(e["event"] == "adopt" for e in queue.events))

    def test_reset_wins_race_before_callback_claims_derived_future(self):
        queue = self.make_queue()
        gated = GatedPlanFuture()
        with patch("rtc_queue.Future", return_value=gated):
            source, prepared = self.request_pending(queue)
        self.assertIs(prepared, gated)
        source.set_running_or_notify_cancel()
        with self.assertNoLogs("concurrent.futures", level="ERROR"):
            producer, produced, errors = self.launch(lambda: source.set_result(self.reply()))
            try:
                self.assertTrue(gated.claim_entered.wait(3))
                queue.reset()
                self.assertTrue(gated.cancelled())
                gated.release_claim.set()
                self.finish_thread(producer, produced, errors)
            finally:
                gated.release_claim.set()
                producer.join(3)
        self.assertEqual(self.anchors, [])
        self.assert_reset(queue)

    def test_real_source_failure_propagates_at_reset_and_adoption(self):
        for operation in ("reset", "adopt"):
            with self.subTest(operation=operation):
                queue = self.make_queue()
                source, prepared = self.request_pending(queue)
                error = ValueError("Inference actually failed")
                source.set_exception(error)
                self.assertIs(prepared.exception(), error)
                if operation == "adopt":
                    self.reach_adoption(queue)
                with self.assertRaises(ValueError) as caught:
                    if operation == "reset":
                        queue.reset()
                    else:
                        queue.next(26, 26 / 30, {})
                self.assertIs(caught.exception, error)

    def test_cancelled_error_raised_by_inference_is_not_silenced(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        error = CancelledError("Inference raised this; Future was not cancelled")
        source.set_exception(error)
        self.assertFalse(source.cancelled())
        self.assertFalse(prepared.cancelled())
        with self.assertRaises(CancelledError) as caught:
            queue.reset()
        self.assertIs(caught.exception, error)

    def test_real_plan_failure_propagates_at_reset_and_adoption(self):
        for operation in ("reset", "adopt"):
            with self.subTest(operation=operation):
                queue = self.make_queue()
                error = ValueError("Projection actually failed")

                def fail_plan(actions, anchor):
                    raise error

                queue.plan_from_anchor = fail_plan
                source, prepared = self.request_pending(queue)
                source.set_result(self.reply())
                self.assertIs(prepared.exception(), error)
                if operation == "adopt":
                    self.reach_adoption(queue)
                with self.assertRaises(ValueError) as caught:
                    if operation == "reset":
                        queue.reset()
                    else:
                        queue.next(26, 26 / 30, {})
                self.assertIs(caught.exception, error)

    def test_finished_result_can_be_reset_without_adoption(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        source.set_result(self.reply())
        self.assertTrue(prepared.done())
        self.assertFalse(prepared.cancelled())
        queue.reset()
        self.assert_reset(queue)
        self.assertFalse(any(e["event"] == "adopt" for e in queue.events))

    def test_prefix_commit_period_and_normal_adoption_are_unchanged(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        request = self.requests[-1]["rtc"]
        self.assertEqual(request["commit_steps"], 14)
        self.assertEqual(request["start_step"], 12)
        self.assertEqual(request["period_s"], 1 / 30)
        np.testing.assert_array_equal(request["prefix"], self.initial[12:32])
        self.reach_adoption(queue)
        source.set_result(self.reply())
        self.assertEqual(len(prepared.result()["planned_packets"]), 18)
        np.testing.assert_array_equal(self.anchors[0], self.initial[25])
        packet = queue.next(26, 26 / 30, {"sequence": 26})
        np.testing.assert_array_equal(packet["command"], self.initial[14] + 100)
        self.assertEqual(
            queue.events,
            [{"event": "request", "step": 12, "prefix_length": 20},
             {"event": "adopt", "step": 26, "committed": 14, "roundtrip_s": 0.4},
             {"event": "request", "step": 26, "prefix_length": 18}],
        )
        queue.reset()
        self.assert_reset(queue)

    def test_unfinished_source_still_fails_at_exact_deadline(self):
        queue = self.make_queue()
        source, prepared = self.request_pending(queue)
        self.reach_adoption(queue)
        with self.assertRaisesRegex(RuntimeError, "deadline missed"):
            queue.next(26, 26 / 30, {})
        source.cancel()
        self.assertTrue(prepared.cancelled())
        self.assertFalse(any(e["event"] == "adopt" for e in queue.events))

    def test_completed_source_still_rejected_when_timeline_slips(self):
        queue = self.make_queue()
        source, _ = self.request_pending(queue)
        self.reach_adoption(queue)
        source.set_result(self.reply())
        with self.assertRaisesRegex(RuntimeError, "timeline slipped"):
            queue.next(26, 26 / 30 + 0.1, {})
        self.assertFalse(any(e["event"] == "adopt" for e in queue.events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
