"""Deadline-checked asynchronous queue of committed hardware targets."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import copy
import numpy as np


class ThreadOwnedClient:
    """Create and use all ZMQ sockets on one inference thread."""
    def __init__(self, factory):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.client = self.pool.submit(factory).result()
    def submit(self, message):
        return self.pool.submit(self.client.infer, message)
    def close(self):
        self.pool.submit(self.client.close).result()
        self.pool.shutdown(wait=True)


class RTCQueue:
    def __init__(self, submit, plan, commit_steps=14, prefix_capacity=20, hz=30, slack_s=1/30,
                 plan_from_anchor=None, clock=None):
        if not 0 < commit_steps < prefix_capacity <= 32 or 2*commit_steps > 32:
            raise ValueError("Insufficient prediction horizon for RTC commitment")
        self.submit = submit
        self.plan = plan
        self.plan_from_anchor = plan_from_anchor
        self.clock = clock
        self.commit_steps = commit_steps
        self.prefix_capacity = prefix_capacity
        self.period = 1/hz
        self.slack = slack_s
        self.queue = deque()
        self.pending = None
        self.events = []
    def prime(self, observation):
        reply = self.submit(observation).result()
        self.queue = deque(self.plan(reply["action"]))
        if len(self.queue) != 32:
            raise ValueError("RTC requires 32 planned actions")
    def next(self, step, now, observation):
        if self.pending is not None:
            p = self.pending
            if step >= p["adopt_step"]:
                self.pending = None
                if step != p["adopt_step"] or not p["future"].done():
                    raise RuntimeError("RTC deadline missed; no stale result dispatched")
                if now > p["start_time"] + self.commit_steps*self.period + self.slack:
                    raise RuntimeError("RTC timeline slipped; result discarded")
                reply = p["future"].result()
                raw = np.asarray(reply["action"])
                if raw.shape != (32,80) or not np.isfinite(raw).all():
                    raise ValueError("Invalid RTC action result")
                if "planned_packets" in reply:
                    packets=reply["planned_packets"]
                    if len(packets)!=32-self.commit_steps:raise ValueError("Invalid preplanned RTC tail")
                    self.queue=deque(packets)
                else:
                    self.queue = deque(self.plan(raw[self.commit_steps:]))
                self.events.append({"event":"adopt","step":step,"committed":self.commit_steps,
                                    "roundtrip_s":reply["roundtrip_seconds"]})
        if not self.queue:
            raise RuntimeError("RTC action queue exhausted")
        if self.pending is None and len(self.queue) <= self.prefix_capacity:
            if len(self.queue) < self.commit_steps:
                raise RuntimeError("RTC prefix cannot cover inference delay")
            prefix = np.stack([p["physical"] for p in list(self.queue)[:self.prefix_capacity]])
            request = copy.deepcopy(observation)
            request["rtc"] = {"prefix":prefix,"commit_steps":self.commit_steps,
                              "start_step":step,"period_s":self.period}
            anchor=copy.deepcopy(list(self.queue)[self.commit_steps-1])
            start_time=now if self.clock is None else self.clock()
            source=self.submit(request)
            future=source
            if self.plan_from_anchor is not None:
                future=Future()
                def prepare(done, output=future, committed=anchor):
                    # Preserve an intentionally cancelled source as cancellation,
                    # rather than converting done.result() into a failed plan.
                    if done.cancelled():
                        output.cancel()
                        return
                    if not output.set_running_or_notify_cancel():return
                    try:
                        reply=done.result()
                        raw=np.asarray(reply["action"])
                        if raw.shape!=(32,80) or not np.isfinite(raw).all():
                            raise ValueError("Invalid RTC action result")
                        packets=self.plan_from_anchor(raw[self.commit_steps:],committed)
                        output.set_result(dict(reply,planned_packets=packets))
                    except BaseException as exc:
                        output.set_exception(exc)
                source.add_done_callback(prepare)
            self.pending = {"future":future,"source":source,"start_time":start_time,"adopt_step":step+self.commit_steps}
            self.events.append({"event":"request","step":step,"prefix_length":len(prefix)})
        return self.queue.popleft()
    def reset(self):
        self.queue.clear()
        if self.pending is not None:
            pending=self.pending
            self.pending=None
            source=pending["source"]
            future=pending["future"]
            if not source.cancel():source.result()
            if future is not source and not future.cancel():future.result()
        self.events.append({"event":"reset"})
