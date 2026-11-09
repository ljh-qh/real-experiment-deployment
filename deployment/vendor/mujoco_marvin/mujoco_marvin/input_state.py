"""Body input admission, freshness, prediction timeout and explicit recovery."""

import time


class BodyInputState:
    def __init__(self, config, hardware=False):
        self.config = config
        self.hardware = hardware
        self.state = "WAITING"
        self.reason = "Waiting for Full Body / Send"
        self.enabled = True
        self.engaged = False
        self._sequence = None
        self._stamp = None
        self._stamp_progress_at = None
        self._prediction_at = None

    def hold(self, reason, require_resume=None):
        self.state, self.reason = "HOLD", reason
        if require_resume is None:
            require_resume = self.engaged
        if require_resume:
            self.enabled = False
        self.engaged = False
        return False

    def pause(self):
        return self.hold("Paused; SPACE resumes after reference countdown", require_resume=True)

    def resume(self):
        self.enabled = True
        self.engaged = False
        self.state = "WAITING"
        self.reason = "Waiting for fresh Body input"
        self._prediction_at = None
        self._sequence = None
        self._stamp = None
        self._stamp_progress_at = None

    def evaluate(self, frame, now=None):
        now = time.monotonic() if now is None else now
        if frame is None:
            return self.hold("No Body payload; select Full Body and Send")
        if now - frame.received_at > self.config["freshness_seconds"]:
            return self.hold("Body payload stopped; Head packets do not refresh it")
        if frame.sequence != self._sequence:
            self._sequence = frame.sequence
            stamp = frame.source_timestamp_ns
            if stamp > 0:
                if self._stamp is not None and stamp < self._stamp:
                    self._stamp = stamp
                    self._stamp_progress_at = now
                    return self.hold("Body source clock restarted; re-record reference", True)
                if stamp != self._stamp:
                    self._stamp, self._stamp_progress_at = stamp, now
            elif self._stamp is not None:
                return self.hold("Body source timestamp disappeared", True)
        if (self._stamp_progress_at is not None and
                now - self._stamp_progress_at > self.config["freshness_seconds"]):
            return self.hold("Body source timestamp is stale")
        if not frame.complete:
            return self.hold(frame.error or f"Incomplete skeleton: {int(frame.valid.sum())}/24 valid")
        if frame.quality == "INVALID":
            return self.hold("Source reports invalid / uncalibrated tracking")
        if self.hardware and frame.source != "pico_body":
            return self.hold("Hardware requires live PICO Body input")
        if self.hardware and frame.quality == "UNKNOWN" and \
                self.config.get("allow_unknown_quality") is not True:
            return self.hold("Hardware waits for explicit Body quality (current source: UNKNOWN)")
        if not self.enabled:
            self.state = "HOLD"
            return False
        if frame.quality == "PREDICTED":
            if not self.engaged:
                return self.hold("Reference requires non-predicted input")
            if self._prediction_at is None:
                self._prediction_at = now
            if now - self._prediction_at > self.config["prediction_seconds"]:
                return self.hold("Prediction interval exceeded; release and resume")
            self.state, self.reason = "PREDICTED", "Bounded source prediction"
        else:
            self._prediction_at = None
            self.state = "TRACKING" if frame.quality == "TRACKING" else "UNKNOWN"
            self.reason = ("Source reports tracking" if frame.quality == "TRACKING" else
                           "Payload received; measured/predicted quality not supplied (simulation)")
        return True
