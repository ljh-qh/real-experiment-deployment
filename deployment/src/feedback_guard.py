"""Confirm tracking-error excursions without advancing any new target."""
import math
import time


def confirm_tracking(read_errors, arm_limit, hand_limit, confirm_s=0.1, poll_s=0.02,
                     clock=time.monotonic, sleep=time.sleep):
    if not 0 <= confirm_s <= 0.1 or not 0 < poll_s <= 0.02:
        raise ValueError("Feedback confirmation must be bounded to 100 ms")
    start = clock()
    readings = []
    while True:
        # read_errors must raise immediately for native faults or stale feedback.
        arm, hand = read_errors()
        if not math.isfinite(arm) or not math.isfinite(hand):
            raise RuntimeError("Non-finite tracking feedback")
        readings.append({"t":clock()-start,"arm_deg":arm,"hand_deg":hand})
        if arm <= arm_limit and hand <= hand_limit:
            return {"recovered":True,"waited_s":clock()-start,"readings":readings}
        remaining = confirm_s - (clock()-start)
        if remaining <= 0:
            return {"recovered":False,"waited_s":clock()-start,"readings":readings}
        sleep(min(poll_s,remaining))
