"""Validate host-monotonic observation timestamps without hardware imports.

The 250 ms freshness and 50 ms modality skew limits match the existing
``vendor/bridge/shadow_observation.py:prepare_snapshot`` contract. They are
observation checks, not model inference parameters. Input timestamps must
come from acquisition; a new message-packaging time cannot renew a frame.

This function validates one snapshot. Camera mapping order is irrelevant;
tracking sequence numbers or timestamp regressions across snapshots belongs
to the caller, because no prior observation is passed to this pure function.
"""
from collections.abc import Mapping
from numbers import Integral
import time


CAMERA_ROLES = ("main", "wrist2", "wrist")
MAX_OBSERVATION_AGE_NS = 250_000_000
MAX_MODALITY_SKEW_NS = 50_000_000



class ObservationTimingError(ValueError):
    """A rejected snapshot with its unmodified acquisition timestamps."""

    def __init__(self, reason, message, diagnostics):
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics


class ObservationSkewError(ObservationTimingError):
    """Only this transient timing failure is eligible for reacquisition."""


def acquire_synchronized(capture, events, *, clock=time):
    """Reacquire a complete snapshot for at most 100 ms after a skew failure.

    No commands or inference run in this helper. Other failures propagate
    immediately. Source timestamps are never rewritten to pass validation.
    The caller must check measured tracking before issuing its next target.
    """
    event = None
    deadline_ns = None
    while True:
        try:
            result = capture()
        except ObservationTimingError as exc:
            now_ns = clock.monotonic_ns()
            if event is None:
                event = {
                    "status": "retrying", "started_ns": now_ns, "attempts": 1,
                    "first_failure": exc.diagnostics, "last_failure": exc.diagnostics,
                    "reason": exc.reason, "waited_ms": 0.0,
                }
                events.append(event)
                deadline_ns = now_ns + 100_000_000
            else:
                event["attempts"] += 1
                event["last_failure"] = exc.diagnostics
            event["waited_ms"] = (now_ns - event["started_ns"]) / 1_000_000
            if not isinstance(exc, ObservationSkewError) or now_ns >= deadline_ns:
                event["status"] = "aborted"
                event["reason"] = exc.reason
                raise
            clock.sleep(min(0.005, (deadline_ns - now_ns) / 1_000_000_000))
        except BaseException as exc:
            if event is not None:
                event.update(status="aborted", error=str(exc),
                             waited_ms=(clock.monotonic_ns() - event["started_ns"]) / 1_000_000)
            raise
        else:
            if event is not None:
                now_ns = clock.monotonic_ns()
                event["attempts"] += 1
                event["waited_ms"] = (now_ns - event["started_ns"]) / 1_000_000
                if now_ns > deadline_ns:
                    event["status"] = "aborted"
                    event["reason"] = "reacquisition_timeout"
                    raise ObservationTimingError(
                        "reacquisition_timeout", "Observation reacquisition exceeded 100 ms",
                        event["last_failure"])
                event["status"] = "recovered"
                event["recovered_timing"] = result[-1]
            return result

def _timestamp_ns(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer monotonic timestamp in nanoseconds")
    stamp = int(value)
    if stamp < 0:
        raise ValueError(f"{name} must be non-negative")
    return stamp


def validate_observation_timing(camera_stamps_ns, state_stamp_ns, now_ns=None):
    """Return packet metadata while preserving each source acquisition time.

    Require exactly ``main``, ``wrist2`` and ``wrist`` timestamps. The oldest
    source must be at most 250 ms old, and all camera/state timestamps must
    span at most 50 ms. Equality at either boundary is allowed. Any missing,
    malformed, negative or future timestamp raises ``ValueError``.
    """
    if not isinstance(camera_stamps_ns, Mapping):
        raise ValueError("camera_stamps_ns must contain the three camera timestamps")
    if set(camera_stamps_ns) != set(CAMERA_ROLES):
        raise ValueError("Expected camera timestamps for exactly main, wrist2 and wrist")
    cameras = {role: _timestamp_ns(camera_stamps_ns[role], f"camera {role}")
               for role in CAMERA_ROLES}
    state = _timestamp_ns(state_stamp_ns, "state")
    now = _timestamp_ns(time.monotonic_ns() if now_ns is None else now_ns, "now")
    stamps = (*cameras.values(), state)
    oldest, newest = min(stamps), max(stamps)
    sources = {**cameras, "state": state}
    diagnostics = {
        "checked_at_ns": now, "camera_timestamps_ns": cameras,
        "state_timestamp_ns": state,
        "age_ms": {name: (now - stamp) / 1_000_000 for name, stamp in sources.items()},
        "modality_skew_ms": (newest - oldest) / 1_000_000,
        "oldest_sources": [name for name, stamp in sources.items() if stamp == oldest],
        "newest_sources": [name for name, stamp in sources.items() if stamp == newest],
    }
    if newest > now:
        raise ObservationTimingError("future", "Observation source has a future timestamp", diagnostics)
    if now - oldest > MAX_OBSERVATION_AGE_NS:
        raise ObservationTimingError("stale", "Observation source is stale (older than 250 ms)", diagnostics)
    if newest - oldest > MAX_MODALITY_SKEW_NS:
        raise ObservationSkewError(
            "skew", "Observation modalities are not synchronized (skew exceeds 50 ms)", diagnostics)
    return {
        "timestamp_ns": now,
        "source_timestamp_ns": oldest,
        "camera_timestamps_ns": cameras,
        "state_timestamp_ns": state,
        "oldest_capture_ns": oldest,
        "modality_skew_ms": (newest - oldest) / 1_000_000,
    }
