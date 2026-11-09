"""Optional tap for existing measured data; never opens a hardware device."""
import threading

import numpy as np
import zmq


class SnapshotPublisher:
    """Single-slot background publisher; offer only already-measured snapshots.

    Create only when explicitly enabling the tap in the owning collector.
    SDK reads, camera acquisition and control are outside this module.
    """
    def __init__(self, endpoint, codec):
        self.endpoint = endpoint
        self.codec = codec
        self.condition = threading.Condition()
        self.latest = None
        self.stopped = False
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._publish, name='observation-tap', daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError('Snapshot publisher did not start')
        if self.error is not None:
            raise RuntimeError(self.error)

    def offer(self, snapshot):
        if self.error is not None or self.stopped:
            return False
        # Copy references only. Caller must pass immutable, newly captured arrays.
        # Images are encoded off the collector thread; never wait for a receiver.
        with self.condition:
            self.latest = snapshot
            self.condition.notify()
        return True

    def _publish(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2)
        try:
            socket.bind(self.endpoint)
            self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self.ready.set()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or self.latest is not None)
                    if self.stopped:
                        break
                    snapshot, self.latest = self.latest, None
                socket.send_multipart([b'measured_observation'] + self.codec.encode_message(snapshot))
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
        finally:
            socket.close()
            context.term()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=2)


def snapshot_from_measured(stream_id, sequence, arms_rad, hands_rad, images, timestamps_ns):
    """All units/routing explicit; targets and commands are not accepted here."""
    return {'type': 'measured_observation', 'source_kind': 'live',
            'stream_id': stream_id, 'sequence': sequence,
            'arm_q_rad': np.array(arms_rad, dtype=np.float64, copy=True),
            'hand_q_rad': np.array(hands_rad, dtype=np.float64, copy=True),
            'images': {name: np.array(image, dtype=np.uint8, copy=True)
                       for name, image in images.items()},
            'timestamps_ns': dict(timestamps_ns)}
