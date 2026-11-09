"""Measured-only snapshot contract for a passive observation subscriber."""
import time

import numpy as np

from adapter import CAMERAS
from zmq_offline_bridge import REPRESENTATION


STAMP_KEYS = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'main', 'wrist2', 'wrist')


def prepare_snapshot(raw, robot, expected_kind, max_age_ms=250, max_skew_ms=50, now_ns=None):
    from kinematics import canonical80
    if raw['type'] != 'measured_observation' or raw['source_kind'] != expected_kind:
        raise ValueError('Expected explicitly identified measured observation source')
    if not raw['stream_id'] or raw['sequence'] < 0:
        raise ValueError('Missing stream identity or invalid sequence')
    stamps = [int(raw['timestamps_ns'][name]) for name in STAMP_KEYS]
    now = time.monotonic_ns() if now_ns is None else now_ns
    if min(stamps) > now or max(stamps) > now or now - min(stamps) > max_age_ms * 1e6:
        raise ValueError('Source is stale or has future timestamps')
    if max(stamps) - min(stamps) > max_skew_ms * 1e6:
        raise ValueError('Measured modalities are not synchronized')
    joints = np.concatenate([np.asarray(raw['arm_q_rad']), np.asarray(raw['hand_q_rad'])])
    if np.asarray(raw['arm_q_rad']).shape != (14,) or np.asarray(raw['hand_q_rad']).shape != (44,):
        raise ValueError('Expected measured arm14 and hand44 in radians')
    images = {}
    for role in CAMERAS.values():
        image = np.asarray(raw['images'][role])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected three uint8 RGB images')
        images[role] = image
    reference = max(int(raw['timestamps_ns'][side]) for side in ('left_arm', 'right_arm'))
    return {
        'type': 'observation', 'sequence': int(raw['sequence']),
        'timestamp_ns': reference, 'source_timestamp_ns': reference,
        'source_stream_id': str(raw['stream_id']), 'source_kind': expected_kind,
        'task': 'badminton', 'representation': REPRESENTATION,
        'oldest_capture_ns': min(stamps),
        'modality_skew_ms': (max(stamps) - min(stamps)) / 1e6,
        'observation': {'state': canonical80(joints[None], robot)[0], 'images': images},
    }
