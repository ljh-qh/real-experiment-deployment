"""Observation/action contract only; this module never opens robot devices."""
import base64
import io
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from kinematics import canonical80

CAMERAS = {'head_camera': 'main', 'left_wrist_camera': 'wrist2', 'right_wrist_camera': 'wrist'}

def observation(images, joints58, robot):
    encoded = {}
    for key, role in CAMERAS.items():
        value = np.asarray(images[role])
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(f'{role}: expected uint8 RGB HWC')
        buffer = io.BytesIO()
        Image.fromarray(value).save(buffer, format='PNG')
        encoded[key] = base64.b64encode(buffer.getvalue()).decode('ascii')
    state = canonical80(np.asarray(joints58)[None], robot)[0]
    return {'type': 'obs', 'images': encoded, 'state': state.tolist(), 'prompt': 'badminton'}

def decode_action(action):
    """Input is already in physical units; do not unnormalize again."""
    action = np.asarray(action, dtype=float)
    if action.shape != (80,) or not np.isfinite(action).all():
        raise ValueError('Expected one finite action80, not a chunk')
    arms = {}
    for side, start in [('left', 0), ('right', 34)]:
        a, b = action[start+3:start+6], action[start+6:start+9]
        if np.linalg.norm(a) < 1e-6:
            raise ValueError('Degenerate rotation first column')
        x = a / np.linalg.norm(a)
        y = b - x * np.dot(x, b)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError('Degenerate rotation second column')
        y /= np.linalg.norm(y)
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
        matrix[:3, 3] = action[start:start+3]
        arms[side] = matrix
    return arms, np.concatenate([action[10:32], action[44:66]])

def ik_targets(action, world_from_mount, elbow_world):
    """Convert mount-frame TCPs to existing MarvinTeleop.set_targets format.

    Caller must supply the URDF fixed transform and measured elbow positions.
    No PICO reference mapping is applied to model-generated targets.
    """
    arms, hands = decode_action(action)
    transform = np.asarray(world_from_mount)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected finite world_from_mount transform')
    targets = {}
    for side, tcp in arms.items():
        world = transform @ tcp
        targets[side] = (world[:3, 3], Rotation.from_matrix(world[:3, :3]), np.asarray(elbow_world[side]))
    return targets, hands
