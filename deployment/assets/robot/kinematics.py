"""Mesh-free URDF forward kinematics; URDF RPY is fixed-axis XYZ."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class URDFKinematics:
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        self.by_child = {j.find('child').get('link'): j for j in root.findall('joint')}

    def chain(self, base, tip):
        result = []
        while tip != base:
            if tip not in self.by_child:
                raise ValueError(f'{base} is not an ancestor of {tip}')
            joint = self.by_child[tip]
            result.append(joint)
            tip = joint.find('parent').get('link')
        return result[::-1]

    def forward(self, positions, joint_names, base, tip):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != len(joint_names):
            raise ValueError('Joint array and names disagree')
        index = {name: i for i, name in enumerate(joint_names)}
        chain = self.chain(base, tip)
        moving = [j.get('name') for j in chain if j.get('type') != 'fixed']
        if set(moving) != set(joint_names):
            raise ValueError(f'Expected chain joints {moving}, got {joint_names}')
        transforms = np.broadcast_to(np.eye(4), (len(positions), 4, 4)).copy()
        for joint in chain:
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            fixed = np.eye(4)
            fixed[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            fixed[:3, 3] = xyz
            transforms = transforms @ fixed
            if joint.get('type') == 'fixed':
                continue
            if joint.get('type') not in ('revolute', 'continuous'):
                raise ValueError(f'Unsupported joint type: {joint.get("type")}')
            axis_node = joint.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1., 0., 0.])
            axis /= np.linalg.norm(axis)
            moving_tf = np.broadcast_to(np.eye(4), transforms.shape).copy()
            moving_tf[:, :3, :3] = Rotation.from_rotvec(positions[:, index[joint.get('name')], None] * axis).as_matrix()
            transforms = transforms @ moving_tf
        return transforms


def canonical80(joints58, robot):
    """Real L/R arm joints + real L/R fingers -> OpenWAM canonical EEF80."""
    joints58 = np.asarray(joints58)
    if joints58.ndim != 2 or joints58.shape[1] != 58 or not np.isfinite(joints58).all():
        raise ValueError('Expected finite (N, 58) measured joint states in radians')
    result = np.zeros((len(joints58), 80), dtype=np.float32)
    for side, source, dest in [('L', 0, 0), ('R', 7, 34)]:
        tf = robot.forward(joints58[:, source:source+7], [f'Joint{i}_{side}' for i in range(1, 8)],
                           'dual_arm_mount', f'TCP_Link_{side}')
        result[:, dest:dest+3] = tf[:, :3, 3]
        result[:, dest+3:dest+9] = np.concatenate([tf[:, :3, 0], tf[:, :3, 1]], axis=1)
    result[:, 10:32] = joints58[:, 14:36]
    result[:, 44:66] = joints58[:, 36:58]
    return result


ACTIVE = np.zeros(80, dtype=bool)
ACTIVE[0:9] = ACTIVE[10:32] = ACTIVE[34:43] = ACTIVE[44:66] = True
ROTATION_DIMS = list(range(3, 9)) + list(range(37, 43))
