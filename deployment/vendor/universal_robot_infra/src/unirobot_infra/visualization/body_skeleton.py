"""Kinematic full-body skeleton drawing in MuJoCo's user scene."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from unirobot_infra.devices.pico_body import PARENTS


def draw_body(window, frame, observation, controller, preview=False, fresh=True):
    from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

    if frame is None or observation is None:
        return
    points_body = (frame.poses[:, :3] - observation.origin) @ observation.body_to_raw
    anchor = controller.data.xpos[controller.model.body("dual_arm_mount").id].copy()
    anchor += ROBOT_TO_SIM @ np.array([0., 0. if preview else .9, .12])
    points = points_body @ ROBOT_TO_SIM.T + anchor
    alpha = 1. if fresh else .25

    def add(kind, position, size, color, matrix=None):
        scene = window.user_scn
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size), position,
                           np.eye(3).ravel() if matrix is None else matrix.ravel(), np.array(color))
        scene.ngeom += 1
        return geom

    with window.lock():
        for index, parent in enumerate(PARENTS):
            if not frame.valid[index]:
                continue
            side_color = ([.2, .8, .55, alpha] if index in (1,4,7,10,13,16,18,20,22) else
                          [.25,.55,1.,alpha] if index in (2,5,8,11,14,17,19,21,23) else
                          [.8,.8,.8,alpha])
            geom = add(mujoco.mjtGeom.mjGEOM_SPHERE, points[index], [.02]*3, side_color)
            if geom is not None:
                geom.label = str(index)
            if parent >= 0 and frame.valid[parent]:
                geom = add(mujoco.mjtGeom.mjGEOM_CAPSULE, points[index], [.01]*3, side_color)
                if geom is not None:
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012,
                                        points[parent], points[index])
            if index in (16,17,18,19,20,21):
                rotation = (ROBOT_TO_SIM @ observation.body_to_raw.T @
                            Rotation.from_quat(frame.poses[index,3:]).as_matrix())
                for axis, color in enumerate(([1.,.2,.2,alpha],[.2,1.,.2,alpha],[.2,.3,1.,alpha])):
                    geom = add(mujoco.mjtGeom.mjGEOM_ARROW, points[index], [.003]*3, color)
                    if geom is not None:
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .003,
                                            points[index], points[index] + .09*rotation[:,axis])
