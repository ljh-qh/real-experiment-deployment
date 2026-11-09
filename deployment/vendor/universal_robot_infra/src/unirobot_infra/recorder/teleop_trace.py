"""Record tracker inputs and commanded IK state for reproducible diagnosis."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class TeleopRecorder:
    def __init__(self, path, controller):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8", buffering=1)
        self.started = time.monotonic()
        self.stream.write(json.dumps({
            "type": "metadata", "version": 1,
            "tracker_serials": controller.tracker_sns,
            "pico_to_model": controller.pico_to_sim.tolist(),
            "home_rad": controller.home.tolist(),
            "joint_limits_rad": controller.model.jnt_range.tolist(),
            "scale": controller.scale,
        }) + "\n")
        print(f"Recording tracker / IK diagnostics: {path}", flush=True)

    def record(self, controller, poses, frame):
        now = time.monotonic()
        state = {
            "type": "sample", "elapsed_s": now - self.started,
            "frame": int(frame), "returning_home": controller.returning_home,
            "elbow_gain": controller.elbow_gain,
            "poses": {sn: np.asarray(p).tolist() for sn, p in poses.items()},
            "reference": None if controller.reference is None else {
                sn: p.tolist() for sn, p in controller.reference.items()},
            "command_rad": controller.command.tolist(), "arms": {},
        }
        for side, (position, rotation, elbow) in controller.targets.items():
            qids = controller.qpos_ids[side]
            site = controller.site_ids[side]["tcp"]
            actual_position = controller.data.site_xpos[site]
            actual_rotation = Rotation.from_matrix(controller.data.site_xmat[site].reshape(3, 3))
            q = controller.command[qids]
            limits = controller.model.jnt_range[controller.joint_ids[side]]
            margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
            jp = np.zeros((3, controller.model.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jacSite(controller.model, controller.data, jp, jr, site)
            dofs = controller.dof_ids[side]
            sv = np.linalg.svd(np.vstack((jp[:, dofs], .5 * jr[:, dofs])), compute_uv=False)
            state["arms"][side] = {
                "target_position_m": position.tolist(),
                "target_quat_xyzw": rotation.as_quat().tolist(),
                "target_elbow_m": elbow.tolist(),
                "actual_position_m": actual_position.tolist(),
                "actual_quat_xyzw": actual_rotation.as_quat().tolist(),
                "position_error_m": float(np.linalg.norm(position - actual_position)),
                "rotation_error_deg": float(np.degrees((rotation * actual_rotation.inv()).magnitude())),
                "joint_margin_deg": np.degrees(margin).tolist(),
                "min_singular_value": float(sv[-1]),
            }
        self.stream.write(json.dumps(state) + "\n")

    def close(self):
        self.stream.close()
