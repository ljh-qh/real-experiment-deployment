"""Chest-relative Full Body mapping, independent of IK and output backend."""

from dataclasses import dataclass
from collections import deque
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from unirobot_infra.robots.arm.marvin_model import CONFIG_DIR


def default_body_config_path():
    """``--body-config`` / ``MUJOCO_MARVIN_BODY_CONFIG`` / package config."""
    return Path(os.environ.get("MUJOCO_MARVIN_BODY_CONFIG")
                or CONFIG_DIR / "device" / "pico_body.yaml")


def load_body_config(path=None):
    config = yaml.safe_load(Path(path or default_body_config_path()).read_text())
    if not isinstance(config, dict):
        raise ValueError("Body config must be a YAML mapping")
    for key in ("scale", "yaw_filter_seconds", "freshness_seconds", "prediction_seconds",
                "max_joint_speed_deg_s", "reset_speed_deg_s", "control_hz", "max_dt_seconds",
                "position_jump_m", "max_hand_speed_m_s", "rotation_jump_degrees",
                "max_hand_angular_speed_deg_s", "bone_length_change_fraction", "max_body_yaw_step_deg",
                "max_consecutive_rejects"):
        if key not in config or not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
        config[key] = float(config[key])
    up = np.asarray(config["raw_up_axis"], dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        raise ValueError("raw_up_axis must be a finite nonzero 3-vector")
    config["raw_up_axis"] = (up / np.linalg.norm(up)).tolist()
    if config["reference_frame"] not in ("torso_yaw", "fixed_yaw"):
        raise ValueError("reference_frame must be torso_yaw or fixed_yaw")
    if config["orientation_joint"] not in ("wrist", "hand"):
        raise ValueError("orientation_joint must be wrist or hand")
    for key in ("hardware_requires_explicit_quality", "allow_unknown_quality"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"{key} must be true or false")
    # The shared core clamps its own IK integration dt at 0.04 s (teleop.py
    # solve/step); max_dt_seconds here also feeds the finger rate limit and
    # must reflect the real body-frame interval (e.g. ~0.2 s at 5 Hz), so it
    # is bounded well above the core's internal clamp.
    if config["max_dt_seconds"] > .5:
        raise ValueError("max_dt_seconds must not exceed 0.5 s")
    if config["bone_length_change_fraction"] >= 1.:
        raise ValueError("bone_length_change_fraction must be less than 1")
    return config


@dataclass
class BodyObservation:
    origin: np.ndarray
    body_to_raw: np.ndarray
    positions: dict
    rotations: dict
    lengths: dict
    received_at: float


class BodyRetarget:
    def __init__(self, config):
        self.config = config
        self.up = np.asarray(config["raw_up_axis"], dtype=float)
        self.reference = None
        self.baseline = None
        self.previous = None
        self.previous_raw_heading = None
        self._length_history = deque(maxlen=5)

    def reset(self):
        self.reference = None
        self.baseline = None
        self.previous = None
        self.previous_raw_heading = None
        self._length_history.clear()

    def observe(self, frame):
        if not frame.complete:
            raise ValueError(frame.error or f"Need 24 valid Body nodes; got {int(frame.valid.sum())}")
        points = frame.poses[:, :3]
        shoulder_delta = points[16] - points[17]
        shoulder_width = np.linalg.norm(shoulder_delta)
        left = shoulder_delta - np.dot(shoulder_delta, self.up) * self.up
        if not .12 <= shoulder_width <= .8 or np.linalg.norm(left) < .10:
            raise ValueError("Shoulder frame is degenerate or shoulder width is invalid")
        left /= np.linalg.norm(left)
        body_to_raw = np.column_stack((np.cross(left, self.up), left, self.up))
        raw_heading = body_to_raw.copy()
        origin = .5 * (points[16] + points[17])
        previous = self.previous
        if previous is not None:
            difference = Rotation.from_matrix(body_to_raw @ previous.body_to_raw.T)
            angle = Rotation.from_matrix(body_to_raw @ self.previous_raw_heading.T).magnitude()
            if np.degrees(angle) > self.config["max_body_yaw_step_deg"]:
                raise ValueError("Body heading jumped; re-record the reference")
            dt = max(0., frame.received_at - previous.received_at)
            if self.config["reference_frame"] == "fixed_yaw":
                body_to_raw = previous.body_to_raw.copy()
            else:
                alpha = 1. - math.exp(-dt / self.config["yaw_filter_seconds"])
                body_to_raw = Rotation.from_rotvec(alpha * difference.as_rotvec()).as_matrix() @ previous.body_to_raw
        positions = {}
        rotations = {}
        lengths = {}
        for side, ids in (("left", (16, 18, 20, 22)), ("right", (17, 19, 21, 23))):
            shoulder, elbow, wrist, hand = ids
            for role, index in zip(("shoulder", "elbow", "wrist", "hand"), ids):
                positions[f"{side}_{role}"] = body_to_raw.T @ (points[index] - origin)
            index = wrist if self.config["orientation_joint"] == "wrist" else hand
            rotations[side] = Rotation.from_matrix(body_to_raw.T) * Rotation.from_quat(frame.poses[index, 3:])
            lengths[side] = np.array([np.linalg.norm(points[elbow] - points[shoulder]),
                                      np.linalg.norm(points[wrist] - points[elbow])])
            if np.any(lengths[side] < .08) or np.any(lengths[side] > .65):
                raise ValueError(f"{side} arm has implausible segment lengths")
        observation = BodyObservation(origin, body_to_raw, positions, rotations, lengths, frame.received_at)
        if previous is not None:
            dt = max(0., frame.received_at - previous.received_at)
            for side in ("left", "right"):
                dp = np.linalg.norm(positions[f"{side}_wrist"] - previous.positions[f"{side}_wrist"])
                dr = np.degrees((rotations[side] * previous.rotations[side].inv()).magnitude())
                if dp > self.config["position_jump_m"] + self.config["max_hand_speed_m_s"] * dt:
                    raise ValueError(f"{side} wrist position jumped")
                if dr > self.config["rotation_jump_degrees"] + self.config["max_hand_angular_speed_deg_s"] * dt:
                    raise ValueError(f"{side} wrist rotation jumped")
                if self.reference is not None:
                    ratio = np.abs(lengths[side] / self.reference.lengths[side] - 1.)
                    if np.any(ratio > self.config["bone_length_change_fraction"]):
                        raise ValueError(f"{side} reconstructed bone length changed")
        self.previous = observation
        self.previous_raw_heading = raw_heading
        self._length_history.append({side: lengths[side].copy() for side in ("left", "right")})
        return observation

    def capture(self, observation, controller, save_path=None):
        controller.capture_baseline()
        self.reference = observation
        if self._length_history:
            # Bone lengths reconstructed from forearm trackers are noisy; the
            # rejection baseline must be a robust median, not one lucky frame.
            self.reference.lengths = {
                side: np.median(np.stack([h[side] for h in self._length_history]), axis=0)
                for side in ("left", "right")
            }
        self.baseline = controller.baseline
        if save_path is not None:
            record = {
                "version": 1, "source": "pico_body", "config": self.config,
                "origin_raw": observation.origin.tolist(),
                "body_to_raw": observation.body_to_raw.tolist(),
                "positions_body_m": {k: v.tolist() for k, v in observation.positions.items()},
                "rotations_body_xyzw": {k: v.as_quat().tolist() for k, v in observation.rotations.items()},
                "arm_lengths_m": {k: v.tolist() for k, v in observation.lengths.items()},
                "robot_reference_rad": controller.arm_command().tolist(),
                "note": "Session reference only; always capture fresh after startup/recovery",
            }
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    def targets(self, observation):
        from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

        if self.reference is None:
            raise RuntimeError("Body reference has not been recorded")
        result = {}
        transform = Rotation.from_matrix(ROBOT_TO_SIM)
        for side in ("left", "right"):
            delta = observation.positions[f"{side}_wrist"] - self.reference.positions[f"{side}_wrist"]
            elbow_delta = observation.positions[f"{side}_elbow"] - self.reference.positions[f"{side}_elbow"]
            turn = observation.rotations[side] * self.reference.rotations[side].inv()
            position, rotation = self.baseline[side]["tcp"]
            elbow, _ = self.baseline[side]["elbow"]
            result[side] = (
                position + self.config["scale"] * ROBOT_TO_SIM @ delta,
                transform * turn * transform.inv() * rotation,
                elbow + self.config["scale"] * ROBOT_TO_SIM @ elbow_delta,
            )
        return result
