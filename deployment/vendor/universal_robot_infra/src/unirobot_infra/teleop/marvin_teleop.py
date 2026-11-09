#!/usr/bin/env python3
"""PICO Motion Trackers -> dual Marvin M6 arms (simulated and/or real).

Unified teleop: one mapping core (record tracker and robot poses together,
apply tracker pose deltas, guide the elbow in the wrist task's null space,
solve IK against the Marvin URDF in MuJoCo) streams 7 joint angles per arm
to switchable output backends:

- ``sim``:   MuJoCo viewer / headless kinematic rendering (default)
- ``robot``: real arms via the Tianji SDK joint-position stream
             (tianji_marvin_backend.TianjiJointBackend)
- ``both``:  viewer and real arms together

The real-robot backend reuses the same standing-pose keyframe as the sim, so
the operator reference posture is identical on both targets.

Model/config/SDK locations are resolved with the priority
``CLI argument > environment variable > package default``; see
``unirobot_infra.robots.arm.marvin_model``.
"""

import argparse
import math
import os
from queue import SimpleQueue
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from unirobot_infra.devices.pico_tracker import (
    ROLES, ForearmRedundancy, PoseFreezeGuard, TrackerActivity, read_tracker_frame,
)
from unirobot_infra.calibration.tracker_alignment import load_alignment, rotation_matrix
from unirobot_infra.kinematics.differential import wrist_and_elbow_steps
from unirobot_infra.robots.arm.marvin_model import (
    ASSETS_DIR, CONFIG_DIR, SIDES, home_profile_names, load_hand_mounts,
    load_home_joints, resolve_model_dir, resolve_model_urdf, resolve_robot_config,
    resolve_robot_ip,
)

# Wuji Hand description (optional; see assets/robot/README.md). The hand is
# attached to each arm TCP as a passive visual reference — no joint data
# drives it. Missing descriptions degrade the model to hands-free.
HAND_PACKAGE_PREFIX = "package://wuji_description/meshes/"
HAND_RGBA = "0.9 0.82 0.75 1"

_PICO_SDK_HELP = (
    "PICO SDK 'xrobotoolkit_sdk' is not importable. Install the PICO PC "
    "Service, put the prebuilt SDK directory (xrobotoolkit_sdk*.so) on "
    "PYTHONPATH and its folder on LD_LIBRARY_PATH, or set PICO_SDK_DIR to "
    "the directory containing it. --demo and --replay need no SDK."
)

# This exact configuration is used by both the model keyframe and real-arm
# startup ramp. Values are degrees in URDF / SDK joint order, Joint1..Joint7.
HOME_JOINTS_DEG = load_home_joints()


def hand_description_dir():
    """``WUJI_DESCRIPTION_DIR`` (wuji_description root) /
    ``assets/robot/wuji_description`` -> ``hand/body``."""
    override = os.environ.get("WUJI_DESCRIPTION_DIR")
    base = Path(override) if override else ASSETS_DIR / "wuji_description"
    return base / "hand" / "body"


def hand_paths():
    description = hand_description_dir()
    return {
        "description": description,
        "urdf": {side: description / "urdf" / f"{side}-ros.urdf" for side in SIDES},
        "mesh_dir": description / "meshes",
    }


def default_tracker_config_path():
    """Repo config: ``configs/device/pico_input.yaml`` when present, else the template."""
    filled = CONFIG_DIR / "device" / "pico_input.yaml"
    return filled if filled.is_file() else CONFIG_DIR / "device" / "pico_input.yaml.template"


def resolve_tracker_config(cli_path=None):
    return Path(cli_path or os.environ.get("MUJOCO_MARVIN_TRACKER_CONFIG")
                or default_tracker_config_path())


def default_body_config_path():
    return Path(os.environ.get("MUJOCO_MARVIN_BODY_CONFIG")
                or CONFIG_DIR / "device" / "pico_body.yaml")


def default_body_reference_path():
    return Path.cwd() / ".local" / "pico-body-reference.json"


def floats(value):
    return np.fromstring(value, sep=" ")


def origin(parent):
    item = parent.find("origin")
    if item is None:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    xyz = floats(item.get("xyz", "0 0 0"))
    quat_xyzw = Rotation.from_euler("xyz", floats(item.get("rpy", "0 0 0"))).as_quat()
    return xyz, quat_xyzw[[3, 0, 1, 2]]


def vec(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def add_body(parent, name, joint, link, assets, model_dir):
    xyz, quat = origin(joint)
    body = ET.SubElement(parent, "body", name=name, pos=vec(xyz), quat=vec(quat))
    inertial = link.find("inertial")
    if inertial is not None and "TCP_Link" not in name:
        ipos, iquat = origin(inertial)
        tensor = inertial.find("inertia")
        values = {k: float(tensor.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        inertia = np.array([[values['ixx'], values['ixy'], values['ixz']],
                            [values['ixy'], values['iyy'], values['iyz']],
                            [values['ixz'], values['iyz'], values['izz']]])
        inertial_rot = Rotation.from_quat(iquat[[1, 2, 3, 0]]).as_matrix()
        inertia = inertial_rot @ inertia @ inertial_rot.T
        ET.SubElement(body, "inertial", pos=vec(ipos),
                      mass=inertial.find("mass").get("value"),
                      fullinertia=vec([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                       inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
    if joint.get("type") != "fixed":
        limit = joint.find("limit")
        ET.SubElement(body, "joint", name=joint.get("name"), type="hinge",
                      axis=joint.find("axis").get("xyz"),
                      range=f"{limit.get('lower')} {limit.get('upper')}",
                      damping="2", armature="0.02")
    visual = link.find("visual")
    if visual is not None:
        mesh = visual.find("geometry/mesh")
        if mesh is not None and "TCP_Link" not in name:
            filename = mesh.get("filename")
            prefix = "package://marvin_dual_arm/"
            if not filename.startswith(prefix):
                raise ValueError(f"Unsupported model mesh URI: {filename}")
            mesh_path = model_dir / filename[len(prefix):]
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            mesh_name = name + "_mesh"
            ET.SubElement(assets, "mesh", name=mesh_name, file=str(mesh_path),
                          scale=mesh.get("scale", "1 1 1"))
            vpos, vquat = origin(visual)
            ET.SubElement(body, "geom", name=name + "_visual", type="mesh",
                          mesh=mesh_name, pos=vec(vpos), quat=vec(vquat),
                          rgba=("0.8 0.82 0.86 1" if name.endswith("_L") else
                                "0.58 0.7 0.9 1" if name.endswith("_R") else "0.3 0.35 0.4 1"),
                          contype="0", conaffinity="0")
    return body


def hand_frame_axes(side):
    """Hand-root anatomical frame and direction toward the thumb base.

    Columns = [third, palm normal, finger direction]. The palm normal's sign
    follows the URDF's positive finger-flexion direction. The index-to-pinky
    row is mirrored between hands, so the cross-product order differs.
    """
    parsed = ET.parse(hand_paths()["urdf"][side]).getroot()
    bases = {}
    for joint in parsed.findall("joint"):
        name = joint.get("name")
        if "tip" in name or "joint1" not in name:
            continue
        digit_part = next(p for p in name.split("_") if "finger" in p)
        digit = int("".join(ch for ch in digit_part if ch.isdigit()))
        bases[digit], _ = origin(joint)
    four = np.mean([bases[i] for i in sorted(bases) if i != 1], axis=0)
    finger_dir = four / np.linalg.norm(four)                    # wrist -> knuckles
    row = bases[5] - bases[2]                                   # index -> pinky
    row -= np.dot(row, finger_dir) * finger_dir
    row /= np.linalg.norm(row)
    thumb = bases[1] - four
    thumb -= np.dot(thumb, finger_dir) * finger_dir
    thumb /= np.linalg.norm(thumb)
    if np.dot(thumb, row) > 0:                                  # thumb sits on the index side
        row = -row
    # Confirmed against finger3's positive flexion axes in the source URDF:
    # curl toward the palm, not the back of the hand. With fingers forward
    # and this normal down, the left thumb points right and vice versa.
    palm = np.cross(row, finger_dir) if side == "left" else np.cross(finger_dir, row)
    third = np.cross(palm, finger_dir)
    return np.column_stack([third / np.linalg.norm(third), palm, finger_dir]), thumb


def hand_desired_world_axes(side):
    """Hand orientation matching the standing pose: forearms reach forward,
    palms down (双手向下). Same [third, palm, fingers] column order as
    hand_frame_axes; identical for both hands."""
    from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

    finger = ROBOT_TO_SIM @ np.array([1., 0., 0.])
    palm = ROBOT_TO_SIM @ np.array([0., 0., -1.])
    third = np.cross(palm, finger)
    return np.column_stack([third, palm, finger])


def hand_mount_quats():
    """Use the saved mounting from the accepted palms-down, thumbs-in model.

    Home-pose search must preserve this physical relationship. Refitting the
    hand attachment to each candidate would hide an incorrect wrist pose.
    The saved visual mounting has not been measured on the real hardware.
    """
    return load_hand_mounts()


def attach_hand(parent, side, mount_quat, assets, movable=False):
    """Attach the original hand hierarchy, optionally with its 20 joints."""
    paths = hand_paths()
    parsed = ET.parse(paths["urdf"][side]).getroot()
    links = {link.get("name"): link for link in parsed.findall("link")}
    joints = parsed.findall("joint")
    children = {}
    for joint in joints:
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    child_links = {joint.find("child").get("link") for joint in joints}
    root_link = next(link_name for link_name in links if link_name not in child_links)
    def add_visual(body, link_name):
        visual = links[link_name].find("visual")
        if visual is None:
            return
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            return
        filename = mesh.get("filename")
        rel = filename.replace(HAND_PACKAGE_PREFIX, "")
        mesh_file = paths["mesh_dir"] / rel
        if not mesh_file.exists():
            return
        mesh_name = f"{side}_hand_{Path(rel).stem}"
        if assets.find(f"mesh[@name='{mesh_name}']") is None:
            ET.SubElement(assets, "mesh", name=mesh_name, file=str(mesh_file))
        vpos, vquat = origin(visual)
        ET.SubElement(body, "geom", name=f"{side}_hand_{link_name}_visual",
                      type="mesh", mesh=mesh_name, pos=vec(vpos), quat=vec(vquat),
                      rgba=HAND_RGBA, contype="0", conaffinity="0")

    root_body = ET.SubElement(parent, "body", name=f"{side}_hand_{root_link}",
                              pos="0 0 0", quat=vec(mount_quat))
    add_visual(root_body, root_link)

    def recurse(parent_elem, link_name):
        for joint in children.get(link_name, []):
            child_name = joint.find("child").get("link")
            xyz, quat = origin(joint)
            body = ET.SubElement(parent_elem, "body", name=f"{side}_hand_{child_name}",
                                 pos=vec(xyz), quat=vec(quat))
            if movable and joint.get("type") != "fixed":
                limit = joint.find("limit")
                ET.SubElement(body, "joint", name=joint.get("name"), type="hinge",
                              axis=joint.find("axis").get("xyz"),
                              range=f"{limit.get('lower')} {limit.get('upper')}",
                              damping="0.01", armature="0.001")
                inertial = links[child_name].find("inertial")
                if inertial is None:
                    raise ValueError(f"Moving hand link lacks inertia: {child_name}")
                ipos, iquat = origin(inertial)
                tensor = inertial.find("inertia")
                v = {k: float(tensor.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
                matrix = np.array([[v['ixx'],v['ixy'],v['ixz']], [v['ixy'],v['iyy'],v['iyz']],
                                   [v['ixz'],v['iyz'],v['izz']]])
                r = Rotation.from_quat(iquat[[1, 2, 3, 0]]).as_matrix()
                matrix = r @ matrix @ r.T
                ET.SubElement(body, "inertial", pos=vec(ipos), mass=inertial.find("mass").get("value"),
                              fullinertia=vec([matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2]]))
            add_visual(body, child_name)
            recurse(body, child_name)

    recurse(root_body, root_link)


def build_xml(mount_quats=None, home_joints=None, movable_hands=False):
    model_dir = resolve_model_dir()
    hand_urdfs = hand_paths()["urdf"]
    root = ET.Element("mujoco", model="marvin_dual_teleop")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false",
                  balanceinertia="true")
    ET.SubElement(root, "option", timestep="0.005", gravity="0 0 -9.81")
    ET.SubElement(root, "visual")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05", rgba="0.9 0.9 0.9 1")
    actuator = ET.SubElement(root, "actuator")
    keyframe = ET.SubElement(root, "keyframe")
    parsed = ET.parse(resolve_model_urdf(model_dir)).getroot()
    links = {link.get("name"): link for link in parsed.findall("link")}
    children = {}
    for joint in parsed.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    joint_order = []
    visited = {"world"}

    def add_subtree(parent_body, parent_link):
        # Explicit branch order keeps qpos and actuators left J1..J7 then
        # right J1..J7, independently of <joint> declaration order in URDF.
        for joint in sorted(children.get(parent_link, []), key=lambda j: j.get("name")):
            child_name = joint.find("child").get("link")
            if child_name in visited:
                raise ValueError(f"Duplicate/cyclic URDF link {child_name}")
            visited.add(child_name)
            body = add_body(parent_body, child_name, joint, links[child_name], assets, model_dir)
            for side, suffix in (("left", "L"), ("right", "R")):
                if child_name == f"Link4_{suffix}":
                    ET.SubElement(body, "site", name=f"{side}_elbow", size="0.012",
                                  rgba="0.9 0.5 0.1 1")
                elif child_name == f"TCP_Link_{suffix}":
                    ET.SubElement(body, "site", name=f"{side}_tcp", size="0.018",
                                  rgba="0.15 0.9 0.3 1")
                    if mount_quats is not None and hand_urdfs[side].exists():
                        attach_hand(body, side, mount_quats[side], assets, movable=movable_hands)
            if joint.get("type") != "fixed":
                name = joint.get("name")
                joint_order.append(name)
                ET.SubElement(actuator, "position", name=f"motor_{name}", joint=name,
                              kp="450", kv="40",
                              ctrlrange=f"{joint.find('limit').get('lower')} {joint.find('limit').get('upper')}")
            add_subtree(body, child_name)

    add_subtree(world, "world")
    expected = [f"Joint{i}_{suffix}" for suffix in ("L", "R") for i in range(1, 8)]
    if joint_order != expected or visited != set(links):
        raise ValueError("URDF must be a connected dual-arm tree with seven joints per arm")
    home_joints = HOME_JOINTS_DEG if home_joints is None else home_joints
    initial = np.deg2rad(home_joints["left"] + home_joints["right"])
    if initial.shape != (14,) or not np.isfinite(initial).all():
        raise ValueError("Home must contain seven finite joint angles per arm")
    for name, angle in zip(joint_order, initial):
        limit = parsed.find(f"joint[@name='{name}']/limit")
        if not float(limit.get("lower")) <= angle <= float(limit.get("upper")):
            raise ValueError(f"Home angle for {name} is outside the URDF limits")
    # Hand joints may be interleaved with arm branches in MuJoCo qpos order.
    arm_home = dict(zip(joint_order, initial))
    qpos = [arm_home[j.get("name")] if j.get("name") in arm_home else
            float(np.clip(0., *floats(j.get("range")))) for j in world.iter("joint")]
    ET.SubElement(keyframe, "key", name="home", qpos=vec(qpos), ctrl=vec(initial))
    return root


def build_model(home_joints=None, movable_hands=False):
    """Load the complete marvin_dual_arm URDF, including its stand and mounts.

    Hand mounts are fixed in configuration and independent of the home pose.
    Both hardware and simulation use the same selected joint-angle profile.
    """
    mounts = hand_mount_quats() if all(hand_paths()["urdf"][s].exists() for s in SIDES) else None
    if movable_hands and mounts is None:
        raise FileNotFoundError("Both Wuji Hand descriptions are required for movable hands")
    xml = ET.tostring(build_xml(mounts, home_joints, movable_hands), encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def _pose_move_step(start, target, progress):
    """Smoothstep interpolation between two absolute 7-joint configurations."""
    from unirobot_infra.robots.arm.tianji_marvin_backend import smoothstep

    blend = smoothstep(min(max(progress, 0.), 1.))
    start = np.asarray(start, dtype=float)
    return start + blend * (np.asarray(target, dtype=float) - start)


class MarvinTeleop:
    def __init__(self, model, tracker_sns, pico_to_sim, scale=1.0, elbow_gain=.35,
                 stall_threshold_mm=30., stall_ticks=135):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ik_data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, model.key("home").id)
        mujoco.mj_resetDataKeyframe(model, self.ik_data, model.key("home").id)
        mujoco.mj_forward(model, self.data)
        mujoco.mj_forward(model, self.ik_data)
        self.home = self.ik_data.qpos.copy()
        self.command = self.home.copy()
        self._reset_start = None
        self._reset_target = self.home.copy()
        self.tracker_sns = tracker_sns
        self.pico_to_sim = rotation_matrix(pico_to_sim)
        self.scale = scale
        if not math.isfinite(elbow_gain) or not 0. <= elbow_gain <= 1.:
            raise ValueError("elbow_gain must be between 0 and 1")
        self.elbow_gain = elbow_gain
        if not math.isfinite(stall_threshold_mm) or stall_threshold_mm <= 0:
            raise ValueError("stall_threshold_mm must be positive and finite")
        if not isinstance(stall_ticks, int) or stall_ticks < 1:
            raise ValueError("stall_ticks must be a positive integer")
        self.stall_threshold_mm = stall_threshold_mm
        self.stall_ticks = stall_ticks
        self.reference = None
        self.targets = {}
        # Stall guard: a large wrist residual that stops improving means the
        # differential IK is leaking toward a limit or stuck in a local
        # minimum; recording is blocked while `stalled` is set.
        self.stalled = False
        self.stall_residual = 0.
        self._stall_best = {}
        self._stall_count = {}
        # Per-arm null-space home overrides from SDK arm-angle planning (B).
        self.plan_bias = {}
        self.plan_bias_angle = {}
        self._pose_move = None
        self.last_frame = None
        self.last_fresh = time.monotonic()
        self.freshness_seconds = .25
        self.last_sns = set()
        self.site_ids = {side: {part: model.site(f"{side}_{part}").id for part in ("tcp", "elbow")}
                         for side in SIDES}
        self.joint_ids = {side: [model.joint(f"Joint{i}_{side[0].upper()}").id for i in range(1, 8)]
                          for side in SIDES}
        self.qpos_ids = {side: np.array([model.jnt_qposadr[j] for j in ids])
                         for side, ids in self.joint_ids.items()}
        self.dof_ids = {side: np.array([model.jnt_dofadr[j] for j in ids])
                        for side, ids in self.joint_ids.items()}
        self.arm_qpos_ids = np.concatenate([self.qpos_ids[side] for side in SIDES])
        self.actuator_qpos_ids = np.array([model.jnt_qposadr[int(j)] for j in model.actuator_trnid[:, 0]])

    def arm_command(self):
        """Canonical 14-element stream; never include finger joints."""
        return self.command[self.arm_qpos_ids].copy()

    def sync_controls(self):
        self.data.ctrl[:] = self.command[self.actuator_qpos_ids]

    def capture_baseline(self):
        self.ik_data.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.ik_data)
        self.baseline = {side: {part: (
            self.ik_data.site_xpos[site].copy(),
            Rotation.from_matrix(self.ik_data.site_xmat[site].reshape(3, 3)))
            for part, site in self.site_ids[side].items()} for side in SIDES}

    def set_targets(self, targets, received_at=None):
        """Semantic target interface for Body input; no fake tracker serials."""
        if self.returning_home:
            return
        if set(targets) != {"left", "right"}:
            raise ValueError("Both arm targets are required")
        for position, rotation, elbow in targets.values():
            if (np.shape(position) != (3,) or np.shape(elbow) != (3,) or
                    not np.isfinite(position).all() or not np.isfinite(elbow).all() or
                    not isinstance(rotation, Rotation)):
                raise ValueError("Invalid arm target")
        self.targets = targets
        self.reference = {}  # BodyRetarget owns the human reference.
        self.last_fresh = time.monotonic() if received_at is None else received_at
        self._clear_stall()

    def accept(self, poses, frame):
        """Accept one coherent SDK frame indexed by serial number."""
        if self.returning_home or frame == self.last_frame or frame == 0:
            return False
        self.last_frame = frame
        self.last_sns = set(poses)
        expected = set(self.tracker_sns.values())
        if not expected.issubset(poses):
            self.reset_reference()
            return False
        for serial in expected:
            pose = np.asarray(poses[serial], dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 0.05:
                self.reset_reference()
                return False
        self.last_fresh = time.monotonic()
        if self.reference is None:
            self.reference = {sn: np.array(poses[sn], dtype=float) for sn in expected}
            self.capture_baseline()
            print("Four trackers acquired; teleoperation reference pose recorded.", flush=True)
        transform = Rotation.from_matrix(self.pico_to_sim)
        for side in SIDES:
            wrist = np.asarray(poses[self.tracker_sns[f"{side}_wrist"]], dtype=float)
            arm = np.asarray(poses[self.tracker_sns[f"{side}_arm"]], dtype=float)
            ref_wrist = self.reference[self.tracker_sns[f"{side}_wrist"]]
            ref_arm = self.reference[self.tracker_sns[f"{side}_arm"]]
            wrist_pos, wrist_rot = self.baseline[side]["tcp"]
            arm_pos, _ = self.baseline[side]["elbow"]
            delta = Rotation.from_quat(wrist[3:]) * Rotation.from_quat(ref_wrist[3:]).inv()
            world_delta = transform * delta * transform.inv()
            self.targets[side] = (
                wrist_pos + self.scale * self.pico_to_sim @ (wrist[:3] - ref_wrist[:3]),
                world_delta * wrist_rot,
                arm_pos + self.scale * self.pico_to_sim @ (arm[:3] - ref_arm[:3]),
            )
        return True

    def reset_reference(self):
        """Hold the last joint command and rebase on the next complete frame."""
        self.reference = None
        self.targets.clear()
        self._clear_stall()
        self.command[self.arm_qpos_ids] = self.data.qpos[self.arm_qpos_ids]
        self.sync_controls()

    def _clear_stall(self):
        self.stalled = False
        self.stall_residual = 0.
        self._stall_best = {}
        self._stall_count = {}

    def _update_stall_guard(self):
        """Track the best wrist residual per arm; stall after stall_ticks ticks
        without improvement above stall_threshold_mm (residuals in meters)."""
        worst = 0.
        for side in SIDES:
            if side not in self.targets:
                self._stall_best.pop(side, None)
                self._stall_count.pop(side, None)
                continue
            target_pos = self.targets[side][0]
            tcp = self.site_ids[side]["tcp"]
            residual = float(np.linalg.norm(target_pos - self.data.site_xpos[tcp]))
            worst = max(worst, residual)
            if residual <= self.stall_threshold_mm * 1e-3:
                self._stall_best.pop(side, None)
                self._stall_count.pop(side, None)
                continue
            if side not in self._stall_best or residual < self._stall_best[side] - 1e-6:
                self._stall_best[side] = residual
                self._stall_count[side] = 0
                continue
            self._stall_count[side] += 1
            if self._stall_count[side] >= self.stall_ticks:
                self.stalled = True
        self.stall_residual = worst

    def set_arm_plan_bias(self, side, q7_rad, angle_deg=None):
        """Override the null-space home term for one arm (SDK arm-angle plan)."""
        bias = np.asarray(q7_rad, dtype=float)
        if side not in SIDES or bias.shape != (7,) or not np.isfinite(bias).all():
            raise ValueError("plan bias needs seven finite angles for left/right")
        self.plan_bias[side] = bias.copy()
        self.plan_bias_angle[side] = angle_deg

    def clear_arm_plan_bias(self, side):
        self.plan_bias.pop(side, None)
        self.plan_bias_angle.pop(side, None)

    @property
    def pose_moving(self):
        return self._pose_move is not None

    def start_pose_move(self, side, target_q7_rad, speed_deg_s=60.):
        """Absolute joint-space move for one arm; tracking is paused (G key)."""
        if side not in SIDES:
            raise ValueError("side must be left or right")
        if not math.isfinite(speed_deg_s) or speed_deg_s <= 0:
            raise ValueError("Pose move speed must be positive and finite")
        qids = self.qpos_ids[side]
        limits = self.model.jnt_range[self.joint_ids[side]]
        target = np.clip(np.asarray(target_q7_rad, dtype=float), limits[:, 0], limits[:, 1])
        start = self.command[qids].copy()
        delta = float(np.max(np.abs(target - start)))
        # smoothstep's max derivative is 1.5, same budget as reset_home.
        duration = max(1 / 90, 1.5 * delta / math.radians(speed_deg_s))
        self._pose_move = {"side": side, "start": start, "target": target,
                           "elapsed": 0., "duration": duration}
        print(f"POSE {side}: max joint delta {math.degrees(delta):.1f} deg "
              f"in {duration:.2f}s", flush=True)

    def _step_pose_move(self, dt):
        move = self._pose_move
        move["elapsed"] += dt
        progress = min(1., move["elapsed"] / move["duration"])
        self.command[self.qpos_ids[move["side"]]] = _pose_move_step(
            move["start"], move["target"], progress)
        if progress >= 1.:
            self._pose_move = None
            print("POSE move finished; tracking re-anchors on the next frame.", flush=True)

    def status_marks(self):
        """Compact HUD marks: STALLED / POSE / PLAN state, empty when idle."""
        marks = []
        if self.stalled:
            marks.append(f"STALLED ({self.stall_residual * 1e3:.0f} mm)")
        if self._pose_move is not None:
            move = self._pose_move
            pct = 100. * min(1., move["elapsed"] / move["duration"])
            marks.append(f"POSE {move['side']} -> "
                         f"{np.round(np.degrees(move['target']), 1).tolist()} {pct:.0f}%")
        for side in SIDES:
            if side in self.plan_bias_angle:
                marks.append(f"PLAN {side} {self.plan_bias_angle[side]:+.0f}deg")
        return " | ".join(marks)

    @property
    def returning_home(self):
        return self._reset_start is not None

    def reset_home(self, speed_deg_s=60.):
        """Pause tracking and return both arms to the selected home profile."""
        if not math.isfinite(speed_deg_s) or speed_deg_s <= 0:
            raise ValueError("Reset speed must be positive and finite")
        self._pose_move = None
        self.reset_reference()
        self.baseline = {}
        self._reset_start = self.command.copy()
        start = self._reset_start[self.arm_qpos_ids]
        home = self.home[self.arm_qpos_ids]
        # Per-joint shortest angular path: home +/- 360k that stays in range.
        wrapped = (home - start + np.pi) % (2 * np.pi) - np.pi
        equivalent = start + wrapped
        ids = np.concatenate([self.qpos_ids[side] for side in SIDES])
        lower = self.model.jnt_range[np.concatenate([self.joint_ids[side] for side in SIDES])][:, 0]
        upper = self.model.jnt_range[np.concatenate([self.joint_ids[side] for side in SIDES])][:, 1]
        valid = (equivalent >= lower) & (equivalent <= upper)
        self._reset_target = np.where(valid, equivalent, home)
        delta = float(np.max(np.abs(self._reset_target - start)))
        self._reset_duration = max(1 / 90, 1.5 * delta / math.radians(speed_deg_s))
        self._reset_elapsed = 0.
        print("R: returning both arms to home; reference countdown starts after reset.", flush=True)
        print(f"R: reset start(deg)={np.round(np.degrees(start), 1).tolist()} "
              f"target(deg)={np.round(np.degrees(self._reset_target), 1).tolist()}", flush=True)

    def cancel_reset(self):
        """Pause an in-progress home move at the current command."""
        self._reset_start = None
        self.reset_reference()

    def diagnostic_lines(self):
        """Compare mapped targets to IK output in operator forward/left/up axes."""
        from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

        lines = []
        for side in SIDES:
            if side not in self.targets:
                continue
            start = self.baseline[side]["tcp"][0]
            target = 100. * ROBOT_TO_SIM.T @ (self.targets[side][0] - start)
            actual = 100. * ROBOT_TO_SIM.T @ (
                self.data.site_xpos[self.site_ids[side]["tcp"]] - start)
            error = np.linalg.norm(target - actual)
            current_rot = Rotation.from_matrix(self.data.site_xmat[self.site_ids[side]["tcp"]].reshape(3, 3))
            angle = np.degrees((self.targets[side][1] * current_rot.inv()).magnitude())
            lines.append(f"{side}: goal {np.round(target, 1)} / model {np.round(actual, 1)}; error {error:.1f} cm / {angle:.1f} deg")
            q = self.command[self.qpos_ids[side]]
            limits = self.model.jnt_range[self.joint_ids[side]]
            margin = np.degrees(np.minimum(q - limits[:, 0], limits[:, 1] - q))
            near = ", ".join(f"J{i + 1} {v:.1f}deg" for i, v in enumerate(margin) if v < 5.)
            closest = int(np.argmin(margin))
            lines.append(f"{side} joints deg {np.round(np.degrees(q), 1)} | " +
                         (f"NEAR LIMIT: {near}" if near else f"min margin J{closest + 1}: {margin[closest]:.1f}deg"))
        return lines

    def solve(self, dt=1 / 90, max_speed_deg_s=120.):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_speed_deg_s) or max_speed_deg_s <= 0:
            raise ValueError("Control dt and speed must be positive and finite")
        dt = min(dt, .04)
        if not self.targets:
            self._clear_stall()
            return
        self.ik_data.qpos[:] = self.command
        for _ in range(8):
            for side in SIDES:
                mujoco.mj_forward(self.model, self.ik_data)
                tcp, elbow = self.site_ids[side]["tcp"], self.site_ids[side]["elbow"]
                target_pos, target_rot, target_elbow = self.targets[side]
                current_rot = Rotation.from_matrix(self.ik_data.site_xmat[tcp].reshape(3, 3))
                position_error = np.clip(target_pos - self.ik_data.site_xpos[tcp], -0.05, 0.05)
                rotation_error = (target_rot * current_rot.inv()).as_rotvec()
                elbow_error = np.clip(target_elbow - self.ik_data.site_xpos[elbow], -0.05, 0.05)
                jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
                je = np.zeros_like(jp); unused = np.zeros_like(jp)
                mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, tcp)
                mujoco.mj_jacSite(self.model, self.ik_data, je, unused, elbow)
                dofs = self.dof_ids[side]
                jac = np.vstack((jp[:, dofs], 0.5 * jr[:, dofs]))
                error = np.r_[position_error, 0.5 * rotation_error]
                qids = self.qpos_ids[side]
                q = self.ik_data.qpos[qids].copy()
                limits = self.model.jnt_range[self.joint_ids[side]]
                # The SDK arm-angle plan (B) replaces the home null-space
                # target; reset_home still returns to the real home profile.
                home = self.plan_bias[side] if side in self.plan_bias else self.home[qids]
                primary, secondary = wrist_and_elbow_steps(
                    jac, error, je[:, dofs], elbow_error, q, home,
                    limits, self.elbow_gain)
                primary_q = np.clip(q + primary, limits[:, 0], limits[:, 1])
                self.ik_data.qpos[qids] = primary_q
                if np.linalg.norm(secondary) < 1e-9:
                    continue
                mujoco.mj_forward(self.model, self.ik_data)
                primary_pos_error = np.linalg.norm(target_pos - self.ik_data.site_xpos[tcp])
                primary_rot = Rotation.from_matrix(self.ik_data.site_xmat[tcp].reshape(3, 3))
                primary_rot_error = (target_rot * primary_rot.inv()).magnitude()
                # Null-space preservation is first-order only. Backtrack if
                # actual FK moves the wrist outside a 0.5 mm / 0.1 deg band,
                # or makes a larger primary error worse.
                for fraction in (1., .5, .25, .125):
                    self.ik_data.qpos[qids] = primary_q + fraction * secondary
                    mujoco.mj_forward(self.model, self.ik_data)
                    pos_error = np.linalg.norm(target_pos - self.ik_data.site_xpos[tcp])
                    rot = Rotation.from_matrix(self.ik_data.site_xmat[tcp].reshape(3, 3))
                    rot_error = (target_rot * rot.inv()).magnitude()
                    if (pos_error <= max(primary_pos_error, .0005) + 1e-8
                            and rot_error <= max(primary_rot_error, math.radians(.1)) + 1e-8):
                        break
                else:
                    self.ik_data.qpos[qids] = primary_q
        # The simulated command changes at a bounded joint rate.
        max_step = math.radians(max_speed_deg_s) * dt
        for qids in self.qpos_ids.values():
            delta = self.ik_data.qpos[qids] - self.command[qids]
            fraction = min(1., max_step / max(np.max(np.abs(delta)), 1e-12))
            self.command[qids] += fraction * delta
        self.sync_controls()
        self._update_stall_guard()

    def step(self, dt=1 / 90, max_speed_deg_s=120.):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_speed_deg_s) or max_speed_deg_s <= 0:
            raise ValueError("Control dt and speed must be positive and finite")
        dt = min(dt, .04)
        if self.returning_home:
            self._reset_elapsed += dt
            progress = min(1., self._reset_elapsed / self._reset_duration)
            blend = progress * progress * (3. - 2. * progress)
            ids = self.arm_qpos_ids
            self.command[ids] = self._reset_start[ids] + blend * (self._reset_target[ids] - self._reset_start[ids])
            self.data.qvel[:] = 0.
            if progress >= 1.:
                self.command[ids] = self.home[ids]
                self.ik_data.qpos[:] = self.command
                self.ik_data.qvel[:] = 0.
                mujoco.mj_forward(self.model, self.ik_data)
                self._reset_start = None
                print("Both arm commands reset to home.", flush=True)
        elif self._pose_move is not None:
            # One arm interpolates to the SDK pose target; the other holds.
            self._step_pose_move(dt)
        elif time.monotonic() - self.last_fresh > self.freshness_seconds:
            self.reset_reference()
        else:
            self.solve(dt, max_speed_deg_s)
        # MuJoCo is used as a kinematic viewer. The CAD inertias and visual
        # meshes are not a validated physics model for dynamic simulation.
        self.data.qpos[:] = self.command
        self.sync_controls()
        mujoco.mj_forward(self.model, self.data)


def read_trackers(sdk):
    return read_tracker_frame(sdk)[0]


def dispatch_command(controller, robot=None, dt=1 / 90, max_speed_deg_s=120.):
    """Single IK result for model and hardware; hardware performs unit conversion only."""
    controller.step(dt, max_speed_deg_s)
    command = controller.arm_command()
    if robot is not None:
        robot.write(command)
    return command


def draw_diagnostics(window, controller):
    from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

    with window.lock():
        scene = window.user_scn
        scene.ngeom = 0
        origin_xyz = controller.data.xpos[controller.model.body("dual_arm_mount").id] + np.array([0., 0., .25])
        colors = ([1., .2, .2, 1.], [.2, 1., .2, 1.], [.3, .5, 1., 1.])
        for i, color in enumerate(colors):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                               origin_xyz, np.eye(3).ravel(), np.array(color))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .006,
                                 origin_xyz, origin_xyz + .22 * ROBOT_TO_SIM[:, i])
            scene.ngeom += 1
        for side, color in (("left", [0., 1., 1., .65]), ("right", [1., .6, 0., .65])):
            if side in controller.targets:
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.015] * 3), controller.targets[side][0],
                                   np.eye(3).ravel(), np.array(color))
                scene.ngeom += 1


def model_joint_limits_deg(model):
    """Per-arm 7-joint limits (deg) from the MuJoCo model — backend safety clip."""
    limits = {}
    for side in SIDES:
        ids = [model.joint(f"Joint{i}_{side[0].upper()}").id for i in range(1, 8)]
        limits[side] = np.degrees(model.jnt_range[ids]).tolist()
    return limits


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-source", choices=("object", "body"), default="object",
                        help="Independent Object trackers or Full Body skeleton")
    parser.add_argument("--preview-body", action="store_true", help="Read-only 24-node Body preview")
    parser.add_argument("--body-config", type=Path, default=None,
                        help="Full Body mapping config (default: package config/pico_body.yaml)")
    parser.add_argument("--body-reference", type=Path, default=None,
                        help="Saved Body reference JSON (default: ./.local/pico-body-reference.json)")
    parser.add_argument("--body-scale", type=float, default=None,
                        help="Override the Body config's translation gain")
    parser.add_argument("--replay", type=Path, help="Replay a Body JSONL recording (sim only)")
    parser.add_argument("--hand-input", choices=("none", "ros"), default="none",
                        help="Body mode: optional retargeted finger targets over ROS")
    parser.add_argument("--control-file", type=Path,
                        help="Body mode: append JSON lines with action pause/resume/reset/stop")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI for checks")
    parser.add_argument("--duration", type=float, default=0, help="Seconds to run (0: until window closes)")
    parser.add_argument("--live-duration", type=float, default=0,
                        help="Stop this many seconds after the first tracker reference is recorded")
    parser.add_argument("--scale", type=float, default=1.0, help="Tracker translation scale")
    parser.add_argument("--elbow-gain", type=float, default=.35,
                        help="Elbow null-space guide gain, 0 disables it (default .35); wrist has priority")
    parser.add_argument("--home-profile", choices=home_profile_names(), default="default",
                        help="Shared sim/hardware home posture; default is the evaluated teleop pose")
    parser.add_argument("--demo", action="store_true", help="Use moving synthetic tracker poses")
    parser.add_argument("--backend", choices=("sim", "robot", "both"), default="sim",
                        help="Where joint targets go: MuJoCo sim, real Tianji arms, or both")
    parser.add_argument("--tracker-yaw", type=float, default=None,
                        help="Explicit manual yaw override; otherwise use saved motion calibration")
    parser.add_argument("--calibrate", action="store_true",
                        help="Record operator forward/left/up motions before starting teleop")
    parser.add_argument("--calibration-file", type=Path, default=None,
                        help="Shared tracker-axis calibration (default: ./.local/pico-axis-calibration.json)")
    parser.add_argument("--ref-delay", type=float, default=5.0,
                        help="Seconds to assume the robot reference posture after tracking arrives")
    parser.add_argument("--freeze-frames", type=int, default=10,
                        help="Bit-identical frames before a tracker is treated as frozen (default 10)")
    parser.add_argument("--ramp-seconds", type=float, default=.6,
                        help="Seconds to blend back to the live pose after a freeze (default 0.6)")
    parser.add_argument("--forearm-length", type=float, default=0.,
                        help="Wrist-to-elbow length (m) for the redundant forearm elbow hint; "
                             "0 measures tracker spacing + 0.10 m (default)")
    parser.add_argument("--record", type=Path,
                        help="Write tracker inputs, joint limits and IK errors to a new JSONL file")
    parser.add_argument("--wrist-camera", action="store_true",
                        help="Enable the RealSense wrist camera: mp4+depth recording and a live view")
    parser.add_argument("--wrist-camera-serial", default=None,
                        help="RealSense serial number (default: first device found)")
    parser.add_argument("--wrist-camera-dir", type=Path, default=None,
                        help="Camera output dir (default: <record>.wrist/ or .local/wrist-camera-<ts>/)")
    parser.add_argument("--wrist-camera-depth-hz", type=float, default=10.,
                        help="Depth PNG rate; color mp4 records at full frame rate (default 10)")
    parser.add_argument("--wrist-camera-rotate", type=int, choices=(0, 90, 180, 270), default=90,
                        help="Camera mounting rotation, applied to live view AND recordings "
                             "(default 90 = our wrist mount's upright frame)")
    parser.add_argument("--wrist-camera-no-live", action="store_true",
                        help="Record without opening the live view window")
    parser.add_argument("--wrist-camera-paused", action="store_true",
                        help="Start with recording paused; C key or the live-view button toggles")
    parser.add_argument("--main-camera", action="store_true",
                        help="Enable a second (main-view) RealSense camera alongside the wrist camera")
    parser.add_argument("--main-camera-serial", default=None,
                        help="Main camera serial (default: the device not used by the wrist camera)")
    parser.add_argument("--main-camera-dir", type=Path, default=None,
                        help="Main camera output dir (default: <record>.main/ or .local/main-camera-<ts>/)")
    parser.add_argument("--main-camera-depth-hz", type=float, default=10.,
                        help="Main camera depth PNG rate (default 10)")
    parser.add_argument("--main-camera-rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Main camera mounting rotation (default 0)")
    parser.add_argument("--main-camera-no-live", action="store_true",
                        help="Main camera records without a live view window")
    parser.add_argument("--main-camera-paused", action="store_true",
                        help="Main camera starts paused")
    parser.add_argument("--wrist2-camera", action="store_true",
                        help="Enable the second wrist camera (other wrist)")
    parser.add_argument("--wrist2-camera-serial", default=None,
                        help="Second wrist camera serial (default: auto-assign remaining device)")
    parser.add_argument("--wrist2-camera-dir", type=Path, default=None,
                        help="Second wrist camera output dir (default: <record>.wrist2/)")
    parser.add_argument("--wrist2-camera-depth-hz", type=float, default=10.,
                        help="Second wrist camera depth PNG rate (default 10)")
    parser.add_argument("--wrist2-camera-rotate", type=int, choices=(0, 90, 180, 270), default=90,
                        help="Second wrist camera mounting rotation (default 90)")
    parser.add_argument("--wrist2-camera-no-live", action="store_true",
                        help="Second wrist camera records without a live view window")
    parser.add_argument("--wrist2-camera-paused", action="store_true",
                        help="Second wrist camera starts paused")
    parser.add_argument("--robot-ip", default=None,
                        help="Tianji controller IP (default: TIANJI_ROBOT_IP, then robot_ip from marvin_robot.yaml)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dual-arm model directory (default: MUJOCO_MARVIN_MODEL_DIR, then assets/robot/marvin_dual_arm)")
    parser.add_argument("--robot-config", type=Path, default=None,
                        help="Tianji robot YAML (default: MUJOCO_MARVIN_ROBOT_CONFIG, then configs/robot/marvin_robot.yaml)")
    parser.add_argument("--tracker-config", type=Path, default=None,
                        help="PICO tracker YAML (default: configs/device/pico_input.yaml or its .template)")
    parser.add_argument("--sdk-dir", type=Path, default=None,
                        help="Tianji vendor SDK directory (default: MARVIN_SDK_DIR, then ./vendor)")
    parser.add_argument("--pose", default=None,
                        help='SDK closed-form absolute pose "x,y,z,a,b,c" in the arm base frame '
                             '(mm, deg); G key moves --arm to it once')
    parser.add_argument("--arm", choices=("left", "right"), default="left",
                        help="Arm used by --pose and the B arm-angle plan (default left)")
    parser.add_argument("--pose-speed", type=float, default=60.,
                        help="Pose move joint speed limit in deg/s (default 60)")
    parser.add_argument("--kine-config", type=Path, default=None,
                        help="SDK kine config .MvKDCfg (default: MARVIN_KINE_CFG, then "
                             "$SDK_DIR/CommonConfig/ccs_m6_40.MvKDCfg)")
    parser.add_argument("--elbow-angle", type=float, default=15.,
                        help="Arm-angle offset in deg for the B-key plan bias (default 15)")
    parser.add_argument("--stall-threshold", type=float, default=30.,
                        help="Wrist residual in mm above which the stall guard may rise (default 30)")
    parser.add_argument("--stall-ticks", type=int, default=135,
                        help="Consecutive ticks without residual improvement before the stall "
                             "guard blocks recording (default 135 ~= 1.5s @90Hz)")
    return parser


def main(argv=None):
    from unirobot_infra.devices.pico_tracker import DEFAULT_ALIGNMENT

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_dir is not None:
        # Resolve early so every model-dependent constant below — and the
        # ones resolved lazily by robot_description — agrees with the CLI.
        os.environ["MUJOCO_MARVIN_MODEL_DIR"] = str(args.model_dir)
    if args.robot_config is not None:
        os.environ["MUJOCO_MARVIN_ROBOT_CONFIG"] = str(args.robot_config)
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be positive and finite")
    if not math.isfinite(args.elbow_gain) or not 0 <= args.elbow_gain <= 1:
        parser.error("--elbow-gain must be between 0 and 1")
    if not math.isfinite(args.ref_delay) or args.ref_delay < 0:
        parser.error("--ref-delay must be nonnegative and finite")
    if not math.isfinite(args.live_duration) or args.live_duration < 0:
        parser.error("--live-duration must be nonnegative and finite")
    if not isinstance(args.freeze_frames, int) or args.freeze_frames < 1:
        parser.error("--freeze-frames must be a positive integer")
    if not math.isfinite(args.ramp_seconds) or args.ramp_seconds <= 0:
        parser.error("--ramp-seconds must be positive and finite")
    if not math.isfinite(args.forearm_length) or args.forearm_length < 0:
        parser.error("--forearm-length must be nonnegative and finite")
    if not math.isfinite(args.wrist_camera_depth_hz) or args.wrist_camera_depth_hz <= 0:
        parser.error("--wrist-camera-depth-hz must be positive and finite")
    if not math.isfinite(args.main_camera_depth_hz) or args.main_camera_depth_hz <= 0:
        parser.error("--main-camera-depth-hz must be positive and finite")
    if not math.isfinite(args.wrist2_camera_depth_hz) or args.wrist2_camera_depth_hz <= 0:
        parser.error("--wrist2-camera-depth-hz must be positive and finite")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be nonnegative and finite")
    if not math.isfinite(args.pose_speed) or args.pose_speed <= 0:
        parser.error("--pose-speed must be positive and finite")
    if not math.isfinite(args.elbow_angle):
        parser.error("--elbow-angle must be finite")
    if not math.isfinite(args.stall_threshold) or args.stall_threshold <= 0:
        parser.error("--stall-threshold must be positive and finite")
    if not isinstance(args.stall_ticks, int) or args.stall_ticks < 1:
        parser.error("--stall-ticks must be a positive integer")
    pose_xyzabc = None
    if args.pose is not None:
        pose_xyzabc = np.fromstring(args.pose, sep=",")
        if pose_xyzabc.shape != (6,) or not np.isfinite(pose_xyzabc).all():
            parser.error('--pose must be six comma-separated numbers "x,y,z,a,b,c" (mm, deg)')
    robot_cfg = yaml.safe_load(resolve_robot_config(args.robot_config).read_text())
    args.robot_ip = resolve_robot_ip(args.robot_ip, robot_cfg.get("robot_ip"))
    if args.backend in ("robot", "both") and not args.robot_ip:
        parser.error("arms/both output requires --robot-ip, TIANJI_ROBOT_IP, or robot_ip in marvin_robot.yaml")
    try:
        resolve_model_dir(args.model_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.input_source == "body":
        if args.calibrate or args.tracker_yaw is not None or args.calibration_file is not None:
            parser.error("Body uses its own chest frame/reference; Object direction calibration is not used")
        if args.demo and args.replay:
            parser.error("Choose either --demo or --replay")
        if args.backend != "sim" and (args.demo or args.replay or args.preview_body):
            parser.error("Body demo, replay and preview are simulation-only")
        if args.backend != "sim" and not args.robot_ip:
            parser.error("Body hardware output requires --robot-ip or TIANJI_ROBOT_IP")
        if args.preview_body and args.hand_input != "none":
            parser.error("Body preview does not consume hand commands")
        from unirobot_infra.apps.teleop_pico_body import run
        run(args)
        return
    if args.preview_body or args.replay or args.hand_input != "none" or args.body_scale is not None or args.control_file:
        parser.error("Body preview/replay/hand input require --input-source body")
    if args.tracker_yaw is not None and not math.isfinite(args.tracker_yaw):
        parser.error("--tracker-yaw must be finite")
    if args.calibrate and (args.tracker_yaw is not None or args.demo):
        parser.error("--calibrate cannot be combined with --tracker-yaw or --demo")
    if args.demo and args.backend != "sim":
        parser.error("--demo is only available with --backend sim")
    tracker_config = resolve_tracker_config(args.tracker_config)
    try:
        config = yaml.safe_load(tracker_config.read_text())["pico_input_node"]["ros__parameters"]
    except (OSError, KeyError, TypeError) as exc:
        parser.error(f"Cannot read tracker config {tracker_config}: {exc}. "
                     "Copy configs/device/pico_input.yaml.template to configs/device/pico_input.yaml "
                     "and fill in your four tracker serial numbers.")
    raw_sns = {key[len("tracker_sn_"):]: str(value).strip()
               for key, value in config.items()
               if isinstance(key, str) and key.startswith("tracker_sn_")}
    fusion_pairs = None
    if any(role.endswith("_aux") for role in raw_sns):
        # Redundant forearm mode: primary + aux tracker per forearm, fused into
        # wrist/elbow role poses; the *_arm roles become virtual placeholders.
        needed = ("left_wrist", "left_wrist_aux", "right_wrist", "right_wrist_aux")
        if (sorted(raw_sns) != sorted(needed) or len(set(raw_sns.values())) != 4
                or any(not sn or sn.startswith("YOUR_") for sn in raw_sns.values())):
            parser.error(f"Redundant forearm mode needs four distinct tracker_sn_* values in "
                         f"{tracker_config}: tracker_sn_left_wrist, tracker_sn_left_wrist_aux, "
                         "tracker_sn_right_wrist, tracker_sn_right_wrist_aux")
        fusion_pairs = {"left": {"wrist": raw_sns["left_wrist"], "aux": raw_sns["left_wrist_aux"]},
                        "right": {"wrist": raw_sns["right_wrist"], "aux": raw_sns["right_wrist_aux"]}}
        sns = {"left_wrist": raw_sns["left_wrist"], "right_wrist": raw_sns["right_wrist"],
               "left_arm": "virtual:left-elbow", "right_arm": "virtual:right-elbow"}
    else:
        sns = {role: str(config.get(f"tracker_sn_{role}", "")).strip() for role in ROLES}
    if args.demo and (len(set(sns.values())) != 4 or any(not sn or sn.startswith("YOUR_") for sn in sns.values())):
        # Synthetic trackers need no real serial numbers.
        sns = {role: f"DEMO-{role}" for role in ROLES}
    if args.demo:
        fusion_pairs = None  # the demo synthesizes the legacy four-role stream directly
    if len(set(sns.values())) != 4 or any(not sn or sn.startswith("YOUR_") for sn in sns.values()):
        parser.error(f"Set four unique tracker_sn_* values in {tracker_config} "
                     "(copy configs/device/pico_input.yaml.template to configs/device/pico_input.yaml)")
    pico_to_robot = rotation_matrix(robot_cfg["pico_to_robot"])
    needs_calibration = args.calibrate
    calibration_file = Path(args.calibration_file or DEFAULT_ALIGNMENT)
    if args.tracker_yaw is not None:
        pico_to_robot = Rotation.from_euler("z", args.tracker_yaw, degrees=True).as_matrix() @ pico_to_robot
        alignment_label = f"Manual yaw {args.tracker_yaw:g} deg"
    elif args.demo:
        alignment_label = "Demo axes"
    elif calibration_file.exists() and not needs_calibration:
        try:
            pico_to_robot = load_alignment(calibration_file, sns)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(f"{exc} Use --calibrate to replace this calibration.")
        alignment_label = "Saved operator calibration"
    else:
        needs_calibration = True
    if needs_calibration and (args.headless or args.backend == "robot"):
        parser.error("先运行 unirobot-infra-teleop-pico calibrate，在仿真中保存方向标定。")
    robot = None
    sdk = None
    recorder = None
    cameras = []
    try:
        if not args.demo:
            try:
                import xrobotoolkit_sdk as sdk
            except ImportError as exc:
                raise SystemExit(f"{_PICO_SDK_HELP} ({exc})") from exc
            sdk.init()
            print("Waiting for Motion Trackers: Object mode / Send.", flush=True)
            print("Frame timing: use Motion timestamp when nonzero, otherwise overall packet timestamp.", flush=True)
        physical_guard = None
        fusion = None
        if fusion_pairs is not None:
            physical_sns = {f"{side}_{part}": pair[part]
                            for side, pair in fusion_pairs.items()
                            for part in ("wrist", "aux")}
            physical_guard = PoseFreezeGuard(physical_sns, freeze_frames=args.freeze_frames,
                                             ramp_seconds=args.ramp_seconds)
            fusion = ForearmRedundancy(fusion_pairs, sns, forearm_length=args.forearm_length)
            print("Redundant forearm mode: aux tracker covers wrist occlusion; "
                  "elbow hint extrapolated from the forearm axis.", flush=True)

        def fuse_frame(poses, frame, now):
            """Physical freeze guard -> forearm fusion; returns (role poses, transitions)."""
            if fusion is None:
                return poses, []
            guarded, transitions = physical_guard.filter(poses, frame, now)
            live = {sn: modes == (PoseFreezeGuard.LIVE, PoseFreezeGuard.LIVE)
                    for sn, modes in physical_guard.modes().items()}
            return fusion.update(guarded, live, now), transitions

        if needs_calibration:
            from unirobot_infra.calibration.tracker_alignment_gui import calibrate
            frame_source = None
            if fusion is not None:
                def frame_source():
                    raw, frame = read_tracker_frame(sdk)
                    poses, _ = fuse_frame(raw, frame, time.monotonic())
                    return poses, frame
            pico_to_robot = calibrate(sdk, sns, calibration_file, frame_source=frame_source)
            alignment_label = "Operator motion calibration"

        from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM
        pico_to_sim = ROBOT_TO_SIM @ pico_to_robot
        print(f"Shared teleop core | backend={args.backend} | {alignment_label}", flush=True)
        print("Tracker -> operator (X forward, Y left, Z up):\n" +
              np.array2string(pico_to_robot, precision=4), flush=True)
        home_joints = load_home_joints(profile=args.home_profile)
        print(f"URDF: {resolve_model_urdf()}\nHome profile: {args.home_profile}\nHome degrees: {home_joints}", flush=True)
        model = build_model(home_joints)
        controller = MarvinTeleop(model, sns, pico_to_sim, args.scale, args.elbow_gain,
                                  stall_threshold_mm=args.stall_threshold,
                                  stall_ticks=args.stall_ticks)
        if args.record is not None:
            from unirobot_infra.recorder.teleop_trace import TeleopRecorder
            recorder = TeleopRecorder(args.record, controller)
        cameras = []
        cam_specs = []
        if args.wrist_camera:
            cam_specs.append(("wrist", args.wrist_camera_serial, args.wrist_camera_rotate,
                              args.wrist_camera_dir, args.wrist_camera_depth_hz,
                              args.wrist_camera_paused, args.wrist_camera_no_live))
        if args.main_camera:
            cam_specs.append(("main", args.main_camera_serial, args.main_camera_rotate,
                              args.main_camera_dir, args.main_camera_depth_hz,
                              args.main_camera_paused, args.main_camera_no_live))
        if args.wrist2_camera:
            cam_specs.append(("wrist2", args.wrist2_camera_serial, args.wrist2_camera_rotate,
                              args.wrist2_camera_dir, args.wrist2_camera_depth_hz,
                              args.wrist2_camera_paused, args.wrist2_camera_no_live))
        if cam_specs:
            from unirobot_infra.camera.realsense_d405 import CameraRecorder, WristCamera, default_camera_dir

            def resolve_serial(preferred, taken):
                if preferred:
                    return preferred
                import pyrealsense2 as rs
                for dev in rs.context().query_devices():
                    sn = dev.get_info(rs.camera_info.serial_number)
                    if sn not in taken:
                        return sn
                raise SystemExit("找不到未被占用的 RealSense 设备——检查相机连接")

            taken = set()
            for role, serial, rotate, cam_dir, depth_hz, paused, no_live in cam_specs:
                serial = resolve_serial(serial, taken)
                taken.add(serial)
                cam_dir = cam_dir or default_camera_dir(args.record, role)
                cam_recorder = CameraRecorder(cam_dir, depth_hz=depth_hz,
                                              recording=False)  # 启动默认不录，按钮/C 键开录
                cam = WristCamera(serial=serial, rotate=rotate, recorder=cam_recorder,
                                  live=not no_live, role=role)
                cam.start()
                cameras.append(cam)
                print(f"{role} camera [{serial[-4:]}]: {cam_dir} "
                      f"({'recording' if cam_recorder.recording else 'PAUSED — C key / button starts'})",
                      flush=True)
        if cameras:
            from unirobot_infra.recorder.pico_episode import EPISODES
            EPISODES.configure(cameras)
        if args.backend in ("robot", "both"):
            from unirobot_infra.robots.arm.tianji_marvin_backend import TianjiJointBackend
            robot = TianjiJointBackend(args.robot_ip, home_joints,
                                       limits_deg=model_joint_limits_deg(model),
                                       sdk_dir=args.sdk_dir)

        start = time.monotonic()
        frames = 0
        live_started = None
        begin_at = None
        input_frame = None
        input_at = float("-inf")
        keys = SimpleQueue()
        activity = TrackerActivity(sns)
        guard = PoseFreezeGuard(sns, freeze_frames=args.freeze_frames,
                                ramp_seconds=args.ramp_seconds)
        tracker_states = {}
        fusion_sources = {}
        paused = False
        def next_frame():
            if not args.demo:
                return read_tracker_frame(sdk)
            t = time.monotonic() - start
            poses = {serial: [0.2 * (i % 2) + 0.04 * math.sin(t),
                             1.0 + 0.03 * math.sin(0.7 * t),
                             -0.3 + 0.03 * math.cos(t + i), 0, 0, 0, 1]
                     for i, serial in enumerate(sns.values())}
            return poses, int((t + 1) * 1e9)

        def ik_reason(result):
            reasons = []
            if result.out_of_range:
                reasons.append("out_of_range")
            if result.joint_exceeded:
                tags = [f"J{i + 1}" for i, t in enumerate(result.exceeded_tags) if t]
                reasons.append(f"joint_exceeded {','.join(tags)}")
            if result.degenerate:
                reasons.append("degenerate")
            return ", ".join(reasons) or "no solution"

        def sdk_kine(side):
            from unirobot_infra.kinematics.sdk_backend import SdkKinematics

            return SdkKinematics(kine_config=args.kine_config,
                                 arm_type=0 if side == "left" else 1)

        def run_pose_move():
            side = args.arm
            controller.reset_reference()  # 暂停跟踪，命令保持
            kine = sdk_kine(side)
            ref_q7_deg = np.degrees(controller.command[controller.qpos_ids[side]])
            result = kine.solve_pose(pose_xyzabc, ref_q7_deg)
            if not result.ok:
                print(f"G: SDK IK failed ({ik_reason(result)}); pose move aborted",
                      flush=True)
                return
            controller.start_pose_move(side, np.radians(result.q7_deg), args.pose_speed)
            print(f"G: POSE {side} -> {np.round(result.q7_deg, 1).tolist()} (deg)",
                  flush=True)

        def run_plan_bias():
            side = args.arm
            if side in controller.plan_bias:
                controller.clear_arm_plan_bias(side)
                print(f"B: PLAN {side} cleared; null-space home term back to profile",
                      flush=True)
                return
            if side not in controller.targets:
                print(f"B: no active wrist target for {side}; arm-angle planning "
                      "needs a tracking target", flush=True)
                return
            kine = sdk_kine(side)
            ref_q7_deg = np.degrees(controller.command[controller.qpos_ids[side]])
            # SDK 自洽目标：对参考构型 FK，保证 TCP 不动只转臂角。
            target_xyzabc = kine.current_xyzabc(ref_q7_deg)
            result = kine.solve_pose(target_xyzabc, ref_q7_deg, zsp_type=1,
                                     zsp_angle=args.elbow_angle)
            if not result.ok:
                print(f"B: SDK arm-angle IK failed ({ik_reason(result)}); bias unchanged",
                      flush=True)
                return
            controller.set_arm_plan_bias(side, np.radians(result.q7_deg),
                                         angle_deg=args.elbow_angle)
            print(f"B: PLAN {side} {args.elbow_angle:+.0f}deg bias set "
                  f"-> {np.round(result.q7_deg, 1).tolist()} (deg)", flush=True)

        def tick():
            nonlocal frames, begin_at, input_frame, input_at, live_started, paused
            while not keys.empty():
                key = keys.get()
                if key == ord("R"):
                    controller.reset_home()
                    begin_at = None
                elif key == ord("E"):
                    controller.elbow_gain = 0. if controller.elbow_gain else (args.elbow_gain or .35)
                    print(f"Wrist priority IK: elbow guide={controller.elbow_gain:g}", flush=True)
                elif key == ord("C"):
                    if controller.stalled:
                        print(f"tracking stalled (residual {controller.stall_residual * 1e3:.0f} mm); "
                              "recording blocked", flush=True)
                    elif cameras:
                        from unirobot_infra.camera.realsense_d405 import toggle_cameras
                        toggle_cameras(cameras)
                elif key == ord("G"):
                    if args.pose is None:
                        print("G: --pose is not configured; pass --pose x,y,z,a,b,c "
                              "(SDK arm base frame, mm/deg)", flush=True)
                    elif controller.pose_moving:
                        print("G: pose move already in progress", flush=True)
                    else:
                        try:
                            run_pose_move()
                        except (RuntimeError, FileNotFoundError) as exc:
                            print(f"G: SDK kinematics unavailable: {exc}", flush=True)
                elif key == ord("B"):
                    try:
                        run_plan_bias()
                    except (RuntimeError, FileNotFoundError) as exc:
                        print(f"B: SDK kinematics unavailable: {exc}", flush=True)
                elif key == ord(" "):
                    paused = not paused
                    if paused:
                        controller.reset_reference()
                        begin_at = None
                        print("PAUSED: holding pose; SPACE resumes after the reference countdown.",
                              flush=True)
                    else:
                        print("Resumed; assume reference posture for the countdown.", flush=True)
            if paused:
                # Hold the current command; the reference recaptures on resume.
                dispatch_command(controller, robot)
                return
            poses, frame = next_frame()
            now = time.monotonic()
            poses, fusion_transitions = fuse_frame(poses, frame, now)
            for role, serial, component, old, new in fusion_transitions:
                print(f"Tracker {role} [{serial[-5:]}] {component}: {old} -> {new}", flush=True)
            if fusion is not None:
                for side in SIDES:
                    source = fusion.sources.get(side)
                    if fusion_sources.get(side) != source:
                        fusion_sources[side] = source
                        if source is not None:
                            print(f"Forearm {side}: wrist source -> {source.upper()} tracker",
                                  flush=True)
            activity.observe(poses, frame, now)
            for role in ROLES:
                status, age = activity.status(role, now)
                if tracker_states.get(role) != status:
                    tracker_states[role] = status
                    detail = f" ({age:.1f}s)" if age is not None and status != "UPDATING" else ""
                    print(f"Tracker {role} [{sns[role][-5:]}]: {status}{detail}", flush=True)
            poses, transitions = guard.filter(poses, frame, now)
            for role, serial, component, old, new in transitions:
                print(f"Tracker {role} [{serial[-5:]}] {component}: {old} -> {new}", flush=True)
            if frame and frame != input_frame:
                input_frame, input_at = frame, now
            complete = bool(frame) and now - input_at < .25 and set(sns.values()).issubset(poses)
            if controller.returning_home or (controller.reference is None and not complete):
                begin_at = None
            if not controller.returning_home and begin_at is None and complete:
                begin_at = time.monotonic() + (0. if args.demo else args.ref_delay)
                print(f"Assume reference posture; capturing in {0 if args.demo else args.ref_delay:g}s.", flush=True)
            if complete and begin_at is not None and time.monotonic() >= begin_at:
                if controller.accept(poses, frame):
                    frames += 1
                    if live_started is None:
                        live_started = time.monotonic()
                        if args.live_duration:
                            print(f"Recording operation for {args.live_duration:g}s from NOW.", flush=True)
                elif controller.reference is None:
                    begin_at = None
            was_ready = controller.reference is not None
            dispatch_command(controller, robot)
            if was_ready and controller.reference is None:
                begin_at = None
            from unirobot_infra.recorder.pico_episode import EPISODES
            if EPISODES.active:
                EPISODES.record_robot(controller, robot, poses, frame)
            if recorder is not None:
                recorder.record(controller, poses, frame)

        def keep_running():
            now = time.monotonic()
            return ((args.duration <= 0 or now - start < args.duration)
                    and (args.live_duration <= 0 or live_started is None
                         or now - live_started < args.live_duration))

        if args.headless or args.backend == "robot":
            last_marks = None
            while keep_running():
                tick()
                if cameras:
                    from unirobot_infra.camera.realsense_d405 import _VIEW
                    _VIEW.sync()
                    _VIEW.update()
                marks = controller.status_marks()
                if marks != last_marks:
                    last_marks = marks
                    if marks:
                        print(f"[status] {marks}", flush=True)
                time.sleep(1 / 90)
        else:
            from mujoco import viewer
            with viewer.launch_passive(model, controller.data, key_callback=keys.put) as window:
                window.cam.distance = 2.2
                # From behind: robot left appears on screen left; forward is away.
                forward = ROBOT_TO_SIM[:, 0]
                window.cam.azimuth = math.degrees(math.atan2(-forward[1], -forward[0]))
                window.cam.elevation = -20
                window.cam.lookat[:] = controller.data.xpos[model.body("dual_arm_mount").id] + .15 * forward
                while window.is_running() and keep_running():
                    tick()
                    if cameras:
                        from unirobot_infra.camera.realsense_d405 import _VIEW
                        _VIEW.sync()
                        _VIEW.update()
                    draw_diagnostics(window, controller)
                    ready = controller.reference is not None
                    countdown = max(0., begin_at - time.monotonic()) if begin_at is not None else args.ref_delay
                    lines = [f"{args.backend.upper()} | home={args.home_profile} | {alignment_label}",
                             "X FORWARD (red) | Y LEFT (green) | Z UP (blue)",
                             "View from behind. R: reset arms to home + re-record reference",
                             "SPACE: pause/resume (resume recaptures the reference) | "
                             f"E: toggle elbow guide (now {controller.elbow_gain:g}) | "
                             f"B: arm-angle plan bias ({args.arm}) | "
                             "G: SDK pose move" + (" | C: camera record" if cameras else ""),
                             ("PAUSED: holding pose; SPACE resumes" if paused else
                              "RESET: returning both arms to home; tracking paused" if controller.returning_home else
                              "LIVE: goal / model TCP delta [X Y Z], cm" if ready else
                              f"Waiting for trackers / reference: {countdown:.1f}s")]
                    marks = controller.status_marks()
                    if marks:
                        lines.insert(1, marks)
                    if cameras:
                        recording = sum(cam.recording for cam in cameras)
                        lines.insert(4, f"相机：● 录制中（{recording}/{len(cameras)}）" if recording
                                     else "相机：已暂停（C 或相机窗口按钮开始）")
                    if args.live_duration and live_started is not None:
                        remaining = max(0., args.live_duration - (time.monotonic() - live_started))
                        lines.insert(0, f"RECORDING: {remaining:.1f}s remaining (window closes automatically)")
                    lines.extend(controller.diagnostic_lines())
                    lines.extend(activity.diagnostic_lines())
                    window.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(lines), ""))
                    window.sync()
                    time.sleep(1 / 90)
        print(f"Teleop finished: {frames} valid tracker frames; qpos={controller.data.qpos.tolist()}")
        if frames == 0:
            raise RuntimeError("No Motion Tracker data; check Object mode, Send, and PC Service")
    finally:
        from unirobot_infra.recorder.pico_episode import EPISODES
        EPISODES.close()
        for cam in cameras:
            cam.stop()
        if cameras:
            from unirobot_infra.camera.realsense_d405 import _VIEW
            _VIEW.shutdown()
        if robot is not None:
            robot.close()
        if sdk is not None:
            sdk.close()
        if recorder is not None:
            recorder.close()


def __getattr__(name):
    """Lazily resolve model/hand-dependent constants so this module imports
    without vendor assets.  ``ROBOT_TO_SIM`` comes from pico_tracker;
    ``HAND_URDF``/``HAND_MESH_DIR``/``HAND_DESCRIPTION`` describe the optional
    Wuji hand description."""
    if name == "ROBOT_TO_SIM":
        from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM
        globals()[name] = ROBOT_TO_SIM
        return ROBOT_TO_SIM
    if name in ("HAND_URDF", "HAND_MESH_DIR", "HAND_DESCRIPTION"):
        paths = hand_paths()
        values = {
            "HAND_DESCRIPTION": paths["description"],
            "HAND_URDF": paths["urdf"],
            "HAND_MESH_DIR": paths["mesh_dir"],
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
