"""Shared arm description, home pose, and path resolution.

Model and configuration locations are resolved with the priority

    explicit argument > environment variable > package default

and the module no longer depends on any fixed checkout layout:

- model dir:   ``--model-dir`` / ``MUJOCO_MARVIN_MODEL_DIR`` /
               ``assets/marvin_dual_arm`` (see ``assets/README.md``)
- robot cfg:   ``MUJOCO_MARVIN_ROBOT_CONFIG`` / ``config/tianji_robot.yaml``

Model descriptions ship under ``assets/`` with the provenance and redistribution
restrictions documented in ``assets/README.md``. A missing model raises an error
pointing there. Hand description lookup lives in teleop.py and degrades
gracefully when absent.
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


PACKAGE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = PACKAGE_DIR / "assets"
CONFIG_DIR = PACKAGE_DIR / "config"
DEFAULT_MODEL_DIR = ASSETS_DIR / "marvin_dual_arm"
DEFAULT_MODEL_URDF = DEFAULT_MODEL_DIR / "urdf" / "marvin_dual_arm.urdf"
DEFAULT_ROBOT_CONFIG = CONFIG_DIR / "tianji_robot.yaml"
SIDES = ("left", "right")

_MODEL_HELP = (
    "Dual-arm model not found. Place the vendor marvin_dual_arm URDF and "
    "meshes in the model directory (urdf/marvin_dual_arm.urdf + meshes/) and "
    "point --model-dir or MUJOCO_MARVIN_MODEL_DIR at it. Layout and licensing "
    "notes: assets/README.md."
)


def resolve_model_dir(model_dir=None):
    """Resolve the dual-arm model directory and verify the URDF exists."""
    candidate = Path(model_dir or os.environ.get("MUJOCO_MARVIN_MODEL_DIR")
                     or DEFAULT_MODEL_DIR)
    if not (candidate / "urdf" / "marvin_dual_arm.urdf").is_file():
        raise FileNotFoundError(f"{_MODEL_HELP} (looked in {candidate})")
    return candidate


def resolve_model_urdf(model_dir=None):
    return resolve_model_dir(model_dir) / "urdf" / "marvin_dual_arm.urdf"


def resolve_robot_config(config_path=None):
    candidate = Path(config_path or os.environ.get("MUJOCO_MARVIN_ROBOT_CONFIG")
                     or DEFAULT_ROBOT_CONFIG)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Tianji robot config not found: {candidate}. Pass --robot-config "
            f"or set MUJOCO_MARVIN_ROBOT_CONFIG.")
    return candidate


def resolve_robot_ip(cli_ip=None, config_ip=None):
    """Robot IP from CLI argument, TIANJI_ROBOT_IP env, or config value."""
    ip = (cli_ip or os.environ.get("TIANJI_ROBOT_IP") or config_ip or "")
    ip = str(ip).strip().strip("'\"")
    if not ip or ip.startswith("YOUR_"):
        return None
    return ip


def load_home_joints(config_path=None, profile="default"):
    raw = yaml.safe_load(resolve_robot_config(config_path).read_text())
    values = raw["init_joints"] if profile == "default" else raw["home_profiles"][profile]
    result = {}
    for side in SIDES:
        joints = np.asarray(values[side], dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"{side} init_joints must contain seven finite angles in degrees")
        result[side] = joints.tolist()
    return result


def load_tool_dyn(config_path=None):
    """Per-arm tool dynamics [mass kg, com_x mm, com_y mm, com_z mm, I..].

    Returns None when the config has no tool_dyn section — the backend then
    falls back to its built-in constant. Values are per physical mount, not
    per hand model: change them whenever the mounted end-effector changes.
    """
    raw = yaml.safe_load(resolve_robot_config(config_path).read_text())
    section = raw.get("tool_dyn")
    if section is None:
        return None
    result = {}
    for side in SIDES:
        values = np.asarray(section[side], dtype=float)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(f"{side} tool_dyn must contain ten finite values")
        result[side] = values.tolist()
    return result


def home_profile_names(config_path=None):
    raw = yaml.safe_load(resolve_robot_config(config_path).read_text())
    return ["default", *raw.get("home_profiles", {})]


def load_hand_mounts(config_path=None):
    """Fixed TCP-to-hand rotations; never refit them when home joints change."""
    raw = yaml.safe_load(resolve_robot_config(config_path).read_text())
    result = {}
    for side in SIDES:
        quat = np.asarray(raw["hand_mount_quat_wxyz"][side], dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1.) > 1e-5:
            raise ValueError(f"{side} hand mount must be a unit quaternion [w,x,y,z]")
        result[side] = quat / np.linalg.norm(quat)
    return result


def robot_frame_in_model(model_urdf=None):
    """Get forward/left/up from the URDF's fixed arm-base placement.

    The URDF world is kept intact, including the stand and mount transforms.
    Tracker calibration maps to operator axes first; this fixed model rotation
    then carries those axes into the new assembly. No room yaw is guessed.
    """
    urdf = Path(model_urdf or resolve_model_urdf())
    root = ET.parse(urdf).getroot()
    children = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    poses = {"world": np.eye(4)}

    def visit(parent):
        for joint in children.get(parent, []):
            if joint.get("type") != "fixed":
                continue
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            child = joint.find("child").get("link")
            poses[child] = poses[parent] @ transform
            visit(child)

    visit("world")
    left = poses["Base_L"][:3, 3] - poses["Base_R"][:3, 3]
    up = np.array([0., 0., 1.])
    left -= np.dot(left, up) * up
    if np.linalg.norm(left) < 1e-6:
        raise ValueError("URDF must place left and right bases apart horizontally")
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    return np.column_stack([forward, left, up])


def __getattr__(name):
    """Lazily resolve model-dependent constants so importing this package
    never requires the vendor CAD.  First access reads the environment;
    the result is cached in module globals."""
    if name == "MODEL_DIR":
        value = resolve_model_dir()
    elif name == "MODEL_URDF":
        value = resolve_model_urdf()
    elif name == "ROBOT_CONFIG":
        value = resolve_robot_config()
    elif name == "ROBOT_TO_MODEL":
        value = robot_frame_in_model()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
