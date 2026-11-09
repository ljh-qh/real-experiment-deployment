"""Tianji dual-arm joint-stream backend for the unified teleop.

The teleop core (teleop.py) solves IK once and produces 7 joint angles per
arm; this backend streams them to the real Marvin arms through the vendor
SDK's joint-position interface (`set_joint_cmd_pose`), the same interface
the ROS2 `tianji_output` driver uses.

The vendor modules are loaded by file path on purpose: importing the
`tianji_output` package would execute its `__init__.py`, which pulls in
`ament_index_python`/rclpy — dependencies the simulation venv does not
have. `fx_robot.py` / `structure_data.py` themselves are pure ctypes plus
the bundled `lib/libMarvinSDK.so`.

The SDK directory is resolved with the priority
``sdk_dir argument > MARVIN_SDK_DIR environment variable > ./vendor`` and
must contain ``fx_robot.py``, ``structure_data.py`` and ``lib/`` (see the
error in :func:`load_sdk` for setup instructions). Only the real-robot
backends need it; pure simulation never imports this module.

Shutdown path mirrors `TianjiChestDriver.disable_and_release`: request
state 0, poll for confirmation, then release the TCP session. Marvin
accepts a single TCP session, so this must run before any other tool
(e.g. a Monitor `brake` UI) can connect.
"""

import importlib.util
import logging
import math
import os
from pathlib import Path
import time

import numpy as np

from unirobot_infra.robots.arm.marvin_model import load_home_joints, load_tool_dyn, resolve_robot_ip

# Joint-impedance servo parameters (identical to TianjiChestDriver defaults).
JOINT_IMPEDANCE_K = [2, 2, 2, 1.6, 1, 1, 1]
JOINT_IMPEDANCE_D = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
# Tool dynamics: 0.95 kg hand, COM +90 mm along tool Z, zero inertia.
TOOL_DYN = [0.95, 0, 0, 90, 0, 0, 0, 0, 0, 0]

# Hard safety clip applied to every outgoing joint target (degrees), taken
# from the MuJoCo model joint limits. Same source the IK core clips against.
DEFAULT_JOINT_LIMITS_DEG = {
    "left": [[-178.0, 178.0], [-120.0, 120.0], [-178.0, 178.0],
             [-145.0, 60.0], [-178.0, 178.0], [-60.0, 60.0], [-90.0, 90.0]],
    "right": [[-178.0, 178.0], [-120.0, 120.0], [-178.0, 178.0],
              [-145.0, 60.0], [-178.0, 178.0], [-60.0, 60.0], [-90.0, 90.0]],
}

_SDK_HELP = (
    "Tianji vendor SDK not found. Provide a directory that contains "
    "fx_robot.py, structure_data.py and lib/libMarvinSDK.so (copy them from "
    "the tianji_output package), then either pass --sdk-dir, set the "
    "MARVIN_SDK_DIR environment variable, or drop them in ./vendor. "
    "Pure simulation (--backend sim) does not need the SDK."
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_sdk_dir(sdk_dir=None):
    """Resolve the vendor SDK directory (argument > env > ./vendor)."""
    candidate = Path(sdk_dir or os.environ.get("MARVIN_SDK_DIR") or Path.cwd() / "vendor")
    if not candidate.is_dir():
        raise FileNotFoundError(f"{_SDK_HELP} (looked in {candidate})")
    return candidate


def load_sdk(sdk_dir=None):
    """Load Marvin_Robot and DCSS without importing the ROS2 package.

    Raises FileNotFoundError with installation guidance when the vendor SDK
    is not available at the resolved directory. Two vendor layouts are
    supported: the split layout (`fx_robot.py` + `structure_data.py` + `lib/`,
    as shipped with the ROS2 package) and the self-contained layout
    (`fx_robot.py` carrying `DCSS` itself, as shipped with MarvinPlatform).
    """
    directory = resolve_sdk_dir(sdk_dir)
    fx_robot_path = directory / "fx_robot.py"
    if not fx_robot_path.is_file():
        raise FileNotFoundError(f"{_SDK_HELP} (missing fx_robot.py in {directory})")
    fx_robot = _load_module("marvin_fx_robot", fx_robot_path)
    structure_data_path = directory / "structure_data.py"
    if structure_data_path.is_file():
        structure_data = _load_module("marvin_structure_data", structure_data_path)
        return fx_robot.Marvin_Robot, structure_data.DCSS
    return fx_robot.Marvin_Robot, fx_robot.DCSS


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


class TianjiJointBackend:
    """Stream joint-angle targets to the real dual-arm controller.

    Lifecycle: connect -> clear errors -> tool dynamics -> position mode ->
    ramp to shared home -> verify feedback -> joint impedance. After that
    `write()` is called once per control tick with the IK output (radians,
    both arms concatenated). `close()` powers down and releases the session.
    """

    def __init__(self, robot_ip, home_deg=None, limits_deg=None,
                 ramp_speed_deg_s=60.0, ramp_hz=100.0, logger=None,
                 sdk_dir=None, config_path=None):
        self.logger = logger or logging.getLogger("TianjiJointBackend")
        self._limits = limits_deg or DEFAULT_JOINT_LIMITS_DEG
        robot_ip = resolve_robot_ip(robot_ip)
        if not robot_ip:
            raise ValueError(
                "Tianji robot IP is required: pass --robot-ip, set TIANJI_ROBOT_IP, "
                "or configure robot_ip in tianji_robot.yaml")
        home_deg = load_home_joints(config_path) if home_deg is None else home_deg
        self._tool_dyn = load_tool_dyn(config_path) or {"left": TOOL_DYN, "right": TOOL_DYN}
        for side in ("left", "right"):
            angles = np.asarray(home_deg[side], dtype=float)
            limits = np.asarray(self._limits[side], dtype=float)
            if (angles.shape != (7,) or not np.isfinite(angles).all()
                    or np.any(angles < limits[:, 0]) or np.any(angles > limits[:, 1])):
                raise ValueError(f"{side} home joints are invalid or outside limits")
        Marvin_Robot, DCSS = load_sdk(sdk_dir)
        self._DCSS = DCSS
        self.robot = Marvin_Robot()

        if self.robot.connect(robot_ip) == 0:
            raise ConnectionError(
                f"Connect to {robot_ip} failed: SDK returned 0 (port in use?)")
        try:
            self._initialize(home_deg, ramp_speed_deg_s, ramp_hz, robot_ip)
        except BaseException:
            self.close()
            raise

    def _initialize(self, home_deg, ramp_speed_deg_s, ramp_hz, robot_ip):
        time.sleep(0.5)
        self.robot.clear_set()
        self.robot.clear_error('A')
        self.robot.clear_error('B')
        self.robot.send_cmd()
        time.sleep(0.5)
        if not self._verify_connection():
            raise ConnectionError(f"No telemetry from {robot_ip}")
        self.logger.info("Connected to Tianji controller at %s", robot_ip)

        self.robot.clear_set()
        self.robot.set_tool(arm='A', kineParams=[0] * 6, dynamicParams=self._tool_dyn["left"])
        self.robot.set_tool(arm='B', kineParams=[0] * 6, dynamicParams=self._tool_dyn["right"])
        self.robot.send_cmd()
        time.sleep(0.3)

        # Position mode reaches home before enabling the softer teleop
        # impedance gains. SDK writes alone do not prove physical arrival.
        left, right = self.read_joints_deg()
        measured = np.array([left, right], dtype=float)
        if measured.shape != (2, 7) or not np.isfinite(measured).all():
            raise RuntimeError("Invalid joint feedback before entering position mode")
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm='A', joints=left.tolist())
        self.robot.set_joint_cmd_pose(arm='B', joints=right.tolist())
        self.robot.set_state(arm='A', state=1)
        self.robot.set_state(arm='B', state=1)
        self.robot.set_vel_acc(arm='A', velRatio=60, AccRatio=60)
        self.robot.set_vel_acc(arm='B', velRatio=60, AccRatio=60)
        self.robot.send_cmd()
        if not self._poll_states(1, timeout=5.0):
            raise RuntimeError("Initial position mode failed; teleoperation was not enabled")

        left, right = self.read_joints_deg()
        self.logger.info("Current joints (deg) left=%s right=%s",
                         np.round(left, 2).tolist(), np.round(right, 2).tolist())

        self._ramp_to_home(np.asarray(home_deg["left"], dtype=float),
                           np.asarray(home_deg["right"], dtype=float),
                           speed_deg_s=ramp_speed_deg_s, hz=ramp_hz)
        self._wait_for_home(home_deg)
        self._servo_on_joint_impedance()
        self._wait_for_home(home_deg)
        self.logger.info("Shared home verified within 2 degrees on every joint; ready for teleoperation")

    # -------------------- SDK helpers --------------------

    def _subscribe(self):
        return self.robot.subscribe(self._DCSS())

    def _verify_connection(self, attempts=5):
        seen = set()
        for _ in range(attempts):
            sub = self._subscribe()
            serial = sub["outputs"][0]["frame_serial"]
            if serial != 0:
                seen.add(serial)
            time.sleep(0.1)
        return len(seen) > 1

    def _poll_states(self, target, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sub = self._subscribe()
            a = sub["states"][0]["cur_state"]
            b = sub["states"][1]["cur_state"]
            if a == target and b == target:
                return True
            time.sleep(0.05)
        self.logger.warning("State switch to %d timed out (left=%d right=%d)",
                            target, a, b)
        return False

    def _servo_on_joint_impedance(self):
        self.robot.clear_set()
        self.robot.set_joint_kd_params(arm='A', K=JOINT_IMPEDANCE_K, D=JOINT_IMPEDANCE_D)
        self.robot.set_joint_kd_params(arm='B', K=JOINT_IMPEDANCE_K, D=JOINT_IMPEDANCE_D)
        self.robot.set_impedance_type(arm='A', type=1)
        self.robot.set_impedance_type(arm='B', type=1)
        self.robot.send_cmd()
        time.sleep(0.5)
        # Servo-on after params are loaded (avoids first-frame jitter).
        self.robot.clear_set()
        self.robot.set_state(arm='A', state=3)
        self.robot.set_state(arm='B', state=3)
        self.robot.set_vel_acc(arm='A', velRatio=60, AccRatio=60)
        self.robot.set_vel_acc(arm='B', velRatio=60, AccRatio=60)
        self.robot.send_cmd()
        if not self._poll_states(3, timeout=5.0):
            raise RuntimeError("Joint impedance mode failed; teleoperation was not enabled")
        self.logger.info("Joint impedance mode active (cur_state=3)")

    def _send_joints(self, left_deg, right_deg):
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm='A', joints=[float(v) for v in left_deg])
        self.robot.set_joint_cmd_pose(arm='B', joints=[float(v) for v in right_deg])
        self.robot.send_cmd()

    def read_joints_deg(self):
        """Return (left, right) current joint angles in degrees."""
        sub = self._subscribe()
        left = sub["outputs"][0]["fb_joint_pos"]
        right = sub["outputs"][1]["fb_joint_pos"]
        return np.asarray(left, dtype=float), np.asarray(right, dtype=float)

    # -------------------- teleop backend API --------------------

    def _clip(self, side, joints_deg):
        limits = np.asarray(self._limits[side], dtype=float)
        return np.clip(joints_deg, limits[:, 0], limits[:, 1])

    def write(self, qpos_rad):
        """Send one IK tick: qpos = 14 joint angles (rad), left then right."""
        qpos_rad = np.asarray(qpos_rad, dtype=float)
        if qpos_rad.shape != (14,) or not np.isfinite(qpos_rad).all():
            raise ValueError("Tianji output requires exactly 14 finite arm angles")
        left = self._clip("left", np.degrees(qpos_rad[:7]))
        right = self._clip("right", np.degrees(qpos_rad[7:]))
        self._send_joints(left, right)

    def _ramp_to_home(self, home_left, home_right, speed_deg_s, hz):
        if not math.isfinite(speed_deg_s) or speed_deg_s <= 0 or not math.isfinite(hz) or hz <= 0:
            raise ValueError("Home ramp speed and rate must be positive and finite")
        start_left, start_right = self.read_joints_deg()
        # Per-joint shortest angular path, falling back if it would leave the limits.
        def wrap_target(home, start, limits):
            limits = np.asarray(limits, dtype=float)
            equivalent = start + (home - start + 180.) % 360. - 180.
            valid = (equivalent >= limits[:, 0]) & (equivalent <= limits[:, 1])
            return np.where(valid, equivalent, home)
        target_left = wrap_target(home_left, start_left, self._limits["left"])
        target_right = wrap_target(home_right, start_right, self._limits["right"])
        delta = max(np.abs(target_left - start_left).max(),
                    np.abs(target_right - start_right).max())
        # The maximum derivative of cubic smoothstep is 1.5.
        steps = max(1, math.ceil(1.5 * delta / (speed_deg_s / hz)))
        self.logger.info("Ramp to standing pose: %.1f deg in %d steps",
                         delta, steps)
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            self._send_joints(start_left + s * (target_left - start_left),
                              start_right + s * (target_right - start_right))
            time.sleep(1.0 / hz)

    def _wait_for_home(self, home_deg, timeout=10.0, tolerance_deg=5.0):
        target = np.array([home_deg['left'], home_deg['right']], dtype=float)
        deadline = time.monotonic() + timeout
        worst = float('inf')
        while True:
            measured = np.array(self.read_joints_deg(), dtype=float)
            if measured.shape != (2, 7) or not np.isfinite(measured).all():
                raise RuntimeError("Invalid joint feedback at initialization")
            worst = float(np.max(np.abs(measured - target)))
            if worst <= tolerance_deg:
                return
            if time.monotonic() >= deadline:
                break
            self._send_joints(target[0], target[1])
            time.sleep(.02)
        raise RuntimeError(f"Arms did not reach shared home: max error {worst:.2f} degrees "
                           f"> {tolerance_deg:.2f}; teleoperation was not enabled")

    def close(self):
        """Servo off and release the TCP session (idempotent)."""
        if self.robot is None:
            return
        try:
            self.robot.clear_set()
            self.robot.set_state(arm='A', state=0)
            self.robot.set_state(arm='B', state=0)
            self.robot.send_cmd()
            self._poll_states(0, timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            self.logger.warning("set_state(0) on shutdown failed (ignored): %s", exc)
        try:
            self.robot.release_robot()
            self.logger.info("Robot connection released")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("release_robot failed (ignored): %s", exc)
        self.robot = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
