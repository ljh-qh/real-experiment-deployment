"""Sharpa SDK joint order, feedback-only discovery, and explicit replay output.

Importing this module never loads the vendor SDK or connects to a device.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np

SIDES = ("left", "right")
# Native SDK order, also used by the vendor gesture_workflow sample.
JOINT_NAMES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
# Same native-joint limits as the external Sharpa joint_teleop controller.
LOWER_RAD = np.array([
    -.1745, -.3491, -.5236, -.3491, 0., -.17453293, -.3491, 0., 0.,
    -.17453293, -.3491, 0., 0., -.17453293, -.3491, 0., 0.,
    0., -.17453293, -.3491, 0., 0.,
])
UPPER_RAD = np.array([
    1.9199, .3491, 1.3963, .3491, 1.7453, 1.5708, .3491, 1.7453, 1.3963,
    1.5708, .3491, 1.7453, 1.3963, 1.5708, .3491, 1.7453, 1.3963,
    .2618, 1.5708, .3491, 1.7453, 1.3963,
])


def load_sharpa(sdk_path=None):
    path = sdk_path or os.environ.get("SHARPA_SDK_PY", "/opt/sharpa-wave-sdk/python")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("sharpa")


def check_error(error, operation):
    if int(error.code) != 0:
        raise RuntimeError(f"Sharpa {operation}: {getattr(error, 'message', error)}")


def read_hand_rad(hand):
    error, degrees = hand.get_joint_position_degree()
    check_error(error, "joint feedback")
    values = np.asarray(degrees, dtype=float)
    if values.shape != (22,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Sharpa feedback must contain 22 finite angles")
    return np.deg2rad(values)


def discover(sdk, timeout=5.):
    """Discover only; never enable, change modes or issue a joint command."""
    manager = sdk.SharpaWaveManager.get_instance()
    deadline = time.monotonic() + timeout
    while True:
        devices = list(manager.get_all_devices())
        hands = [d for d in devices if d.device_type == sdk.DeviceType.HAND]
        if len(hands) >= 2 or time.monotonic() >= deadline:
            return manager, hands
        time.sleep(.1)


class SharpaReplayBackend:
    def __init__(self, serials, sdk_path=None, timeout=5.):
        if set(serials) != set(SIDES) or not all(serials.values()) or len(set(serials.values())) != 2:
            raise ValueError("Configure two distinct SHARPA_LEFT_SN / SHARPA_RIGHT_SN values")
        self.sdk = load_sharpa(sdk_path)
        self.manager = None
        self.hands = {}
        self._owned = set()
        try:
            self.manager, devices = discover(self.sdk, timeout)
            by_sn = {d.sn: d for d in devices}
            for side in SIDES:
                sn = serials[side]
                if sn not in by_sn:
                    raise RuntimeError(f"Configured {side} Sharpa hand was not discovered")
                expected = getattr(self.sdk.HandSide, side.upper())
                if by_sn[sn].hand_side != expected:
                    raise RuntimeError(f"Configured {side} SN belongs to the other hand")
                self.hands[side] = self.manager.connect(sn, skip_tactile=True)
            self.serials = dict(serials)
            self.initial_state = {}
            for side, hand in self.hands.items():
                error, source = hand.get_control_source()
                check_error(error, f"{side} control-source preflight")
                error, mode = hand.get_control_mode()
                check_error(error, f"{side} control-mode preflight")
                self.initial_state[side] = (source, mode)
            self.check_disabled()
            self.read()
        except BaseException:
            self.close()
            raise

    # 现场确认无害的故障码：UNKNOWN_ERROR 仅落在触觉指尖模组节点 24-28（固件版本误报），
    # ERROR_POWER_MCU_VERSION_LOW 为左手电源 MCU 版本提示——两者遥操全程存在且不阻塞控制
    _BENIGN_TACTILE_NODES = frozenset((24, 25, 26, 27, 28))

    def _faults_benign(self, faults):
        fc = self.sdk.FaultCode
        for code, nodes in faults.items():
            if code == fc.NORMAL:
                continue
            if code == fc.ERROR_POWER_MCU_VERSION_LOW:
                continue
            if code == fc.UNKNOWN_ERROR and set(nodes) <= self._BENIGN_TACTILE_NODES:
                continue
            return False
        return True

    def read(self):
        values = []
        for side in SIDES:
            hand = self.hands[side]
            faults = hand.get_fault_code()
            if not self._faults_benign(faults):
                raise RuntimeError(f"{side} Sharpa reports a hardware fault: {faults}")
            try:
                angles = read_hand_rad(hand)
                tolerance = np.deg2rad(1.)
                if np.any(angles < LOWER_RAD - tolerance) or np.any(angles > UPPER_RAD + tolerance):
                    raise RuntimeError("feedback outside native joint limits")
                values.append(angles)
            except RuntimeError as exc:
                raise RuntimeError(f"{side} Sharpa feedback: {exc}") from exc
        return np.concatenate(values)

    def check_disabled(self):
        """Check both hands before taking ownership of either hand."""
        for side, hand in self.hands.items():
            error, enabled = hand.get_enable_state()
            check_error(error, f"{side} enable-state preflight")
            if enabled:
                raise RuntimeError(f"{side} Sharpa is enabled; disable both hands before replay")

    def enable(self):
        # Ownership begins only here. Preflight failures leave the hand's mode
        # and enable state untouched; motion failures disable every owned hand.
        try:
            self.check_disabled()
            measured = self.read()  # Validate both sides before mutating either.
            for i, side in enumerate(SIDES):
                hand = self.hands[side]
                current = np.clip(measured[i * 22:(i + 1) * 22], LOWER_RAD, UPPER_RAD)
                self._owned.add(side)
                check_error(hand.set_control_source(self.sdk.ControlSource.SDK), "SDK source")
                check_error(hand.set_joint_position(current.tolist(), True), "preload measured pose")
                check_error(hand.set_control_mode(self.sdk.ControlMode.POSITION), "position mode")
                check_error(hand.set_enable_state(True), "enable")
                error, enabled = hand.get_enable_state()
                check_error(error, "enable feedback")
                if not enabled:
                    raise RuntimeError(f"{side} Sharpa did not enable")
        except BaseException:
            self.close()
            raise

    def write(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (44,) or not np.all(np.isfinite(joints)):
            raise ValueError("Sharpa replay output requires 44 finite radians")
        for i, side in enumerate(SIDES):
            values = joints[i * 22:(i + 1) * 22]
            if np.any(values < LOWER_RAD) or np.any(values > UPPER_RAD):
                raise ValueError(f"{side} Sharpa target outside native joint limits")
            check_error(self.hands[side].set_joint_position(values.tolist(), False), f"{side} joint write")

    def close(self):
        failures = []
        for side in SIDES:
            if side not in self._owned:
                continue
            hand = self.hands[side]
            # Attempt FLOATING only after the disable is confirmed: issuing the
            # mode change while the hand is still enabled cancels the disable.
            # Never restore a POSITION mode until feedback confirms disabled.
            try:
                check_error(hand.set_enable_state(False), "disable")
            except Exception as exc:
                failures.append(f"{side}: {exc}")
            try:
                # 失能状态经 UDP 回传有延迟，立刻读会拿到旧的 enabled=True——轮询确认
                enabled = None
                for _ in range(20):
                    error, enabled = hand.get_enable_state()
                    check_error(error, "disabled-state confirmation")
                    if not enabled:
                        break
                    time.sleep(0.1)
                if enabled:
                    raise RuntimeError("hand remains enabled after shutdown request")
                check_error(hand.set_control_mode(self.sdk.ControlMode.FLOATING), "floating mode")
                source, mode = self.initial_state[side]
                check_error(hand.set_control_source(source), "restore control source")
                check_error(hand.set_control_mode(mode), "restore control mode")
                # 恢复 GLOVE 源后 Pilot/手套会自动重新使能——收尾必须再失能并确认
                check_error(hand.set_enable_state(False), "final disable")
                enabled = None
                for _ in range(20):
                    error, enabled = hand.get_enable_state()
                    check_error(error, "final disabled-state confirmation")
                    if not enabled:
                        break
                    time.sleep(0.1)
                if enabled:
                    raise RuntimeError("hand re-enabled itself after shutdown")
            except Exception as exc:
                failures.append(f"{side}: {exc}")
        if self.manager is not None:
            try:
                self.manager.disconnect_all()
            except Exception as exc:
                failures.append(str(exc))
            self.manager = None
        self._owned.clear()
        if failures:
            raise RuntimeError("Sharpa shutdown failed; check hardware: " + "; ".join(failures))
