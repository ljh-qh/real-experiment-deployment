from dataclasses import dataclass, field
import importlib
import sys
import time
from threading import RLock
from typing import Any

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import (
    SharpaHandState,
    SharpaTactileFrame,
)
from unirobot_infra.utils.sdk_paths import SharpaManusSdkPaths, preload_current_libpython, prepend_library_paths


_TACTILE_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_SHARPA_SDK_LOCK = RLock()


@dataclass
class SharpaHand:
    cfg: DictConfig
    side: str
    sdk_paths: SharpaManusSdkPaths
    name: str = "sharpa"
    dof: int = 22
    connected: bool = False
    _symbols: dict[str, Any] = field(default_factory=dict, init=False)
    _manager: Any = field(default=None, init=False)
    _hand: Any = field(default=None, init=False)
    _sn: str = field(default="", init=False)
    _enabled: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)

    def connect(self) -> None:
        with _SHARPA_SDK_LOCK:
            self._symbols = self._import_symbols()
            manager_cls = self._symbols["SharpaWaveManager"]
            device_type = self._symbols["DeviceType"]
            hand_side = self._symbols["HandSide"]
            expected_side = self._expected_sdk_side(hand_side)
            self._manager = manager_cls.get_instance()

            deadline = time.monotonic() + float(self.cfg.discovery_timeout_s)
            matching_info = None
            while time.monotonic() < deadline:
                device_infos = self._manager.get_all_devices()
                for info in device_infos:
                    if self._matches_hand_info(info, device_type.HAND, expected_side):
                        matching_info = info
                        break
                if matching_info is not None:
                    break
                time.sleep(0.2)

            if matching_info is None:
                self._manager.disconnect_all()
                raise RuntimeError(self._discovery_error_message())

            self._sn = str(matching_info.sn)
            self._hand = self._manager.connect(matching_info.sn, bool(self.cfg.skip_tactile))
            self._initialize_hand()
            started = self._hand.start()
            if not started:
                raise RuntimeError(f"Sharpa hand {self._sn} failed to start")
            self._started = True
            self.connected = True

    def read_state(self) -> SharpaHandState:
        with _SHARPA_SDK_LOCK:
            if not self.connected or self._hand is None:
                raise RuntimeError("SharpaHand is not connected")
            state = self._hand.get_states()
            return SharpaHandState(
                name=self.name,
                timestamp_ns=now_ns(),
                sdk_timestamp=int(state.timestamp),
                sequence=int(state.sequence),
                joint_pos=_float_tuple(state.angles, self.dof, "angles"),
                joint_vel=_float_tuple(state.velocities, self.dof, "velocities"),
                joint_torque=_float_tuple(state.torques, self.dof, "torques"),
            )

    def read_tactile_frame(self, channel: int, timeout_s: float) -> SharpaTactileFrame | None:
        with _SHARPA_SDK_LOCK:
            if not self.connected or self._hand is None:
                raise RuntimeError("SharpaHand is not connected")
            frame = self._hand.fetch_tactile_frame(int(channel), float(timeout_s))
        if frame is None:
            return None
        frame_channel = int(frame["channel"])
        side, finger = _tactile_channel_side_finger(frame_channel)
        if side != self.side:
            raise RuntimeError(f"Sharpa tactile channel {frame_channel} does not belong to {self.side} hand")
        content = frame["content"]
        shape = frame["shape"]
        raw_data = content["RAW"]
        deform_data = content["DEFORM"]
        f6_data = content["F6"]
        contact_point_data = content["CONTACT_POINT"]
        return SharpaTactileFrame(
            side=side,
            finger=finger,
            channel=frame_channel,
            timestamp_ns=now_ns(),
            sdk_timestamp_s=float(frame["ts"]),
            frame_id=int(frame["frame_id"]),
            raw_image=_tactile_uint8_image(raw_data, shape["RAW"], "RAW"),
            deform_image=_tactile_uint8_image(deform_data, shape["DEFORM"], "DEFORM"),
            f6=_float_values_tuple(f6_data),
            f6_shape=_float_values_shape(f6_data),
            contact_point=_float_values_tuple(contact_point_data),
            contact_point_shape=_float_values_shape(contact_point_data),
        )

    def send_joint_position_rad(self, q_rad: tuple[float, ...]) -> None:
        with _SHARPA_SDK_LOCK:
            if not self.connected or self._hand is None:
                raise RuntimeError("SharpaHand is not connected")
            if len(q_rad) != self.dof:
                raise ValueError("end effector q_rad length does not match dof")
            error = self._hand.set_joint_position([float(v) for v in q_rad], bool(self.cfg.interpolate))
            self._raise_if_error(error, "Sharpa set_joint_position failed")

    def close(self) -> None:
        with _SHARPA_SDK_LOCK:
            failures = []
            identity = f"{self.side} hand {self._sn}"
            # Try every release step; a disable failure must remain visible.
            if self._hand is not None:
                if self._enabled:
                    try:
                        error = self._hand.set_enable_state(False)
                        self._raise_if_error(error, "Sharpa set_enable_state(false) failed")
                    except Exception as exc:
                        failures.append(f"disable: {exc}")
                if self._started:
                    try:
                        if self._hand.stop() is False:
                            raise RuntimeError("Sharpa stop failed")
                    except Exception as exc:
                        failures.append(f"stop: {exc}")
            if self._manager is not None and self._sn != "":
                try:
                    if self._manager.disconnect(self._sn) is False:
                        raise RuntimeError("Sharpa disconnect failed")
                except Exception as exc:
                    failures.append(f"disconnect: {exc}")
            self._hand = None
            self._manager = None
            self._sn = ""
            self._enabled = False
            self._started = False
            self.connected = False
            if failures:
                raise RuntimeError(f"Sharpa {identity} cleanup failed: " + "; ".join(failures))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def sn(self) -> str:
        return self._sn

    def _initialize_hand(self) -> None:
        if not bool(self.cfg.initialize_control_on_connect):
            if self.cfg.enable_on_connect:
                raise ValueError("Sharpa enable_on_connect requires initialize_control_on_connect")
            return
        control_mode = self._symbols["ControlMode"]
        control_source = self._symbols["ControlSource"]
        for error in (
            self._hand.set_control_mode(control_mode.POSITION),
            self._hand.set_speed_coeff(float(self.cfg.speed_coeff)),
            self._hand.set_current_coeff(float(self.cfg.current_coeff)),
            self._hand.set_control_source(control_source.SDK),
        ):
            self._raise_if_error(error, "Sharpa init failed")
        if self.cfg.enable_on_connect:
            error = self._hand.set_enable_state(True)
            self._raise_if_error(error, "Sharpa set_enable_state(true) failed")
            self._enabled = True

    def _import_symbols(self) -> dict[str, Any]:
        sdk_root = self.sdk_paths.sharpa_sdk_root
        if self.sdk_paths.python_package_layout:
            for path in (self.sdk_paths.sharpa_python_dir, self.sdk_paths.sharpa_python_module):
                if not path.exists():
                    raise FileNotFoundError(str(path))
            preload_current_libpython()
        else:
            for path in (sdk_root, sdk_root / "lib", self.sdk_paths.sharpa_python_dir, self.sdk_paths.sharpa_python_module):
                if not path.exists():
                    raise FileNotFoundError(str(path))
            prepend_library_paths((sdk_root / "lib",))
            preload_current_libpython()
        python_dir = self.sdk_paths.sharpa_python_dir
        python_dir_str = str(python_dir)
        if python_dir_str not in sys.path:
            sys.path.insert(0, python_dir_str)
        sharpa = importlib.import_module("sharpa")
        return {
            "ControlMode": sharpa.ControlMode,
            "ControlSource": sharpa.ControlSource,
            "DeviceType": sharpa.DeviceType,
            "HandSide": sharpa.HandSide,
            "SharpaWaveManager": sharpa.SharpaWaveManager,
        }

    def _expected_sdk_side(self, hand_side: Any) -> Any:
        if self.side == "left":
            return hand_side.LEFT
        if self.side == "right":
            return hand_side.RIGHT
        raise ValueError("side must be left or right")

    def _matches_hand_info(self, info: Any, hand_device_type: Any, expected_side: Any) -> bool:
        if info.device_type != hand_device_type or info.hand_side != expected_side:
            return False
        expected_sn = self.cfg.hands[self.side].expected_sn
        if expected_sn is not None and str(info.sn) != str(expected_sn):
            return False
        expected_ip = self.cfg.hands[self.side].expected_ip
        if expected_ip is None:
            return True
        return _device_ip_str(info.ip) == str(expected_ip)

    def _discovery_error_message(self) -> str:
        expected_sn = self.cfg.hands[self.side].expected_sn
        expected_ip = self.cfg.hands[self.side].expected_ip
        constraints = []
        if expected_sn is not None:
            constraints.append(f"SN {expected_sn}")
        if expected_ip is not None:
            constraints.append(f"IP {expected_ip}")
        if not constraints:
            return f"No Sharpa {self.side} HAND device discovered"
        return f"No Sharpa {self.side} HAND device discovered matching {', '.join(constraints)}"

    def _raise_if_error(self, error: Any, message: str) -> None:
        if error.code != 0:
            raise RuntimeError(f"{message}: {error.message}")

    def _read_error_value(self, result: tuple[Any, Any], message: str) -> Any:
        error, value = result
        self._raise_if_error(error, message)
        return value


def _device_ip_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple | list):
        if len(value) != 4:
            raise ValueError("Sharpa DeviceInfo ip length is not 4")
        return ".".join(str(int(part)) for part in value)
    raise TypeError(f"unsupported Sharpa DeviceInfo ip type: {type(value).__name__}")


def _tactile_channel_side_finger(channel: int) -> tuple[str, str]:
    if 0 <= channel < 5:
        return "right", _TACTILE_FINGERS[channel]
    if 5 <= channel < 10:
        return "left", _TACTILE_FINGERS[channel - 5]
    raise ValueError("Sharpa tactile channel must be in [0, 9]")


def _tactile_uint8_image(data: Any, shape: Any, field_name: str) -> np.ndarray | None:
    if data is None:
        return None
    array = np.asarray(data, dtype=np.uint8)
    if array.size == 0:
        return None
    height = int(shape[1])
    width = int(shape[2])
    image = array.reshape(height, width)
    if image.ndim != 2:
        raise RuntimeError(f"Sharpa tactile {field_name} image must be 2D, got shape={image.shape}")
    return np.ascontiguousarray(image)


def _float_tuple(values, length: int, field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise RuntimeError(f"Sharpa {field_name} length is not {length}")
    return result


def _float_values_tuple(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(float(value) for value in np.asarray(values, dtype=np.float32).reshape(-1))


def _float_values_shape(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in np.asarray(values, dtype=np.float32).shape)
