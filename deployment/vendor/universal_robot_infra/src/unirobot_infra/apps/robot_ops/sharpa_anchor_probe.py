from pathlib import Path
from threading import Lock
from typing import Any
import importlib
import time

import hydra
import yaml
from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import RobotAction
from unirobot_infra.devices.joint_gui import JointGui, build_joint_gui_parts
from unirobot_infra.robots.end_effector.sharpa import _device_ip_str
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")
SHARPA_DOF = 22
_FINGER_SLICES = {
    "thumb": slice(0, 5),
    "index": slice(5, 9),
    "middle": slice(9, 13),
    "ring": slice(13, 17),
    "pinky": slice(17, 22),
}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/sharpa_anchor_probe/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.debug("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_sharpa_anchor_probe(cfg, logger, console)


def run_sharpa_anchor_probe(cfg: DictConfig, logger, console: Console | None = None) -> None:
    side = str(cfg.anchor_probe.side)
    _validate_probe_side(cfg, side)
    ee_cfg = cfg.robot.hardware.end_effector
    output_dir = Path(str(cfg.anchor_probe.output_dir))
    joint_order = _joint_order(ee_cfg)
    symbols = _import_sharpa_symbols(ee_cfg)
    manager = symbols["SharpaWaveManager"].get_instance()
    info = _discover_hand(manager, ee_cfg, side, symbols)
    sn = str(info.sn)
    hand = None
    started = False
    gui = None
    key_listener = None
    records = []
    try:
        hand = manager.connect(info.sn, _skip_tactile(cfg))
        _configure_position_hand(hand, ee_cfg, symbols)
        started = hand.start()
        if not started:
            raise RuntimeError(f"Sharpa hand {sn} failed to start")
        if bool(cfg.anchor_probe.zero_on_start):
            _send_zero_pose(hand, ee_cfg)
            max_abs_q = _wait_for_zero_pose(
                hand,
                timeout_s=float(cfg.anchor_probe.zero_timeout_s),
                tolerance_rad=float(cfg.anchor_probe.zero_tolerance_rad),
            )
            logger.debug("Sharpa 全零初始姿态已到位 side=%s sn=%s max_abs_q=%.4f", side, sn, max_abs_q)
        mode = _read_error_value(hand.get_control_mode(), "Sharpa get_control_mode failed")
        enable_state = _read_error_value(hand.get_enable_state(), "Sharpa get_enable_state failed")
        _validate_position_state(mode, enable_state, symbols, sn)
        target = _LatestGuiTarget([0.0] * SHARPA_DOF)
        gui = _build_anchor_joint_gui(cfg, side, target, console)
        gui.start()
        key_listener = _AnchorProbeKeyListener()
        logger.debug(
            "Sharpa anchor probe 启动 side=%s sn=%s mode=%s enable=%s output=%s",
            side,
            sn,
            _control_mode_name(mode),
            enable_state,
            output_dir,
        )
        sent_target_version = target.version()
        running = True
        while running:
            loop_start = time.monotonic()
            target_q_rad, target_version = target.read_with_version()
            if target_version != sent_target_version:
                _send_joint_pose(hand, ee_cfg, target_q_rad)
                sent_target_version = target_version
            key = key_listener.pop_key()
            while key is not None:
                if key == "quit":
                    running = False
                elif key == "save":
                    target_q_rad, target_version = target.read_with_version()
                    if target_version != sent_target_version:
                        _send_joint_pose(hand, ee_cfg, target_q_rad)
                        sent_target_version = target_version
                    record = _record_from_target(
                        target_q_rad,
                        hand.get_states(),
                        bool(cfg.anchor_probe.save_torque),
                    )
                    records.append(record)
                    _write_anchor_records(output_dir, side, joint_order, records)
                    logger.info(
                        "已保存 sample=%s anchor_count=%s %s dir=%s",
                        len(records) - 1,
                        len(records),
                        _saved_record_summary(record),
                        output_dir,
                    )
                key = key_listener.pop_key()
            _sleep_remaining(loop_start, 1.0 / float(cfg.anchor_probe.poll_hz))
    finally:
        if key_listener is not None:
            key_listener.close()
        if gui is not None:
            gui.close()
        if hand is not None and bool(ee_cfg.enable_on_connect):
            _raise_if_error(hand.set_enable_state(False), "Sharpa set_enable_state(false) failed")
        if hand is not None and started:
            stopped = hand.stop()
            if stopped is False:
                raise RuntimeError("Sharpa stop failed")
        if hand is not None:
            manager.disconnect(sn)
        manager.disconnect_all()


def _validate_probe_side(cfg: DictConfig, side: str) -> None:
    if side not in cfg.robot.sides:
        raise ValueError(f"anchor_probe.side={side} not found in robot.sides")
    if "end_effector" not in cfg.robot.sides[side].parts:
        raise ValueError(f"anchor_probe.side={side} must have end_effector part")


def _import_sharpa_symbols(ee_cfg: DictConfig) -> dict[str, Any]:
    sdk_paths = sharpa_manus_sdk_paths(Path(str(ee_cfg.sdk.vendor_root)))
    configure_sharpa_runtime(sdk_paths)
    sharpa = importlib.import_module("sharpa")
    return {
        "ControlMode": sharpa.ControlMode,
        "ControlSource": sharpa.ControlSource,
        "DeviceType": sharpa.DeviceType,
        "HandSide": sharpa.HandSide,
        "SharpaWaveManager": sharpa.SharpaWaveManager,
    }


def _discover_hand(manager: Any, ee_cfg: DictConfig, side: str, symbols: dict[str, Any]) -> Any:
    expected_side = _expected_sdk_side(symbols["HandSide"], side)
    deadline = time.monotonic() + float(ee_cfg.discovery_timeout_s)
    while time.monotonic() < deadline:
        device_infos = manager.get_all_devices()
        for info in device_infos:
            if _matches_hand_info(info, ee_cfg, side, symbols["DeviceType"].HAND, expected_side):
                return info
        time.sleep(0.2)
    manager.disconnect_all()
    raise RuntimeError(_discovery_error_message(ee_cfg, side))


def _expected_sdk_side(hand_side: Any, side: str) -> Any:
    if side == "left":
        return hand_side.LEFT
    if side == "right":
        return hand_side.RIGHT
    raise ValueError("side must be left or right")


def _matches_hand_info(info: Any, ee_cfg: DictConfig, side: str, hand_device_type: Any, expected_side: Any) -> bool:
    if info.device_type != hand_device_type or info.hand_side != expected_side:
        return False
    expected_sn = ee_cfg.hands[side].expected_sn
    if expected_sn is not None and str(info.sn) != str(expected_sn):
        return False
    expected_ip = ee_cfg.hands[side].expected_ip
    if expected_ip is None:
        return True
    return _device_ip_str(info.ip) == str(expected_ip)


def _discovery_error_message(ee_cfg: DictConfig, side: str) -> str:
    expected_sn = ee_cfg.hands[side].expected_sn
    expected_ip = ee_cfg.hands[side].expected_ip
    constraints = []
    if expected_sn is not None:
        constraints.append(f"SN {expected_sn}")
    if expected_ip is not None:
        constraints.append(f"IP {expected_ip}")
    if not constraints:
        return f"No Sharpa {side} HAND device discovered"
    return f"No Sharpa {side} HAND device discovered matching {', '.join(constraints)}"


def _skip_tactile(cfg: DictConfig) -> bool:
    return bool(cfg.robot.hardware.end_effector.skip_tactile) or not bool(cfg.anchor_probe.enable_tactile)


def _configure_position_hand(hand: Any, ee_cfg: DictConfig, symbols: dict[str, Any]) -> None:
    for error in (
        hand.set_control_source(symbols["ControlSource"].SDK),
        hand.set_speed_coeff(float(ee_cfg.speed_coeff)),
        hand.set_current_coeff(float(ee_cfg.current_coeff)),
        hand.set_control_mode(symbols["ControlMode"].POSITION),
    ):
        _raise_if_error(error, "Sharpa position init failed")
    if bool(ee_cfg.enable_on_connect):
        _raise_if_error(hand.set_enable_state(True), "Sharpa set_enable_state(true) failed")


def _validate_position_state(mode: Any, enable_state: bool, symbols: dict[str, Any], sn: str) -> None:
    if mode != symbols["ControlMode"].POSITION:
        raise RuntimeError(f"Sharpa hand {sn} control mode is {mode}, expected POSITION")
    if not bool(enable_state):
        raise RuntimeError(f"Sharpa hand {sn} enable state is false, expected true for position")


def _send_zero_pose(hand: Any, ee_cfg: DictConfig) -> None:
    _send_joint_pose(hand, ee_cfg, [0.0] * SHARPA_DOF)


def _send_joint_pose(hand: Any, ee_cfg: DictConfig, q_rad: list[float]) -> None:
    if len(q_rad) != SHARPA_DOF:
        raise ValueError("Sharpa q_rad length must be 22")
    error = hand.set_joint_position([float(value) for value in q_rad], bool(ee_cfg.interpolate))
    _raise_if_error(error, "Sharpa set joint position failed")


def _wait_for_zero_pose(hand: Any, timeout_s: float, tolerance_rad: float) -> float:
    deadline = time.monotonic() + float(timeout_s)
    max_abs_q = float("inf")
    while time.monotonic() < deadline:
        q_rad = _float_list(hand.get_states().angles, SHARPA_DOF, "angles")
        max_abs_q = max(abs(value) for value in q_rad)
        if max_abs_q <= float(tolerance_rad):
            return max_abs_q
        time.sleep(0.05)
    raise RuntimeError(f"Sharpa zero pose did not settle: max_abs_q={max_abs_q:.4f}")


def _raise_if_error(error: Any, message: str) -> None:
    if error.code != 0:
        raise RuntimeError(f"{message}: {error.message}")


def _read_error_value(result: tuple[Any, Any], message: str) -> Any:
    error, value = result
    _raise_if_error(error, message)
    return value


def _joint_order(ee_cfg: DictConfig) -> list[str]:
    return [_neutral_joint_name(str(joint.name)) for joint in ee_cfg.joints]


def _neutral_joint_name(name: str) -> str:
    if name.startswith("left_"):
        return name[len("left_") :]
    if name.startswith("right_"):
        return name[len("right_") :]
    return name


def _build_anchor_joint_gui(cfg: DictConfig, side: str, target: "_LatestGuiTarget", console: Console | None) -> JointGui:
    gui = JointGui(
        cfg.anchor_probe.gui,
        parts=build_joint_gui_parts(cfg.robot, cfg.anchor_probe.gui),
        console=console,
    )
    gui.connect()
    gui.set_initial_actions(
        (
            RobotAction(
                side=side,
                timestamp_ns=now_ns(),
                end_effector_q_rad=(0.0,) * SHARPA_DOF,
            ),
        )
    )
    gui.set_action_callback(lambda action: target.update(_action_end_effector_q_rad(action)))
    return gui


def _action_end_effector_q_rad(action: RobotAction) -> list[float]:
    if action.end_effector_q_rad is None:
        raise ValueError("Joint GUI action missing end_effector_q_rad")
    return _float_list(action.end_effector_q_rad, SHARPA_DOF, "end_effector_q_rad")


def _record_from_target(q_rad: list[float], state: Any, save_torque: bool) -> dict:
    record = {
        "timestamp_ns": now_ns(),
        "sdk_timestamp": int(state.timestamp),
        "sequence": int(state.sequence),
        "source": "joint_gui_target",
        "q_rad": _float_list(q_rad, SHARPA_DOF, "target q_rad"),
        "feedback_q_rad": _float_list(state.angles, SHARPA_DOF, "angles"),
        "feedback_dq_rad_s": _float_list(state.velocities, SHARPA_DOF, "velocities"),
    }
    if save_torque:
        record["feedback_joint_torque"] = _float_list(state.torques, SHARPA_DOF, "torques")
    return record


def _float_list(values: Any, length: int, field_name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length:
        raise RuntimeError(f"Sharpa {field_name} length is not {length}")
    return result


def _write_anchor_records(output_dir: Path, side: str, joint_order: list[str], records: list[dict]) -> None:
    _validate_existing_anchor_dir(output_dir, side, joint_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_anchor_files(output_dir)
    for sample_index, record in enumerate(records):
        anchor_record = dict(record)
        anchor_record["sample_index"] = sample_index
        data = {
            "side": side,
            "joint_order": joint_order,
            "sample_count": len(records),
            "anchor": anchor_record,
        }
        with (output_dir / _anchor_filename(sample_index)).open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def _validate_existing_anchor_dir(output_dir: Path, side: str, joint_order: list[str]) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"anchor output path must be a directory: {output_dir}")
    for path in sorted(output_dir.glob("*.yaml")):
        if not path.is_file():
            raise ValueError(f"anchor path is not a file: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if data is None:
            raise ValueError(f"anchor file is empty: {path}")
        if str(data["side"]) != side:
            raise ValueError(f"anchor file side={data['side']} does not match {side}")
        if [str(name) for name in data["joint_order"]] != joint_order:
            raise ValueError(f"anchor file joint_order does not match current Sharpa config: {path}")
        data["anchor"]


def _clear_anchor_files(output_dir: Path) -> None:
    for path in sorted(output_dir.glob("*.yaml")):
        if not path.is_file():
            raise ValueError(f"anchor path is not a file: {path}")
        path.unlink()


def _anchor_filename(sample_index: int) -> str:
    return f"{int(sample_index):03d}.yaml"


def _saved_record_summary(record: dict) -> str:
    q_rad = record["q_rad"]
    text = (
        f"seq={record['sequence']} "
        f"max|target|={max(abs(value) for value in q_rad):.3f} "
        f"{_finger_summary(q_rad)}"
    )
    if "feedback_joint_torque" in record:
        text += f" max|tau|={max(abs(value) for value in record['feedback_joint_torque']):.3f}"
    return text


def _finger_summary(q_rad: list[float]) -> str:
    parts = []
    for name, finger_slice in _FINGER_SLICES.items():
        values = q_rad[finger_slice]
        parts.append(f"{name}[{min(values):+.2f},{max(values):+.2f}]")
    return " ".join(parts)


def _control_mode_name(mode: Any) -> str:
    text = str(mode)
    if "." in text:
        return text.rsplit(".", maxsplit=1)[1]
    return text


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


class _AnchorProbeKeyListener:
    def __init__(self) -> None:
        from pynput import keyboard

        self._keyboard = keyboard
        self._keys: list[str] = []
        self._lock = Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def pop_key(self) -> str | None:
        with self._lock:
            if not self._keys:
                return None
            return self._keys.pop(0)

    def close(self) -> None:
        self._listener.stop()

    def _on_press(self, key) -> None:
        with self._lock:
            if key == self._keyboard.Key.enter or key == self._keyboard.Key.esc:
                self._keys.append("quit")
            elif hasattr(key, "char") and key.char == "q":
                self._keys.append("quit")
            elif hasattr(key, "char") and key.char == "s":
                self._keys.append("save")


class _LatestGuiTarget:
    def __init__(self, q_rad: list[float]) -> None:
        self._q_rad = _float_list(q_rad, SHARPA_DOF, "q_rad")
        self._version = 0
        self._lock = Lock()

    def read(self) -> list[float]:
        q_rad, _ = self.read_with_version()
        return q_rad

    def read_with_version(self) -> tuple[list[float], int]:
        with self._lock:
            return list(self._q_rad), self._version

    def version(self) -> int:
        with self._lock:
            return self._version

    def update(self, q_rad: list[float]) -> None:
        values = _float_list(q_rad, SHARPA_DOF, "q_rad")
        with self._lock:
            self._q_rad = values
            self._version += 1


if __name__ == "__main__":
    main()
