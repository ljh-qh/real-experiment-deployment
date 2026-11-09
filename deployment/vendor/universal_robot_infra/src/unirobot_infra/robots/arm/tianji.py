from dataclasses import dataclass, field
from pathlib import Path
import importlib
import sys
import time
from typing import Any

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import TianjiArmState
from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.utils.logging import get_logger


ARM_STATE_POSITION = 1
ARM_STATE_ERROR = 100


@dataclass(frozen=True)
class TianjiArmStatus:
    cur_state: int
    cmd_state: int
    err_code: int


def sdk_arm_to_index(arm: str) -> int:
    arm_name = arm.strip().upper()
    if arm_name == "A":
        return 0
    if arm_name == "B":
        return 1
    raise ValueError("Tianji arm must be A or B")


@dataclass
class TianjiRobot:
    cfg: DictConfig
    connected: bool = False
    _robot: Any = field(default=None, init=False)
    _dcss: Any = field(default=None, init=False)

    def connect(self) -> None:
        fx_robot = self._import_fx_robot()
        self._robot = fx_robot.Marvin_Robot()
        self._dcss = fx_robot.DCSS()
        connected = self._robot.connect(self.cfg.robot_ip)
        if not connected:
            raise RuntimeError(f"Failed to connect Tianji robot at {self.cfg.robot_ip}")
        time.sleep(float(self.cfg.connect_settle_s))
        self._wait_until_feedback_ready()
        self.connected = True

    def clear_errors(self, arm_ids: tuple[str, ...]) -> None:
        if not self.connected or self._robot is None:
            raise RuntimeError("TianjiRobot is not connected")
        normalized_arm_ids = _normalize_arm_ids(arm_ids)
        self._robot.clear_set()
        for arm_id in normalized_arm_ids:
            clear_ok = self._robot.clear_error(arm_id)
            if not clear_ok:
                raise RuntimeError(f"Tianji clear_error failed for arm {arm_id}")
        send_ok = self._robot.send_cmd()
        if not send_ok:
            raise RuntimeError(f"Tianji send_cmd clear errors failed for arms {','.join(normalized_arm_ids)}")

    def read_joint_position_deg(self, arm_id: str) -> tuple[float, ...]:
        _, output, _ = self._read_arm_sdk_data(arm_id)
        value = output["fb_joint_pos"]
        q_deg = tuple(float(v) for v in value)
        if len(q_deg) != 7:
            raise RuntimeError("Tianji fb_joint_pos length is not 7")
        return q_deg

    def read_joint_command_deg(self, arm_id: str) -> tuple[float, ...]:
        _, _, input_state = self._read_arm_sdk_data(arm_id)
        value = input_state["joint_cmd_pos"]
        q_deg = tuple(float(v) for v in value)
        if len(q_deg) != 7:
            raise RuntimeError("Tianji joint_cmd_pos length is not 7")
        return q_deg

    def read_arm_status(self, arm_id: str) -> TianjiArmStatus:
        state, _, _ = self._read_arm_sdk_data(arm_id)
        return TianjiArmStatus(
            cur_state=int(state["cur_state"]),
            cmd_state=int(state["cmd_state"]),
            err_code=int(state["err_code"]),
        )

    def read_arm_state(self, arm_id: str, name: str) -> TianjiArmState:
        state, output, input_state = self._read_arm_sdk_data(arm_id)
        return TianjiArmState(
            name=name,
            timestamp_ns=now_ns(),
            cur_state=int(state["cur_state"]),
            cmd_state=int(state["cmd_state"]),
            err_code=int(state["err_code"]),
            frame_serial=int(output["frame_serial"]),
            traj_state=_sdk_int(output["traj_state"]),
            tip_di=_sdk_int(output["tip_di"]),
            low_speed_flag=_sdk_int(output["low_speed_flag"]),
            joint_pos=_deg_tuple(output["fb_joint_pos"], 7, "fb_joint_pos"),
            joint_vel=_deg_tuple(output["fb_joint_vel"], 7, "fb_joint_vel"),
            joint_pos_external=_deg_tuple(output["fb_joint_posE"], 7, "fb_joint_posE"),
            joint_pos_commanded=_deg_tuple(output["fb_joint_cmd"], 7, "fb_joint_cmd"),
            joint_current=_float_tuple(output["fb_joint_cToq"], 7, "fb_joint_cToq"),
            joint_torque=_float_tuple(output["fb_joint_sToq"], 7, "fb_joint_sToq"),
            joint_temperature=_float_tuple(output["fb_joint_them"], 7, "fb_joint_them"),
            est_joint_friction=_float_tuple(output["est_joint_firc"], 7, "est_joint_firc"),
            est_joint_friction_derivative=_float_tuple(output["est_joint_firc_dot"], 7, "est_joint_firc_dot"),
            est_joint_force=_float_tuple(output["est_joint_force"], 7, "est_joint_force"),
            est_cart_force=_float_tuple(output["est_cart_fn"], 6, "est_cart_fn"),
            rt_in_switch=int(input_state["rt_in_switch"]),
            imp_type=int(input_state["imp_type"]),
            in_frame_serial=int(input_state["in_frame_serial"]),
            frame_miss_cnt=int(input_state["frame_miss_cnt"]),
            max_frame_miss_cnt=int(input_state["max_frame_miss_cnt"]),
            sys_cyc=int(input_state["sys_cyc"]),
            sys_cyc_miss_cnt=int(input_state["sys_cyc_miss_cnt"]),
            max_sys_cyc_miss_cnt=int(input_state["max_sys_cyc_miss_cnt"]),
            joint_cmd_pos=_deg_tuple(input_state["joint_cmd_pos"], 7, "joint_cmd_pos"),
            joint_vel_ratio=int(input_state["joint_vel_ratio"]),
            joint_acc_ratio=int(input_state["joint_acc_ratio"]),
            joint_k=_float_tuple(input_state["joint_k"], 7, "joint_k"),
            joint_d=_float_tuple(input_state["joint_d"], 7, "joint_d"),
            drag_sp_type=int(input_state["drag_sp_type"]),
            drag_sp_para=_float_tuple(input_state["drag_sp_para"], 6, "drag_sp_para"),
            cart_kd_type=int(input_state["cart_kd_type"]),
            cart_k=_float_tuple(input_state["cart_k"], 6, "cart_k"),
            cart_d=_float_tuple(input_state["cart_d"], 6, "cart_d"),
            cart_kn=float(input_state["cart_kn"]),
            cart_dn=float(input_state["cart_dn"]),
            force_fb_type=int(input_state["force_fb_type"]),
            force_type=int(input_state["force_type"]),
            force_dir=_float_tuple(input_state["force_dir"], 6, "force_dir"),
            force_pidul=_float_tuple(input_state["force_pidul"], 7, "force_pidul"),
            force_adj_lmt=float(input_state["force_adj_lmt"]),
            force_cmd=float(input_state["force_cmd"]),
            set_tags=_int_tuple(input_state["set_tags"], 16, "set_tags"),
            update_tags=_int_tuple(input_state["update_tags"], 16, "update_tags"),
            pvt_id=int(input_state["pvt_id"]),
            pvt_id_update=int(input_state["pvt_id_update"]),
            pvt_run_id=int(input_state["pvt_run_id"]),
            pvt_run_state=int(input_state["pvt_run_state"]),
        )

    def send_joint_position_deg(self, arm_id: str, q_deg: tuple[float, ...], blocking: bool = True) -> None:
        self.send_joint_positions_deg({arm_id: q_deg}, blocking=blocking)

    def send_joint_positions_deg(
        self,
        commands_by_arm: dict[str, tuple[float, ...]],
        blocking: bool = True,
    ) -> None:
        if not self.connected or self._robot is None:
            raise RuntimeError("TianjiRobot is not connected")
        if not commands_by_arm:
            raise ValueError("Tianji joint commands are empty")
        send_ok = self._send_joint_positions_deg_with_retry(commands_by_arm)
        if not send_ok:
            raise RuntimeError(f"Tianji send_cmd failed after {int(self.cfg.send_cmd_retries) + 1} attempts")
        if not blocking:
            return
        time.sleep(float(self.cfg.command_settle_s))
        tolerance_deg = float(self.cfg.command_tolerance_deg)
        for arm_id in commands_by_arm:
            q_deg = commands_by_arm[arm_id]
            status = self.read_arm_status(arm_id)
            if status.err_code != 0:
                raise RuntimeError(f"Tianji arm {arm_id} error code {status.err_code} after send_cmd={send_ok}")
            command = self.read_joint_command_deg(arm_id)
            if not _joint_positions_close(command, q_deg, tolerance_deg):
                max_error_deg = _max_joint_error_deg(command, q_deg)
                get_logger(__name__).warning(
                    "Tianji joint command readback mismatch arm=%s max_error_deg=%.6f "
                    "tolerance_deg=%.6f expected=%s actual=%s",
                    arm_id,
                    max_error_deg,
                    tolerance_deg,
                    _format_joint_deg(q_deg),
                    _format_joint_deg(command),
                )

    def _send_joint_positions_deg_with_retry(self, commands_by_arm: dict[str, tuple[float, ...]]) -> bool:
        send_cmd_retries = int(self.cfg.send_cmd_retries)
        retry_sleep_s = float(self.cfg.send_cmd_retry_sleep_s)
        if send_cmd_retries < 0:
            raise ValueError("Tianji send_cmd_retries must be non-negative")
        if retry_sleep_s < 0.0:
            raise ValueError("Tianji send_cmd_retry_sleep_s must be non-negative")

        for attempt_index in range(send_cmd_retries + 1):
            self._robot.clear_set()
            for arm_id in commands_by_arm:
                q_deg = commands_by_arm[arm_id]
                if len(q_deg) != 7:
                    raise ValueError("Tianji joint command length is not 7")
                set_ok = self._robot.set_joint_cmd_pose(arm=arm_id, joints=list(q_deg))
                if not set_ok:
                    raise RuntimeError(f"Tianji set_joint_cmd_pose failed for arm {arm_id}")
            send_ok = self._robot.send_cmd()
            if send_ok:
                return True
            if attempt_index < send_cmd_retries:
                get_logger(__name__).warning(
                    "Tianji send_cmd failed; retrying attempt=%d/%d arms=%s",
                    attempt_index + 1,
                    send_cmd_retries,
                    ",".join(commands_by_arm),
                )
                time.sleep(retry_sleep_s)
        return False

    def send_joint_positions_rad(
        self,
        commands_by_arm: dict[str, tuple[float, ...]],
        blocking: bool = True,
    ) -> None:
        commands_deg_by_arm = {
            arm_id: tuple(rad_to_deg(value) for value in commands_by_arm[arm_id])
            for arm_id in commands_by_arm
        }
        self.send_joint_positions_deg(commands_deg_by_arm, blocking=blocking)

    def enter_position_mode(self, arm_id: str) -> None:
        self.enter_position_modes((arm_id,))

    def enter_position_modes(self, arm_ids: tuple[str, ...]) -> None:
        if not self.connected or self._robot is None:
            raise RuntimeError("TianjiRobot is not connected")
        normalized_arm_ids = _normalize_arm_ids(arm_ids)

        status_by_arm = {arm_id: self.read_arm_status(arm_id) for arm_id in normalized_arm_ids}
        error_arm_ids = tuple(
            arm_id for arm_id in normalized_arm_ids
            if status_by_arm[arm_id].cur_state == ARM_STATE_ERROR or status_by_arm[arm_id].err_code != 0
        )
        if error_arm_ids:
            diagnostics_by_arm = self._read_arm_diagnostics(normalized_arm_ids)
            raise RuntimeError(
                f"Tianji arms {','.join(error_arm_ids)} have errors before entering position mode: "
                + _format_arm_diagnostics(diagnostics_by_arm)
            )
        pending_arm_ids = tuple(
            arm_id for arm_id in normalized_arm_ids if status_by_arm[arm_id].cur_state != ARM_STATE_POSITION
        )
        if not pending_arm_ids:
            return

        q_deg_by_arm = {arm_id: self.read_joint_position_deg(arm_id) for arm_id in pending_arm_ids}
        self._enter_position_modes_together(pending_arm_ids, q_deg_by_arm)

    def _enter_position_modes_together(
        self,
        arm_ids: tuple[str, ...],
        q_deg_by_arm: dict[str, tuple[float, ...]],
    ) -> None:
        self._send_position_hold_targets_deg(q_deg_by_arm)
        self._send_position_mode_requests(arm_ids)
        status_by_arm = self._wait_until_position_modes(arm_ids)
        if not all(
            status_by_arm[arm_id].cur_state == ARM_STATE_POSITION and status_by_arm[arm_id].err_code == 0
            for arm_id in arm_ids
        ):
            diagnostics_by_arm = self._read_arm_diagnostics(arm_ids)
            raise RuntimeError(
                f"Tianji arms {','.join(arm_ids)} failed to enter position mode: "
                + _format_arm_diagnostics(diagnostics_by_arm)
            )
        self._send_position_hold_targets_deg(q_deg_by_arm)

    def _wait_until_position_modes(self, arm_ids: tuple[str, ...]) -> dict[str, TianjiArmStatus]:
        timeout_s = float(self.cfg.position_mode_wait_timeout_s)
        poll_s = float(self.cfg.position_mode_wait_poll_s)
        if timeout_s <= 0.0:
            raise ValueError("Tianji position_mode_wait_timeout_s must be positive")
        if poll_s <= 0.0:
            raise ValueError("Tianji position_mode_wait_poll_s must be positive")
        deadline = time.monotonic() + timeout_s
        status_by_arm = {arm_id: self.read_arm_status(arm_id) for arm_id in arm_ids}
        while (
            all(status_by_arm[arm_id].err_code == 0 for arm_id in arm_ids)
            and any(status_by_arm[arm_id].cur_state != ARM_STATE_POSITION for arm_id in arm_ids)
            and time.monotonic() < deadline
        ):
            time.sleep(poll_s)
            status_by_arm = {arm_id: self.read_arm_status(arm_id) for arm_id in arm_ids}
        return status_by_arm

    def _read_arm_diagnostics(self, arm_ids: tuple[str, ...]) -> dict[str, TianjiArmState]:
        return {arm_id: self.read_arm_state(arm_id, arm_id) for arm_id in arm_ids}

    def _send_position_hold_targets_deg(self, q_deg_by_arm: dict[str, tuple[float, ...]]) -> None:
        self._robot.clear_set()
        for arm_id in q_deg_by_arm:
            self._set_position_hold_target_deg(arm_id, q_deg_by_arm[arm_id])
        self._send_position_mode_cmd(
            f"Tianji send_cmd position hold target failed for arms {','.join(q_deg_by_arm)}"
        )
        time.sleep(float(self.cfg.position_mode_settle_s))

    def _send_position_mode_requests(self, arm_ids: tuple[str, ...]) -> None:
        self._robot.clear_set()
        for arm_id in arm_ids:
            state_ok = self._robot.set_state(arm=arm_id, state=ARM_STATE_POSITION)
            if not state_ok:
                raise RuntimeError(f"Tianji set_state position failed for arm {arm_id}")
        send_ok = self._robot.send_cmd()
        if not send_ok:
            raise RuntimeError(f"Tianji send_cmd position mode failed for arms {','.join(arm_ids)}")
        time.sleep(float(self.cfg.position_mode_settle_s))

    def _send_position_mode_cmd(self, message: str) -> None:
        send_ok = self._robot.send_cmd()
        if not send_ok:
            raise RuntimeError(message)

    def _set_position_hold_target_deg(self, arm_id: str, q_deg: tuple[float, ...]) -> None:
        self._set_joint_command_deg(arm_id, q_deg)
        self._set_vel_acc(arm_id)

    def _set_joint_command_deg(self, arm_id: str, q_deg: tuple[float, ...]) -> None:
        if len(q_deg) != 7:
            raise ValueError("Tianji joint command length is not 7")
        set_ok = self._robot.set_joint_cmd_pose(arm=arm_id, joints=list(q_deg))
        if not set_ok:
            raise RuntimeError(f"Tianji set_joint_cmd_pose failed for arm {arm_id}")

    def _set_vel_acc(self, arm_id: str) -> None:
        vel_acc_ok = self._robot.set_vel_acc(
            arm=arm_id,
            velRatio=int(self.cfg.vel_ratio),
            AccRatio=int(self.cfg.acc_ratio),
        )
        if not vel_acc_ok:
            raise RuntimeError(f"Tianji set_vel_acc failed for arm {arm_id}")

    def close(self) -> None:
        if self._robot is not None:
            release_ok = self._robot.release_robot()
            if not release_ok:
                raise RuntimeError("Tianji release_robot failed")
        self._robot = None
        self._dcss = None
        self.connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _import_fx_robot(self) -> Any:
        sdk_dir = Path(self.cfg.sdk_dir)
        if not sdk_dir.exists():
            raise FileNotFoundError(str(sdk_dir))
        sdk_dir_str = str(sdk_dir)
        if sdk_dir_str not in sys.path:
            sys.path.insert(0, sdk_dir_str)
        return importlib.import_module("fx_robot")

    def _read_arm_sdk_data(self, arm_id: str) -> tuple[dict, dict, dict]:
        if not self.connected or self._robot is None or self._dcss is None:
            raise RuntimeError("TianjiRobot is not connected")
        data = self._robot.subscribe(self._dcss)
        if not isinstance(data, dict):
            raise RuntimeError("Tianji subscribe did not return dict")
        arm_index = sdk_arm_to_index(arm_id)
        return data["states"][arm_index], data["outputs"][arm_index], data["inputs"][arm_index]

    def _wait_until_feedback_ready(self) -> None:
        if self._robot is None or self._dcss is None:
            raise RuntimeError("TianjiRobot is not connected")
        timeout_s = float(self.cfg.feedback_ready_timeout_s)
        poll_s = float(self.cfg.feedback_ready_poll_s)
        if timeout_s <= 0.0:
            raise ValueError("Tianji feedback_ready_timeout_s must be positive")
        if poll_s <= 0.0:
            raise ValueError("Tianji feedback_ready_poll_s must be positive")

        deadline = time.monotonic() + timeout_s
        frame_serials_by_arm: dict[str, int] = {}
        while time.monotonic() < deadline:
            data = self._robot.subscribe(self._dcss)
            if not isinstance(data, dict):
                raise RuntimeError("Tianji subscribe did not return dict")
            frame_serials_by_arm = {}
            for side in self.cfg.arms:
                arm_id = str(self.cfg.arms[side].arm_id)
                output = data["outputs"][sdk_arm_to_index(arm_id)]
                frame_serials_by_arm[arm_id] = int(output["frame_serial"])
            if all(frame_serials_by_arm[arm_id] > 0 for arm_id in frame_serials_by_arm):
                return
            time.sleep(poll_s)

        raise RuntimeError(f"Tianji feedback frame_serial not ready: {frame_serials_by_arm}")


def _joint_positions_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    tolerance_deg: float,
) -> bool:
    return len(actual) == len(expected) and all(
        abs(actual_value - expected_value) <= tolerance_deg
        for actual_value, expected_value in zip(actual, expected)
    )


def _max_joint_error_deg(actual: tuple[float, ...], expected: tuple[float, ...]) -> float:
    if len(actual) != len(expected):
        raise ValueError("Tianji joint command readback length mismatch")
    return max(abs(actual_value - expected_value) for actual_value, expected_value in zip(actual, expected))


def _format_joint_deg(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _normalize_arm_ids(arm_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not arm_ids:
        raise ValueError("Tianji arm ids are empty")
    normalized_arm_ids = tuple(dict.fromkeys(str(arm_id).strip().upper() for arm_id in arm_ids))
    for arm_id in normalized_arm_ids:
        sdk_arm_to_index(arm_id)
    return normalized_arm_ids


def _format_arm_diagnostics(states_by_arm: dict[str, TianjiArmState]) -> str:
    return ", ".join(
        f"{arm_id}.cur_state={state.cur_state} cmd_state={state.cmd_state} err_code={state.err_code} "
        f"est_force={_format_float_values(state.est_joint_force)} "
        f"s_toq={_format_float_values(state.joint_torque)} "
        f"c_toq={_format_float_values(state.joint_current)}"
        for arm_id, state in states_by_arm.items()
    )


def _format_float_values(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _float_tuple(values, length: int, field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise RuntimeError(f"Tianji {field_name} length is not {length}")
    return result


def _int_tuple(values, length: int, field_name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != length:
        raise RuntimeError(f"Tianji {field_name} length is not {length}")
    return result


def _deg_tuple(values, length: int, field_name: str) -> tuple[float, ...]:
    return tuple(deg_to_rad(value) for value in _float_tuple(values, length, field_name))


def _sdk_int(value) -> int:
    if isinstance(value, bytes):
        return int.from_bytes(value[:1], "little", signed=False)
    return int(value)


@dataclass
class TianjiArm:
    cfg: DictConfig
    side: str
    robot: TianjiRobot
    name: str = "tianji"
    dof: int = 7

    def connect(self) -> None:
        if not self.robot.connected:
            self.robot.connect()
        self.robot.enter_position_mode(self.arm_id)

    @property
    def arm_id(self) -> str:
        return str(self.cfg.arms[self.side].arm_id)

    def read_state(self) -> TianjiArmState:
        return self.robot.read_arm_state(self.arm_id, self.name)

    def read_joint_position_deg(self) -> tuple[float, ...]:
        return self.robot.read_joint_position_deg(self.arm_id)

    def send_joint_position_rad(self, q_rad: tuple[float, ...], blocking: bool = True) -> None:
        if len(q_rad) != self.dof:
            raise ValueError("arm q_rad length does not match dof")
        q_deg = tuple(rad_to_deg(v) for v in q_rad)
        self.robot.send_joint_position_deg(self.arm_id, q_deg, blocking=blocking)

    def close(self) -> None:
        self.robot.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
