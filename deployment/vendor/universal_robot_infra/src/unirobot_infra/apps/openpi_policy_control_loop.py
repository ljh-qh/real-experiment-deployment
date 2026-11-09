from pathlib import Path
import json
import time
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from unirobot_infra.apps.teleop import _hold_action_from_sent_actions, _robot_sides_summary, _sleep_remaining
from unirobot_infra.controllers.delta_ee import DeltaEeCommandController, arm_q_rad
from unirobot_infra.envs.builder import build_robot_group
from unirobot_infra.policy.action_chunk import ActionChunkBuffer
from unirobot_infra.policy.client import PolicyClient, build_policy_client
from unirobot_infra.policy.observation import build_policy_server_observation
from unirobot_infra.recorder import DataRecorder
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiTcpPoseMapper, rotation_6d_from_matrix
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.wrappers.base import EnvWrapper
from unirobot_infra.wrappers.builder import build_env
from unirobot_infra.wrappers.camera_observation import CameraObservationWrapper
from unirobot_infra.wrappers.hand_alpha_action import hand_state_alpha
from unirobot_infra.wrappers.tianji_delta_ee_action import TianjiDeltaEeActionWrapper


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/openpi_policy_control_loop/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        logger.info(
            "openpi_policy_control_loop 启动 robot=%s sides=%s rate_hz=%s server=%s",
            cfg.robot.name,
            _robot_sides_summary(cfg),
            cfg.env.runtime.rate_hz,
            cfg.policy.server.type,
        )
        _run_openpi_policy_control_loop(cfg, logger, console)


def _run_openpi_policy_control_loop(cfg: DictConfig, logger, console: Console | None = None) -> None:
    _validate_execution_cfg(cfg)
    robot_group = build_robot_group(cfg)
    env = None
    client: PolicyClient | None = None
    recorder = None
    try:
        env = build_env(cfg, robot_group=robot_group, console=console)
        client = build_policy_client(cfg.policy.server)
        recorder = _build_recorder_from_config(cfg, robot_group, env)
        tactile_reader = _TactileObservationReader(cfg.policy.observation.modalities.tactile, robot_group)
        _run_openpi_policy_control_loop_loop(cfg, env, client, tactile_reader, recorder, logger)
    finally:
        if client is not None:
            client.close()
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()


def _run_openpi_policy_control_loop_loop(
    cfg: DictConfig,
    env,
    client: PolicyClient,
    tactile_reader: "_TactileObservationReader",
    recorder: DataRecorder | None,
    logger,
) -> None:
    period_s = 1.0 / float(cfg.env.runtime.rate_hz)
    frame_limit = int(cfg.env.runtime.frames)
    policy_state_builder = _PolicyStateBuilder.from_config(cfg)
    observation, info = env.reset()
    info = policy_state_builder.attach(observation, info)
    previous_action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
    if bool(cfg.policy.execution.initial_action.enabled):
        info, previous_action = _move_to_initial_action(cfg, env, info, logger)
        info = policy_state_builder.attach(observation, info)
    delta_ee_controller = _delta_ee_controller_for_policy(cfg, env, info)
    action_buffer = ActionChunkBuffer(env.action_layout.size, int(cfg.policy.action.execute_steps))
    delta_ee_chunk_active = False
    consecutive_send_failures = 0
    if recorder is not None:
        recorder.start_episode()
    logger.info(
        "openpi_policy_control_loop 已连接 server=%s host=%s port=%s action_dim=%s dry_run=%s",
        cfg.policy.server.type,
        cfg.policy.server.host,
        cfg.policy.server.port,
        env.action_layout.size,
        cfg.policy.execution.dry_run,
    )
    try:
        for frame_index in _frame_indices(frame_limit):
            loop_start = time.monotonic()
            action = action_buffer.pop_next()
            if action is None:
                action_buffer.replace(
                    _next_action_chunk(
                        cfg=cfg,
                        env=env,
                        client=client,
                        observation=observation,
                        info=info,
                        previous_action=previous_action,
                        tactile_reader=tactile_reader,
                    )
                )
                if (
                    delta_ee_controller is not None
                    and _delta_ee_execution_mode(cfg) == "chunk_blocking"
                    and not bool(cfg.policy.execution.dry_run)
                ):
                    delta_ee_controller.begin_chunk()
                    delta_ee_chunk_active = True
                action = action_buffer.pop_next()
                if action is None:
                    raise RuntimeError("policy response produced an empty action chunk")

            policy_action = tuple(float(value) for value in action)
            current_state = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
            send_action = _limit_action_to_state(cfg.policy.action, policy_action, current_state)
            if bool(cfg.policy.execution.dry_run):
                info = _info_from_current_state(env)
                info = policy_state_builder.attach(observation, info)
                sent_action = send_action
                terminated = False
                truncated = False
            else:
                try:
                    observation, _, terminated, truncated, info = _send_policy_action(
                        cfg=cfg,
                        env=env,
                        delta_ee_controller=delta_ee_controller,
                        action=send_action,
                        frame_index=frame_index,
                    )
                except RuntimeError as exc:
                    if "Tianji send_cmd failed" not in str(exc):
                        raise
                    consecutive_send_failures += 1
                    logger.warning(
                        "Tianji send_cmd failed frame=%s consecutive=%s/%s; clearing action buffer and replanning",
                        frame_index,
                        consecutive_send_failures,
                        int(cfg.policy.execution.send_failure.max_consecutive),
                    )
                    if consecutive_send_failures >= int(cfg.policy.execution.send_failure.max_consecutive):
                        raise
                    info = _info_from_current_state(env)
                    info = policy_state_builder.attach(observation, info)
                    previous_action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
                    action_buffer.clear()
                    if delta_ee_chunk_active and delta_ee_controller is not None:
                        delta_ee_controller.end_chunk()
                        delta_ee_chunk_active = False
                    _sleep_remaining(loop_start, period_s)
                    continue
                consecutive_send_failures = 0
                info = policy_state_builder.attach(observation, info)
                sent_action = (
                    _hold_action_from_sent_actions(env, previous_action, info)
                    if delta_ee_controller is None
                    else send_action
                )
                if (
                    delta_ee_controller is not None
                    and _delta_ee_execution_mode(cfg) == "chunk_blocking"
                    and action_buffer.remaining == 0
                ):
                    chunk_end_states = delta_ee_controller.wait_until_current_target_reached(
                        f"openpi policy chunk ending at frame {frame_index} delta-ee target"
                    )
                    info = dict(info)
                    info["robot_states"] = tuple(chunk_end_states)
                    info = policy_state_builder.attach(observation, info)
                    delta_ee_controller.end_chunk()
                    delta_ee_chunk_active = False
            previous_action = sent_action
            _record_policy_action(recorder, policy_action, sent_action)
            _record_camera_frames(recorder, info)
            if recorder is not None and recorder.has_error:
                raise RuntimeError(recorder.error_message)
            if terminated or truncated:
                logger.info("openpi_policy_control_loop episode finished terminated=%s truncated=%s", terminated, truncated)
                break
            _sleep_remaining(loop_start, period_s)
    finally:
        if delta_ee_chunk_active and delta_ee_controller is not None:
            delta_ee_controller.end_chunk()
        if recorder is not None and recorder.recording:
            recorder.stop_episode()


def _next_action_chunk(
    *,
    cfg: DictConfig,
    env,
    client: PolicyClient,
    observation: tuple[float, ...],
    info: dict,
    previous_action: tuple[float, ...],
    tactile_reader: "_TactileObservationReader",
) -> np.ndarray:
    policy_observation = build_policy_server_observation(
        cfg=cfg.policy.observation.server,
        prompt=str(cfg.policy.task),
        observation=observation,
        info=info,
        action_layout=env.action_layout,
        tactile=tactile_reader.read(),
    )
    response = client.infer(policy_observation)
    return _sanitize_action_chunk(
        cfg=cfg.policy.action,
        action_chunk=_response_action_chunk(response, str(cfg.policy.action.response_key), env.action_layout.size),
        env=env,
        previous_action=previous_action,
    )


def _response_action_chunk(response: dict[str, Any], response_key: str, action_dim: int) -> np.ndarray:
    action_chunk = np.asarray(response[response_key], dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)
    if action_chunk.ndim != 2:
        raise RuntimeError(f"policy actions must be [H,D], got {action_chunk.shape}")
    if action_chunk.shape[1] != action_dim:
        raise RuntimeError(f"policy action dim {action_chunk.shape[1]} does not match env action_dim {action_dim}")
    if action_chunk.shape[0] == 0:
        raise RuntimeError("policy returned an empty action horizon")
    return action_chunk


def _sanitize_action_chunk(
    *,
    cfg: DictConfig,
    action_chunk: np.ndarray,
    env,
    previous_action: tuple[float, ...],
) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32).copy()
    if not np.all(np.isfinite(chunk)):
        raise RuntimeError("policy actions contain NaN or Inf")
    if bool(cfg.clip_to_action_space):
        chunk = np.clip(chunk, np.asarray(env.action_space.low, dtype=np.float32), np.asarray(env.action_space.high, dtype=np.float32))
    max_delta = float(cfg.max_delta)
    if max_delta > 0.0:
        limited = []
        base = np.asarray(previous_action, dtype=np.float32)
        for action in chunk:
            limited_action = np.clip(action, base - max_delta, base + max_delta)
            limited.append(limited_action)
            base = limited_action
        chunk = np.stack(limited, axis=0).astype(np.float32)
    chunk = _force_hand_alpha_if_enabled(cfg, chunk, env)
    return chunk


def _force_hand_alpha_if_enabled(cfg: DictConfig, chunk: np.ndarray, env) -> np.ndarray:
    if not bool(cfg.force_hand_alpha.enabled):
        return chunk
    value = float(cfg.force_hand_alpha.value)
    if value < 0.0 or value > 1.0:
        raise ValueError("policy.action.force_hand_alpha.value must be in [0, 1]")
    output = chunk.copy()
    forced_segments = 0
    for segment in env.action_layout.segments:
        if segment.role != "end_effector":
            continue
        if segment.size != 1:
            raise ValueError("policy.action.force_hand_alpha requires 1D hand-alpha end_effector segments")
        output[:, segment.start] = value
        forced_segments += 1
    if forced_segments == 0:
        raise ValueError("policy.action.force_hand_alpha requires end_effector action segments")
    return output


def _limit_action_to_state(cfg: DictConfig, action: tuple[float, ...], state: tuple[float, ...]) -> tuple[float, ...]:
    max_delta = float(cfg.max_state_delta)
    action_array = np.asarray(action, dtype=np.float32)
    if max_delta <= 0.0:
        return tuple(float(value) for value in action_array)
    state_array = np.asarray(state, dtype=np.float32)
    if action_array.shape != state_array.shape:
        raise RuntimeError(f"state/action shape mismatch: {state_array.shape} vs {action_array.shape}")
    limited = np.clip(action_array, state_array - max_delta, state_array + max_delta).astype(np.float32)
    return tuple(float(value) for value in limited)


def _record_policy_action(recorder: DataRecorder | None, policy_action: tuple[float, ...], sent_action: tuple[float, ...]) -> None:
    if recorder is None:
        return
    recorder.record_action(
        policy_action=tuple(policy_action),
        sent_action=tuple(sent_action),
        teleop_active=False,
    )


def _send_policy_action(
    *,
    cfg: DictConfig,
    env,
    delta_ee_controller: DeltaEeCommandController | None,
    action: tuple[float, ...],
    frame_index: int,
) -> tuple[tuple[float, ...], float, bool, bool, dict]:
    if delta_ee_controller is None:
        return env.step(action)
    mode = _delta_ee_execution_mode(cfg)
    if mode == "step_blocking":
        observation, reward, terminated, truncated, info = delta_ee_controller.send_action(action)
        states = delta_ee_controller.wait_until_current_target_reached(
            f"openpi policy step {frame_index} delta-ee target"
        )
        info = dict(info)
        info["robot_states"] = tuple(states)
        return observation, reward, terminated, truncated, info
    if mode == "chunk_blocking":
        return delta_ee_controller.send_chunk_action(action)
    raise ValueError(f"unsupported policy.execution.delta_ee.mode: {mode}")


def _move_to_initial_action(cfg: DictConfig, env, info: dict, logger) -> tuple[dict, tuple[float, ...]]:
    target = _load_initial_action(Path(str(cfg.policy.execution.initial_action.path)), env.action_layout.size)
    target = np.clip(target, np.asarray(env.action_space.low, dtype=np.float32), np.asarray(env.action_space.high, dtype=np.float32))
    current = np.asarray(env.action_layout.hold_action_from_states(tuple(info["robot_states"])), dtype=np.float32)
    max_delta = float(cfg.policy.execution.initial_action.max_delta)
    move_steps = int(cfg.policy.execution.initial_action.move_steps)
    if move_steps <= 0:
        raise ValueError("policy.execution.initial_action.move_steps must be positive")
    if max_delta > 0.0:
        delta_steps = int(np.ceil(float(np.max(np.abs(target - current))) / max_delta))
        move_steps = max(move_steps, delta_steps, 1)
    logger.info(
        "移动到 OpenPI 初始 action path=%s steps=%s dry_run=%s",
        cfg.policy.execution.initial_action.path,
        move_steps,
        cfg.policy.execution.dry_run,
    )
    period_s = 1.0 / float(cfg.env.runtime.rate_hz)
    latest_info = info
    for step_index in range(1, move_steps + 1):
        loop_start = time.monotonic()
        alpha = step_index / float(move_steps)
        action = tuple(float(value) for value in (current * (1.0 - alpha) + target * alpha))
        if bool(cfg.policy.execution.dry_run):
            latest_info = _info_from_current_state(env)
        else:
            _, _, _, _, latest_info = env.step(action)
        _sleep_remaining(loop_start, period_s)
    if bool(cfg.policy.execution.dry_run):
        latest_info = _info_from_current_state(env)
    logger.info("OpenPI 初始 action 移动完成")
    return latest_info, tuple(float(value) for value in target)


def _load_initial_action(path: Path, action_dim: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"initial action file must contain an object: {path}")
    action = np.asarray(payload["action"], dtype=np.float32).reshape(-1)
    if action.shape != (action_dim,):
        raise ValueError(f"initial action file must contain shape ({action_dim},), got {action.shape}: {path}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"initial action file contains NaN or Inf: {path}")
    return action


def _info_from_current_state(env) -> dict[str, Any]:
    states = _base_env(env).controller.read_state()
    info = {
        "robot_states": states,
        "target_actions": (),
        "sent_actions": (),
        "device_frames": (),
    }
    _attach_camera_frames_from_wrappers(env, info)
    return info


def _delta_ee_controller_for_policy(cfg: DictConfig, env, info: dict) -> DeltaEeCommandController | None:
    mode = _delta_ee_execution_mode(cfg)
    if mode == "none":
        return None
    wrapper = _find_delta_ee_action_wrapper(env)
    return DeltaEeCommandController(
        env=wrapper.env,
        expander=wrapper.action_expander,
        initial_states=tuple(info["robot_states"]),
    )


def _delta_ee_execution_mode(cfg: DictConfig) -> str:
    return str(cfg.policy.execution.delta_ee.mode)


def _find_delta_ee_action_wrapper(env) -> TianjiDeltaEeActionWrapper:
    current = env
    while isinstance(current, EnvWrapper):
        if isinstance(current, TianjiDeltaEeActionWrapper):
            return current
        current = current.env
    raise RuntimeError("policy.execution.delta_ee.mode requires wrappers.tianji_delta_ee_action.enabled=true")


def _base_env(env):
    current = env
    while isinstance(current, EnvWrapper):
        current = current.env
    return current


def _attach_camera_frames_from_wrappers(env, info: dict) -> None:
    current = env
    while isinstance(current, EnvWrapper):
        if isinstance(current, CameraObservationWrapper):
            current.attach_camera_frames(info)
        current = current.env


def _build_recorder_from_config(cfg: DictConfig, robot_group, env) -> DataRecorder | None:
    if not bool(cfg.policy.recording.enabled):
        return None
    if bool(cfg.policy.recording.include_cameras):
        camera_manager = _camera_observation_manager(cfg, env)
        if camera_manager is None:
            return DataRecorder(cfg.recording, robot_group=robot_group, cameras=cfg.camera.cameras)
        return DataRecorder(
            cfg.recording,
            robot_group=robot_group,
            cameras=cfg.camera.cameras,
            camera_manager=camera_manager,
            close_camera_manager=False,
            sample_camera_threads=False,
        )
    recording_cfg = OmegaConf.create(OmegaConf.to_container(cfg.recording, resolve=True))
    recording_cfg.camera_names = []
    return DataRecorder(recording_cfg, robot_group=robot_group, cameras=None)


def _camera_observation_manager(cfg: DictConfig, env):
    if not bool(cfg.wrappers.camera_observation.enabled):
        return None
    current = env
    while isinstance(current, EnvWrapper):
        if isinstance(current, CameraObservationWrapper):
            if current.camera_manager is None:
                raise RuntimeError("camera_observation.enabled requires camera manager")
            return current.camera_manager
        current = current.env
    raise RuntimeError("camera_observation.enabled=true but CameraObservationWrapper was not found")


def _record_camera_frames(recorder: DataRecorder | None, info: dict) -> None:
    if recorder is None:
        return
    if not recorder.records_camera_frames_manually:
        return
    recorder.record_camera_frames(info["camera_frames"])


def _validate_execution_cfg(cfg: DictConfig) -> None:
    delta_ee_mode = _delta_ee_execution_mode(cfg)
    if delta_ee_mode not in {"none", "step_blocking", "chunk_blocking"}:
        raise ValueError("policy.execution.delta_ee.mode must be none, step_blocking, or chunk_blocking")
    if delta_ee_mode != "none":
        if not bool(cfg.wrappers.tianji_delta_ee_action.enabled):
            raise ValueError("policy.execution.delta_ee.mode requires wrappers.tianji_delta_ee_action.enabled=true")
        if str(cfg.policy.observation.server.state.source) != "policy_state":
            raise ValueError("policy.execution.delta_ee.mode requires policy.observation.server.state.source=policy_state")
        if float(cfg.policy.action.max_state_delta) > 0.0:
            raise ValueError("policy.execution.delta_ee.mode requires policy.action.max_state_delta=0.0")
        if bool(cfg.policy.execution.initial_action.enabled):
            raise ValueError("policy.execution.delta_ee.mode requires policy.execution.initial_action.enabled=false")
        if delta_ee_mode == "chunk_blocking" and bool(cfg.env.runtime.action_blocking):
            raise ValueError("policy.execution.delta_ee.mode=chunk_blocking requires env.runtime.action_blocking=false")
    if not bool(cfg.policy.execution.dry_run) and not bool(cfg.policy.execution.confirm_real_execution):
        raise RuntimeError(
            "Refusing to execute real robot actions without policy.execution.confirm_real_execution=true"
        )
    if bool(cfg.policy.execution.dry_run):
        state_cfg = cfg.policy.observation.server.state
        if bool(state_cfg.enabled) and str(state_cfg.source) == "observation":
            raise ValueError("dry_run requires policy.observation.server.state.source=hold_action")
    if int(cfg.policy.action.execute_steps) <= 0:
        raise ValueError("policy.action.execute_steps must be positive")
    if bool(cfg.policy.action.force_hand_alpha.enabled):
        force_hand_alpha = float(cfg.policy.action.force_hand_alpha.value)
        if force_hand_alpha < 0.0 or force_hand_alpha > 1.0:
            raise ValueError("policy.action.force_hand_alpha.value must be in [0, 1]")
    if int(cfg.policy.execution.send_failure.max_consecutive) <= 0:
        raise ValueError("policy.execution.send_failure.max_consecutive must be positive")


def _frame_indices(frame_limit: int):
    frame_index = 0
    while frame_limit == 0 or frame_index < frame_limit:
        yield frame_index
        frame_index += 1


class _TactileObservationReader:
    def __init__(self, cfg: DictConfig, robot_group) -> None:
        self.cfg = cfg
        self.samplers = _build_tactile_observation_samplers(cfg, robot_group)

    def read(self) -> dict[str, Any]:
        if not bool(self.cfg.enabled):
            return {}
        f6_by_side: dict[str, list[np.ndarray]] = {}
        deform_by_side: dict[str, list[np.ndarray]] = {}
        for sampler in self.samplers:
            frame = sampler["read_tactile_frame"](sampler["channel"], float(self.cfg.timeout_s))
            if frame is None:
                continue
            side = str(frame.side)
            if side not in f6_by_side:
                f6_by_side[side] = []
                deform_by_side[side] = []
            if bool(self.cfg.include_f6):
                f6_by_side[side].append(np.asarray(frame.f6, dtype=np.float32).reshape(-1))
            if bool(self.cfg.include_deform) and frame.deform_image is not None:
                deform_by_side[side].append(np.asarray(frame.deform_image, dtype=np.uint8))
        output: dict[str, Any] = {}
        if bool(self.cfg.include_f6):
            output["f6"] = {
                side: np.stack(values, axis=0)
                for side, values in f6_by_side.items()
                if values
            }
        if bool(self.cfg.include_deform):
            output["deform"] = {
                side: np.stack(values, axis=0)
                for side, values in deform_by_side.items()
                if values
            }
        return output


def _build_tactile_observation_samplers(cfg: DictConfig, robot_group) -> tuple[dict[str, Any], ...]:
    if not bool(cfg.enabled):
        return ()
    unit_by_side = {unit.side: unit for unit in robot_group.units}
    samplers = []
    for hand_cfg in cfg.hands:
        side = str(hand_cfg.side)
        unit = unit_by_side[side]
        if unit.end_effector is None:
            raise ValueError(f"policy tactile observation requires {side} end_effector")
        for channel in hand_cfg.channels:
            samplers.append(
                {
                    "side": side,
                    "channel": int(channel),
                    "read_tactile_frame": unit.end_effector.read_tactile_frame,
                }
            )
    if not samplers:
        raise ValueError("policy tactile observation hands must not be empty when enabled")
    return tuple(samplers)


class _PolicyStateBuilder:
    def __init__(self, cfg: DictConfig) -> None:
        self.source = str(cfg.policy.observation.server.state.source)
        self.sides = tuple(str(side) for side in cfg.robot.sides)
        self.mappers_by_side = {}
        if self.source == "policy_state":
            self.mappers_by_side = {
                side: TianjiTcpPoseMapper.fk_from_config(cfg.robot.hardware.arm, side)
                for side in self.sides
            }
            self.hand_open_state = {
                side: tuple(float(value) for value in cfg.wrappers.hand_alpha_action.open_state[side])
                for side in self.sides
            }
            self.hand_grasp_state = {
                side: tuple(float(value) for value in cfg.wrappers.hand_alpha_action.grasp_state[side])
                for side in self.sides
            }

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "_PolicyStateBuilder":
        return cls(cfg)

    def attach(self, _observation: tuple[float, ...], info: dict) -> dict:
        if self.source != "policy_state":
            return info
        output = dict(info)
        output["policy_state"] = self._delta_ee_tcp_pose_world_with_hand_alpha(info)
        return output

    def _delta_ee_tcp_pose_world_with_hand_alpha(self, info: dict) -> np.ndarray:
        state_by_side = {state.side: state for state in info["robot_states"]}
        values: list[float] = []
        for side in self.sides:
            state = state_by_side[side]
            tcp_pose_world = np.asarray(
                self.mappers_by_side[side].tcp_pose_world_from_q_rad(arm_q_rad(state)),
                dtype=np.float64,
            )
            values.extend(float(value) for value in tcp_pose_world[:3, 3])
            values.extend(rotation_6d_from_matrix(tcp_pose_world[:3, :3]))
            values.append(
                hand_state_alpha(
                    _hand_q_rad(state),
                    self.hand_open_state[side],
                    self.hand_grasp_state[side],
                )
            )
        return np.asarray(values, dtype=np.float32)


def _hand_q_rad(state) -> tuple[float, ...]:
    if state.end_effector_joint is not None:
        return tuple(float(value) for value in state.end_effector_joint.q_rad)
    if state.sharpa is not None:
        return tuple(float(value) for value in state.sharpa.observation_values("joint_pos"))
    raise RuntimeError(f"RobotState missing end-effector joint state for side {state.side}")


if __name__ == "__main__":
    main()
