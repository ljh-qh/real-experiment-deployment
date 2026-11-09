from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import RobotState


def build_policy_observation(
    *,
    cfg: DictConfig,
    sequence: int,
    observation: tuple[float, ...],
    info: dict,
    images: dict[str, np.ndarray],
    tactile: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "observation",
        "sequence": int(sequence),
        "timestamp_ns": now_ns(),
        "observation": {},
    }
    if str(cfg.task):
        payload["task"] = str(cfg.task)
    if bool(cfg.modalities.state.enabled):
        payload["observation"]["state"] = np.asarray(observation, dtype=np.float32)
    if bool(cfg.modalities.robot_states.enabled):
        payload["robot_state"] = _robot_state_stream_payload(info["robot_states"], cfg.modalities.robot_states.streams)
    if bool(cfg.modalities.images.enabled):
        payload["observation"]["images"] = {
            camera_name: np.asarray(image, dtype=np.uint8)
            for camera_name, image in images.items()
        }
    if bool(cfg.modalities.tactile.enabled):
        payload["observation"]["tactile"] = tactile
    return payload


def build_policy_server_observation(
    *,
    cfg: DictConfig,
    prompt: str,
    observation: tuple[float, ...],
    info: dict,
    action_layout,
    tactile: dict[str, Any],
) -> dict[str, Any]:
    if str(cfg.format) == "openpi_flat":
        return _build_openpi_flat_observation(
            cfg=cfg,
            prompt=prompt,
            observation=observation,
            info=info,
            action_layout=action_layout,
            tactile=tactile,
        )
    raise ValueError(f"unsupported policy observation format: {cfg.format}")


def _build_openpi_flat_observation(
    *,
    cfg: DictConfig,
    prompt: str,
    observation: tuple[float, ...],
    info: dict,
    action_layout,
    tactile: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        str(cfg.prompt_key): str(prompt),
    }
    if bool(cfg.state.enabled):
        payload[str(cfg.state.key)] = _policy_state_array(cfg.state, observation, info, action_layout)
    if bool(cfg.images.enabled):
        camera_frames = info["camera_frames"]
        for mapping in cfg.images.mappings:
            payload[str(mapping.key)] = prepare_policy_image(
                camera_frames[str(mapping.camera_name)],
                cfg=cfg.images,
            )
    if bool(cfg.tactile.enabled):
        payload[str(cfg.tactile.key)] = tactile
    return payload


def _policy_state_array(cfg: DictConfig, observation: tuple[float, ...], info: dict, action_layout) -> np.ndarray:
    source = str(cfg.source)
    if source == "observation":
        return np.asarray(observation, dtype=np.float32)
    if source == "hold_action":
        return np.asarray(action_layout.hold_action_from_states(tuple(info["robot_states"])), dtype=np.float32)
    if source == "policy_state":
        return np.asarray(info["policy_state"], dtype=np.float32)
    raise ValueError(f"unsupported policy state source: {source}")


def prepare_policy_image(image: np.ndarray, cfg: DictConfig) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"policy image must have shape [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"policy image must be uint8, got {array.dtype}")
    if bool(cfg.resize.enabled):
        width = int(cfg.resize.width)
        height = int(cfg.resize.height)
        if width <= 0 or height <= 0:
            raise ValueError("policy observation image resize width/height must be positive")
        if array.shape[1] != width or array.shape[0] != height:
            array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
            array = np.asarray(array, dtype=np.uint8)
    layout = str(cfg.layout)
    if layout != "hwc":
        raise ValueError(f"OpenPI policy image layout must be hwc, got {layout}")
    return np.ascontiguousarray(array)


def _robot_state_stream_payload(states: tuple[RobotState, ...], streams) -> dict[str, dict[str, Any]]:
    if not streams:
        raise ValueError("policy robot_states.streams must not be empty when enabled")
    state_by_side = {state.side: state for state in states}
    output: dict[str, dict[str, Any]] = {}
    for stream_cfg in streams:
        stream_name = str(stream_cfg.name)
        side, part = _stream_name_side_part(stream_name)
        fields = tuple(str(field) for field in stream_cfg.fields)
        if not fields:
            raise ValueError("policy robot state stream fields must not be empty")
        state = state_by_side[side]
        output[stream_name] = {
            field: _state_component_value(state, part, field)
            for field in fields
        }
    return output


def _stream_name_side_part(stream_name: str) -> tuple[str, str]:
    if stream_name == "left_arm":
        return "left", "arm"
    if stream_name == "right_arm":
        return "right", "arm"
    if stream_name == "left_end_effector":
        return "left", "end_effector"
    if stream_name == "right_end_effector":
        return "right", "end_effector"
    raise ValueError(f"unsupported policy robot state stream: {stream_name}")


def _state_component_value(state: RobotState, part: str, token: str) -> Any:
    if token == "timestamp_ns":
        return int(state.timestamp_ns)
    if part == "arm":
        return _arm_state_value(state, token)
    if part == "end_effector":
        return _end_effector_state_value(state, token)
    raise ValueError("policy robot state part must be arm or end_effector")


def _arm_state_value(state: RobotState, token: str) -> Any:
    if state.arm_joint is None:
        raise ValueError(f"policy robot state requires {state.side} arm")
    if token == "joint_pos":
        return np.asarray(state.arm_joint.q_rad, dtype=np.float32)
    if token == "pose":
        if state.ee_pose is None:
            raise ValueError(f"policy robot state requires {state.side} arm pose")
        return np.asarray(state.ee_pose.xyz_m + state.ee_pose.quat_xyzw, dtype=np.float32)
    raise ValueError(f"unsupported policy arm state token: {token}")


def _end_effector_state_value(state: RobotState, token: str) -> Any:
    if state.end_effector_joint is None:
        raise ValueError(f"policy robot state requires {state.side} end_effector")
    if token == "joint_pos":
        return np.asarray(state.end_effector_joint.q_rad, dtype=np.float32)
    if token == "gripper":
        values = tuple(float(value) for value in state.end_effector_joint.q_rad)
        if len(values) != 1:
            raise ValueError(f"policy gripper requires {state.side} end_effector joint_pos length 1")
        if values[0] not in {0.0, 1.0}:
            raise ValueError("policy gripper must be discrete 0 or 1")
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"unsupported policy end_effector state token: {token}")
