from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    stream: str
    field: str

    @classmethod
    def from_config(cls, value: str | dict[str, Any]) -> "FieldSpec":
        if isinstance(value, str):
            parts = value.split(".", 1)
            if len(parts) != 2:
                raise ValueError("dataset field spec must be '<stream>.<field>'")
            return cls(stream=parts[0], field=parts[1])
        return cls(stream=str(value["stream"]), field=str(value["field"]))

    @property
    def token(self) -> str:
        return f"{self.stream}.{self.field}"


@dataclass(frozen=True)
class CropConfig:
    top: int
    bottom: int
    left: int
    right: int

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "CropConfig":
        return cls(
            top=int(value["top"]),
            bottom=int(value["bottom"]),
            left=int(value["left"]),
            right=int(value["right"]),
        )


@dataclass(frozen=True)
class ConvertCameraConfig:
    source: str
    crop: CropConfig
    resize: tuple[int, int] | None

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "ConvertCameraConfig":
        resize = None
        if value["resize"] is not None:
            resize_values = tuple(int(item) for item in value["resize"])
            if len(resize_values) != 2:
                raise ValueError("camera resize must have length 2")
            resize = (resize_values[0], resize_values[1])
        return cls(source=str(value["source"]), crop=CropConfig.from_config(value["crop"]), resize=resize)


@dataclass(frozen=True)
class ConvertTactileImageConfig:
    name: str
    side: str
    finger: str
    modality: str
    resize: tuple[int, int] | None

    @classmethod
    def from_config(cls, name: str, value: dict[str, Any]) -> "ConvertTactileImageConfig":
        resize = None
        if value["resize"] is not None:
            resize_values = tuple(int(item) for item in value["resize"])
            if len(resize_values) != 2:
                raise ValueError("tactile resize must have length 2")
            resize = (resize_values[0], resize_values[1])
        side = str(value["side"])
        if side not in {"left", "right"}:
            raise ValueError("tactile side must be left or right")
        modality = str(value["modality"])
        if modality not in {"raw", "deform"}:
            raise ValueError("tactile modality must be raw or deform")
        return cls(
            name=str(name),
            side=side,
            finger=str(value["finger"]),
            modality=modality,
            resize=resize,
        )


@dataclass(frozen=True)
class ConvertTactileForceSensorConfig:
    side: str
    finger: str

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "ConvertTactileForceSensorConfig":
        side = str(value["side"])
        if side not in {"left", "right"}:
            raise ValueError("tactile force side must be left or right")
        return cls(side=side, finger=str(value["finger"]))

    @property
    def name(self) -> str:
        return f"{self.side}_{self.finger}"


@dataclass(frozen=True)
class ConvertTactileForceConfig:
    field: str
    sensors: tuple[ConvertTactileForceSensorConfig, ...]

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "ConvertTactileForceConfig":
        sensors = tuple(ConvertTactileForceSensorConfig.from_config(item) for item in value["sensors"])
        if not sensors:
            raise ValueError("tactile_force.sensors must not be empty")
        return cls(field=str(value["field"]), sensors=sensors)


@dataclass(frozen=True)
class HandAlphaConfig:
    open_state: dict[str, tuple[float, ...]]
    grasp_state: dict[str, tuple[float, ...]]

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "HandAlphaConfig":
        return cls(
            open_state=_hand_state_by_side(value["open_state"]),
            grasp_state=_hand_state_by_side(value["grasp_state"]),
        )


@dataclass(frozen=True)
class StateConfig:
    type: str
    keys: tuple[FieldSpec, ...] = ()
    sides: tuple[str, ...] = ()
    arm_keys: dict[str, FieldSpec] | None = None
    hand_keys: dict[str, FieldSpec] | None = None
    hand_alpha: HandAlphaConfig | None = None

    @classmethod
    def from_field_specs(cls, keys: tuple[FieldSpec, ...]) -> "StateConfig":
        if not keys:
            raise ValueError("state_keys must not be empty")
        return cls(type="field_values", keys=keys)

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "StateConfig":
        state_type = str(value["type"])
        if state_type == "field_values":
            keys = tuple(FieldSpec.from_config(item) for item in value["keys"])
            if not keys:
                raise ValueError("state.keys must not be empty")
            return cls(type=state_type, keys=keys)
        if state_type in {"tcp_pose_world_with_hand_alpha", "tcp_pose_world_with_hand_joint"}:
            sides = tuple(str(side) for side in value["sides"])
            if not sides:
                raise ValueError(f"{state_type} state requires sides")
            arm_keys = {
                side: FieldSpec.from_config(value["arm_keys"][side])
                for side in sides
            }
            hand_keys = {
                side: FieldSpec.from_config(value["hand_keys"][side])
                for side in sides
            }
            hand_alpha = None
            if state_type == "tcp_pose_world_with_hand_alpha":
                hand_alpha = HandAlphaConfig.from_config(value["hand_alpha"])
            return cls(
                type=state_type,
                sides=sides,
                arm_keys=arm_keys,
                hand_keys=hand_keys,
                hand_alpha=hand_alpha,
            )
        raise ValueError(f"State type {state_type!r} is not supported")

    @property
    def source_specs(self) -> tuple[FieldSpec, ...]:
        if self.type == "field_values":
            return self.keys
        if self.type == "tcp_pose_world_with_hand_alpha":
            if self.arm_keys is None or self.hand_keys is None:
                raise ValueError("tcp_pose_world_with_hand_alpha state is missing arm_keys or hand_keys")
            specs = []
            for side in self.sides:
                specs.append(self.arm_keys[side])
                specs.append(self.hand_keys[side])
            return tuple(specs)
        if self.type == "tcp_pose_world_with_hand_joint":
            if self.arm_keys is None or self.hand_keys is None:
                raise ValueError("tcp_pose_world_with_hand_joint state is missing arm_keys or hand_keys")
            specs = []
            for side in self.sides:
                specs.append(self.arm_keys[side])
                specs.append(self.hand_keys[side])
            return tuple(specs)
        raise ValueError(f"State type {self.type!r} is not supported")

    @property
    def tokens(self) -> tuple[str, ...]:
        if self.type == "field_values":
            return tuple(spec.token for spec in self.keys)
        if self.type == "tcp_pose_world_with_hand_alpha":
            tokens = []
            for side in self.sides:
                tokens.append(f"{side}_tcp_pose_world_xyz[3]")
                tokens.append(f"{side}_tcp_pose_world_rotation_6d[6]")
                tokens.append(f"{side}_hand_alpha[1]")
            return tuple(tokens)
        if self.type == "tcp_pose_world_with_hand_joint":
            tokens = []
            for side in self.sides:
                tokens.append(f"{side}_tcp_pose_world_xyz[3]")
                tokens.append(f"{side}_tcp_pose_world_rotation_6d[6]")
                tokens.append(f"{side}_hand_joint_pos[22]")
            return tuple(tokens)
        raise ValueError(f"State type {self.type!r} is not supported")


@dataclass(frozen=True)
class FrameFilterConfig:
    enabled: bool
    type: str
    max_tcp_translation_m: float
    max_rotation_6d: float
    max_hand_alpha: float
    min_run_frames: int

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "FrameFilterConfig":
        filter_type = str(value["type"])
        if filter_type not in {"none", "stationary_tcp_pose_world_with_hand_alpha"}:
            raise ValueError(f"Frame filter type {filter_type!r} is not supported")
        min_run_frames = int(value["min_run_frames"])
        if min_run_frames <= 0:
            raise ValueError("frame_filter.min_run_frames must be positive")
        max_tcp_translation_m = float(value["max_tcp_translation_m"])
        max_rotation_6d = float(value["max_rotation_6d"])
        max_hand_alpha = float(value["max_hand_alpha"])
        if max_tcp_translation_m < 0.0 or max_rotation_6d < 0.0 or max_hand_alpha < 0.0:
            raise ValueError("frame_filter thresholds must be non-negative")
        return cls(
            enabled=bool(value["enabled"]),
            type=filter_type,
            max_tcp_translation_m=max_tcp_translation_m,
            max_rotation_6d=max_rotation_6d,
            max_hand_alpha=max_hand_alpha,
            min_run_frames=min_run_frames,
        )


@dataclass(frozen=True)
class ActionConfig:
    type: str
    source: str
    keys: tuple[FieldSpec, ...]
    sides: tuple[str, ...] = ()
    arm_keys: dict[str, FieldSpec] | None = None
    hand_keys: dict[str, FieldSpec] | None = None
    hand_alpha: HandAlphaConfig | None = None

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "ActionConfig":
        keys = tuple(FieldSpec.from_config(item) for item in value["keys"])
        action_type = str(value["type"])
        action_source = str(value["source"])
        if action_type in {"delta_ee_world_with_hand_alpha", "delta_ee_world_with_hand_joint"}:
            if action_source != "next_state":
                raise ValueError(f"{action_type} action requires source=next_state")
            sides = tuple(str(side) for side in value["sides"])
            if not sides:
                raise ValueError(f"{action_type} action requires sides")
            arm_keys = {
                side: FieldSpec.from_config(value["arm_keys"][side])
                for side in sides
            }
            hand_keys = {
                side: FieldSpec.from_config(value["hand_keys"][side])
                for side in sides
            }
            hand_alpha = None
            if action_type == "delta_ee_world_with_hand_alpha":
                hand_alpha = HandAlphaConfig.from_config(value["hand_alpha"])
            return cls(
                type=action_type,
                source=action_source,
                keys=keys,
                sides=sides,
                arm_keys=arm_keys,
                hand_keys=hand_keys,
                hand_alpha=hand_alpha,
            )
        return cls(type=action_type, source=action_source, keys=keys)

    @property
    def source_specs(self) -> tuple[FieldSpec, ...]:
        if self.type == "delta_ee_world_with_hand_alpha":
            if self.arm_keys is None or self.hand_keys is None:
                raise ValueError("delta_ee_world_with_hand_alpha action is missing arm_keys or hand_keys")
            specs = []
            for side in self.sides:
                specs.append(self.arm_keys[side])
                specs.append(self.hand_keys[side])
            return tuple(specs)
        if self.type == "delta_ee_world_with_hand_joint":
            if self.arm_keys is None or self.hand_keys is None:
                raise ValueError("delta_ee_world_with_hand_joint action is missing arm_keys or hand_keys")
            specs = []
            for side in self.sides:
                specs.append(self.arm_keys[side])
                specs.append(self.hand_keys[side])
            return tuple(specs)
        return self.keys

    @property
    def tokens(self) -> tuple[str, ...]:
        if self.type == "delta_ee_world_with_hand_alpha":
            tokens = []
            for side in self.sides:
                tokens.append(f"{side}_delta_ee_world[9]")
                tokens.append(f"{side}_hand_alpha[1]")
            return tuple(tokens)
        if self.type == "delta_ee_world_with_hand_joint":
            tokens = []
            for side in self.sides:
                tokens.append(f"{side}_delta_ee_world[9]")
                tokens.append(f"{side}_hand_joint_pos[22]")
            return tuple(tokens)
        return tuple(spec.token for spec in self.keys)


@dataclass(frozen=True)
class ConvertConfig:
    tactile_image_format: str
    target_fps: int
    repo_id: str
    robot_type: str
    state: StateConfig
    state_keys: tuple[FieldSpec, ...]
    frame_filter: FrameFilterConfig
    action: ActionConfig
    cameras: dict[str, ConvertCameraConfig]
    tactile_images: dict[str, ConvertTactileImageConfig]
    tactile_force: ConvertTactileForceConfig | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConvertConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        logger.info("Loaded convert config from %s", config_path)
        return cls.from_config(data["convert"])

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "ConvertConfig":
        if "state" in data:
            state = StateConfig.from_config(data["state"])
            state_keys = state.source_specs
        else:
            state_keys = tuple(FieldSpec.from_config(item) for item in data["state_keys"])
            state = StateConfig.from_field_specs(state_keys)
        cameras = {
            str(camera_name): ConvertCameraConfig.from_config(camera_cfg)
            for camera_name, camera_cfg in data["cameras"].items()
        }
        tactile_images = {
            str(image_name): ConvertTactileImageConfig.from_config(str(image_name), image_cfg)
            for image_name, image_cfg in data["tactile_images"].items()
        }
        tactile_force = None
        if data["tactile_force"] is not None:
            tactile_force = ConvertTactileForceConfig.from_config(data["tactile_force"])
        tactile_image_format = str(data["tactile_image_format"])
        if tactile_image_format not in {"mp4", "png"}:
            raise ValueError("tactile_image_format must be mp4 or png")
        return cls(
            tactile_image_format=tactile_image_format,
            target_fps=int(data["target_fps"]),
            repo_id=str(data["repo_id"]),
            robot_type=str(data["robot_type"]),
            state=state,
            state_keys=state_keys,
            frame_filter=FrameFilterConfig.from_config(data["frame_filter"]),
            action=ActionConfig.from_config(data["action"]),
            cameras=cameras,
            tactile_images=tactile_images,
            tactile_force=tactile_force,
        )

    @property
    def state_tokens(self) -> tuple[str, ...]:
        return self.state.tokens

    @property
    def action_tokens(self) -> tuple[str, ...]:
        return self.action.tokens

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.cameras.keys())

    @property
    def camera_source_names(self) -> tuple[str, ...]:
        return tuple(camera.source for camera in self.cameras.values())

    @property
    def tactile_image_names(self) -> tuple[str, ...]:
        return tuple(self.tactile_images.keys())

    @property
    def tactile_image_specs(self) -> tuple[ConvertTactileImageConfig, ...]:
        return tuple(self.tactile_images.values())


def _hand_state_by_side(value: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    return {
        str(side): tuple(float(item) for item in q_rad)
        for side, q_rad in value.items()
    }
