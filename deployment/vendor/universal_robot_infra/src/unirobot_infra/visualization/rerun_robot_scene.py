from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from unirobot_infra.calibration.urdf_visualization import UrdfScene, load_urdf_scene
from unirobot_infra.core.types import RobotState


class RerunRobotSceneLogger:
    def __init__(self, rr: Any, cfg: DictConfig) -> None:
        self.rr = rr
        self.cfg = cfg
        self.robot_entity = f"{cfg.frame_name}/{cfg.robot.entity_name}"
        self._logged_assets: set[str] = set()
        self._log_world_root()

    def log_robot_states(self, states: tuple[RobotState, ...]) -> None:
        if not bool(self.cfg.robot.enabled) and not bool(self.cfg.hands.enabled):
            return
        robot_scene = self._load_robot_scene(states)
        if bool(self.cfg.robot.enabled):
            self._log_robot_scene(robot_scene)
            if bool(self.cfg.robot.sdk_tcp_frames.enabled):
                self._log_sdk_tcp_frames(robot_scene)
        if bool(self.cfg.hands.enabled):
            self._log_hands(robot_scene, states)

    def _log_world_root(self) -> None:
        self.rr.log("/", self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.rr.log(
            str(self.cfg.frame_name),
            self.rr.Transform3D(
                translation=(0.0, 0.0, 0.0),
                mat3x3=np.eye(3),
                relation=self.rr.TransformRelation.ParentFromChild,
            ),
            static=True,
        )
        self.rr.log(
            self.robot_entity,
            self.rr.Transform3D(
                translation=(0.0, 0.0, 0.0),
                mat3x3=np.eye(3),
                relation=self.rr.TransformRelation.ParentFromChild,
            ),
            static=True,
        )

    def _load_robot_scene(self, states: tuple[RobotState, ...]) -> UrdfScene:
        return load_urdf_scene(
            self.cfg.robot.urdf_path,
            str(self.cfg.robot.base_link),
            joint_positions_rad=self._robot_joint_positions_rad(states),
        )

    def _robot_joint_positions_rad(self, states: tuple[RobotState, ...]) -> dict[str, float]:
        joint_positions = {}
        for state in states:
            if state.arm_joint is not None:
                joint_names = tuple(str(name) for name in self.cfg.robot.arm_joint_names[state.side])
                if len(joint_names) != len(state.arm_joint.q_rad):
                    raise ValueError(
                        f"wrappers.rerun_visualization.scene.robot.arm_joint_names.{state.side} length mismatch"
                    )
                for joint_name, q_rad in zip(joint_names, state.arm_joint.q_rad, strict=True):
                    joint_positions[joint_name] = float(q_rad)
        return joint_positions

    def _log_robot_scene(self, scene: UrdfScene) -> None:
        for visual in scene.visuals:
            if self._robot_visual_hidden(visual.link_name):
                continue
            visual_entity = f"{self.robot_entity}/meshes/{_safe_entity_name(visual.link_name)}_{visual.visual_index}"
            self._log_transform(visual_entity, visual.base_from_visual, visual.scale)
            self._log_asset(visual_entity, visual.mesh_path, _asset_albedo_factor(float(self.cfg.robot.asset_alpha)))

    def _robot_visual_hidden(self, link_name: str) -> bool:
        for prefix in self.cfg.robot.hide_link_prefixes:
            if link_name.startswith(str(prefix)):
                return True
        return False

    def _log_sdk_tcp_frames(self, robot_scene: UrdfScene) -> None:
        axis_length_m = float(self.cfg.robot.sdk_tcp_frames.axis_length_m)
        if axis_length_m <= 0.0:
            raise ValueError(
                "wrappers.rerun_visualization.scene.robot.sdk_tcp_frames.axis_length_m must be positive"
            )
        for side in ("left", "right"):
            if side in self.cfg.robot.sdk_tcp_frames:
                self._log_sdk_tcp_frame(side, self.cfg.robot.sdk_tcp_frames[side], robot_scene, axis_length_m)

    def _log_sdk_tcp_frame(self, side: str, frame_cfg: DictConfig, robot_scene: UrdfScene, axis_length_m: float) -> None:
        parent_link = str(frame_cfg.parent_link)
        if parent_link not in robot_scene.link_transforms:
            raise ValueError(
                f"wrappers.rerun_visualization.scene.robot.sdk_tcp_frames.{side}.parent_link={parent_link} not found in URDF"
            )
        base_from_tcp = robot_scene.link_transforms[parent_link] @ _pose_from_xyz_rpy(
            frame_cfg.parent_from_tcp.xyz,
            frame_cfg.parent_from_tcp.rpy,
        )
        self.rr.log(
            f"{self.robot_entity}/frames/{_safe_entity_name(str(frame_cfg.frame_name))}",
            self.rr.Transform3D(
                translation=base_from_tcp[:3, 3],
                mat3x3=base_from_tcp[:3, :3],
                relation=self.rr.TransformRelation.ParentFromChild,
                axis_length=axis_length_m,
            ),
        )

    def _log_hands(self, robot_scene: UrdfScene, states: tuple[RobotState, ...]) -> None:
        for state in states:
            if state.end_effector_joint is not None:
                self._log_hand(state.side, state.end_effector_joint.q_rad, robot_scene)

    def _log_hand(self, side: str, q_rad: tuple[float, ...], robot_scene: UrdfScene) -> None:
        hand_cfg = self.cfg.hands[side]
        if not bool(hand_cfg.enabled):
            return
        parent_link = str(hand_cfg.parent_link)
        if parent_link not in robot_scene.link_transforms:
            raise ValueError(
                f"wrappers.rerun_visualization.scene.hands.{side}.parent_link={parent_link} not found in robot URDF"
            )
        joint_names = tuple(str(name) for name in hand_cfg.joint_names)
        if len(joint_names) != len(q_rad):
            raise ValueError(f"wrappers.rerun_visualization.scene.hands.{side}.joint_names length mismatch")
        hand_scene = load_urdf_scene(
            hand_cfg.urdf_path,
            str(hand_cfg.base_link),
            joint_positions_rad={
                joint_name: float(value)
                for joint_name, value in zip(joint_names, q_rad, strict=True)
            },
            package_paths=_package_paths(hand_cfg),
        )
        hand_entity = f"{self.robot_entity}/hands/{hand_cfg.entity_name}"
        base_from_hand = robot_scene.link_transforms[parent_link] @ _pose_from_xyz_rpy(
            hand_cfg.mount.xyz,
            hand_cfg.mount.rpy,
        )
        self._log_transform(hand_entity, base_from_hand, (1.0, 1.0, 1.0))
        if bool(self.cfg.hands.local_frame.enabled):
            self._log_hand_local_frame(hand_entity, str(hand_cfg.entity_name))
        for visual in hand_scene.visuals:
            visual_entity = f"{hand_entity}/meshes/{_safe_entity_name(visual.link_name)}_{visual.visual_index}"
            self._log_transform(visual_entity, visual.base_from_visual, visual.scale)
            self._log_asset(visual_entity, visual.mesh_path, _asset_albedo_factor(float(self.cfg.hands.asset_alpha)))

    def _log_hand_local_frame(self, hand_entity: str, hand_name: str) -> None:
        entity = f"{hand_entity}/local_frame"
        if entity in self._logged_assets:
            return
        length_m = float(self.cfg.hands.local_frame.length_m)
        radius_m = float(self.cfg.hands.local_frame.radius_m)
        if length_m <= 0.0:
            raise ValueError("wrappers.rerun_visualization.scene.hands.local_frame.length_m must be positive")
        if radius_m <= 0.0:
            raise ValueError("wrappers.rerun_visualization.scene.hands.local_frame.radius_m must be positive")
        self.rr.log(
            entity,
            self.rr.Arrows3D(
                origins=[(0.0, 0.0, 0.0)] * 3,
                vectors=[(length_m, 0.0, 0.0), (0.0, length_m, 0.0), (0.0, 0.0, length_m)],
                radii=[radius_m] * 3,
                colors=((255, 0, 0, 255), (0, 200, 0, 255), (0, 90, 255, 255)),
                labels=[f"{hand_name}_{axis}" for axis in ("X", "Y", "Z")],
                show_labels=bool(self.cfg.hands.local_frame.show_labels),
            ),
            static=True,
        )
        self._logged_assets.add(entity)

    def _log_transform(self, entity: str, transform: np.ndarray, scale: tuple[float, float, float]) -> None:
        self.rr.log(
            entity,
            self.rr.Transform3D(
                translation=transform[:3, 3],
                mat3x3=transform[:3, :3],
                scale=scale,
                relation=self.rr.TransformRelation.ParentFromChild,
            ),
        )

    def _log_asset(self, entity: str, mesh_path: Path, albedo_factor: tuple[int, int, int, int]) -> None:
        if entity in self._logged_assets:
            return
        self.rr.log(entity, self.rr.Asset3D(path=mesh_path, albedo_factor=albedo_factor), static=True)
        self._logged_assets.add(entity)


def _pose_from_xyz_rpy(xyz_cfg, rpy_cfg) -> np.ndarray:
    xyz = tuple(float(value) for value in xyz_cfg)
    rpy = tuple(float(value) for value in rpy_cfg)
    if len(xyz) != 3:
        raise ValueError("xyz must have length 3")
    if len(rpy) != 3:
        raise ValueError("rpy must have length 3")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _package_paths(hand_cfg: DictConfig) -> dict[str, Path]:
    package_paths = {}
    for package_name, package_path in hand_cfg.package_paths.items():
        package_paths[str(package_name)] = Path(str(package_path))
    return package_paths


def _asset_albedo_factor(alpha: float) -> tuple[int, int, int, int]:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("asset alpha must be between 0 and 1")
    return (255, 255, 255, int(round(alpha * 255.0)))


def _safe_entity_name(name: str) -> str:
    return name.replace("/", "_")
