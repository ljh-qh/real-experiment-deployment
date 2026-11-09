from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from scipy.spatial.transform import Rotation

from unirobot_infra.calibration.hand_eye import load_transform
from unirobot_infra.calibration.urdf_visualization import UrdfScene, load_urdf_scene
from unirobot_infra.camera.zed_capture import ZedCapture
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.perception.frame_transform import RigidTransform, matrix3x3_tuple, offset_tuple
from unirobot_infra.perception.object_localization import localize_detections_3d
from unirobot_infra.perception.pointcloud import rgbd_to_pointcloud
from unirobot_infra.perception.types import Detection2D, ObjectPoint3D, PerceptionResult
from unirobot_infra.perception.yoloe import YoloEDetector
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/perception/perception_loop")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level, console=_terminal_console())
    logger = get_logger("perception_loop")
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_perception_loop(cfg, logger)


def _terminal_console() -> Console:
    terminal_stderr = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1)
    return Console(file=terminal_stderr, stderr=True)


def run_perception_loop(cfg: DictConfig, logger) -> None:
    logger.info(
        "开始初始化感知循环 side=%s target_label=%s classes=%s camera=%s",
        cfg.perception.arm_side,
        cfg.perception.robot_preview.target_label,
        list(cfg.perception.detector.classes),
        cfg.perception.camera_name,
    )
    detector = YoloEDetector(cfg.perception.detector)
    logger.info("检测模型初始化完成")
    camera = _build_zed_rgbd_camera(cfg)
    logger.info("相机初始化完成 camera=%s", cfg.perception.camera_name)
    base_from_camera = _base_from_camera_transform(cfg)
    visualizer = _build_rerun_visualizer(cfg)
    logger.info("rerun 初始化完成 enabled=%s", bool(cfg.visualize.rerun.enabled))
    kinematics = _build_tianji_kinematics(cfg) if bool(cfg.perception.robot_preview.enabled) else None
    if kinematics is not None:
        logger.info("Tianji IK 初始化完成 side=%s", cfg.perception.arm_side)
    preview_reference_q_deg = _initial_arm_q_deg(cfg, str(cfg.perception.arm_side)) if bool(cfg.perception.robot_preview.enabled) else None
    logger.info(
        "感知循环启动 side=%s target_label=%s classes=%s camera=%s",
        cfg.perception.arm_side,
        cfg.perception.robot_preview.target_label,
        list(cfg.perception.detector.classes),
        cfg.perception.camera_name,
    )
    frame_index = 0
    try:
        while _should_continue(frame_index, int(cfg.app.max_frames)):
            frame_start = time.monotonic()
            ok, rgbd = camera.read_rgbd()
            if not ok or rgbd is None:
                raise RuntimeError("ZED RGB-D read failed")
            detections = detector.detect(rgbd.rgb)
            objects = localize_detections_3d(
                detections=detections,
                depth_m=rgbd.depth_m,
                intrinsics=rgbd.intrinsics,
                base_from_camera=base_from_camera,
                offset_m=offset_tuple(cfg.perception.target.offset_m),
                offset_frame=str(cfg.perception.target.offset_frame),
                base_xyz_from_robot_world=_base_xyz_from_robot_world(cfg),
                min_depth_m=float(cfg.perception.depth.min_m),
                max_depth_m=float(cfg.perception.depth.max_m),
                depth_search_radius_px=int(cfg.perception.depth.search_radius_px),
            )
            result = PerceptionResult(
                rgb=rgbd.rgb,
                depth_m=rgbd.depth_m,
                detections=detections,
                objects=objects,
            )
            pointcloud_xyz, pointcloud_rgb = rgbd_to_pointcloud(
                rgb=rgbd.rgb,
                depth_m=rgbd.depth_m,
                intrinsics=rgbd.intrinsics,
                stride=int(cfg.perception.pointcloud.stride),
                min_depth_m=float(cfg.perception.depth.min_m),
                max_depth_m=float(cfg.perception.depth.max_m),
            )
            if visualizer is not None:
                _log_perception_result(
                    visualizer,
                    cfg,
                    frame_index,
                    result,
                    pointcloud_xyz,
                    pointcloud_rgb,
                    kinematics,
                    preview_reference_q_deg,
                    logger,
                )
            if _should_log_frame(frame_index, int(cfg.app.log_every_n_frames)):
                logger.info("感知帧完成 frame=%s detections=%s objects=%s", frame_index, len(detections), len(objects))
            frame_index += 1
            _sleep_remaining(frame_start, 1.0 / float(cfg.app.rate_hz))
    finally:
        camera.close()


def _build_zed_rgbd_camera(cfg: DictConfig) -> ZedCapture:
    camera_name = str(cfg.perception.camera_name)
    camera_cfg = cfg.camera.cameras[camera_name]
    if camera_cfg.type != "zed_sdk":
        raise ValueError("perception_loop requires a zed_sdk camera")
    return ZedCapture(camera_name, camera_cfg)


def _build_tianji_kinematics(cfg: DictConfig) -> TianjiKinematics:
    side = str(cfg.perception.arm_side)
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    return kinematics


def _base_from_camera_transform(cfg: DictConfig) -> RigidTransform:
    camera_from_base = load_transform(cfg.perception.transform.camera_from_base_path)
    return RigidTransform(camera_from_base).inverse()


def _base_xyz_from_robot_world(cfg: DictConfig) -> tuple[tuple[float, float, float], ...]:
    side = str(cfg.perception.arm_side)
    return matrix3x3_tuple(
        cfg.robot.hardware.arm.arms[side].sdk_xyz_from_robot_world,
        f"robot.hardware.arm.arms.{side}.sdk_xyz_from_robot_world",
    )


def _build_rerun_visualizer(cfg: DictConfig):
    if not bool(cfg.visualize.rerun.enabled):
        return None
    import rerun as rr

    rr.init(str(cfg.visualize.rerun.app_id), spawn=bool(cfg.visualize.rerun.spawn))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        cfg.visualize.frame_name,
        rr.Transform3D(
            translation=(0.0, 0.0, 0.0),
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    robot_scene = _load_initial_robot_scene(cfg)
    if bool(cfg.perception.robot_preview.enabled):
        _log_robot_root(rr, _robot_entity(cfg), static=True)
    else:
        _log_urdf_scene_visuals(rr, cfg, _robot_entity(cfg), robot_scene, static=True)
        _log_robot_sdk_tcp_frames(rr, cfg, robot_scene, _robot_entity(cfg))
        if bool(cfg.visualize.hands.enabled):
            _log_sharpa_hands(rr, cfg, robot_scene, _robot_entity(cfg))
    camera_parent_entity = _selected_arm_base_entity(rr, cfg, robot_scene)
    camera_from_base = load_transform(cfg.perception.transform.camera_from_base_path)
    camera_entity = f"{camera_parent_entity}/{cfg.visualize.camera_frame_name}"
    rr.log(
        camera_entity,
        rr.Transform3D(
            translation=camera_from_base[:3, 3],
            mat3x3=camera_from_base[:3, :3],
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=True,
    )
    _log_camera_frustum(rr, cfg, camera_entity)
    return {
        "rr": rr,
        "camera_entity": camera_entity,
        "camera_parent_entity": camera_parent_entity,
        "target_tcp_rotation": _target_tcp_rotation_base_from_tcp(cfg),
    }


def _log_perception_result(
    visualizer,
    cfg: DictConfig,
    frame_index: int,
    result: PerceptionResult,
    pointcloud_xyz: np.ndarray,
    pointcloud_rgb: np.ndarray,
    kinematics: TianjiKinematics | None,
    preview_reference_q_deg: tuple[float, float, float, float, float, float, float] | None,
    logger,
) -> None:
    rr = visualizer["rr"]
    rr.set_time_sequence("frame", frame_index)
    rr.log("perception/rgb", rr.Image(result.rgb))
    if result.detections:
        rr.log(
            "perception/rgb/detections",
            rr.Boxes2D(
                mins=_bbox_mins(result.detections),
                sizes=_bbox_sizes(result.detections),
                labels=[f"{detection.label} {detection.score:.2f}" for detection in result.detections],
                show_labels=True,
            ),
        )
    rr.log(
        f"{visualizer['camera_entity']}/pointcloud",
        rr.Points3D(pointcloud_xyz, colors=pointcloud_rgb, radii=float(cfg.visualize.point_radius_m)),
    )
    if not result.objects:
        _log_rerun_status(rr, "WARN", f"no_3d_target detections={_detection_label_summary(result.detections)}")
        if _should_log_frame(frame_index, int(cfg.app.log_every_n_frames)):
            logger.warning(
                "没有可定位的 3D 目标 frame=%s detections=%s，可能是没有检测到目标或深度无效",
                frame_index,
                _detection_label_summary(result.detections),
            )
        return

    rr.log(
        f"{visualizer['camera_entity']}/detections",
        rr.Points3D(
            [obj.camera_xyz_m for obj in result.objects],
            colors=[255, 80, 40],
            labels=[obj.label for obj in result.objects],
            show_labels=True,
            radii=float(cfg.visualize.object_radius_m),
        ),
    )
    rr.log(
        f"{visualizer['camera_parent_entity']}/objects_base",
        rr.Points3D(
            [obj.base_xyz_m for obj in result.objects],
            colors=[80, 160, 255],
            labels=[f"{obj.label}_base" for obj in result.objects],
            show_labels=True,
            radii=float(cfg.visualize.object_radius_m),
        ),
    )
    rr.log(
        f"{visualizer['camera_parent_entity']}/targets",
        rr.Points3D(
            [obj.target_xyz_m for obj in result.objects],
            colors=[40, 220, 120],
            labels=[f"{obj.label}_target" for obj in result.objects],
            show_labels=True,
            radii=float(cfg.visualize.target_radius_m),
        ),
    )
    if bool(cfg.visualize.target_tcp.enabled):
        _log_target_tcp_frames(rr, visualizer, cfg, result)
    if bool(cfg.perception.robot_preview.enabled):
        if kinematics is None or preview_reference_q_deg is None:
            raise RuntimeError("perception.robot_preview.enabled requires Tianji kinematics")
        target = _select_preview_target(cfg, result)
        if target is None:
            _log_rerun_status(
                rr,
                "WARN",
                f"no_target_label target_label={cfg.perception.robot_preview.target_label} localized={_object_label_summary(result.objects)}",
            )
            if _should_log_frame(frame_index, int(cfg.app.log_every_n_frames)):
                logger.warning(
                    "没有找到 IK 目标 frame=%s target_label=%s localized=%s",
                    frame_index,
                    cfg.perception.robot_preview.target_label,
                    _object_label_summary(result.objects),
                )
            return
        target_xyzabc = None
        try:
            target_pose = _target_tcp_pose_mat_mm(target, visualizer["target_tcp_rotation"])
            target_xyzabc = kinematics.mat4x4_to_xyzabc_mm_deg(target_pose)
            _log_ik_robot_preview(rr, cfg, target_xyzabc, kinematics, preview_reference_q_deg)
        except RuntimeError as exc:
            if not bool(cfg.perception.robot_preview.continue_on_ik_error):
                raise
            logger.warning(
                "IK 预览失败，已跳过当前帧 frame=%s side=%s target_label=%s target_xyz_m=%s target_xyzabc=%s error=%s",
                frame_index,
                cfg.perception.arm_side,
                cfg.perception.robot_preview.target_label,
                _format_xyz_m(target.target_xyz_m),
                _format_xyzabc_mm_deg(target_xyzabc),
                exc,
            )
            _log_rerun_status(
                rr,
                "ERROR",
                f"ik_failed side={cfg.perception.arm_side} target_xyz_m={_format_xyz_m(target.target_xyz_m)} "
                f"target_xyzabc={_format_xyzabc_mm_deg(target_xyzabc)} error={exc}",
            )
            return
        _log_rerun_status(
            rr,
            "INFO",
            f"ik_success side={cfg.perception.arm_side} target_label={target.label} "
            f"target_xyz_m={_format_xyz_m(target.target_xyz_m)} target_xyzabc={_format_xyzabc_mm_deg(target_xyzabc)}",
        )
        if _should_log_frame(frame_index, int(cfg.app.log_every_n_frames)):
            logger.info(
                "IK 预览成功 frame=%s side=%s target_label=%s target_xyz_m=%s target_xyzabc=%s",
                frame_index,
                cfg.perception.arm_side,
                target.label,
                _format_xyz_m(target.target_xyz_m),
                _format_xyzabc_mm_deg(target_xyzabc),
            )


def _log_rerun_status(rr, level: str, message: str) -> None:
    rr.log("perception/status", rr.TextLog(message, level=level))


def _load_initial_robot_scene(cfg: DictConfig) -> UrdfScene:
    return _load_robot_scene(cfg, _initial_pose_joint_positions_rad(cfg))


def _load_robot_scene(cfg: DictConfig, joint_positions_rad: dict[str, float]) -> UrdfScene:
    return load_urdf_scene(
        cfg.visualize.robot.urdf_path,
        str(cfg.visualize.robot.base_link),
        joint_positions_rad=joint_positions_rad,
    )


def _log_urdf_scene_visuals(rr, cfg: DictConfig, robot_entity: str, scene: UrdfScene, static: bool) -> None:
    _log_robot_root(rr, robot_entity, static)
    for visual in scene.visuals:
        if _robot_visual_hidden(cfg, visual.link_name):
            continue
        visual_entity = f"{robot_entity}/meshes/{_safe_entity_name(visual.link_name)}_{visual.visual_index}"
        rr.log(
            visual_entity,
            rr.Transform3D(
                translation=visual.base_from_visual[:3, 3],
                mat3x3=visual.base_from_visual[:3, :3],
                scale=visual.scale,
                relation=rr.TransformRelation.ParentFromChild,
            ),
            static=static,
        )
        rr.log(visual_entity, rr.Asset3D(path=visual.mesh_path, albedo_factor=_asset_albedo_factor(cfg)), static=True)


def _log_robot_root(rr, robot_entity: str, static: bool) -> None:
    rr.log(
        robot_entity,
        rr.Transform3D(
            translation=(0.0, 0.0, 0.0),
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=static,
    )


def _robot_visual_hidden(cfg: DictConfig, link_name: str) -> bool:
    if not bool(cfg.visualize.hands.enabled):
        return False
    for prefix in cfg.visualize.hands.hide_robot_link_prefixes:
        if link_name.startswith(str(prefix)):
            return True
    return False


def _selected_arm_base_entity(rr, cfg: DictConfig, robot_scene: UrdfScene) -> str:
    camera_parent_link = _selected_arm_base_link(cfg)
    if camera_parent_link not in robot_scene.link_transforms:
        raise ValueError(f"visualize.arm_base_links.{cfg.perception.arm_side}={camera_parent_link} not found in URDF")
    robot_entity = _robot_entity(cfg)
    rr.log(
        f"{robot_entity}/frames/{_safe_entity_name(camera_parent_link)}",
        rr.Transform3D(
            translation=robot_scene.link_transforms[camera_parent_link][:3, 3],
            mat3x3=robot_scene.link_transforms[camera_parent_link][:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            axis_length=float(cfg.visualize.link_axis_length_m),
        ),
        static=True,
    )
    return f"{robot_entity}/frames/{_safe_entity_name(camera_parent_link)}"


def _selected_arm_base_link(cfg: DictConfig) -> str:
    side = str(cfg.perception.arm_side)
    if side == "left":
        return str(cfg.visualize.arm_base_links.left)
    if side == "right":
        return str(cfg.visualize.arm_base_links.right)
    raise ValueError("perception.arm_side must be left or right")


def _log_robot_sdk_tcp_frames(
    rr,
    cfg: DictConfig,
    robot_scene: UrdfScene,
    robot_entity: str,
    static: bool = True,
) -> set[str]:
    if not bool(cfg.visualize.robot.sdk_tcp_frames.enabled):
        return set()
    axis_length_m = float(cfg.visualize.robot.sdk_tcp_frames.axis_length_m)
    if axis_length_m <= 0.0:
        raise ValueError("visualize.robot.sdk_tcp_frames.axis_length_m must be positive")
    side = str(cfg.perception.arm_side)
    if side == "left":
        frame_name = _log_robot_sdk_tcp_frame(
            rr,
            robot_entity,
            "left",
            cfg.visualize.robot.sdk_tcp_frames.left,
            robot_scene,
            axis_length_m,
            static,
        )
    elif side == "right":
        frame_name = _log_robot_sdk_tcp_frame(
            rr,
            robot_entity,
            "right",
            cfg.visualize.robot.sdk_tcp_frames.right,
            robot_scene,
            axis_length_m,
            static,
        )
    else:
        raise ValueError("perception.arm_side must be left or right")
    return {frame_name}


def _log_robot_sdk_tcp_frame(
    rr,
    robot_entity: str,
    side: str,
    frame_cfg: DictConfig,
    robot_scene: UrdfScene,
    axis_length_m: float,
    static: bool,
) -> str:
    parent_link = str(frame_cfg.parent_link)
    tcp_frame_name = str(frame_cfg.frame_name)
    if parent_link not in robot_scene.link_transforms:
        raise ValueError(f"visualize.robot.sdk_tcp_frames.{side}.parent_link={parent_link} not found in URDF")
    base_from_tcp = robot_scene.link_transforms[parent_link] @ _pose_from_xyz_rpy(
        frame_cfg.parent_from_tcp.xyz,
        frame_cfg.parent_from_tcp.rpy,
    )
    rr.log(
        f"{robot_entity}/frames/{_safe_entity_name(tcp_frame_name)}",
        rr.Transform3D(
            translation=base_from_tcp[:3, 3],
            mat3x3=base_from_tcp[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            axis_length=axis_length_m,
        ),
        static=static,
    )
    return tcp_frame_name


def _log_sharpa_hands(rr, cfg: DictConfig, robot_scene: UrdfScene, robot_entity: str, static: bool = True) -> None:
    _log_sharpa_hand(rr, cfg, robot_scene, robot_entity, cfg.visualize.hands.left, "left", static)
    _log_sharpa_hand(rr, cfg, robot_scene, robot_entity, cfg.visualize.hands.right, "right", static)


def _log_sharpa_hand(
    rr,
    cfg: DictConfig,
    robot_scene: UrdfScene,
    robot_entity: str,
    hand_cfg: DictConfig,
    side: str,
    static: bool,
) -> None:
    if not bool(hand_cfg.enabled):
        return
    parent_link = str(hand_cfg.parent_link)
    if parent_link not in robot_scene.link_transforms:
        raise ValueError(f"visualize.hands.{side}.parent_link={parent_link} not found in robot URDF")
    scene = load_urdf_scene(
        hand_cfg.urdf_path,
        str(hand_cfg.base_link),
        joint_positions_rad=_hand_joint_positions_rad(hand_cfg),
        package_paths=_hand_package_paths(hand_cfg),
    )
    hand_entity = f"{robot_entity}/hands/{hand_cfg.entity_name}"
    base_from_parent = robot_scene.link_transforms[parent_link]
    parent_from_hand_base = _pose_from_xyz_rpy(hand_cfg.mount.xyz, hand_cfg.mount.rpy)
    base_from_hand_base = base_from_parent @ parent_from_hand_base
    rr.log(
        hand_entity,
        rr.Transform3D(
            translation=base_from_hand_base[:3, 3],
            mat3x3=base_from_hand_base[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=static,
    )
    if bool(cfg.visualize.hands.local_frame.enabled):
        _log_hand_local_frame(rr, cfg, hand_entity, str(hand_cfg.entity_name), static)
    for visual in scene.visuals:
        visual_entity = f"{hand_entity}/meshes/{_safe_entity_name(visual.link_name)}_{visual.visual_index}"
        rr.log(
            visual_entity,
            rr.Transform3D(
                translation=visual.base_from_visual[:3, 3],
                mat3x3=visual.base_from_visual[:3, :3],
                scale=visual.scale,
                relation=rr.TransformRelation.ParentFromChild,
            ),
            static=static,
        )
        rr.log(visual_entity, rr.Asset3D(path=visual.mesh_path, albedo_factor=_hand_asset_albedo_factor(cfg)), static=True)


def _log_hand_local_frame(rr, cfg: DictConfig, hand_entity: str, hand_name: str, static: bool) -> None:
    length_m = float(cfg.visualize.hands.local_frame.length_m)
    radius_m = float(cfg.visualize.hands.local_frame.radius_m)
    if length_m <= 0.0:
        raise ValueError("visualize.hands.local_frame.length_m must be positive")
    if radius_m <= 0.0:
        raise ValueError("visualize.hands.local_frame.radius_m must be positive")
    entity = f"{hand_entity}/local_frame"
    rr.log(
        entity,
        rr.Transform3D(
            translation=(0.0, 0.0, 0.0),
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=static,
    )
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[(0.0, 0.0, 0.0)] * 3,
            vectors=[(length_m, 0.0, 0.0), (0.0, length_m, 0.0), (0.0, 0.0, length_m)],
            radii=[radius_m] * 3,
            colors=((255, 0, 0, 255), (0, 200, 0, 255), (0, 90, 255, 255)),
            labels=[f"{hand_name}_{axis}" for axis in ("X", "Y", "Z")],
            show_labels=bool(cfg.visualize.hands.local_frame.show_labels),
        ),
        static=static,
    )


def _hand_joint_positions_rad(hand_cfg: DictConfig) -> dict[str, float]:
    joint_positions = {}
    for joint_name, q_rad in hand_cfg.joint_positions_rad.items():
        joint_positions[str(joint_name)] = float(q_rad)
    return joint_positions


def _hand_package_paths(hand_cfg: DictConfig) -> dict[str, Path]:
    package_paths = {}
    for package_name, package_path in hand_cfg.package_paths.items():
        package_paths[str(package_name)] = Path(str(package_path))
    return package_paths


def _log_target_tcp_frames(rr, visualizer, cfg: DictConfig, result: PerceptionResult) -> None:
    target_rotation = visualizer["target_tcp_rotation"]
    for index, obj in enumerate(result.objects):
        rr.log(
            f"{visualizer['camera_parent_entity']}/target_tcp/{index:02d}_{_safe_entity_name(obj.label)}",
            rr.Transform3D(
                translation=obj.target_xyz_m,
                mat3x3=target_rotation,
                relation=rr.TransformRelation.ParentFromChild,
                axis_length=float(cfg.visualize.target_tcp.axis_length_m),
            ),
        )


def _log_ik_robot_preview(
    rr,
    cfg: DictConfig,
    target_xyzabc: tuple[float, float, float, float, float, float],
    kinematics: TianjiKinematics,
    reference_q_deg: tuple[float, float, float, float, float, float, float],
) -> None:
    target_q_deg = kinematics.ik_xyzabc_mm_deg(target_xyzabc, reference_q_deg)
    joint_positions_rad = _preview_joint_positions_rad(cfg, target_q_deg)
    scene = load_urdf_scene(
        cfg.visualize.robot.urdf_path,
        str(cfg.visualize.robot.base_link),
        joint_positions_rad=joint_positions_rad,
    )
    robot_entity = _robot_entity(cfg)
    _log_urdf_scene_visuals(rr, cfg, robot_entity, scene, static=False)
    _log_robot_sdk_tcp_frames(rr, cfg, scene, robot_entity, static=False)
    if bool(cfg.visualize.hands.enabled):
        _log_sharpa_hands(rr, cfg, scene, robot_entity, static=False)


def _select_preview_target(cfg: DictConfig, result: PerceptionResult) -> ObjectPoint3D | None:
    target_label = str(cfg.perception.robot_preview.target_label)
    candidates = tuple(obj for obj in result.objects if obj.label == target_label)
    if not candidates:
        return None
    return max(candidates, key=lambda obj: obj.score)


def _target_tcp_pose_mat_mm(
    target: ObjectPoint3D,
    rotation_base_from_tcp: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (rotation_base_from_tcp[0][0], rotation_base_from_tcp[0][1], rotation_base_from_tcp[0][2], target.target_xyz_m[0] * 1000.0),
        (rotation_base_from_tcp[1][0], rotation_base_from_tcp[1][1], rotation_base_from_tcp[1][2], target.target_xyz_m[1] * 1000.0),
        (rotation_base_from_tcp[2][0], rotation_base_from_tcp[2][1], rotation_base_from_tcp[2][2], target.target_xyz_m[2] * 1000.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _preview_joint_positions_rad(
    cfg: DictConfig,
    target_q_deg: tuple[float, float, float, float, float, float, float],
) -> dict[str, float]:
    joint_positions = _initial_pose_joint_positions_rad(cfg)
    side = str(cfg.perception.arm_side)
    if side == "left":
        joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.left_joint_names)
    elif side == "right":
        joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.right_joint_names)
    else:
        raise ValueError("perception.arm_side must be left or right")
    if len(joint_names) != len(target_q_deg):
        raise ValueError("preview joint name count mismatch")
    for joint_name, q_deg in zip(joint_names, target_q_deg, strict=True):
        joint_positions[joint_name] = deg_to_rad(q_deg)
    return joint_positions


def _log_camera_frustum(rr, cfg: DictConfig, camera_entity: str) -> None:
    width, height = (int(value) for value in cfg.visualize.camera_frustum.resolution)
    rr.log(
        f"{camera_entity}/image",
        rr.Pinhole(
            fov_y=float(np.deg2rad(float(cfg.visualize.camera_frustum.fov_y_deg))),
            aspect_ratio=float(width) / float(height),
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=float(cfg.visualize.camera_frustum.image_plane_distance_m),
        ),
        static=True,
    )


def _target_tcp_rotation_base_from_tcp(cfg: DictConfig) -> tuple[tuple[float, float, float], ...]:
    x_axis = _unit_vector3(cfg.perception.target_tcp.local_x_axis, "perception.target_tcp.local_x_axis")
    y_axis = _unit_vector3(cfg.perception.target_tcp.local_y_axis, "perception.target_tcp.local_y_axis")
    if abs(_dot(x_axis, y_axis)) > 1e-6:
        raise ValueError("perception.target_tcp local_x_axis and local_y_axis must be perpendicular")
    z_axis = _cross(x_axis, y_axis)
    if str(cfg.perception.target_tcp.orientation_frame) == "robot_world":
        base_from_world = _base_xyz_from_robot_world(cfg)
        x_axis = _transform_vector(base_from_world, x_axis)
        y_axis = _transform_vector(base_from_world, y_axis)
        z_axis = _transform_vector(base_from_world, z_axis)
    elif str(cfg.perception.target_tcp.orientation_frame) != "arm_base":
        raise ValueError("perception.target_tcp.orientation_frame must be arm_base or robot_world")
    return _rotation_from_axes(x_axis, y_axis, z_axis)


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


def _unit_vector3(values, name: str) -> tuple[float, float, float]:
    vector = np.asarray(offset_tuple(values), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    unit = vector / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    result = np.cross(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return (float(result[0]), float(result[1]), float(result[2]))


def _transform_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2]),
        float(matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2]),
        float(matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]),
    )


def _rotation_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    rotation = np.asarray(
        [
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]],
        ],
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("perception.target_tcp orientation axes must form an orthonormal basis")
    if float(np.linalg.det(rotation)) < 0.999999:
        raise ValueError("perception.target_tcp orientation axes must form a right-handed basis")
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
    )


def _initial_pose_joint_positions_rad(cfg: DictConfig) -> dict[str, float]:
    left_joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.left_joint_names)
    right_joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.right_joint_names)
    left_q_deg = tuple(float(value) for value in cfg.env.reset.targets.left.arm_q_deg)
    right_q_deg = tuple(float(value) for value in cfg.env.reset.targets.right.arm_q_deg)
    if len(left_joint_names) != len(left_q_deg):
        raise ValueError("left initial pose joint name count mismatch")
    if len(right_joint_names) != len(right_q_deg):
        raise ValueError("right initial pose joint name count mismatch")
    joint_positions = {}
    for joint_name, q_deg in zip(left_joint_names, left_q_deg, strict=True):
        joint_positions[joint_name] = deg_to_rad(q_deg)
    for joint_name, q_deg in zip(right_joint_names, right_q_deg, strict=True):
        joint_positions[joint_name] = deg_to_rad(q_deg)
    return joint_positions


def _initial_arm_q_deg(cfg: DictConfig, side: str) -> tuple[float, float, float, float, float, float, float]:
    if side == "left":
        q_deg = tuple(float(value) for value in cfg.env.reset.targets.left.arm_q_deg)
    elif side == "right":
        q_deg = tuple(float(value) for value in cfg.env.reset.targets.right.arm_q_deg)
    else:
        raise ValueError("perception.arm_side must be left or right")
    if len(q_deg) != 7:
        raise ValueError("initial arm_q_deg must have length 7")
    return (q_deg[0], q_deg[1], q_deg[2], q_deg[3], q_deg[4], q_deg[5], q_deg[6])


def _bbox_mins(detections: tuple[Detection2D, ...]) -> list[tuple[float, float]]:
    return [(detection.bbox.x1, detection.bbox.y1) for detection in detections]


def _bbox_sizes(detections: tuple[Detection2D, ...]) -> list[tuple[float, float]]:
    return [
        (detection.bbox.x2 - detection.bbox.x1, detection.bbox.y2 - detection.bbox.y1)
        for detection in detections
    ]


def _detection_label_summary(detections: tuple[Detection2D, ...]) -> str:
    if not detections:
        return "none"
    return ", ".join(f"{detection.label}:{detection.score:.2f}" for detection in detections)


def _object_label_summary(objects: tuple[ObjectPoint3D, ...]) -> str:
    if not objects:
        return "none"
    return ", ".join(f"{obj.label}:{obj.score:.2f}" for obj in objects)


def _format_xyz_m(xyz_m: tuple[float, float, float]) -> str:
    return f"({xyz_m[0]:.3f}, {xyz_m[1]:.3f}, {xyz_m[2]:.3f})"


def _format_xyzabc_mm_deg(xyzabc: tuple[float, float, float, float, float, float] | None) -> str:
    if xyzabc is None:
        return "none"
    return (
        f"({xyzabc[0]:.1f}, {xyzabc[1]:.1f}, {xyzabc[2]:.1f}, "
        f"{xyzabc[3]:.1f}, {xyzabc[4]:.1f}, {xyzabc[5]:.1f})"
    )


def _robot_entity(cfg: DictConfig) -> str:
    return f"{cfg.visualize.frame_name}/{cfg.visualize.robot.entity_name}"


def _safe_entity_name(name: str) -> str:
    return name.replace("/", "_")


def _asset_albedo_factor(cfg: DictConfig) -> tuple[int, int, int, int]:
    return _alpha_to_albedo_factor(float(cfg.visualize.robot.asset_alpha), "visualize.robot.asset_alpha")


def _hand_asset_albedo_factor(cfg: DictConfig) -> tuple[int, int, int, int]:
    return _alpha_to_albedo_factor(float(cfg.visualize.hands.asset_alpha), "visualize.hands.asset_alpha")


def _alpha_to_albedo_factor(alpha: float, config_name: str) -> tuple[int, int, int, int]:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{config_name} must be between 0 and 1")
    return (255, 255, 255, int(round(alpha * 255.0)))


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def _should_continue(frame_index: int, max_frames: int) -> bool:
    return max_frames <= 0 or frame_index < max_frames


def _should_log_frame(frame_index: int, log_every_n_frames: int) -> bool:
    return log_every_n_frames > 0 and frame_index % log_every_n_frames == 0


if __name__ == "__main__":
    main()
