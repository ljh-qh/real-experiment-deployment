from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from unirobot_infra.calibration.hand_eye import default_transform_path, load_intrinsics, load_transform
from unirobot_infra.calibration.urdf_visualization import UrdfScene, load_urdf_scene
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")
_AXIS_LABELS = ("X", "Y", "Z")
_AXIS_COLORS_RGBA = (
    (255, 0, 0, 255),
    (0, 200, 0, 255),
    (0, 90, 255, 255),
)


@dataclass(frozen=True)
class VisualizedCalibration:
    side: str
    mode: str
    transform_path: Path
    transform: np.ndarray
    camera_parent_link: str
    camera_frame_name: str


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/calibration/visualize_camera")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    calibrations = _load_visualized_calibrations(cfg)
    for calibration in calibrations:
        logger.info(
            "已加载标定外参 side=%s path=%s mode=%s parent=%s frame=%s",
            calibration.side,
            calibration.transform_path,
            calibration.mode,
            calibration.camera_parent_link,
            calibration.camera_frame_name,
        )
        logger.info("外参矩阵 side=%s\n%s", calibration.side, np.array2string(calibration.transform, precision=5, suppress_small=True))
        logger.info(
            "平移_m side=%s %s quat_xyzw=%s",
            calibration.side,
            _format_vec(calibration.transform[:3, 3]),
            _format_vec(Rotation.from_matrix(calibration.transform[:3, :3]).as_quat()),
        )
    if bool(cfg.visualize.rerun.enabled):
        _visualize_with_rerun(cfg, calibrations)


def _load_visualized_calibrations(cfg: DictConfig) -> tuple[VisualizedCalibration, ...]:
    calibrations = []
    for calibration_cfg in cfg.visualize.calibrations:
        mode = str(calibration_cfg.mode)
        transform_path = (
            default_transform_path(calibration_cfg.data_dir, mode)
            if calibration_cfg.transform_path is None
            else Path(str(calibration_cfg.transform_path))
        )
        calibrations.append(
            VisualizedCalibration(
                side=str(calibration_cfg.side),
                mode=mode,
                transform_path=transform_path,
                transform=load_transform(transform_path),
                camera_parent_link=str(calibration_cfg.camera_parent_link),
                camera_frame_name=str(calibration_cfg.camera_frame_name),
            )
        )
    if not calibrations:
        raise ValueError("visualize.calibrations must contain at least one calibration")
    return tuple(calibrations)


def _visualize_with_rerun(cfg: DictConfig, calibrations: tuple[VisualizedCalibration, ...]) -> None:
    import rerun as rr

    rr.init(str(cfg.visualize.rerun.app_id), spawn=bool(cfg.visualize.rerun.spawn))
    frame_name = str(cfg.visualize.frame_name)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        frame_name,
        rr.Transform3D(
            translation=(0.0, 0.0, 0.0),
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    robot_scene = None
    logged_link_frames: set[str] = set()
    if bool(cfg.visualize.robot.enabled):
        robot_scene = _log_urdf_robot(rr, cfg, frame_name)
        logged_link_frames = _log_robot_base_frames(rr, cfg, frame_name, robot_scene)
        logged_link_frames.update(_log_robot_sdk_tcp_frames(rr, cfg, frame_name, robot_scene))
        get_logger(__name__).info(
            "坐标轴颜色约定: X=红色 Y=绿色 Z=蓝色; base_frames=%s; robot_asset_alpha=%.2f",
            ",".join(str(name) for name in cfg.visualize.robot.base_frame_links),
            float(cfg.visualize.robot.asset_alpha),
        )
    if bool(cfg.visualize.hands.enabled):
        if robot_scene is None:
            raise ValueError("visualize.hands.enabled requires visualize.robot.enabled=true")
        _log_sharpa_hands(rr, cfg, frame_name, robot_scene)
    for calibration in calibrations:
        _log_camera_calibration(rr, cfg, frame_name, robot_scene, logged_link_frames, calibration)


def _log_camera_calibration(
    rr,
    cfg: DictConfig,
    frame_name: str,
    robot_scene: UrdfScene | None,
    logged_link_frames: set[str],
    calibration: VisualizedCalibration,
) -> None:
    camera_parent_entity = _camera_parent_entity(
        rr,
        cfg,
        frame_name,
        robot_scene,
        logged_link_frames,
        calibration.camera_parent_link,
    )
    camera_entity = f"{camera_parent_entity}/{calibration.camera_frame_name}"
    rr.log(
        camera_entity,
        rr.Transform3D(
            translation=calibration.transform[:3, 3],
            mat3x3=calibration.transform[:3, :3],
            relation=_camera_transform_relation(rr, calibration.mode),
        ),
        static=True,
    )
    if bool(cfg.visualize.camera_frustum.enabled):
        _log_camera_frustum(rr, cfg, camera_entity)


def _log_urdf_robot(rr, cfg: DictConfig, frame_name: str) -> UrdfScene:
    joint_positions_rad = _initial_pose_joint_positions_rad(cfg)
    scene = load_urdf_scene(
        cfg.visualize.robot.urdf_path,
        str(cfg.visualize.robot.base_link),
        joint_positions_rad=joint_positions_rad,
    )
    robot_entity = _robot_entity(cfg, frame_name)
    rr.log(
        robot_entity,
        rr.Transform3D(
            translation=(0.0, 0.0, 0.0),
            mat3x3=np.eye(3),
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    logged_visuals = 0
    hidden_visuals = 0
    for visual in scene.visuals:
        if _robot_visual_hidden(cfg, visual.link_name):
            hidden_visuals += 1
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
            static=True,
        )
        rr.log(visual_entity, rr.Asset3D(path=visual.mesh_path, albedo_factor=_asset_albedo_factor(cfg)), static=True)
        logged_visuals += 1
    get_logger(__name__).info(
        "已加载 URDF 机器人 urdf=%s base_link=%s links=%s visuals=%s hidden_visuals=%s initial_joints=%s",
        scene.urdf_path,
        scene.base_link,
        len(scene.link_transforms),
        logged_visuals,
        hidden_visuals,
        len(joint_positions_rad),
    )
    return scene


def _robot_visual_hidden(cfg: DictConfig, link_name: str) -> bool:
    if not bool(cfg.visualize.hands.enabled):
        return False
    for prefix in cfg.visualize.hands.hide_robot_link_prefixes:
        if link_name.startswith(str(prefix)):
            return True
    return False


def _log_sharpa_hands(rr, cfg: DictConfig, frame_name: str, robot_scene: UrdfScene) -> None:
    _log_sharpa_hand(rr, cfg, frame_name, robot_scene, cfg.visualize.hands.left, "left")
    _log_sharpa_hand(rr, cfg, frame_name, robot_scene, cfg.visualize.hands.right, "right")


def _log_sharpa_hand(
    rr,
    cfg: DictConfig,
    frame_name: str,
    robot_scene: UrdfScene,
    hand_cfg: DictConfig,
    side: str,
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
    robot_entity = _robot_entity(cfg, frame_name)
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
        static=True,
    )
    if bool(cfg.visualize.hands.local_frame.enabled):
        _log_hand_local_frame(rr, cfg, hand_entity, str(hand_cfg.entity_name))
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
            static=True,
        )
        rr.log(visual_entity, rr.Asset3D(path=visual.mesh_path, albedo_factor=_hand_asset_albedo_factor(cfg)), static=True)
    get_logger(__name__).info(
        "已加载 Sharpa 灵巧手 side=%s urdf=%s base_link=%s parent_link=%s links=%s visuals=%s",
        side,
        scene.urdf_path,
        scene.base_link,
        parent_link,
        len(scene.link_transforms),
        len(scene.visuals),
    )


def _log_hand_local_frame(rr, cfg: DictConfig, hand_entity: str, hand_name: str) -> None:
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
        static=True,
    )
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[(0.0, 0.0, 0.0)] * 3,
            vectors=[(length_m, 0.0, 0.0), (0.0, length_m, 0.0), (0.0, 0.0, length_m)],
            radii=[radius_m] * 3,
            colors=_AXIS_COLORS_RGBA,
            labels=[f"{hand_name}_{axis}" for axis in _AXIS_LABELS],
            show_labels=bool(cfg.visualize.hands.local_frame.show_labels),
        ),
        static=True,
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


def _pose_from_xyz_rpy(xyz_cfg, rpy_cfg) -> np.ndarray:
    xyz = tuple(float(value) for value in xyz_cfg)
    rpy = tuple(float(value) for value in rpy_cfg)
    if len(xyz) != 3:
        raise ValueError("hand mount xyz must have length 3")
    if len(rpy) != 3:
        raise ValueError("hand mount rpy must have length 3")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _log_camera_frustum(rr, cfg: DictConfig, camera_entity: str) -> None:
    if cfg.visualize.intrinsics_path is not None:
        intrinsics = load_intrinsics(cfg.visualize.intrinsics_path)
        rr.log(
            f"{camera_entity}/image",
            rr.Pinhole(
                image_from_camera=intrinsics.camera_matrix,
                width=intrinsics.resolution[0],
                height=intrinsics.resolution[1],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=float(cfg.visualize.image_plane_distance_m),
            ),
            static=True,
        )
        return
    if str(cfg.visualize.camera_frustum.source) != "approximate":
        raise ValueError("visualize.camera_frustum.source must be approximate when intrinsics_path is null")
    width, height = _frustum_resolution(cfg)
    fov_y_deg = float(cfg.visualize.camera_frustum.fov_y_deg)
    if fov_y_deg <= 0.0:
        raise ValueError("visualize.camera_frustum.fov_y_deg must be positive")
    rr.log(
        f"{camera_entity}/image",
        rr.Pinhole(
            fov_y=float(np.deg2rad(fov_y_deg)),
            aspect_ratio=float(width) / float(height),
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=float(cfg.visualize.image_plane_distance_m),
        ),
        static=True,
    )
    get_logger(__name__).info(
        "未配置相机内参，使用近似视锥 resolution=%sx%s fov_y_deg=%.2f",
        width,
        height,
        fov_y_deg,
    )


def _frustum_resolution(cfg: DictConfig) -> tuple[int, int]:
    resolution = tuple(int(value) for value in cfg.visualize.camera_frustum.resolution)
    if len(resolution) != 2:
        raise ValueError("visualize.camera_frustum.resolution must have length 2")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("visualize.camera_frustum.resolution values must be positive")
    return (resolution[0], resolution[1])


def _initial_pose_joint_positions_rad(cfg: DictConfig) -> dict[str, float]:
    if not bool(cfg.visualize.robot.initial_pose.enabled):
        return {}
    left_joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.left_joint_names)
    right_joint_names = tuple(str(name) for name in cfg.visualize.robot.initial_pose.right_joint_names)
    left_q_deg = tuple(float(value) for value in cfg.env.reset.targets.left.arm_q_deg)
    right_q_deg = tuple(float(value) for value in cfg.env.reset.targets.right.arm_q_deg)
    if len(left_joint_names) != len(left_q_deg):
        raise ValueError("visualize.robot.initial_pose.left_joint_names length must match env.reset.targets.left.arm_q_deg")
    if len(right_joint_names) != len(right_q_deg):
        raise ValueError("visualize.robot.initial_pose.right_joint_names length must match env.reset.targets.right.arm_q_deg")
    joint_positions = {}
    for joint_name, q_deg in zip(left_joint_names, left_q_deg, strict=True):
        joint_positions[joint_name] = deg_to_rad(q_deg)
    for joint_name, q_deg in zip(right_joint_names, right_q_deg, strict=True):
        joint_positions[joint_name] = deg_to_rad(q_deg)
    return joint_positions


def _log_robot_base_frames(rr, cfg: DictConfig, frame_name: str, robot_scene: UrdfScene) -> set[str]:
    robot_entity = _robot_entity(cfg, frame_name)
    logged_links = set()
    for link_name in cfg.visualize.robot.base_frame_links:
        name = str(link_name)
        if name not in robot_scene.link_transforms:
            raise ValueError(f"visualize.robot.base_frame_links contains unknown URDF link: {name}")
        _log_robot_link_frame(
            rr,
            robot_entity,
            name,
            robot_scene.link_transforms[name],
            float(cfg.visualize.robot.base_frame_axis_length_m),
        )
        if bool(cfg.visualize.robot.base_frame_helper_axes.enabled):
            _log_helper_axes(rr, cfg, robot_entity, name, robot_scene.link_transforms[name])
        logged_links.add(name)
    return logged_links


def _log_robot_sdk_tcp_frames(rr, cfg: DictConfig, frame_name: str, robot_scene: UrdfScene) -> set[str]:
    if not bool(cfg.visualize.robot.sdk_tcp_frames.enabled):
        return set()
    robot_entity = _robot_entity(cfg, frame_name)
    axis_length_m = float(cfg.visualize.robot.sdk_tcp_frames.axis_length_m)
    if axis_length_m <= 0.0:
        raise ValueError("visualize.robot.sdk_tcp_frames.axis_length_m must be positive")
    logged_frames = {
        _log_robot_sdk_tcp_frame(rr, robot_entity, "left", cfg.visualize.robot.sdk_tcp_frames.left, robot_scene, axis_length_m),
        _log_robot_sdk_tcp_frame(rr, robot_entity, "right", cfg.visualize.robot.sdk_tcp_frames.right, robot_scene, axis_length_m),
    }
    get_logger(__name__).info(
        "已绘制 SDK TCP 坐标系 frames=%s axis_length_m=%.3f",
        ",".join(sorted(logged_frames)),
        axis_length_m,
    )
    return logged_frames


def _log_robot_sdk_tcp_frame(
    rr,
    robot_entity: str,
    side: str,
    frame_cfg: DictConfig,
    robot_scene: UrdfScene,
    axis_length_m: float,
) -> str:
    parent_link = str(frame_cfg.parent_link)
    tcp_frame_name = str(frame_cfg.frame_name)
    if parent_link not in robot_scene.link_transforms:
        raise ValueError(f"visualize.robot.sdk_tcp_frames.{side}.parent_link={parent_link} not found in URDF")
    base_from_tcp = robot_scene.link_transforms[parent_link] @ _pose_from_xyz_rpy(
        frame_cfg.parent_from_tcp.xyz,
        frame_cfg.parent_from_tcp.rpy,
    )
    _log_robot_link_frame(
        rr,
        robot_entity,
        tcp_frame_name,
        base_from_tcp,
        axis_length_m,
    )
    return tcp_frame_name


def _log_helper_axes(rr, cfg: DictConfig, robot_entity: str, link_name: str, base_from_link: np.ndarray) -> None:
    length_m = float(cfg.visualize.robot.base_frame_helper_axes.length_m)
    radius_m = float(cfg.visualize.robot.base_frame_helper_axes.radius_m)
    if length_m <= 0.0:
        raise ValueError("visualize.robot.base_frame_helper_axes.length_m must be positive")
    if radius_m <= 0.0:
        raise ValueError("visualize.robot.base_frame_helper_axes.radius_m must be positive")
    entity = f"{robot_entity}/helper_axes/{_safe_entity_name(link_name)}"
    rr.log(
        entity,
        rr.Transform3D(
            translation=base_from_link[:3, 3],
            mat3x3=base_from_link[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[(0.0, 0.0, 0.0)] * 3,
            vectors=[(length_m, 0.0, 0.0), (0.0, length_m, 0.0), (0.0, 0.0, length_m)],
            radii=[radius_m] * 3,
            colors=_AXIS_COLORS_RGBA,
            labels=[f"{link_name}_{axis}" for axis in _AXIS_LABELS],
            show_labels=bool(cfg.visualize.robot.base_frame_helper_axes.show_labels),
        ),
        static=True,
    )


def _camera_parent_entity(
    rr,
    cfg: DictConfig,
    frame_name: str,
    robot_scene: UrdfScene | None,
    logged_link_frames: set[str],
    camera_parent_link: str,
) -> str:
    if robot_scene is None:
        return frame_name
    robot_entity = _robot_entity(cfg, frame_name)
    if camera_parent_link in logged_link_frames:
        return f"{robot_entity}/frames/{_safe_entity_name(camera_parent_link)}"
    if camera_parent_link not in robot_scene.link_transforms:
        raise ValueError(f"camera_parent_link={camera_parent_link} not found in URDF or logged frames")
    if camera_parent_link not in logged_link_frames:
        _log_robot_link_frame(
            rr,
            robot_entity,
            camera_parent_link,
            robot_scene.link_transforms[camera_parent_link],
            float(cfg.visualize.robot.link_axis_length_m),
        )
    if bool(cfg.visualize.robot.log_link_frames):
        for link_name, base_from_link in robot_scene.link_transforms.items():
            if link_name not in logged_link_frames and link_name != camera_parent_link:
                _log_robot_link_frame(
                    rr,
                    robot_entity,
                    link_name,
                    base_from_link,
                    float(cfg.visualize.robot.link_axis_length_m),
                )
    return f"{robot_entity}/frames/{_safe_entity_name(camera_parent_link)}"


def _log_robot_link_frame(
    rr,
    robot_entity: str,
    link_name: str,
    base_from_link: np.ndarray,
    axis_length_m: float,
) -> None:
    rr.log(
        f"{robot_entity}/frames/{_safe_entity_name(link_name)}",
        rr.Transform3D(
            translation=base_from_link[:3, 3],
            mat3x3=base_from_link[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            axis_length=axis_length_m,
        ),
        static=True,
    )


def _camera_transform_relation(rr, mode: str):
    if mode == "eye-to-hand":
        return rr.TransformRelation.ChildFromParent
    if mode == "eye-in-hand":
        return rr.TransformRelation.ParentFromChild
    raise ValueError("calibration.mode must be eye-to-hand or eye-in-hand")


def _asset_albedo_factor(cfg: DictConfig) -> tuple[int, int, int, int]:
    return _alpha_to_albedo_factor(float(cfg.visualize.robot.asset_alpha), "visualize.robot.asset_alpha")


def _hand_asset_albedo_factor(cfg: DictConfig) -> tuple[int, int, int, int]:
    return _alpha_to_albedo_factor(float(cfg.visualize.hands.asset_alpha), "visualize.hands.asset_alpha")


def _alpha_to_albedo_factor(alpha: float, config_name: str) -> tuple[int, int, int, int]:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{config_name} must be between 0 and 1")
    return (255, 255, 255, int(round(alpha * 255.0)))


def _robot_entity(cfg: DictConfig, frame_name: str) -> str:
    return f"{frame_name}/{cfg.visualize.robot.entity_name}"


def _safe_entity_name(name: str) -> str:
    return name.replace("/", "_")


def _format_vec(values) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


if __name__ == "__main__":
    main()
