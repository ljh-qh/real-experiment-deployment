from __future__ import annotations

import numpy as np

from unirobot_infra.calibration.hand_eye import CameraIntrinsics
from unirobot_infra.perception.depth_projection import project_detection_center_to_camera
from unirobot_infra.perception.frame_transform import RigidTransform, add_xyz_offset, transform_vector
from unirobot_infra.perception.types import Detection2D, ObjectPoint3D


def localize_detections_3d(
    detections: tuple[Detection2D, ...],
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    base_from_camera: RigidTransform,
    offset_m: tuple[float, float, float],
    offset_frame: str,
    base_xyz_from_robot_world: tuple[tuple[float, float, float], ...],
    min_depth_m: float,
    max_depth_m: float,
    depth_search_radius_px: int,
) -> tuple[ObjectPoint3D, ...]:
    base_offset_m = target_offset_in_base(
        offset_m=offset_m,
        offset_frame=offset_frame,
        base_xyz_from_robot_world=base_xyz_from_robot_world,
    )
    objects = []
    for detection in detections:
        camera_xyz_m = project_detection_center_to_camera(
            detection=detection,
            depth_m=depth_m,
            intrinsics=intrinsics,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            depth_search_radius_px=depth_search_radius_px,
        )
        if camera_xyz_m is None:
            continue
        base_xyz_m = base_from_camera.transform_point(camera_xyz_m)
        target_xyz_m = add_xyz_offset(base_xyz_m, base_offset_m)
        objects.append(
            ObjectPoint3D(
                label=detection.label,
                score=detection.score,
                bbox=detection.bbox,
                camera_xyz_m=camera_xyz_m,
                base_xyz_m=base_xyz_m,
                target_xyz_m=target_xyz_m,
            )
        )
    return tuple(objects)


def target_offset_in_base(
    offset_m: tuple[float, float, float],
    offset_frame: str,
    base_xyz_from_robot_world: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    if offset_frame == "arm_base":
        return offset_m
    if offset_frame == "robot_world":
        return transform_vector(base_xyz_from_robot_world, offset_m)
    raise ValueError("offset_frame must be arm_base or robot_world")
