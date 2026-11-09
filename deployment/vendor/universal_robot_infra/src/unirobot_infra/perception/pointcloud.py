from __future__ import annotations

import numpy as np

from unirobot_infra.calibration.hand_eye import CameraIntrinsics


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape HxW")
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError("rgb and depth_m resolution mismatch")
    if int(stride) <= 0:
        raise ValueError("pointcloud stride must be positive")

    sampled_depth = depth_m[:: int(stride), :: int(stride)]
    sampled_rgb = rgb[:: int(stride), :: int(stride)]
    height, width = sampled_depth.shape
    y_index, x_index = np.mgrid[0:height, 0:width]
    u = x_index.astype(np.float64) * float(stride)
    v = y_index.astype(np.float64) * float(stride)
    z = sampled_depth.astype(np.float64)

    valid = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    camera_matrix = intrinsics.camera_matrix
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera intrinsics fx/fy must be positive")

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)[valid]
    colors = sampled_rgb[valid]
    return points.astype(np.float32), colors.astype(np.uint8)
