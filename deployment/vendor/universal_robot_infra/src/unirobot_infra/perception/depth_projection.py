from __future__ import annotations

import math

import numpy as np

from unirobot_infra.calibration.hand_eye import CameraIntrinsics
from unirobot_infra.perception.types import Detection2D


def project_detection_center_to_camera(
    detection: Detection2D,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth_m: float,
    max_depth_m: float,
    depth_search_radius_px: int,
) -> tuple[float, float, float] | None:
    u_float, v_float = detection.bbox.center_xy
    u = int(round(u_float))
    v = int(round(v_float))
    _validate_pixel(depth_m, u, v)
    z = _valid_center_depth_or_nearby_m(
        depth_m=depth_m,
        u=u,
        v=v,
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
        depth_search_radius_px=int(depth_search_radius_px),
    )
    if z is None:
        return None
    camera_matrix = intrinsics.camera_matrix
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera intrinsics fx/fy must be positive")
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return (float(x), float(y), float(z))


def _valid_center_depth_or_nearby_m(
    depth_m: np.ndarray,
    u: int,
    v: int,
    min_depth_m: float,
    max_depth_m: float,
    depth_search_radius_px: int,
) -> float | None:
    if depth_search_radius_px < 0:
        raise ValueError("depth_search_radius_px must be non-negative")
    center_depth = float(depth_m[v, u])
    if _valid_depth(center_depth, min_depth_m, max_depth_m):
        return center_depth
    if depth_search_radius_px == 0:
        return None

    height, width = depth_m.shape
    u1 = max(0, u - depth_search_radius_px)
    u2 = min(width, u + depth_search_radius_px + 1)
    v1 = max(0, v - depth_search_radius_px)
    v2 = min(height, v + depth_search_radius_px + 1)
    window = depth_m[v1:v2, u1:u2]
    valid_mask = np.isfinite(window) & (window > 0.0) & (window >= min_depth_m) & (window <= max_depth_m)
    valid_depths = window[valid_mask]
    if valid_depths.size == 0:
        return None
    return float(np.median(valid_depths))


def _valid_depth(depth_m: float, min_depth_m: float, max_depth_m: float) -> bool:
    return math.isfinite(depth_m) and depth_m > 0.0 and min_depth_m <= depth_m <= max_depth_m


def _validate_pixel(depth_m: np.ndarray, u: int, v: int) -> None:
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape HxW")
    height, width = depth_m.shape
    if u < 0 or u >= width or v < 0 or v >= height:
        raise ValueError(f"bbox center pixel out of image u={u} v={v} width={width} height={height}")
