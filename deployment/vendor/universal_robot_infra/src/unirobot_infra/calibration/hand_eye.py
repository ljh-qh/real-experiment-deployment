from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

CALIBRATION_MODES = frozenset(("eye-to-hand", "eye-in-hand"))
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass(frozen=True)
class CheckerboardConfig:
    cols: int
    rows: int
    square_size_m: float

    @property
    def size(self) -> tuple[int, int]:
        if self.cols <= 0:
            raise ValueError("checkerboard.cols must be positive")
        if self.rows <= 0:
            raise ValueError("checkerboard.rows must be positive")
        if self.square_size_m <= 0.0:
            raise ValueError("checkerboard.square_size_m must be positive")
        return (self.cols, self.rows)


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    resolution: tuple[int, int]


@dataclass(frozen=True)
class CalibrationSample:
    record_index: int
    image_path: Path
    base_from_ee: np.ndarray
    camera_from_board: np.ndarray


@dataclass(frozen=True)
class HandEyeSolveResult:
    transform: np.ndarray
    samples_used: int
    mode: str
    output_stem: str


@dataclass(frozen=True)
class VerificationStats:
    sample_count: int
    mean_rotation_error_deg: float
    max_rotation_error_deg: float
    mean_translation_error_mm: float
    max_translation_error_mm: float
    mode: str
    checked_transform_name: str


@dataclass(frozen=True)
class VerificationQualityThreshold:
    level: str
    label: str
    mean_translation_error_mm: float
    max_translation_error_mm: float
    mean_rotation_error_deg: float
    max_rotation_error_deg: float
    recommendation: str


@dataclass(frozen=True)
class VerificationQuality:
    level: str
    label: str
    recommendation: str


_VERIFICATION_QUALITY_THRESHOLDS = (
    VerificationQualityThreshold(
        level="excellent",
        label="优秀",
        mean_translation_error_mm=2.0,
        max_translation_error_mm=5.0,
        mean_rotation_error_deg=0.2,
        max_rotation_error_deg=0.5,
        recommendation="可作为高置信外参使用",
    ),
    VerificationQualityThreshold(
        level="good",
        label="良好",
        mean_translation_error_mm=5.0,
        max_translation_error_mm=15.0,
        mean_rotation_error_deg=0.5,
        max_rotation_error_deg=1.5,
        recommendation="可用于当前 demo，接近阶段保留安全距离",
    ),
    VerificationQualityThreshold(
        level="marginal",
        label="临界",
        mean_translation_error_mm=10.0,
        max_translation_error_mm=25.0,
        mean_rotation_error_deg=1.0,
        max_rotation_error_deg=3.0,
        recommendation="精细抓取前建议剔除异常样本或重新标定",
    ),
)


def pose7_to_matrix(pose: list[float] | tuple[float, ...]) -> np.ndarray:
    values = tuple(float(value) for value in pose)
    if len(values) != 7:
        raise ValueError("pose must have length 7: [x, y, z, qx, qy, qz, qw]")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(values[3:7]).as_matrix()
    transform[:3, 3] = np.asarray(values[:3], dtype=np.float64)
    return transform


def matrix_to_pose7(transform: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    matrix = _validated_transform_matrix(transform, "transform")
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return (
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = _read_json(Path(path))
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    resolution_values = tuple(int(value) for value in data["resolution"])
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics.camera_matrix must have shape 3x3")
    if len(resolution_values) != 2:
        raise ValueError("intrinsics.resolution must have length 2")
    if resolution_values[0] <= 0 or resolution_values[1] <= 0:
        raise ValueError("intrinsics.resolution values must be positive")
    return CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        resolution=(resolution_values[0], resolution_values[1]),
    )


def save_intrinsics(path: str | Path, intrinsics: CameraIntrinsics) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "camera_matrix": intrinsics.camera_matrix.tolist(),
                "dist_coeffs": intrinsics.dist_coeffs.reshape(-1).tolist(),
                "resolution": list(intrinsics.resolution),
            },
            file,
            indent=2,
        )


def calibrate_intrinsics_from_checkerboards(
    images_bgr: list[np.ndarray],
    checkerboard: CheckerboardConfig,
) -> tuple[CameraIntrinsics, tuple[int, ...]]:
    object_points = _checkerboard_object_points(checkerboard).astype(np.float32)
    all_object_points = []
    all_image_points = []
    used_indices = []
    for index, image in enumerate(images_bgr):
        gray = _gray_image(image)
        ok, corners = cv2.findChessboardCorners(gray, checkerboard.size)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        all_object_points.append(object_points)
        all_image_points.append(corners)
        used_indices.append(index)

    if len(all_object_points) < 3:
        raise ValueError(f"Need at least 3 checkerboard detections for intrinsics, got {len(all_object_points)}")

    height, width = images_bgr[0].shape[:2]
    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        (width, height),
        None,
        None,
    )
    return (
        CameraIntrinsics(
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
            resolution=(int(width), int(height)),
        ),
        tuple(used_indices),
    )


def detect_checkerboard_pose(
    image_bgr: np.ndarray,
    checkerboard: CheckerboardConfig,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    gray = _gray_image(image_bgr)
    ok, corners = cv2.findChessboardCorners(
        gray,
        checkerboard.size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    ok, rvec, tvec = cv2.solvePnP(
        _checkerboard_object_points(checkerboard),
        corners,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for checkerboard")
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_board = np.eye(4, dtype=np.float64)
    camera_from_board[:3, :3] = rotation
    camera_from_board[:3, 3] = tvec.reshape(3)
    return camera_from_board


def load_calibration_samples(
    data_dir: str | Path,
    camera_name: str | None,
    checkerboard: CheckerboardConfig,
    intrinsics_path: str | Path | None,
) -> tuple[tuple[CalibrationSample, ...], CameraIntrinsics, str]:
    data_path = Path(data_dir)
    records = load_pose_records(data_path / "poses.json")
    selected_camera = select_camera_name(records, camera_name)
    images, base_from_ee_list, record_indices, image_paths = _load_record_images_and_poses(
        data_path,
        records,
        selected_camera,
    )
    if intrinsics_path is None:
        intrinsics, _ = calibrate_intrinsics_from_checkerboards(images, checkerboard)
    else:
        intrinsics = load_intrinsics(intrinsics_path)

    samples = []
    for image, base_from_ee, record_index, image_path in zip(images, base_from_ee_list, record_indices, image_paths, strict=True):
        camera_from_board = detect_checkerboard_pose(image, checkerboard, intrinsics)
        if camera_from_board is None:
            continue
        samples.append(
            CalibrationSample(
                record_index=record_index,
                image_path=image_path,
                base_from_ee=base_from_ee,
                camera_from_board=camera_from_board,
            )
        )
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 checkerboard detections, got {len(samples)}")
    return tuple(samples), intrinsics, selected_camera


def load_pose_records(path: str | Path) -> list[dict[str, Any]]:
    records = _read_json(Path(path))
    if not isinstance(records, list):
        raise ValueError("poses.json must contain a list")
    for index, record in enumerate(records):
        if "cartesian_pose" not in record:
            raise ValueError(f"poses.json record {index} missing cartesian_pose")
        if "images" not in record:
            raise ValueError(f"poses.json record {index} missing images")
    return records


def select_camera_name(records: list[dict[str, Any]], camera_name: str | None) -> str:
    camera_names = []
    for record in records:
        for name in record["images"]:
            if name not in camera_names:
                camera_names.append(str(name))
    if not camera_names:
        raise ValueError("poses.json records do not contain images")
    if camera_name is not None:
        selected = str(camera_name)
        if selected not in camera_names:
            raise ValueError(f"camera {selected} not found in poses.json records")
        return selected
    if len(camera_names) != 1:
        raise ValueError(f"multiple cameras found in poses.json: {camera_names}; set calibration.camera")
    return camera_names[0]


def solve_hand_eye(
    samples: tuple[CalibrationSample, ...],
    mode: str,
    method: str,
) -> HandEyeSolveResult:
    _validate_mode(mode)
    if method not in HAND_EYE_METHODS:
        raise ValueError(f"unsupported hand-eye method: {method}")
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 samples, got {len(samples)}")

    rotation_gripper_to_base = []
    translation_gripper_to_base = []
    rotation_target_to_camera = []
    translation_target_to_camera = []
    for sample in samples:
        base_from_ee = _validated_transform_matrix(sample.base_from_ee, "base_from_ee")
        camera_from_board = _validated_transform_matrix(sample.camera_from_board, "camera_from_board")
        if mode == "eye-to-hand":
            board_from_camera = np.linalg.inv(camera_from_board)
            rotation_gripper_to_base.append(board_from_camera[:3, :3])
            translation_gripper_to_base.append(board_from_camera[:3, 3:])
            rotation_target_to_camera.append(base_from_ee[:3, :3])
            translation_target_to_camera.append(base_from_ee[:3, 3:])
        else:
            rotation_gripper_to_base.append(base_from_ee[:3, :3])
            translation_gripper_to_base.append(base_from_ee[:3, 3:])
            rotation_target_to_camera.append(camera_from_board[:3, :3])
            translation_target_to_camera.append(camera_from_board[:3, 3:])

    rotation, translation = cv2.calibrateHandEye(
        R_gripper2base=rotation_gripper_to_base,
        t_gripper2base=translation_gripper_to_base,
        R_target2cam=rotation_target_to_camera,
        t_target2cam=translation_target_to_camera,
        method=HAND_EYE_METHODS[method],
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return HandEyeSolveResult(
        transform=transform,
        samples_used=len(samples),
        mode=mode,
        output_stem=transform_output_stem(mode),
    )


def verify_hand_eye(
    samples: tuple[CalibrationSample, ...],
    transform: np.ndarray,
    mode: str,
) -> VerificationStats:
    _validate_mode(mode)
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples for verification, got {len(samples)}")
    calibrated_transform = _validated_transform_matrix(transform, "transform")
    constant_transforms = []
    for sample in samples:
        base_from_ee = _validated_transform_matrix(sample.base_from_ee, "base_from_ee")
        camera_from_board = _validated_transform_matrix(sample.camera_from_board, "camera_from_board")
        if mode == "eye-to-hand":
            camera_from_base = calibrated_transform
            ee_from_board = np.linalg.inv(base_from_ee) @ np.linalg.inv(camera_from_base) @ camera_from_board
            constant_transforms.append(ee_from_board)
        else:
            ee_from_camera = calibrated_transform
            base_from_board = base_from_ee @ ee_from_camera @ camera_from_board
            constant_transforms.append(base_from_board)

    rotation_errors_deg, translation_errors_mm = _transform_spread_errors(constant_transforms)
    return VerificationStats(
        sample_count=len(constant_transforms),
        mean_rotation_error_deg=float(np.mean(rotation_errors_deg)),
        max_rotation_error_deg=float(np.max(rotation_errors_deg)),
        mean_translation_error_mm=float(np.mean(translation_errors_mm)),
        max_translation_error_mm=float(np.max(translation_errors_mm)),
        mode=mode,
        checked_transform_name="T_ee_board" if mode == "eye-to-hand" else "T_base_board",
    )


def save_transform(path_stem: str | Path, transform: np.ndarray) -> tuple[Path, Path]:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    matrix = _validated_transform_matrix(transform, "transform")
    npy_path = stem.with_suffix(".npy")
    json_path = stem.with_suffix(".json")
    np.save(npy_path, matrix)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(matrix.tolist(), file, indent=2)
    return npy_path, json_path


def load_transform(path: str | Path) -> np.ndarray:
    transform_path = Path(path)
    if transform_path.suffix == ".npy":
        return _validated_transform_matrix(np.load(transform_path), "transform")
    if transform_path.suffix == ".json":
        return _validated_transform_matrix(np.asarray(_read_json(transform_path), dtype=np.float64), "transform")
    raise ValueError(f"unsupported transform file suffix: {transform_path.suffix}")


def transform_output_stem(mode: str) -> str:
    _validate_mode(mode)
    if mode == "eye-to-hand":
        return "T_cam2base"
    return "T_cam2ee"


def transform_label(mode: str) -> str:
    _validate_mode(mode)
    if mode == "eye-to-hand":
        return "T_cam2base (camera_from_base)"
    return "T_cam2ee (ee_from_camera)"


def default_transform_path(data_dir: str | Path, mode: str) -> Path:
    return Path(data_dir) / f"{transform_output_stem(mode)}.npy"


def format_transform_report(result: HandEyeSolveResult) -> str:
    transform = _validated_transform_matrix(result.transform, "transform")
    angle_deg = Rotation.from_matrix(transform[:3, :3]).magnitude() * 180.0 / np.pi
    distance_cm = float(np.linalg.norm(transform[:3, 3]) * 100.0)
    return (
        f"标定求解结果\n"
        f"外参: {transform_label(result.mode)}\n"
        f"矩阵:\n"
        f"{np.array2string(transform, precision=5, suppress_small=True)}\n"
        f"旋转角={angle_deg:.3f} deg  平移距离={distance_cm:.3f} cm  样本数={result.samples_used}"
    )


def format_verification_report(stats: VerificationStats) -> str:
    quality = assess_verification_quality(stats)
    return (
        f"标定验证结果\n"
        f"一致性检查对象: {stats.checked_transform_name}\n"
        f"样本数: {stats.sample_count}\n"
        f"旋转误差: mean={stats.mean_rotation_error_deg:.3f} deg  max={stats.max_rotation_error_deg:.3f} deg\n"
        f"平移误差: mean={stats.mean_translation_error_mm:.3f} mm  max={stats.max_translation_error_mm:.3f} mm\n"
        f"质量等级: {quality.label}\n"
        f"建议: {quality.recommendation}"
    )


def assess_verification_quality(stats: VerificationStats) -> VerificationQuality:
    for threshold in _VERIFICATION_QUALITY_THRESHOLDS:
        if _verification_stats_within_threshold(stats, threshold):
            return VerificationQuality(
                level=threshold.level,
                label=threshold.label,
                recommendation=threshold.recommendation,
            )
    return VerificationQuality(
        level="recalibrate",
        label="需重新标定",
        recommendation="重新标定后再用于机器人抓取",
    )


def verification_quality_thresholds() -> tuple[VerificationQualityThreshold, ...]:
    return _VERIFICATION_QUALITY_THRESHOLDS


def _load_record_images_and_poses(
    data_dir: Path,
    records: list[dict[str, Any]],
    camera_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    images = []
    poses = []
    record_indices = []
    image_paths = []
    for index, record in enumerate(records):
        if camera_name not in record["images"]:
            continue
        image_path = data_dir / str(record["images"][camera_name])
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read calibration image: {image_path}")
        images.append(image)
        poses.append(pose7_to_matrix(record["cartesian_pose"]))
        record_indices.append(index)
        image_paths.append(image_path)
    if len(images) < 3:
        raise ValueError(f"Need at least 3 image-pose pairs for camera {camera_name}, got {len(images)}")
    return images, poses, record_indices, image_paths


def _checkerboard_object_points(checkerboard: CheckerboardConfig) -> np.ndarray:
    object_points = np.zeros((checkerboard.cols * checkerboard.rows, 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0 : checkerboard.cols, 0 : checkerboard.rows].T.reshape(-1, 2)
    object_points *= float(checkerboard.square_size_m)
    return object_points


def _gray_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("calibration image must have shape HxWx3")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _transform_spread_errors(transforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    reference = transforms[0]
    rotation_errors = []
    translation_errors = []
    for transform in transforms[1:]:
        relative_rotation = reference[:3, :3].T @ transform[:3, :3]
        cos_angle = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cos_angle))))
        translation_errors.append(float(np.linalg.norm(reference[:3, 3] - transform[:3, 3]) * 1000.0))
    return np.asarray(rotation_errors, dtype=np.float64), np.asarray(translation_errors, dtype=np.float64)


def _verification_stats_within_threshold(
    stats: VerificationStats,
    threshold: VerificationQualityThreshold,
) -> bool:
    return (
        stats.mean_translation_error_mm <= threshold.mean_translation_error_mm
        and stats.max_translation_error_mm <= threshold.max_translation_error_mm
        and stats.mean_rotation_error_deg <= threshold.mean_rotation_error_deg
        and stats.max_rotation_error_deg <= threshold.max_rotation_error_deg
    )


def _validated_transform_matrix(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape 4x4")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    return matrix


def _validate_mode(mode: str) -> None:
    if mode not in CALIBRATION_MODES:
        raise ValueError("calibration mode must be eye-to-hand or eye-in-hand")


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
