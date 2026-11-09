from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from unirobot_infra.calibration.hand_eye import matrix_to_pose7
from unirobot_infra.core.time import now_ns
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics


def tianji_tcp_matrix_m(kinematics: TianjiKinematics, q_deg: tuple[float, ...]) -> np.ndarray:
    xyzabc_mm_deg = kinematics.fk_xyzabc_mm_deg(q_deg)
    transform_mm = np.asarray(kinematics.xyzabc_to_mat4x4(xyzabc_mm_deg), dtype=np.float64)
    transform_m = transform_mm.copy()
    transform_m[:3, 3] = transform_m[:3, 3] * 0.001
    return transform_m


def build_tianji_calibration_record(
    *,
    side: str,
    arm_id: str,
    q_deg: tuple[float, ...],
    base_from_ee: np.ndarray,
    images: dict[str, str],
) -> dict[str, Any]:
    q_rad = tuple(deg_to_rad(value) for value in q_deg)
    return {
        "timestamp_ns": now_ns(),
        "side": side,
        "arm_id": arm_id,
        "base_frame": f"tianji_{side}_sdk_base",
        "ee_frame": f"tianji_{side}_tcp",
        "cartesian_pose": list(matrix_to_pose7(base_from_ee)),
        "joint_pos": [float(value) for value in q_rad],
        "joint_pos_deg": [float(value) for value in q_deg],
        "images": images,
    }


def write_pose_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
