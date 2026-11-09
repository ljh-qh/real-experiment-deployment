from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unirobot_infra.calibration.hand_eye import CameraIntrinsics


@dataclass(frozen=True)
class RgbdFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
