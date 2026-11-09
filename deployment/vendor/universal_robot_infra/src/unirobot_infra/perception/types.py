from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_xy(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


@dataclass(frozen=True)
class Detection2D:
    label: str
    score: float
    bbox: BBox2D


@dataclass(frozen=True)
class ObjectPoint3D:
    label: str
    score: float
    bbox: BBox2D
    camera_xyz_m: tuple[float, float, float]
    base_xyz_m: tuple[float, float, float]
    target_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class PerceptionResult:
    rgb: np.ndarray
    depth_m: np.ndarray
    detections: tuple[Detection2D, ...]
    objects: tuple[ObjectPoint3D, ...]
