from __future__ import annotations

import numpy as np


class RigidTransform:
    def __init__(self, target_from_source: np.ndarray) -> None:
        matrix = np.asarray(target_from_source, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("target_from_source must have shape 4x4")
        if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError("target_from_source last row must be [0, 0, 0, 1]")
        self.target_from_source = matrix
        self.source_from_target = np.linalg.inv(matrix)

    def transform_point(self, xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
        point = np.asarray([xyz_m[0], xyz_m[1], xyz_m[2], 1.0], dtype=np.float64)
        transformed = self.target_from_source @ point
        return (float(transformed[0]), float(transformed[1]), float(transformed[2]))

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.source_from_target)


def add_xyz_offset(
    xyz_m: tuple[float, float, float],
    offset_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(xyz_m[0] + offset_m[0]),
        float(xyz_m[1] + offset_m[1]),
        float(xyz_m[2] + offset_m[2]),
    )


def transform_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2]),
        float(matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2]),
        float(matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]),
    )


def matrix3x3_tuple(values, name: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if len(rows) != 3:
        raise ValueError(f"{name} must have 3 rows")
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must have 3 columns")
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )


def offset_tuple(values) -> tuple[float, float, float]:
    offset = tuple(float(value) for value in values)
    if len(offset) != 3:
        raise ValueError("offset_m must have length 3")
    return (offset[0], offset[1], offset[2])
