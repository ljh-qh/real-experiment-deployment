"""Direction calibration fit/persist/load, shared by every teleop backend."""

import json
from pathlib import Path

import numpy as np


def rotation_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError("Expected a finite right-handed 3x3 rotation matrix")
    return matrix


def fit_alignment(displacements):
    """Fit measured forward/left/up strokes to operator X/Y/Z, preserving scale.

    Each row is a displacement measured in the SDK tracking frame. Reject
    parallel, tiny or reflected strokes instead of guessing another yaw.
    """
    strokes = np.asarray(displacements, dtype=float)
    if strokes.shape != (3, 3) or not np.isfinite(strokes).all():
        raise ValueError("Need three finite forward/left/up displacements")
    lengths = np.linalg.norm(strokes, axis=1)
    if np.any(lengths < 0.08):
        raise ValueError("每个方向请移动至少 8 cm，再采集。")
    axes = strokes / lengths[:, None]
    if np.max(np.abs(axes @ axes.T - np.eye(3))) > 0.5:
        raise ValueError("三个动作方向没有明显分开，请从同一零点分别向前、向左、向上移动。")
    if np.linalg.det(axes) <= 0:
        raise ValueError("动作方向包含反向：请确认向左是你自己的左侧，向上是朝天花板。")
    u, _, vt = np.linalg.svd(axes)
    transform = rotation_matrix(u @ vt)
    mapped = (transform @ axes.T).T
    errors = np.degrees(np.arccos(np.clip(np.diag(mapped), -1., 1.)))
    if np.max(errors) > 15.:
        raise ValueError("动作偏离直线过多，请重新采集三个方向。")
    return transform, errors


def save_alignment(path, matrix, serials, strokes, errors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "version": 1, "axes": "X forward, Y left, Z up (operator at calibration)",
        "tracker_to_robot": rotation_matrix(matrix).tolist(),
        "tracker_serials": dict(serials),
        "measured_strokes": np.asarray(strokes).tolist(),
        "fit_error_degrees": np.asarray(errors).tolist(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_alignment(path, serials):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("version") != 1 or record.get("tracker_serials") != dict(serials):
        raise ValueError("Tracker 配置已变化，请重新标定方向。")
    return rotation_matrix(record["tracker_to_robot"])
