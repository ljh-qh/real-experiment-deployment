import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--hand-model", required=True)
    parser.add_argument("--filter-alpha", required=True)
    parser.add_argument("--response-fd", required=True, type=int)
    args = parser.parse_args()

    configure_sharpa_runtime(sharpa_manus_sdk_paths(Path(args.vendor_root)))

    import hand_retargeting_optimizer as optimizer

    hand_model = str(args.hand_model).upper()
    hand_models = optimizer.init_hand_model(hand_model)
    manager = optimizer.MultiprocessOptimizationManager(
        hand_models,
        filter_alpha=float(args.filter_alpha),
        hand_serial=hand_model,
    )
    started = False
    frame_index = 0
    response = os.fdopen(args.response_fd, "w", encoding="utf-8")
    _write(response, {"type": "ready", "joint_counts": _joint_counts(hand_models)})

    for line in sys.stdin:
        message = json.loads(line)
        command = message["command"]
        if command == "start":
            if not started:
                manager.start()
                started = True
            _write(response, {"type": "started"})
        elif command == "process":
            result = _process_frame(manager, message, frame_index)
            frame_index += 1
            _write(response, {"type": "result", "result": result})
        elif command == "stop":
            if started:
                manager.stop()
                started = False
            _write(response, {"type": "stopped"})
            response.close()
            return
        else:
            raise ValueError(f"unsupported retarget worker command: {command}")


def _joint_counts(hand_models: Any) -> dict[str, int]:
    return {
        "left": len(hand_models["left"].joint_names),
        "right": len(hand_models["right"].joint_names),
    }


def _process_frame(manager: Any, message: dict[str, Any], frame_index: int) -> dict[str, list[float] | None]:
    return {
        "left": _process_side(manager, "left", message["left"], frame_index),
        "right": _process_side(manager, "right", message["right"], frame_index),
    }


def _process_side(manager: Any, side: str, points: list[list[float]], frame_index: int) -> list[float] | None:
    if not _side_has_data(points):
        return None
    array = np.asarray(points, dtype=np.float64)
    if manager.update_process_keypoints(side, array, frame_index):
        result = manager.get_result(side)
        q_rad = np.asarray(result.filtered_angles, dtype=np.float64).reshape(-1).tolist()
        if len(q_rad) != 22:
            raise RuntimeError("Sharpa retarget result length is not 22")
        return [float(value) for value in q_rad]
    return None


def _side_has_data(points: list[list[float]]) -> bool:
    return any(value != 0.0 for point in points for value in point)


def _write(response, message: dict[str, Any]) -> None:
    response.write(json.dumps(message, ensure_ascii=True) + "\n")
    response.flush()


if __name__ == "__main__":
    main()
