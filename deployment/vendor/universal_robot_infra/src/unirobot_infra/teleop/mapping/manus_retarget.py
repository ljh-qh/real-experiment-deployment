from dataclasses import dataclass, field
import json
from pathlib import Path
import importlib
import os
import subprocess
from typing import Any
import warnings

from omegaconf import DictConfig

from unirobot_infra.core.types import RobotAction, Side
from unirobot_infra.devices.manus import ManusKeypoints
from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths


_RESOURCE_TRACKER_WARNING = "ignore:resource_tracker:UserWarning:multiprocessing.resource_tracker"
current_warnings = os.environ["PYTHONWARNINGS"] if "PYTHONWARNINGS" in os.environ else ""
if _RESOURCE_TRACKER_WARNING not in current_warnings:
    os.environ["PYTHONWARNINGS"] = (
        f"{current_warnings},{_RESOURCE_TRACKER_WARNING}" if current_warnings else _RESOURCE_TRACKER_WARNING
    )

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked shared_memory objects",
    category=UserWarning,
)


@dataclass(frozen=True)
class RetargetedHandJoints:
    timestamp_ns: int
    left_q_rad: tuple[float, ...] | None
    right_q_rad: tuple[float, ...] | None

    def side_q_rad(self, side: Side) -> tuple[float, ...] | None:
        if side == "left":
            return self.left_q_rad
        if side == "right":
            return self.right_q_rad
        raise ValueError("side must be left or right")


@dataclass
class SharpaRetargeter:
    retarget_cfg: DictConfig
    sharpa_cfg: DictConfig
    _manager: Any = field(default=None, init=False)
    _hand_models: Any = field(default=None, init=False)
    _worker: "_RetargetWorker | None" = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _frame_index: int = field(default=0, init=False)

    def connect(self) -> None:
        python_executable = str(self.retarget_cfg.python_executable)
        if python_executable:
            self._worker = _RetargetWorker(
                python_executable=python_executable,
                vendor_root=Path(self.sharpa_cfg.sdk.vendor_root),
                hand_model=str(self.retarget_cfg.hand_model).upper(),
                filter_alpha=float(self.retarget_cfg.filter_alpha),
                timeout_s=float(self.retarget_cfg.worker_timeout_s),
                log_path=Path(self.retarget_cfg.worker_log_path),
            )
            self._worker.connect()
            return
        configure_sharpa_runtime(sharpa_manus_sdk_paths(Path(self.sharpa_cfg.sdk.vendor_root)))
        module = importlib.import_module("hand_retargeting_optimizer")
        if self.retarget_cfg.hand_model.upper() != "WAVE":
            raise ValueError("Sharpa retargeting supports hand_model=WAVE")
        self._hand_models = module.init_hand_model(self.retarget_cfg.hand_model.upper())
        self._manager = module.MultiprocessOptimizationManager(
            self._hand_models,
            filter_alpha=float(self.retarget_cfg.filter_alpha),
            hand_serial=self.retarget_cfg.hand_model.upper(),
        )

    @property
    def joint_counts(self) -> dict[str, int]:
        if self._worker is not None:
            return self._worker.joint_counts
        self._ensure_connected()
        return {
            "left": len(self._hand_models["left"].joint_names),
            "right": len(self._hand_models["right"].joint_names),
        }

    def start(self) -> None:
        if self._worker is not None:
            self._worker.start()
            self._started = True
            return
        self._ensure_connected()
        if not self._started:
            self._manager.start()
            self._started = True

    def process(self, keypoints: ManusKeypoints) -> RetargetedHandJoints:
        if self._worker is not None:
            return self._worker.process(keypoints)
        self.start()
        frame_index = self._frame_index
        self._frame_index += 1
        left_result = self._process_side("left", keypoints, frame_index)
        right_result = self._process_side("right", keypoints, frame_index)
        return RetargetedHandJoints(
            timestamp_ns=keypoints.timestamp_ns,
            left_q_rad=_angles_from_result(left_result),
            right_q_rad=_angles_from_result(right_result),
        )

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
            self._started = False
            return
        if self._started:
            self._manager.stop()
            self._started = False

    def __enter__(self):
        self.connect()
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _process_side(self, side: Side, keypoints: ManusKeypoints, frame_index: int) -> Any | None:
        if not keypoints.side_has_data(side):
            return None
        import numpy as np

        points = np.asarray(keypoints.side_points(side), dtype=np.float64)
        if self._manager.update_process_keypoints(side, points, frame_index):
            return self._manager.get_result(side)
        return None

    def _ensure_connected(self) -> None:
        if self._worker is not None:
            return
        if self._manager is None or self._hand_models is None:
            raise RuntimeError("SharpaRetargeter is not connected")


def retargeted_joints_to_actions(
    joints: RetargetedHandJoints,
    sides: tuple[Side, ...],
) -> tuple[RobotAction, ...]:
    actions = []
    for side in sides:
        q_rad = joints.side_q_rad(side)
        if q_rad is not None:
            actions.append(
                RobotAction(
                    side=side,
                    timestamp_ns=joints.timestamp_ns,
                    end_effector_q_rad=q_rad,
                )
            )
    return tuple(actions)


def _angles_from_result(result: Any | None) -> tuple[float, ...] | None:
    if result is None:
        return None
    import numpy as np

    angles = result.filtered_angles
    q_rad = tuple(float(value) for value in np.asarray(angles, dtype=np.float64).reshape(-1).tolist())
    if len(q_rad) != 22:
        raise RuntimeError("Sharpa retarget result length is not 22")
    return q_rad


@dataclass
class _RetargetWorker:
    python_executable: str
    vendor_root: Path
    hand_model: str
    filter_alpha: float
    timeout_s: float
    log_path: Path
    process_handle: subprocess.Popen | None = field(default=None, init=False)
    log_file: Any = field(default=None, init=False)
    response_file: Any = field(default=None, init=False)
    joint_counts: dict[str, int] = field(default_factory=dict, init=False)

    def connect(self) -> None:
        python_path = Path(self.python_executable)
        if not python_path.exists():
            raise FileNotFoundError(str(python_path))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        response_read_fd, response_write_fd = os.pipe()
        self.response_file = os.fdopen(response_read_fd, "r", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        src_dir = str(Path(__file__).resolve().parents[3])
        current_pythonpath = env["PYTHONPATH"] if "PYTHONPATH" in env else ""
        env["PYTHONPATH"] = f"{src_dir}:{current_pythonpath}" if current_pythonpath else src_dir
        self.process_handle = subprocess.Popen(
            [
                str(python_path),
                "-m",
                "unirobot_infra.teleop.mapping.manus_retarget_worker",
                "--vendor-root",
                str(self.vendor_root),
                "--hand-model",
                self.hand_model,
                "--filter-alpha",
                str(self.filter_alpha),
                "--response-fd",
                str(response_write_fd),
            ],
            stdin=subprocess.PIPE,
            stdout=self.log_file,
            stderr=self.log_file,
            text=True,
            bufsize=1,
            env=env,
            pass_fds=(response_write_fd,),
        )
        os.close(response_write_fd)
        ready = self._read_message()
        if ready["type"] != "ready":
            raise RuntimeError(f"unexpected Sharpa retarget worker message: {ready}")
        self.joint_counts = {str(side): int(count) for side, count in ready["joint_counts"].items()}

    def start(self) -> None:
        self._write_message({"command": "start"})
        message = self._read_message()
        if message["type"] != "started":
            raise RuntimeError(f"unexpected Sharpa retarget worker message: {message}")

    def process(self, keypoints: ManusKeypoints) -> RetargetedHandJoints:
        self._write_message(
            {
                "command": "process",
                "timestamp_ns": int(keypoints.timestamp_ns),
                "left": keypoints.left,
                "right": keypoints.right,
            }
        )
        message = self._read_message()
        if message["type"] != "result":
            raise RuntimeError(f"unexpected Sharpa retarget worker message: {message}")
        result = message["result"]
        return RetargetedHandJoints(
            timestamp_ns=keypoints.timestamp_ns,
            left_q_rad=_tuple_or_none(result["left"]),
            right_q_rad=_tuple_or_none(result["right"]),
        )

    def stop(self) -> None:
        if self.process_handle is not None and self.process_handle.poll() is None:
            self._write_message({"command": "stop"})
            message = self._read_message()
            if message["type"] != "stopped":
                raise RuntimeError(f"unexpected Sharpa retarget worker message: {message}")
            self.process_handle.wait(timeout=self.timeout_s)
        if self.response_file is not None:
            self.response_file.close()
        if self.log_file is not None:
            self.log_file.close()
        self.process_handle = None
        self.response_file = None
        self.log_file = None

    def _write_message(self, message: dict[str, Any]) -> None:
        if self.process_handle is None or self.process_handle.stdin is None:
            raise RuntimeError("Sharpa retarget worker is not connected")
        self.process_handle.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
        self.process_handle.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        if self.response_file is None:
            raise RuntimeError("Sharpa retarget worker is not connected")
        line = self.response_file.readline()
        if line == "":
            raise RuntimeError(f"Sharpa retarget worker exited; log: {self.log_path}")
        return json.loads(line)


def _tuple_or_none(values: list[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)
