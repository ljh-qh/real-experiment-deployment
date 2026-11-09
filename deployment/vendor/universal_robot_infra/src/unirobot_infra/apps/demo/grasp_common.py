from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
import yaml

from unirobot_infra.calibration.hand_eye import load_transform
from unirobot_infra.camera.zed_capture import ZedCapture
from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.grasp.adaptive import (
    AdaptiveGraspFeedbackConfig,
    AdaptiveGraspHoldCommand,
    AdaptiveGraspHoldConfig,
    AdaptiveGraspPlanner,
    AdaptiveGraspState,
    AdaptiveGraspStateMachine,
    group_indices_from_config,
    load_anchor_trajectory,
)
from unirobot_infra.perception.frame_transform import RigidTransform, add_xyz_offset, matrix3x3_tuple, transform_vector
from unirobot_infra.perception.object_localization import localize_detections_3d
from unirobot_infra.perception.types import ObjectPoint3D
from unirobot_infra.perception.yoloe import YoloEDetector
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics


@dataclass(frozen=True)
class TcpPose:
    xyz_base_m: tuple[float, float, float]
    rotation_base_from_tcp: tuple[tuple[float, float, float], ...]

    def mat4x4_mm(self) -> tuple[tuple[float, float, float, float], ...]:
        return (
            (
                self.rotation_base_from_tcp[0][0],
                self.rotation_base_from_tcp[0][1],
                self.rotation_base_from_tcp[0][2],
                self.xyz_base_m[0] * 1000.0,
            ),
            (
                self.rotation_base_from_tcp[1][0],
                self.rotation_base_from_tcp[1][1],
                self.rotation_base_from_tcp[1][2],
                self.xyz_base_m[1] * 1000.0,
            ),
            (
                self.rotation_base_from_tcp[2][0],
                self.rotation_base_from_tcp[2][1],
                self.rotation_base_from_tcp[2][2],
                self.xyz_base_m[2] * 1000.0,
            ),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class LiveTargetPose:
    side: str
    label: str
    score: float
    timestamp_ns: int
    pose: TcpPose


@dataclass(frozen=True)
class _LivePerceptionTask:
    side: str
    target_label: str
    base_from_camera: RigidTransform
    base_from_world: tuple[tuple[float, float, float], ...]
    target_offset_world_m: tuple[float, float, float]
    rotation_base_from_tcp: tuple[tuple[float, float, float], ...]


class LiveTargetPoseProcess:
    def __init__(self, cfg: DictConfig, sides: tuple[str, ...], rgbd_frame_bus=None) -> None:
        self.cfg_data = OmegaConf.to_container(cfg, resolve=True)
        self.sides = tuple(str(side) for side in sides)
        if not self.sides:
            raise ValueError("live perception sides must not be empty")
        self.rgbd_frame_bus = rgbd_frame_bus
        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        self.process: mp.Process | None = None
        self.latest_by_side: dict[str, LiveTargetPose] = {}
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("LiveTargetPoseProcess is already started")
        if self.rgbd_frame_bus is not None:
            self.rgbd_frame_bus.start()
        self.process = self.ctx.Process(
            target=_live_target_pose_worker,
            args=(self.cfg_data, self.sides, self.queue, self.stop_event, _rgbd_frame_queue(self.rgbd_frame_bus)),
            daemon=True,
        )
        self.process.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.rgbd_frame_bus is not None:
            self.rgbd_frame_bus.stop()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3.0)
        self.process = None

    def __enter__(self) -> "LiveTargetPoseProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def latest_pose(self, side: str) -> TcpPose | None:
        target = self.latest_target(side)
        if target is None:
            return None
        return target.pose

    def latest_target(self, side: str) -> LiveTargetPose | None:
        self._drain_queue()
        self._raise_if_process_failed()
        with self.lock:
            if side not in self.latest_by_side:
                return None
            return self.latest_by_side[side]

    def _drain_queue(self) -> None:
        while True:
            try:
                message = self.queue.get_nowait()
            except queue.Empty:
                break
            with self.lock:
                for side in message:
                    self.latest_by_side[side] = message[side]

    def _raise_if_process_failed(self) -> None:
        if self.process is None:
            raise RuntimeError("LiveTargetPoseProcess is not started")
        if not self.process.is_alive() and self.process.exitcode not in (None, 0):
            raise RuntimeError(f"LiveTargetPoseProcess exited with code {self.process.exitcode}")


class GraspDemoRuntime:
    def __init__(
        self,
        cfg: DictConfig,
        logger,
        env=None,
        camera=None,
        detector=None,
        manage_env: bool = True,
        manage_camera: bool = True,
        use_internal_perception: bool = True,
        io_lock: threading.Lock | None = None,
        controller_lock: threading.Lock | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.side = str(cfg.demo.side)
        self.env = env
        self.camera = camera
        self.detector = detector
        self.manage_env = manage_env
        self.manage_camera = manage_camera
        self.use_internal_perception = use_internal_perception
        self.io_lock = io_lock
        self.controller_lock = controller_lock
        self.stop_event = stop_event
        self._last_send_cmd_failure_log_s = 0.0
        self._last_ik_failure_log_s = 0.0
        self._last_grasp_hold_log_s = 0.0
        self.grasp_state_machine: AdaptiveGraspStateMachine | None = None
        self.base_from_camera = None
        self.base_from_world = matrix3x3_tuple(
            cfg.robot.hardware.arm.arms[self.side].sdk_xyz_from_robot_world,
            f"robot.hardware.arm.arms.{self.side}.sdk_xyz_from_robot_world",
        )
        self.kinematics = _build_kinematics(cfg, self.side)
        self.ik_solver = TianjiIkSolver(
            kinematics=self.kinematics,
            max_joint_step_deg=float(cfg.demo.ik.max_joint_step_deg),
            joint_limit_mode=str(cfg.demo.ik.joint_limit_mode),
        )

    def start(self) -> None:
        if self.use_internal_perception:
            if self.detector is None:
                self.detector = YoloEDetector(self.cfg.demo.detector)
            if self.camera is None:
                self.camera = _build_zed_rgbd_camera(self.cfg)
        self.base_from_camera = _base_from_camera_transform(self.cfg)
        if self.env is None:
            self.env = build_base_env(self.cfg)
            self.env.reset()

    def close(self) -> None:
        if self.manage_camera and self.camera is not None:
            self.camera.close()
        if self.manage_env and self.env is not None:
            self.env.close()

    def __enter__(self) -> "GraspDemoRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def perceive_object_position(self) -> tuple[float, float, float]:
        deadline = time.monotonic() + float(self.cfg.demo.perception_timeout_s)
        period_s = 1.0 / float(self.cfg.demo.perception_rate_hz)
        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            target = self._read_target_object()
            if target is not None:
                self.logger.info("target detected label=%s xyz_base_m=%s", target.label, _format_xyz(target.base_xyz_m))
                return target.base_xyz_m
            _sleep_remaining(loop_start, period_s)
        raise RuntimeError(f"target label={self.cfg.demo.target_label} not detected")

    def target_pose_from_position(self, position_base_m: tuple[float, float, float]) -> TcpPose:
        return self.target_pose(
            position_base_m,
            _vector3_tuple(self.cfg.demo.grasp_target_offset_world_m, "demo.grasp_target_offset_world_m"),
        )

    def target_pose(
        self,
        position_base_m: tuple[float, float, float],
        offset_world_m: tuple[float, float, float],
    ) -> TcpPose:
        offset_base_m = transform_vector(self.base_from_world, offset_world_m)
        return TcpPose(
            xyz_base_m=add_xyz_offset(position_base_m, offset_base_m),
            rotation_base_from_tcp=self.target_rotation_base_from_tcp(),
        )

    def adaptive_grasp(self) -> None:
        state_machine = _build_adaptive_grasp_state_machine(self.cfg)
        self.logger.info(
            "adaptive grasp start side=%s duration_s=%.3f rate_hz=%.3f",
            self.side,
            float(self.cfg.adaptive_grasp.duration_s),
            float(self.cfg.adaptive_grasp.rate_hz),
        )
        settle_q = state_machine.trajectory.interpolate(0.0)
        self.send_actions(
            (
                RobotAction(
                    side=self.side,
                    timestamp_ns=now_ns(),
                    end_effector_q_rad=settle_q,
                ),
            ),
            blocking=False,
        )
        time.sleep(float(self.cfg.adaptive_grasp.initial_settle_s))
        state_machine.reset_for_grasp(self.sharpa_state())
        period_s = 1.0 / float(self.cfg.adaptive_grasp.rate_hz)
        duration_s = float(self.cfg.adaptive_grasp.duration_s)
        start_s = time.monotonic()
        command = None
        while True:
            self._raise_if_stop_requested()
            loop_start = time.monotonic()
            alpha = min(1.0, (loop_start - start_s) / duration_s)
            command = state_machine.grasp_command(alpha, self.sharpa_state())
            self.send_actions(
                (
                    RobotAction(
                        side=self.side,
                        timestamp_ns=now_ns(),
                        end_effector_q_rad=command.q_rad,
                    ),
                ),
                blocking=False,
            )
            _log_new_grasp_stops(command.new_stops, self.logger)
            if alpha >= 1.0 or state_machine.all_groups_stopped:
                break
            _sleep_remaining(loop_start, period_s)
        state = state_machine.finish_grasp()
        self.grasp_state_machine = state_machine
        self.logger.info(
            "adaptive grasp done side=%s state=%s alpha=%.3f stopped=%s",
            self.side,
            state,
            0.0 if command is None else command.alpha,
            ",".join(state_machine.stopped_groups) if state_machine.stopped_groups else "none",
        )
        if state == AdaptiveGraspState.OBJECT_MISSING:
            raise RuntimeError(f"adaptive grasp object missing side={self.side}")

    def servo_to_pose(self, target_pose: TcpPose, timeout_s: float, *, hold_grasp: bool = True) -> None:
        self.logger.info(
            "servo fixed target start side=%s timeout_s=%.3f target_xyz_base_m=%s",
            self.side,
            timeout_s,
            _format_xyz(target_pose.xyz_base_m),
        )
        deadline = time.monotonic() + timeout_s
        servo_period_s = 1.0 / float(self.cfg.demo.servo.rate_hz)
        reached_count = 0
        last_error_log_s = 0.0
        first_command_sent = False
        while time.monotonic() < deadline:
            self._raise_if_stop_requested()
            loop_start = time.monotonic()
            current_pose = self.current_tcp_pose()
            if self.pose_reached_from(current_pose, target_pose):
                reached_count += 1
            else:
                reached_count = 0
                servo_pose = self.step_from_pose_towards(current_pose, target_pose)
                if loop_start - last_error_log_s >= 1.0:
                    last_error_log_s = loop_start
                    self.logger.info(
                        "servo fixed target tracking side=%s position_error_m=%.4f orientation_error_deg=%.2f "
                        "current_xyz_base_m=%s target_xyz_base_m=%s command_xyz_base_m=%s",
                        self.side,
                        _position_error_m(current_pose.xyz_base_m, target_pose.xyz_base_m),
                        _rotation_error_deg(current_pose.rotation_base_from_tcp, target_pose.rotation_base_from_tcp),
                        _format_xyz(current_pose.xyz_base_m),
                        _format_xyz(target_pose.xyz_base_m),
                        _format_xyz(servo_pose.xyz_base_m),
                    )
                action = self._arm_action_to_pose_for_servo(servo_pose)
                if action is not None:
                    self.send_actions((action,), blocking=False)
                    if not first_command_sent:
                        first_command_sent = True
                        self.logger.info(
                            "servo fixed target first command sent side=%s command_xyz_base_m=%s",
                            self.side,
                            _format_xyz(servo_pose.xyz_base_m),
                        )
            if hold_grasp:
                self.hold_grasp()
            if reached_count >= int(self.cfg.demo.reached_consecutive_frames):
                self.logger.info(
                    "servo fixed target reached side=%s target_xyz_base_m=%s",
                    self.side,
                    _format_xyz(target_pose.xyz_base_m),
                )
                return
            _sleep_remaining(loop_start, servo_period_s)
        raise RuntimeError(f"pose not reached within timeout_s={timeout_s}")

    def servo_to_live_target_pose(self, target_source: LiveTargetPoseProcess, timeout_s: float) -> TcpPose:
        self.logger.info(
            "servo live target start side=%s timeout_s=%.3f servo_hz=%.3f perception_timeout_s=%.3f",
            self.side,
            timeout_s,
            float(self.cfg.demo.servo.rate_hz),
            float(self.cfg.demo.perception_timeout_s),
        )
        deadline = time.monotonic() + timeout_s
        servo_period_s = 1.0 / float(self.cfg.demo.servo.rate_hz)
        reached_count = 0
        target_pose = None
        last_target_timestamp_ns = None
        last_wait_log_s = 0.0
        last_error_log_s = 0.0
        first_command_sent = False
        while time.monotonic() < deadline:
            self._raise_if_stop_requested()
            loop_start = time.monotonic()
            live_target = target_source.latest_target(self.side)
            if live_target is None:
                if loop_start - last_wait_log_s >= 1.0:
                    last_wait_log_s = loop_start
                    self.logger.info("servo live target waiting side=%s", self.side)
                _sleep_remaining(loop_start, servo_period_s)
                continue
            target_pose = live_target.pose
            if live_target.timestamp_ns != last_target_timestamp_ns:
                last_target_timestamp_ns = live_target.timestamp_ns
                self.logger.info(
                    "live target update side=%s label=%s score=%.3f target_xyz_base_m=%s",
                    self.side,
                    live_target.label,
                    live_target.score,
                    _format_xyz(target_pose.xyz_base_m),
                )
            current_pose = self.current_tcp_pose()
            if self.pose_reached_from(current_pose, target_pose):
                reached_count += 1
            else:
                reached_count = 0
                servo_pose = self.step_from_pose_towards(current_pose, target_pose)
                if loop_start - last_error_log_s >= 1.0:
                    last_error_log_s = loop_start
                    self.logger.info(
                        "servo live target tracking side=%s position_error_m=%.4f orientation_error_deg=%.2f "
                        "current_xyz_base_m=%s target_xyz_base_m=%s command_xyz_base_m=%s",
                        self.side,
                        _position_error_m(current_pose.xyz_base_m, target_pose.xyz_base_m),
                        _rotation_error_deg(current_pose.rotation_base_from_tcp, target_pose.rotation_base_from_tcp),
                        _format_xyz(current_pose.xyz_base_m),
                        _format_xyz(target_pose.xyz_base_m),
                        _format_xyz(servo_pose.xyz_base_m),
                    )
                action = self._arm_action_to_pose_for_servo(servo_pose)
                if action is not None:
                    self.send_actions((action,), blocking=False)
                    if not first_command_sent:
                        first_command_sent = True
                        self.logger.info(
                            "servo live target first command sent side=%s command_xyz_base_m=%s",
                            self.side,
                            _format_xyz(servo_pose.xyz_base_m),
                        )
            self.hold_grasp()
            if reached_count >= int(self.cfg.demo.reached_consecutive_frames):
                self.logger.info(
                    "servo live target reached side=%s target_xyz_base_m=%s",
                    self.side,
                    _format_xyz(target_pose.xyz_base_m),
                )
                return target_pose
            _sleep_remaining(loop_start, servo_period_s)
        if target_pose is None:
            raise RuntimeError(f"live target pose side={self.side} not received within timeout_s={timeout_s}")
        raise RuntimeError(f"live target pose side={self.side} not reached within timeout_s={timeout_s}")

    def release_grasp(self) -> None:
        self.release_grasp_to_alpha(0.0, "full")

    def release_grasp_to_middle(self) -> None:
        self.release_grasp_to_alpha(float(self.cfg.adaptive_grasp.release.middle_alpha), "middle")

    def release_grasp_to_alpha(self, target_alpha: float, label: str) -> None:
        if target_alpha < 0.0 or target_alpha > 1.0:
            raise ValueError("adaptive grasp release target alpha must be in [0, 1]")
        if self.grasp_state_machine is not None:
            self.grasp_state_machine.start_release()
        planner = _build_adaptive_grasp_planner(self.cfg)
        target_q = planner.trajectory.interpolate(target_alpha)
        current_q = self.sharpa_state().joint_pos
        stage_duration_s = float(self.cfg.adaptive_grasp.release.stage_duration_s)
        rate_hz = float(self.cfg.adaptive_grasp.release.rate_hz)
        settle_s = float(self.cfg.adaptive_grasp.release.settle_s)
        if stage_duration_s <= 0.0:
            raise ValueError("adaptive_grasp.release.stage_duration_s must be positive")
        if rate_hz <= 0.0:
            raise ValueError("adaptive_grasp.release.rate_hz must be positive")
        if settle_s < 0.0:
            raise ValueError("adaptive_grasp.release.settle_s must be non-negative")
        group_indices = group_indices_from_config(self.cfg.adaptive_grasp.stop_groups)
        release_stages = tuple(tuple(str(group) for group in stage) for stage in self.cfg.adaptive_grasp.release.stages)
        _validate_release_stages(release_stages, group_indices, len(target_q))
        for stage in release_stages:
            stage_target_q = _release_stage_target(current_q, target_q, group_indices, stage)
            self.logger.info(
                "release grasp stage start side=%s label=%s target_alpha=%.3f groups=%s",
                self.side,
                label,
                target_alpha,
                ",".join(stage),
            )
            self._send_release_interpolation(current_q, stage_target_q, stage_duration_s, rate_hz)
            current_q = stage_target_q
            self.logger.info(
                "release grasp stage done side=%s label=%s target_alpha=%.3f groups=%s",
                self.side,
                label,
                target_alpha,
                ",".join(stage),
            )
        time.sleep(settle_s)
        if target_alpha == 0.0 and self.grasp_state_machine is not None:
            self.grasp_state_machine.mark_released()
        self.logger.info(
            "release grasp done side=%s label=%s target_alpha=%.3f stage_duration_s=%.3f stages=%d",
            self.side,
            label,
            target_alpha,
            stage_duration_s,
            len(release_stages),
        )

    def post_release_clearance_pose(self) -> TcpPose:
        return self.translated_pose_world(self.current_tcp_pose(), self.post_release_clearance_offset_world())

    def post_release_clearance_offset_world(self) -> tuple[float, float, float]:
        clearance_cfg = self.cfg.demo.post_release_clearance
        return _scaled_direction_world(
            clearance_cfg.directions_world[self.side],
            float(clearance_cfg.distance_m),
            f"demo.post_release_clearance.directions_world.{self.side}",
        )

    def move_post_release_clearance(self, *, hold_grasp: bool = True) -> None:
        self.servo_translate_tcp_world(
            self.post_release_clearance_offset_world(),
            timeout_s=float(self.cfg.demo.post_release_clearance.timeout_s),
            label="post release clearance",
            hold_grasp=hold_grasp,
        )

    def post_replay_anchor_pose(self) -> TcpPose:
        anchor_cfg = self.cfg.demo.post_replay_anchors
        path_text = str(anchor_cfg.paths[self.side])
        if path_text == "":
            raise ValueError(f"demo.post_replay_anchors.paths.{self.side} must not be empty")
        return load_arm_anchor_pose(Path(path_text), self.side)

    def move_post_replay_anchor(self) -> None:
        anchor_cfg = self.cfg.demo.post_replay_anchors
        target_pose = self.post_replay_anchor_pose()
        self.logger.info(
            "post replay anchor servo start side=%s timeout_s=%.3f target_xyz_base_m=%s",
            self.side,
            float(anchor_cfg.timeout_s),
            _format_xyz(target_pose.xyz_base_m),
        )
        self.servo_to_pose(target_pose, timeout_s=float(anchor_cfg.timeout_s))
        self.logger.info("post replay anchor servo done side=%s", self.side)

    def hold_grasp(self) -> None:
        if self.grasp_state_machine is None:
            return
        if self.grasp_state_machine.state != AdaptiveGraspState.HOLDING:
            return
        command = self.grasp_state_machine.hold_command(self.sharpa_state())
        self.send_actions(
            (
                RobotAction(
                    side=self.side,
                    timestamp_ns=now_ns(),
                    end_effector_q_rad=command.q_rad,
                ),
            ),
            blocking=False,
        )
        self._log_grasp_hold(command)
        if command.object_missing:
            raise RuntimeError(f"adaptive grasp object missing side={self.side}")

    def _log_grasp_hold(self, command: AdaptiveGraspHoldCommand) -> None:
        active_groups = _active_hold_groups(command)
        if not active_groups and not command.object_missing:
            return
        if command.object_missing:
            self.logger.error("adaptive grasp hold object missing side=%s", self.side)
            return
        now_s = time.monotonic()
        if now_s - self._last_grasp_hold_log_s < float(self.cfg.adaptive_grasp.hold.log_interval_s):
            return
        self._last_grasp_hold_log_s = now_s
        self.logger.info(
            "adaptive grasp hold side=%s state=%s active=%s",
            self.side,
            command.state,
            ",".join(active_groups),
        )

    def _send_release_interpolation(
        self,
        start_q: tuple[float, ...],
        target_q: tuple[float, ...],
        duration_s: float,
        rate_hz: float,
    ) -> None:
        period_s = 1.0 / rate_hz
        started_s = time.monotonic()
        while True:
            self._raise_if_stop_requested()
            loop_start = time.monotonic()
            alpha = min(1.0, (loop_start - started_s) / duration_s)
            q_rad = _interpolate_tuple(start_q, target_q, alpha)
            self.send_actions(
                (
                    RobotAction(
                        side=self.side,
                        timestamp_ns=now_ns(),
                        end_effector_q_rad=q_rad,
                    ),
                ),
                blocking=False,
            )
            if alpha >= 1.0:
                break
            _sleep_remaining(loop_start, period_s)

    def reset_robot(self) -> None:
        self.env.reset()
        self.logger.info("robot reset done")

    def offset_pose_world(self, pose: TcpPose, offset_world_m: tuple[float, float, float]) -> TcpPose:
        return TcpPose(
            xyz_base_m=add_xyz_offset(pose.xyz_base_m, transform_vector(self.base_from_world, offset_world_m)),
            rotation_base_from_tcp=pose.rotation_base_from_tcp,
        )

    def translated_pose_world(self, pose: TcpPose, offset_world_m: tuple[float, float, float]) -> TcpPose:
        return self.offset_pose_world(pose, _vector3_tuple(offset_world_m, "offset_world_m"))

    def servo_translate_tcp_world(
        self,
        offset_world_m: tuple[float, float, float],
        timeout_s: float,
        *,
        start_pose: TcpPose | None = None,
        label: str,
        hold_grasp: bool = True,
    ) -> TcpPose:
        from_pose = self.current_tcp_pose() if start_pose is None else start_pose
        target_pose = self.translated_pose_world(from_pose, offset_world_m)
        self.logger.info(
            "servo translated tcp world start side=%s label=%s offset_world_m=%s target_xyz_base_m=%s",
            self.side,
            label,
            _format_xyz(_vector3_tuple(offset_world_m, "offset_world_m")),
            _format_xyz(target_pose.xyz_base_m),
        )
        self.servo_to_pose(target_pose, timeout_s=timeout_s, hold_grasp=hold_grasp)
        self.logger.info("servo translated tcp world done side=%s label=%s", self.side, label)
        return target_pose

    def arm_action_to_pose(self, target_pose: TcpPose) -> RobotAction:
        robot_state = self.robot_state()
        if robot_state.arm_joint is None:
            raise RuntimeError("demo requires arm_joint state")
        reference_q_deg = _rad_tuple_to_deg7(robot_state.arm_joint.q_rad)
        target_xyzabc = self.kinematics.mat4x4_to_xyzabc_mm_deg(target_pose.mat4x4_mm())
        target_q_deg = self.ik_solver.solve(
            target_xyzabc_mm_deg=target_xyzabc,
            reference_q_deg=reference_q_deg,
            last_q_deg=reference_q_deg,
        )
        return RobotAction(
            side=self.side,
            timestamp_ns=robot_state.timestamp_ns,
            arm_q_rad=tuple(deg_to_rad(value) for value in target_q_deg),
        )

    def _arm_action_to_pose_for_servo(self, target_pose: TcpPose) -> RobotAction | None:
        try:
            return self.arm_action_to_pose(target_pose)
        except RuntimeError as exc:
            if not self._should_ignore_ik_failure(exc):
                raise
            self._log_ignored_ik_failure(exc, target_pose)
            return None

    def pose_reached(self, target_pose: TcpPose) -> bool:
        return self.pose_reached_from(self.current_tcp_pose(), target_pose)

    def pose_reached_from(self, current_pose: TcpPose, target_pose: TcpPose) -> bool:
        position_error_m = _position_error_m(current_pose.xyz_base_m, target_pose.xyz_base_m)
        orientation_error_deg = _rotation_error_deg(
            current_pose.rotation_base_from_tcp,
            target_pose.rotation_base_from_tcp,
        )
        return (
            position_error_m <= float(self.cfg.demo.tcp_position_tolerance_m)
            and orientation_error_deg <= float(self.cfg.demo.tcp_orientation_tolerance_deg)
        )

    def tcp_pose_error(self, target_pose: TcpPose) -> tuple[float, float]:
        current_pose = self.current_tcp_pose()
        position_error_m = _position_error_m(current_pose.xyz_base_m, target_pose.xyz_base_m)
        orientation_error_deg = _rotation_error_deg(
            current_pose.rotation_base_from_tcp,
            target_pose.rotation_base_from_tcp,
        )
        return position_error_m, orientation_error_deg

    def step_towards_pose(self, target_pose: TcpPose) -> TcpPose:
        return self.step_from_pose_towards(self.current_tcp_pose(), target_pose)

    def step_from_pose_towards(self, current_pose: TcpPose, target_pose: TcpPose) -> TcpPose:
        return TcpPose(
            xyz_base_m=_step_xyz_towards(
                current_pose.xyz_base_m,
                target_pose.xyz_base_m,
                float(self.cfg.demo.servo.max_translation_step_m),
            ),
            rotation_base_from_tcp=_step_rotation_towards(
                current_pose.rotation_base_from_tcp,
                target_pose.rotation_base_from_tcp,
                float(self.cfg.demo.servo.max_orientation_step_deg),
            ),
        )

    def current_tcp_pose(self) -> TcpPose:
        robot_state = self.robot_state()
        if robot_state.arm_joint is None:
            raise RuntimeError("demo requires arm_joint state")
        return self.tcp_pose_from_arm_q_rad(robot_state.arm_joint.q_rad)

    def tcp_pose_from_arm_q_rad(self, q_rad: tuple[float, ...]) -> TcpPose:
        q_deg = _rad_tuple_to_deg7(q_rad)
        xyzabc = self.kinematics.fk_xyzabc_mm_deg(q_deg)
        pose_mat = self.kinematics.xyzabc_to_mat4x4(xyzabc)
        return TcpPose(
            xyz_base_m=(xyzabc[0] * 0.001, xyzabc[1] * 0.001, xyzabc[2] * 0.001),
            rotation_base_from_tcp=_rotation_from_pose_mat(pose_mat),
        )

    def robot_state(self) -> RobotState:
        with self._controller_context():
            return _robot_state_by_side(self.env.controller.read_state(), self.side)

    def send_actions(self, actions: tuple[RobotAction, ...], blocking: bool) -> None:
        self._raise_if_stop_requested()
        with self._controller_context():
            try:
                self.env.controller.send_actions(actions, blocking=blocking)
            except RuntimeError as exc:
                if not self._should_ignore_send_cmd_failure(exc, blocking):
                    raise
                self._log_ignored_send_cmd_failure(exc)

    def _should_ignore_send_cmd_failure(self, exc: RuntimeError, blocking: bool) -> bool:
        return (
            not blocking
            and bool(self.cfg.demo.servo.ignore_send_cmd_failure)
            and "Tianji send_cmd failed" in str(exc)
        )

    def _should_ignore_ik_failure(self, exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            bool(self.cfg.demo.servo.ignore_ik_failure)
            and (
                "Tianji IK failed" in message
                or "Tianji IK target is out of range" in message
                or "Tianji IK joint result exceeds limits" in message
                or "Tianji IK joint step exceeds max_joint_step_deg" in message
            )
        )

    def _log_ignored_send_cmd_failure(self, exc: RuntimeError) -> None:
        now_s = time.monotonic()
        if now_s - self._last_send_cmd_failure_log_s < float(self.cfg.demo.servo.send_cmd_failure_log_interval_s):
            return
        self._last_send_cmd_failure_log_s = now_s
        self.logger.warning(
            "Tianji send_cmd failed; skipped servo frame side=%s error=%s",
            self.side,
            exc,
        )

    def _log_ignored_ik_failure(self, exc: RuntimeError, target_pose: TcpPose) -> None:
        now_s = time.monotonic()
        if now_s - self._last_ik_failure_log_s < float(self.cfg.demo.servo.ik_failure_log_interval_s):
            return
        self._last_ik_failure_log_s = now_s
        self.logger.warning(
            "Tianji IK failed; skipped servo frame side=%s target_xyz_base_m=%s error=%s",
            self.side,
            _format_xyz(target_pose.xyz_base_m),
            exc,
        )

    def _raise_if_stop_requested(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise RuntimeError(f"demo stop requested side={self.side}")

    def _controller_context(self):
        if self.controller_lock is None:
            return nullcontext()
        return self.controller_lock

    def sharpa_state(self):
        state = self.robot_state()
        if state.sharpa is None:
            raise RuntimeError("demo requires Sharpa hand state")
        return state.sharpa

    def target_rotation_base_from_tcp(self) -> tuple[tuple[float, float, float], ...]:
        x_axis = transform_vector(
            self.base_from_world,
            _vector3_tuple(self.cfg.demo.tcp_local_x_axis_world, "demo.tcp_local_x_axis_world"),
        )
        y_axis = transform_vector(
            self.base_from_world,
            _vector3_tuple(self.cfg.demo.tcp_local_y_axis_world, "demo.tcp_local_y_axis_world"),
        )
        return _rotation_from_axes(x_axis, y_axis)

    def _read_target_object(self) -> ObjectPoint3D | None:
        with self._io_context():
            ok, rgbd = self.camera.read_rgbd()
            if not ok or rgbd is None:
                raise RuntimeError("ZED RGB-D read failed")
            detections = self.detector.detect(rgbd.rgb)
        objects = localize_detections_3d(
            detections=detections,
            depth_m=rgbd.depth_m,
            intrinsics=rgbd.intrinsics,
            base_from_camera=self.base_from_camera,
            offset_m=(0.0, 0.0, 0.0),
            offset_frame="arm_base",
            base_xyz_from_robot_world=self.base_from_world,
            min_depth_m=float(self.cfg.demo.depth.min_m),
            max_depth_m=float(self.cfg.demo.depth.max_m),
            depth_search_radius_px=int(self.cfg.demo.depth.search_radius_px),
        )
        candidates = tuple(obj for obj in objects if obj.label == str(self.cfg.demo.target_label))
        if not candidates:
            return None
        return max(candidates, key=lambda obj: obj.score)

    def _io_context(self):
        if self.io_lock is None:
            return nullcontext()
        return self.io_lock


def _build_zed_rgbd_camera(cfg: DictConfig) -> ZedCapture:
    camera_name = str(cfg.demo.camera_name)
    camera_cfg = cfg.camera.cameras[camera_name]
    if camera_cfg.type != "zed_sdk":
        raise ValueError("perception adaptive grasp demo requires a zed_sdk camera")
    return ZedCapture(camera_name, camera_cfg)


def _live_target_pose_worker(cfg_data, sides: tuple[str, ...], output_queue, stop_event, rgbd_frame_queue) -> None:
    cfg = OmegaConf.create(cfg_data)
    tasks = _live_perception_tasks(cfg, sides)
    detector = YoloEDetector(cfg.demo.detector)
    if rgbd_frame_queue is not None:
        _live_target_pose_worker_shared_frames(cfg, tasks, detector, output_queue, stop_event, rgbd_frame_queue)
        return

    camera = _build_zed_rgbd_camera(cfg)
    try:
        period_s = 1.0 / float(cfg.demo.perception_rate_hz)
        while not stop_event.is_set():
            loop_start = time.monotonic()
            ok, rgbd = camera.read_rgbd()
            if not ok or rgbd is None:
                raise RuntimeError("ZED RGB-D read failed")
            _publish_live_targets(cfg, tasks, detector, output_queue, rgbd)
            _sleep_remaining(loop_start, period_s)
    finally:
        camera.close()


def _live_target_pose_worker_shared_frames(cfg, tasks, detector, output_queue, stop_event, rgbd_frame_queue) -> None:
    period_s = 1.0 / float(cfg.demo.perception_rate_hz)
    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            rgbd = rgbd_frame_queue.get(timeout=period_s)
        except queue.Empty:
            continue
        _publish_live_targets(cfg, tasks, detector, output_queue, rgbd)
        _sleep_remaining(loop_start, period_s)


def _publish_live_targets(cfg, tasks, detector, output_queue, rgbd) -> None:
    detections = detector.detect(rgbd.rgb)
    latest_targets = {}
    for task in tasks:
        target = _live_target_pose_for_task(cfg, task, detections, rgbd.depth_m, rgbd.intrinsics)
        if target is not None:
            latest_targets[task.side] = target
    if latest_targets:
        output_queue.put(latest_targets)


def _rgbd_frame_queue(rgbd_frame_bus):
    if rgbd_frame_bus is None:
        return None
    return rgbd_frame_bus.queue


def _live_perception_tasks(cfg: DictConfig, sides: tuple[str, ...]) -> tuple[_LivePerceptionTask, ...]:
    return tuple(_live_perception_task(cfg, side) for side in sides)


def _live_perception_task(cfg: DictConfig, side: str) -> _LivePerceptionTask:
    demo_cfg = _side_demo_cfg(cfg, side)
    base_from_world = matrix3x3_tuple(
        cfg.robot.hardware.arm.arms[side].sdk_xyz_from_robot_world,
        f"robot.hardware.arm.arms.{side}.sdk_xyz_from_robot_world",
    )
    x_axis = transform_vector(
        base_from_world,
        _vector3_tuple(demo_cfg.tcp_local_x_axis_world, f"demo.tasks.{side}.demo.tcp_local_x_axis_world"),
    )
    y_axis = transform_vector(
        base_from_world,
        _vector3_tuple(demo_cfg.tcp_local_y_axis_world, f"demo.tasks.{side}.demo.tcp_local_y_axis_world"),
    )
    return _LivePerceptionTask(
        side=side,
        target_label=str(demo_cfg.target_label),
        base_from_camera=_base_from_camera_transform_from_path(demo_cfg.transform.camera_from_base_path),
        base_from_world=base_from_world,
        target_offset_world_m=_vector3_tuple(
            demo_cfg.grasp_target_offset_world_m,
            f"demo.tasks.{side}.demo.grasp_target_offset_world_m",
        ),
        rotation_base_from_tcp=_rotation_from_axes(x_axis, y_axis),
    )


def _side_demo_cfg(cfg: DictConfig, side: str) -> DictConfig:
    if "tasks" in cfg.demo:
        demo_cfg = OmegaConf.create(OmegaConf.to_container(cfg.demo, resolve=True))
        return OmegaConf.merge(demo_cfg, cfg.demo.tasks[side].demo)
    return cfg.demo


def _base_from_camera_transform_from_path(camera_from_base_path: str) -> RigidTransform:
    camera_from_base = load_transform(camera_from_base_path)
    return RigidTransform(camera_from_base).inverse()


def _live_target_pose_for_task(
    cfg: DictConfig,
    task: _LivePerceptionTask,
    detections,
    depth_m,
    intrinsics,
) -> LiveTargetPose | None:
    objects = localize_detections_3d(
        detections=detections,
        depth_m=depth_m,
        intrinsics=intrinsics,
        base_from_camera=task.base_from_camera,
        offset_m=(0.0, 0.0, 0.0),
        offset_frame="arm_base",
        base_xyz_from_robot_world=task.base_from_world,
        min_depth_m=float(cfg.demo.depth.min_m),
        max_depth_m=float(cfg.demo.depth.max_m),
        depth_search_radius_px=int(cfg.demo.depth.search_radius_px),
    )
    candidates = tuple(obj for obj in objects if obj.label == task.target_label)
    if not candidates:
        return None
    target = max(candidates, key=lambda obj: obj.score)
    offset_base_m = transform_vector(task.base_from_world, task.target_offset_world_m)
    pose = TcpPose(
        xyz_base_m=add_xyz_offset(target.base_xyz_m, offset_base_m),
        rotation_base_from_tcp=task.rotation_base_from_tcp,
    )
    return LiveTargetPose(
        side=task.side,
        label=target.label,
        score=float(target.score),
        timestamp_ns=now_ns(),
        pose=pose,
    )


def _validate_demo_config(cfg: DictConfig) -> None:
    side = str(cfg.demo.side)
    if side not in cfg.robot.sides:
        raise ValueError(f"demo.side={side} not found in robot.sides")
    if "arm" not in cfg.robot.sides[side].parts:
        raise ValueError(f"demo.side={side} must have arm part")
    if "end_effector" not in cfg.robot.sides[side].parts:
        raise ValueError(f"demo.side={side} must have end_effector part")
    if str(cfg.adaptive_grasp.side) != side:
        raise ValueError("adaptive_grasp.side must match demo.side")
    if float(cfg.demo.perception_rate_hz) <= 0.0:
        raise ValueError("demo.perception_rate_hz must be positive")
    if float(cfg.demo.servo.rate_hz) <= 0.0:
        raise ValueError("demo.servo.rate_hz must be positive")
    if float(cfg.demo.servo.max_translation_step_m) <= 0.0:
        raise ValueError("demo.servo.max_translation_step_m must be positive")
    if float(cfg.demo.servo.max_orientation_step_deg) <= 0.0:
        raise ValueError("demo.servo.max_orientation_step_deg must be positive")
    if float(cfg.demo.perception_timeout_s) <= 0.0:
        raise ValueError("demo.perception_timeout_s must be positive")
    if float(cfg.demo.move_timeout_s) <= 0.0:
        raise ValueError("demo.move_timeout_s must be positive")
    if float(cfg.demo.lift_timeout_s) <= 0.0:
        raise ValueError("demo.lift_timeout_s must be positive")
    if float(cfg.demo.post_release_clearance.distance_m) <= 0.0:
        raise ValueError("demo.post_release_clearance.distance_m must be positive")
    if float(cfg.demo.post_release_clearance.timeout_s) <= 0.0:
        raise ValueError("demo.post_release_clearance.timeout_s must be positive")
    if int(cfg.demo.reached_consecutive_frames) <= 0:
        raise ValueError("demo.reached_consecutive_frames must be positive")
    if float(cfg.demo.tcp_position_tolerance_m) <= 0.0:
        raise ValueError("demo.tcp_position_tolerance_m must be positive")
    if float(cfg.demo.tcp_orientation_tolerance_deg) <= 0.0:
        raise ValueError("demo.tcp_orientation_tolerance_deg must be positive")
    _vector3_tuple(cfg.demo.grasp_target_offset_world_m, "demo.grasp_target_offset_world_m")
    _vector3_tuple(cfg.demo.lift_offset_world_m, "demo.lift_offset_world_m")
    _scaled_direction_world(
        cfg.demo.post_release_clearance.directions_world[side],
        float(cfg.demo.post_release_clearance.distance_m),
        f"demo.post_release_clearance.directions_world.{side}",
    )
    x_axis = _vector3_tuple(cfg.demo.tcp_local_x_axis_world, "demo.tcp_local_x_axis_world")
    y_axis = _vector3_tuple(cfg.demo.tcp_local_y_axis_world, "demo.tcp_local_y_axis_world")
    if abs(sum(x * y for x, y in zip(x_axis, y_axis, strict=True))) > 1e-6:
        raise ValueError("demo TCP local x/y axes must be perpendicular")


def load_arm_anchor_pose(path: Path, side: str) -> TcpPose:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        raise ValueError(f"arm anchor file is empty: {path}")
    if str(data["type"]) != "tianji_arm_anchor":
        raise ValueError(f"arm anchor type must be tianji_arm_anchor: {path}")
    if str(data["side"]) != side:
        raise ValueError(f"arm anchor side={data['side']} does not match {side}: {path}")
    anchor = data["anchor"]
    xyz = tuple(float(value) for value in anchor["tcp_xyz_m"])
    if len(xyz) != 3:
        raise ValueError(f"arm anchor tcp_xyz_m length must be 3: {path}")
    rotation = tuple(tuple(float(value) for value in row) for row in anchor["tcp_rotation_base_from_tcp"])
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError(f"arm anchor tcp_rotation_base_from_tcp must be 3x3: {path}")
    return TcpPose(
        xyz_base_m=(xyz[0], xyz[1], xyz[2]),
        rotation_base_from_tcp=rotation,
    )


def _base_from_camera_transform(cfg: DictConfig) -> RigidTransform:
    return _base_from_camera_transform_from_path(cfg.demo.transform.camera_from_base_path)


def _build_kinematics(cfg: DictConfig, side: str) -> TianjiKinematics:
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    return kinematics


def _build_adaptive_grasp_planner(cfg: DictConfig) -> AdaptiveGraspPlanner:
    side = str(cfg.adaptive_grasp.side)
    return AdaptiveGraspPlanner(
        trajectory=load_anchor_trajectory(Path(str(cfg.adaptive_grasp.anchor_dir)), side),
        group_indices=group_indices_from_config(cfg.adaptive_grasp.stop_groups),
        feedback=AdaptiveGraspFeedbackConfig(
            enabled=bool(cfg.adaptive_grasp.feedback.enabled),
            torque_delta_threshold_by_group=_float_by_group(
                cfg.adaptive_grasp.feedback.torque_delta_threshold_by_group
            ),
            position_lag_threshold_rad_by_group=_float_by_group(
                cfg.adaptive_grasp.feedback.position_lag_threshold_rad_by_group
            ),
            min_alpha_for_stop=float(cfg.adaptive_grasp.feedback.min_alpha_for_stop),
        ),
    )


def _build_adaptive_grasp_state_machine(cfg: DictConfig) -> AdaptiveGraspStateMachine:
    return AdaptiveGraspStateMachine(
        planner=_build_adaptive_grasp_planner(cfg),
        hold=AdaptiveGraspHoldConfig(
            enabled=bool(cfg.adaptive_grasp.hold.enabled),
            min_torque_delta_ratio=float(cfg.adaptive_grasp.hold.min_torque_delta_ratio),
            max_torque_delta_ratio=float(cfg.adaptive_grasp.hold.max_torque_delta_ratio),
            missing_torque_delta_ratio=float(cfg.adaptive_grasp.hold.missing_torque_delta_ratio),
            close_step_rad=float(cfg.adaptive_grasp.hold.close_step_rad),
            open_step_rad=float(cfg.adaptive_grasp.hold.open_step_rad),
            max_extra_close_rad=float(cfg.adaptive_grasp.hold.max_extra_close_rad),
            missing_target_margin_rad=float(cfg.adaptive_grasp.hold.missing_target_margin_rad),
            missing_consecutive_frames=int(cfg.adaptive_grasp.hold.missing_consecutive_frames),
            log_interval_s=float(cfg.adaptive_grasp.hold.log_interval_s),
        ),
    )


def _rotation_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    z_axis = np.cross(np.asarray(x_axis, dtype=np.float64), np.asarray(y_axis, dtype=np.float64))
    rotation = np.asarray(
        [
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]],
        ],
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("target TCP axes must form an orthonormal basis")
    if float(np.linalg.det(rotation)) < 0.999999:
        raise ValueError("target TCP axes must form a right-handed basis")
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
    )


def _rotation_from_pose_mat(pose_mat) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in pose_mat)
    if len(rows) != 4:
        raise ValueError("pose_mat must have 4 rows")
    if any(len(row) != 4 for row in rows):
        raise ValueError("pose_mat must have 4 columns")
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )


def _robot_state_by_side(states: tuple[RobotState, ...], side: str) -> RobotState:
    for state in states:
        if state.side == side:
            return state
    raise RuntimeError(f"robot state missing side={side}")


def _position_error_m(
    actual_xyz_m: tuple[float, float, float],
    target_xyz_m: tuple[float, float, float],
) -> float:
    delta = np.asarray(actual_xyz_m, dtype=np.float64) - np.asarray(target_xyz_m, dtype=np.float64)
    return float(np.linalg.norm(delta))


def _rotation_error_deg(
    actual_rotation: tuple[tuple[float, float, float], ...],
    target_rotation: tuple[tuple[float, float, float], ...],
) -> float:
    actual = np.asarray(actual_rotation, dtype=np.float64)
    target = np.asarray(target_rotation, dtype=np.float64)
    delta = target.T @ actual
    cosine = (float(np.trace(delta)) - 1.0) * 0.5
    angle_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.rad2deg(angle_rad))


def _step_xyz_towards(
    actual_xyz_m: tuple[float, float, float],
    target_xyz_m: tuple[float, float, float],
    max_step_m: float,
) -> tuple[float, float, float]:
    actual = np.asarray(actual_xyz_m, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    delta = target - actual
    distance_m = float(np.linalg.norm(delta))
    if distance_m <= max_step_m:
        return target_xyz_m
    stepped = actual + delta * (max_step_m / distance_m)
    return (float(stepped[0]), float(stepped[1]), float(stepped[2]))


def _interpolate_tuple(start: tuple[float, ...], target: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    if len(start) != len(target):
        raise ValueError("interpolate tuple length mismatch")
    return tuple((1.0 - alpha) * start_value + alpha * target_value for start_value, target_value in zip(start, target))


def _release_stage_target(
    current_q: tuple[float, ...],
    target_q: tuple[float, ...],
    group_indices: dict[str, tuple[int, ...]],
    stage: tuple[str, ...],
) -> tuple[float, ...]:
    if len(current_q) != len(target_q):
        raise ValueError("release current/target q length mismatch")
    next_q = list(current_q)
    for group in stage:
        for index in group_indices[group]:
            next_q[index] = target_q[index]
    return tuple(next_q)


def _validate_release_stages(
    stages: tuple[tuple[str, ...], ...],
    group_indices: dict[str, tuple[int, ...]],
    dof: int,
) -> None:
    if not stages:
        raise ValueError("adaptive_grasp.release.stages must not be empty")
    covered_indices = set()
    for stage in stages:
        if not stage:
            raise ValueError("adaptive_grasp.release.stages[] must not be empty")
        for group in stage:
            for index in group_indices[group]:
                if index in covered_indices:
                    raise ValueError("adaptive_grasp.release.stages contain duplicate joint index")
                covered_indices.add(index)
    if covered_indices != set(range(dof)):
        raise ValueError("adaptive_grasp.release.stages must cover every hand joint")


def _step_rotation_towards(
    actual_rotation: tuple[tuple[float, float, float], ...],
    target_rotation: tuple[tuple[float, float, float], ...],
    max_step_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    actual = Rotation.from_matrix(np.asarray(actual_rotation, dtype=np.float64))
    target = Rotation.from_matrix(np.asarray(target_rotation, dtype=np.float64))
    delta = target * actual.inv()
    angle_deg = float(np.rad2deg(delta.magnitude()))
    if angle_deg <= max_step_deg:
        return target_rotation
    stepped = Rotation.from_rotvec(delta.as_rotvec() * (max_step_deg / angle_deg)) * actual
    matrix = stepped.as_matrix()
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def _rad_tuple_to_deg7(q_rad: tuple[float, ...]) -> tuple[float, float, float, float, float, float, float]:
    q_deg = tuple(rad_to_deg(value) for value in q_rad)
    if len(q_deg) != 7:
        raise ValueError("arm q_rad must have length 7")
    return (q_deg[0], q_deg[1], q_deg[2], q_deg[3], q_deg[4], q_deg[5], q_deg[6])


def _vector3_tuple(values, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must have length 3")
    return (vector[0], vector[1], vector[2])


def _scaled_direction_world(values, distance_m: float, name: str = "direction_world") -> tuple[float, float, float]:
    direction = _vector3_tuple(values, name)
    norm = float(np.linalg.norm(np.asarray(direction, dtype=np.float64)))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return tuple(distance_m * value / norm for value in direction)


def _float_by_group(cfg_value) -> dict[str, float]:
    return {str(group): float(cfg_value[group]) for group in cfg_value}


def _log_new_grasp_stops(stops, logger) -> None:
    for stop in stops:
        logger.info(
            "adaptive grasp stop group=%s signal=%s value=%.6f threshold=%.6f alpha=%.3f",
            stop.group,
            stop.signal,
            stop.value,
            stop.threshold,
            stop.alpha,
        )


def _active_hold_groups(command: AdaptiveGraspHoldCommand) -> tuple[str, ...]:
    return tuple(
        f"{group.group}:{group.action}:{group.torque_delta:.3f}"
        for group in command.groups
        if group.action != "hold"
    )


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def _format_xyz(xyz_m: tuple[float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in xyz_m) + "]"
