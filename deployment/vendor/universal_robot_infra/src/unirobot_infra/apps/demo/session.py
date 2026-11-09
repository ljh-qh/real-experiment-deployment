from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from omegaconf import DictConfig, OmegaConf

from unirobot_infra.apps.demo.grasp_common import (
    GraspDemoRuntime,
    LiveTargetPoseProcess,
    TcpPose,
    _validate_demo_config,
)
from unirobot_infra.apps.demo.skills import SkillTask, Together, skill_task
from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import RobotAction
from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.replay.lerobot_episode import load_lerobot_episode_actions
from unirobot_infra.robots.compositional.group import RobotGroup


@dataclass(frozen=True)
class DualDemoRuntimes:
    left: GraspDemoRuntime
    right: GraspDemoRuntime

    @property
    def by_side(self) -> dict[str, GraspDemoRuntime]:
        return {"left": self.left, "right": self.right}

    @property
    def stop_event(self) -> threading.Event | None:
        if self.left.stop_event is not None:
            return self.left.stop_event
        return self.right.stop_event


class RuntimeSkills:
    def __init__(self, runtimes: DualDemoRuntimes, logger) -> None:
        self.runtimes = runtimes
        self.logger = logger

    def start(self) -> None:
        for side in ("left", "right"):
            runtime = self.runtimes.by_side[side]
            self.logger.info("dual grasp runtime start side=%s", side)
            runtime.start()


class SideMotionSkills:
    def __init__(self, runtime: GraspDemoRuntime) -> None:
        self.runtime = runtime

    def move_to_live_target(self, target_source: LiveTargetPoseProcess) -> TcpPose:
        return self.runtime.servo_to_live_target_pose(
            target_source,
            timeout_s=float(self.runtime.cfg.demo.move_timeout_s),
        )

    def translate_after_grasp(self, start_pose: TcpPose) -> TcpPose:
        return self.translate_tcp_world(
            _float3(self.runtime.cfg.demo.lift_offset_world_m),
            timeout_s=float(self.runtime.cfg.demo.lift_timeout_s),
            start_pose=start_pose,
            label="after grasp",
        )

    def translate_tcp_world(
        self,
        offset_world_m: tuple[float, float, float],
        timeout_s: float,
        *,
        start_pose: TcpPose | None = None,
        label: str,
        hold_grasp: bool = True,
    ) -> TcpPose:
        return self.runtime.servo_translate_tcp_world(
            offset_world_m,
            timeout_s=timeout_s,
            start_pose=start_pose,
            label=label,
            hold_grasp=hold_grasp,
        )

    def post_release_clearance(self, *, hold_grasp: bool = True) -> None:
        self.runtime.move_post_release_clearance(hold_grasp=hold_grasp)

    def move_to_post_replay_anchor(self) -> None:
        self.runtime.move_post_replay_anchor()

    def servo_to_pose(self, pose: TcpPose, timeout_s: float, *, hold_grasp: bool = True) -> None:
        self.runtime.servo_to_pose(pose, timeout_s, hold_grasp=hold_grasp)


class DualMotionSkills:
    def __init__(self, session: DualGraspDemoSession) -> None:
        self.session = session
        self.left = SideMotionSkills(session.runtime.left)
        self.right = SideMotionSkills(session.runtime.right)

    @property
    def by_side(self) -> dict[str, SideMotionSkills]:
        return {"left": self.left, "right": self.right}

    def post_replay_anchor_tasks(self) -> list[SkillTask[None]]:
        tasks = []
        for side in ("left", "right"):
            path_text = str(self.session.cfg.demo.post_replay_anchors.paths[side])
            if path_text == "":
                continue
            motion = self.by_side[side]
            tasks.append(skill_task(f"{side}_arm", motion.move_to_post_replay_anchor))
        return tasks


class SideGraspSkills:
    def __init__(self, runtime: GraspDemoRuntime) -> None:
        self.runtime = runtime

    def adaptive(self) -> None:
        self.runtime.adaptive_grasp()

    def release_to_middle(self) -> None:
        self.runtime.release_grasp_to_middle()

    def release_full(self) -> None:
        self.runtime.release_grasp()


class DualGraspSkills:
    def __init__(self, runtimes: DualDemoRuntimes) -> None:
        self.left = SideGraspSkills(runtimes.left)
        self.right = SideGraspSkills(runtimes.right)


class ResetSkills:
    def __init__(self, session: DualGraspDemoSession) -> None:
        self.session = session

    def to_initial_state(self) -> None:
        env = self.session.env
        logger = self.session.logger
        logger.info("dual grasp env reset connect start")
        env.connect()
        logger.info("dual grasp env reset connect done")
        initial_actions = env.initial_state_actions()
        hand_actions = tuple(_hand_only_action(initial_actions, side) for side in ("right", "left"))
        logger.info("dual grasp env reset hands start sides=right,left")
        env.move_to_actions_until_reached(hand_actions, "dual grasp env reset hands")
        logger.info("dual grasp env reset hands done sides=right,left")
        timeout_s = float(self.session.cfg.demo.reset_servo.timeout_s)
        left_pose = _reset_tcp_pose(self.session.runtime.left, initial_actions)
        right_pose = _reset_tcp_pose(self.session.runtime.right, initial_actions)
        logger.info(
            "dual grasp reset tcp servo start left_target_xyz_base_m=%s right_target_xyz_base_m=%s timeout_s=%.3f",
            left_pose.xyz_base_m,
            right_pose.xyz_base_m,
            timeout_s,
        )
        self.session.together("reset tcp").run(
            [
                skill_task("left_arm", lambda: self.session.motion.left.servo_to_pose(left_pose, timeout_s)),
                skill_task("right_arm", lambda: self.session.motion.right.servo_to_pose(right_pose, timeout_s)),
            ]
        )
        logger.info("dual grasp env reset done")


class ReplaySkills:
    def __init__(self, session: DualGraspDemoSession, replay_data) -> None:
        self.session = session
        self.replay_data = replay_data

    def tianji_trajectory(self) -> None:
        replay_cfg = self.session.cfg.demo.replay
        logger = self.session.logger
        if not bool(replay_cfg.enabled):
            logger.info("dual grasp replay skipped enabled=false")
            return
        if self.replay_data is None:
            raise RuntimeError("dual grasp replay is enabled but replay data is not loaded")
        actions, dataset_fps = self.replay_data
        rate_hz = float(replay_cfg.rate_hz)
        start_frame = int(replay_cfg.start_frame)
        end_frame = int(replay_cfg.end_frame)
        if rate_hz <= 0.0:
            raise ValueError("demo.replay.rate_hz must be positive")
        if start_frame < 0:
            raise ValueError("demo.replay.start_frame must be non-negative")
        if end_frame < start_frame:
            raise ValueError("demo.replay.end_frame must be greater than or equal to start_frame")
        if end_frame >= len(actions):
            raise ValueError(f"demo.replay.end_frame={end_frame} is outside episode length {len(actions)}")
        if actions.shape[1] != 14:
            raise ValueError(f"Tianji dual replay action dim must be 14, got {actions.shape[1]}")

        first_action = tuple(float(value) for value in actions[start_frame])
        left_pose = self.session.runtime.left.tcp_pose_from_arm_q_rad(first_action[:7])
        right_pose = self.session.runtime.right.tcp_pose_from_arm_q_rad(first_action[7:14])
        entry_timeout_s = float(replay_cfg.entry_servo_timeout_s)
        logger.info(
            "dual grasp replay entry servo start left_target_xyz_base_m=%s right_target_xyz_base_m=%s timeout_s=%.3f",
            left_pose.xyz_base_m,
            right_pose.xyz_base_m,
            entry_timeout_s,
        )
        self.session.together("replay entry").run(
            [
                skill_task("left_arm", lambda: self.session.motion.left.servo_to_pose(left_pose, entry_timeout_s)),
                skill_task("right_arm", lambda: self.session.motion.right.servo_to_pose(right_pose, entry_timeout_s)),
            ]
        )

        period_s = 1.0 / rate_hz
        logger.info(
            "dual grasp replay Tianji trajectory start frames=%d..%d replay_hz=%.2f dataset_fps=%.2f",
            start_frame,
            end_frame,
            rate_hz,
            dataset_fps,
        )
        for frame_index in range(start_frame, end_frame + 1):
            loop_start = time.monotonic()
            action = tuple(float(value) for value in actions[frame_index])
            self.session.env.controller.send_actions(
                (
                    RobotAction(side="left", timestamp_ns=now_ns(), arm_q_rad=action[:7]),
                    RobotAction(side="right", timestamp_ns=now_ns(), arm_q_rad=action[7:14]),
                ),
                blocking=False,
            )
            _sleep_remaining(loop_start, period_s)
        logger.info("dual grasp replay Tianji trajectory done frames=%d", end_frame - start_frame + 1)


class DualGraspDemoSession:
    def __init__(
        self,
        cfg: DictConfig,
        logger,
        env,
        runtime: DualDemoRuntimes,
        replay_data,
        live_target_rgbd_frame_bus=None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.env = env
        self.runtime = runtime
        self.live_target_rgbd_frame_bus = live_target_rgbd_frame_bus
        self.runtimes = RuntimeSkills(runtime, logger)
        self.motion = DualMotionSkills(self)
        self.grasp = DualGraspSkills(runtime)
        self.reset = ResetSkills(self)
        self.replay = ReplaySkills(self, replay_data)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        logger,
        robot_group: RobotGroup | None = None,
        live_target_rgbd_frame_bus=None,
    ) -> DualGraspDemoSession:
        left_cfg = task_cfg(cfg, "left")
        right_cfg = task_cfg(cfg, "right")
        _validate_demo_config(left_cfg)
        _validate_demo_config(right_cfg)
        replay_data = load_replay_trajectory(cfg.demo.replay, logger)
        env = build_base_env(cfg, robot_group=robot_group)
        controller_lock = threading.Lock()
        stop_event = threading.Event()
        runtime = DualDemoRuntimes(
            left=_runtime(left_cfg, logger, env, controller_lock, stop_event),
            right=_runtime(right_cfg, logger, env, controller_lock, stop_event),
        )
        return cls(cfg, logger, env, runtime, replay_data, live_target_rgbd_frame_bus=live_target_rgbd_frame_bus)

    def __enter__(self) -> DualGraspDemoSession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.env.close()

    def together(self, label: str) -> Together:
        return Together(label, self.logger, self.runtime.stop_event)

    def live_targets(self, sides: tuple[str, ...]) -> LiveTargetPoseProcess:
        return LiveTargetPoseProcess(self.cfg, sides, rgbd_frame_bus=self.live_target_rgbd_frame_bus)


class SingleLiveGraspDemoSession:
    def __init__(self, cfg: DictConfig, logger, runtime: GraspDemoRuntime) -> None:
        self.cfg = cfg
        self.logger = logger
        self.runtime = runtime
        self.motion = SideMotionSkills(runtime)
        self.grasp = SideGraspSkills(runtime)

    @classmethod
    def from_config(cls, cfg: DictConfig, logger) -> SingleLiveGraspDemoSession:
        _validate_demo_config(cfg)
        runtime = GraspDemoRuntime(
            cfg,
            logger,
            use_internal_perception=False,
            controller_lock=threading.Lock(),
            stop_event=threading.Event(),
        )
        return cls(cfg, logger, runtime)

    def __enter__(self) -> SingleLiveGraspDemoSession:
        self.runtime.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.runtime.__exit__(exc_type, exc, traceback)

    def together(self, label: str) -> Together:
        return Together(label, self.logger, self.runtime.stop_event)

    def live_targets(self) -> LiveTargetPoseProcess:
        return LiveTargetPoseProcess(self.cfg, (str(self.cfg.demo.side),))

    @property
    def side(self) -> str:
        return str(self.cfg.demo.side)


def run_single_live_grasp_demo(cfg: DictConfig, logger) -> None:
    with SingleLiveGraspDemoSession.from_config(cfg, logger) as demo:
        with demo.live_targets() as target_source:
            target_pose = demo.motion.move_to_live_target(target_source)
            demo.grasp.adaptive()
            demo.motion.translate_after_grasp(target_pose)
            demo.grasp.release_to_middle()
            demo.together("post release clearance and release full").run(
                single_release_clearance_tasks(demo)
            )
            demo.runtime.reset_robot()


def single_release_clearance_tasks(demo: SingleLiveGraspDemoSession) -> list[SkillTask[None]]:
    return [
        skill_task(
            f"{demo.side}_arm_clearance",
            lambda: demo.motion.post_release_clearance(hold_grasp=False),
        ),
        skill_task(f"{demo.side}_hand_release_full", demo.grasp.release_full),
    ]


def task_cfg(cfg: DictConfig, side: str) -> DictConfig:
    task = cfg.demo.tasks[side]
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    demo_cfg = OmegaConf.create(OmegaConf.to_container(cfg.demo, resolve=True))
    adaptive_grasp_cfg = OmegaConf.create(OmegaConf.to_container(cfg.adaptive_grasp, resolve=True))
    return OmegaConf.merge(
        base_cfg,
        {
            "demo": OmegaConf.merge(demo_cfg, task.demo),
            "adaptive_grasp": OmegaConf.merge(adaptive_grasp_cfg, task.adaptive_grasp),
        },
    )


def load_replay_trajectory(replay_cfg: DictConfig, logger):
    if not bool(replay_cfg.enabled):
        logger.info("dual grasp replay preload skipped enabled=false")
        return None
    logger.info("dual grasp replay preload start")
    actions, dataset_fps = load_lerobot_episode_actions(replay_cfg.dataset, logger)
    logger.info("dual grasp replay preload done frames=%d dataset_fps=%.2f", len(actions), dataset_fps)
    return actions, dataset_fps


def _runtime(
    cfg: DictConfig,
    logger,
    env,
    controller_lock: threading.Lock,
    stop_event: threading.Event,
) -> GraspDemoRuntime:
    return GraspDemoRuntime(
        cfg,
        logger,
        env=env,
        manage_env=False,
        manage_camera=False,
        use_internal_perception=False,
        controller_lock=controller_lock,
        stop_event=stop_event,
    )


def _reset_tcp_pose(runtime: GraspDemoRuntime, actions: tuple[RobotAction, ...]) -> TcpPose:
    for action in actions:
        if action.side == runtime.side and action.arm_q_rad is not None:
            return runtime.tcp_pose_from_arm_q_rad(action.arm_q_rad)
    raise RuntimeError(f"dual grasp reset missing arm action side={runtime.side}")


def _hand_only_action(actions: tuple[RobotAction, ...], side: str) -> RobotAction:
    for action in actions:
        if action.side == side and action.end_effector_q_rad is not None:
            return RobotAction(
                side=side,
                timestamp_ns=now_ns(),
                end_effector_q_rad=action.end_effector_q_rad,
            )
    raise RuntimeError(f"dual grasp reset missing hand action side={side}")


def _float3(values) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError("world offset must have length 3")
    return (vector[0], vector[1], vector[2])


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)
