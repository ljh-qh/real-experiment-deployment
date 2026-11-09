from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.replay.lerobot_episode import LeRobotEpisodeData
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiTcpPoseMapper, rotation_matrix_from_6d


def build_tcp_trajectory_logger(cfg: DictConfig, episode_data: LeRobotEpisodeData, logger):
    trajectory_cfg = cfg.app.replay.trajectory_log
    if not bool(trajectory_cfg.enabled):
        return None
    if str(trajectory_cfg.type) != "delta_ee_tcp":
        raise ValueError("app.replay.trajectory_log.type must be delta_ee_tcp when enabled")
    if not bool(cfg.wrappers.tianji_delta_ee_action.enabled):
        raise ValueError("app.replay.trajectory_log requires wrappers.tianji_delta_ee_action.enabled=true")
    return TcpTrajectoryLogger(cfg, episode_data, logger)


class TcpTrajectoryLogger:
    def __init__(self, cfg: DictConfig, episode_data: LeRobotEpisodeData, logger) -> None:
        self.cfg = cfg
        self.episode_data = episode_data
        self.logger = logger
        self.rows: list[dict] = []
        self.reset_rows: list[dict] = []
        self.chunk_end_rows: list[dict] = []
        self.closed = False

        self.sides = tuple(
            segment.side
            for segment in RobotActionLayout.from_robot_config(cfg.robot).segments
            if segment.role == "arm"
        )
        if not self.sides:
            raise ValueError("app.replay.trajectory_log requires at least one arm segment")
        self.mappers = {
            side: TianjiTcpPoseMapper.fk_from_config(cfg.robot.hardware.arm, side)
            for side in self.sides
        }
        self.gt_pose_by_side = tcp_pose_state_by_side(episode_data.observations, self.sides)
        self.gt_xyz_by_side = {
            side: np.asarray([_pose_xyz_m(pose) for pose in self.gt_pose_by_side[side]], dtype=np.float64)
            for side in self.sides
        }
        dataset_name = _dataset_name(cfg)
        run_name = time.strftime(f"{dataset_name}_episode_{int(cfg.app.dataset.episode_index)}_%Y%m%d_%H%M%S")
        self.output_dir = Path(str(cfg.app.replay.trajectory_log.output_dir)) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TCP trajectory log enabled output_dir=%s", self.output_dir)

    def record_reset(self, info: dict) -> None:
        states_by_side = {state.side: state for state in tuple(info["robot_states"])}
        target_actions_by_side = {action.side: action for action in tuple(info["target_actions"])}
        for side in self.sides:
            if side not in target_actions_by_side:
                raise RuntimeError(f"TCP trajectory log reset comparison missing target action for side={side}")
            gt_pose = self.gt_pose_by_side[side][0]
            gt_xyz = self.gt_xyz_by_side[side][0]
            actual_pose = _actual_tcp_pose_m(self.mappers[side], states_by_side[side])
            target_pose = _target_tcp_pose_m(self.mappers[side], target_actions_by_side[side])
            actual_xyz = _pose_xyz_m(actual_pose)
            target_xyz = _pose_xyz_m(target_pose)
            actual_vs_gt_cm = _error_cm(actual_xyz, gt_xyz)
            target_vs_gt_cm = _error_cm(target_xyz, gt_xyz)
            actual_vs_target_cm = _error_cm(actual_xyz, target_xyz)
            actual_vs_gt_rot_deg = _rotation_error_deg(actual_pose, gt_pose)
            target_vs_gt_rot_deg = _rotation_error_deg(target_pose, gt_pose)
            actual_vs_target_rot_deg = _rotation_error_deg(actual_pose, target_pose)
            row = {
                "dataset_frame_index": int(self.episode_data.frame_indices[0]),
                "timestamp_s": float(self.episode_data.timestamps[0]),
                "side": side,
            }
            row.update(_xyz_columns("gt_frame0", gt_xyz))
            row.update(_xyz_columns("reset_target", target_xyz))
            row.update(_xyz_columns("reset_actual", actual_xyz))
            row.update(_error_columns("reset_actual_vs_gt_frame0", actual_vs_gt_cm))
            row.update(_error_columns("reset_target_vs_gt_frame0", target_vs_gt_cm))
            row.update(_error_columns("reset_actual_vs_target", actual_vs_target_cm))
            row.update(_rotation_error_column("reset_actual_vs_gt_frame0", actual_vs_gt_rot_deg))
            row.update(_rotation_error_column("reset_target_vs_gt_frame0", target_vs_gt_rot_deg))
            row.update(_rotation_error_column("reset_actual_vs_target", actual_vs_target_rot_deg))
            self.reset_rows.append(row)

    def record(self, action_index: int, info: dict) -> None:
        states_by_side = {state.side: state for state in tuple(info["robot_states"])}
        target_actions_by_side = {action.side: action for action in tuple(info["target_actions"])}
        trace_by_side = info["delta_ee_trace"]
        gt_target_index = action_index + 1
        gt_target_available = gt_target_index < len(self.episode_data.observations)
        for side in self.sides:
            gt_current_pose = self.gt_pose_by_side[side][action_index]
            gt_target_pose = self.gt_pose_by_side[side][gt_target_index] if gt_target_available else _nan_pose()
            gt_current_xyz = self.gt_xyz_by_side[side][action_index]
            gt_target_xyz = self.gt_xyz_by_side[side][gt_target_index] if gt_target_available else _nan_xyz()
            delta_ee_trace = trace_by_side[side]
            delta_base_pose = _trace_pose_m(delta_ee_trace.base_pose_world)
            desired_pose = _trace_pose_m(delta_ee_trace.desired_pose_world)
            debug_target_pose = _trace_pose_m(delta_ee_trace.target_pose_world)
            delta_base_xyz = _pose_xyz_m(delta_base_pose)
            desired_xyz = _pose_xyz_m(desired_pose)
            debug_target_xyz = _pose_xyz_m(debug_target_pose)
            delta_xyz = _trace_xyz_m(delta_ee_trace.delta_xyz_m)
            actual_pose = _actual_tcp_pose_m(self.mappers[side], states_by_side[side])
            target_pose = _target_tcp_pose_m(self.mappers[side], target_actions_by_side[side])
            actual_xyz = _pose_xyz_m(actual_pose)
            target_xyz = _pose_xyz_m(target_pose)
            row = {
                "action_index": int(action_index),
                "dataset_frame_index": int(self.episode_data.frame_indices[action_index]),
                "timestamp_s": float(self.episode_data.timestamps[action_index]),
                "gt_target_action_index": int(gt_target_index) if gt_target_available else -1,
                "gt_target_dataset_frame_index": int(self.episode_data.frame_indices[gt_target_index])
                if gt_target_available
                else -1,
                "side": side,
            }
            row.update(_xyz_columns("gt_current", gt_current_xyz))
            row.update(_xyz_columns("gt_target", gt_target_xyz))
            row.update(_xyz_columns("delta_base", delta_base_xyz))
            row.update(_xyz_columns("delta_xyz", delta_xyz))
            row.update(_xyz_columns("desired", desired_xyz))
            row.update(_xyz_columns("debug_target", debug_target_xyz))
            row.update(_xyz_columns("target", target_xyz))
            row.update(_xyz_columns("actual", actual_xyz))
            _add_error_columns(
                row,
                "delta_base_vs_gt_current",
                delta_base_xyz,
                gt_current_xyz,
                delta_base_pose,
                gt_current_pose,
            )
            _add_error_columns(row, "desired_vs_gt_target", desired_xyz, gt_target_xyz, desired_pose, gt_target_pose)
            _add_error_columns(row, "target_vs_desired", target_xyz, desired_xyz, target_pose, desired_pose)
            _add_error_columns(row, "debug_target_vs_target", debug_target_xyz, target_xyz, debug_target_pose, target_pose)
            _add_error_columns(row, "actual_vs_gt_target", actual_xyz, gt_target_xyz, actual_pose, gt_target_pose)
            _add_error_columns(row, "target_vs_gt_target", target_xyz, gt_target_xyz, target_pose, gt_target_pose)
            _add_error_columns(row, "actual_vs_target", actual_xyz, target_xyz, actual_pose, target_pose)
            self.rows.append(row)

    def record_chunk_end(
        self,
        chunk_start_action_index: int,
        chunk_end_action_index: int,
        states: tuple,
        info: dict,
    ) -> None:
        states_by_side = {state.side: state for state in tuple(states)}
        target_actions_by_side = {action.side: action for action in tuple(info["target_actions"])}
        gt_target_index = chunk_end_action_index + 1
        gt_target_available = gt_target_index < len(self.episode_data.observations)
        for side in self.sides:
            gt_target_pose = self.gt_pose_by_side[side][gt_target_index] if gt_target_available else _nan_pose()
            gt_target_xyz = self.gt_xyz_by_side[side][gt_target_index] if gt_target_available else _nan_xyz()
            actual_pose = _actual_tcp_pose_m(self.mappers[side], states_by_side[side])
            target_pose = _target_tcp_pose_m(self.mappers[side], target_actions_by_side[side])
            actual_xyz = _pose_xyz_m(actual_pose)
            target_xyz = _pose_xyz_m(target_pose)
            row = {
                "chunk_start_action_index": int(chunk_start_action_index),
                "chunk_end_action_index": int(chunk_end_action_index),
                "chunk_size": int(chunk_end_action_index - chunk_start_action_index + 1),
                "chunk_end_dataset_frame_index": int(self.episode_data.frame_indices[chunk_end_action_index]),
                "chunk_end_timestamp_s": float(self.episode_data.timestamps[chunk_end_action_index]),
                "gt_target_action_index": int(gt_target_index) if gt_target_available else -1,
                "gt_target_dataset_frame_index": int(self.episode_data.frame_indices[gt_target_index])
                if gt_target_available
                else -1,
                "side": side,
            }
            row.update(_xyz_columns("gt_target", gt_target_xyz))
            row.update(_xyz_columns("target", target_xyz))
            row.update(_xyz_columns("actual_after_wait", actual_xyz))
            _add_error_columns(row, "target_vs_gt_target", target_xyz, gt_target_xyz, target_pose, gt_target_pose)
            _add_error_columns(
                row,
                "actual_after_wait_vs_gt_target",
                actual_xyz,
                gt_target_xyz,
                actual_pose,
                gt_target_pose,
            )
            _add_error_columns(row, "actual_after_wait_vs_target", actual_xyz, target_xyz, actual_pose, target_pose)
            self.chunk_end_rows.append(row)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.reset_rows:
            self._write_reset_rows()
        if self.rows:
            self._write_step_rows()
        else:
            self.logger.info("TCP trajectory log skipped step output because no rows were recorded")
        if self.chunk_end_rows:
            self._write_chunk_end_rows()

    def _write_reset_rows(self) -> None:
        import pandas as pd

        reset_csv_path = self.output_dir / "reset_pose_compare.csv"
        reset_summary_path = self.output_dir / "reset_summary.csv"
        reset_df = pd.DataFrame(self.reset_rows)
        reset_df.to_csv(reset_csv_path, index=False)
        reset_summary = _reset_summary(reset_df, self.sides)
        reset_summary.to_csv(reset_summary_path, index=False)
        self.logger.info(
            "TCP trajectory log wrote reset_csv=%s reset_summary=%s",
            reset_csv_path,
            reset_summary_path,
        )

    def _write_step_rows(self) -> None:
        import pandas as pd

        csv_path = self.output_dir / "actual_tcp_trajectory_compare.csv"
        summary_path = self.output_dir / "summary.csv"
        df = pd.DataFrame(self.rows)
        df.to_csv(csv_path, index=False)
        summary = _trajectory_summary(df, self.sides)
        summary.to_csv(summary_path, index=False)
        self.logger.info("TCP trajectory log wrote csv=%s summary=%s", csv_path, summary_path)
        if bool(self.cfg.app.replay.trajectory_log.plot):
            plot_path = self.output_dir / "actual_tcp_trajectory_compare.png"
            _plot_actual_trajectory(df, plot_path, self.sides)
            self.logger.info("TCP trajectory log wrote plot=%s", plot_path)

    def _write_chunk_end_rows(self) -> None:
        import pandas as pd

        chunk_csv_path = self.output_dir / "chunk_end_tcp_compare.csv"
        chunk_summary_path = self.output_dir / "chunk_end_summary.csv"
        chunk_df = pd.DataFrame(self.chunk_end_rows)
        chunk_df.to_csv(chunk_csv_path, index=False)
        chunk_summary = _chunk_end_summary(chunk_df, self.sides)
        chunk_summary.to_csv(chunk_summary_path, index=False)
        self.logger.info(
            "TCP trajectory log wrote chunk_csv=%s chunk_summary=%s",
            chunk_csv_path,
            chunk_summary_path,
        )
        if bool(self.cfg.app.replay.trajectory_log.plot):
            chunk_plot_path = self.output_dir / "chunk_end_tcp_compare.png"
            _plot_chunk_end_trajectory(chunk_df, chunk_plot_path, self.sides)
            self.logger.info("TCP trajectory log wrote chunk_plot=%s", chunk_plot_path)


def _dataset_name(cfg: DictConfig) -> str:
    if cfg.app.dataset.root is not None:
        name = Path(str(cfg.app.dataset.root)).name
    else:
        name = str(cfg.app.dataset.repo_id)
    return name.replace("/", "_")


def tcp_pose_state_by_side(observations: np.ndarray, sides: tuple[str, ...]) -> dict[str, np.ndarray]:
    if observations.ndim != 2:
        raise ValueError("delta_ee_tcp trajectory log requires observation.state to be a 2D array")
    expected_dim = len(sides) * 10
    if observations.shape[1] != expected_dim:
        raise ValueError(
            "delta_ee_tcp trajectory log requires observation.state from "
            f"convert.state.type=tcp_pose_world_with_hand_alpha with dim {expected_dim}, "
            f"got {observations.shape[1]}"
        )
    return {
        side: np.asarray(
            [
                _tcp_pose_from_state_values(values)
                for values in observations[:, side_index * 10 : side_index * 10 + 10]
            ],
            dtype=np.float64,
        )
        for side_index, side in enumerate(sides)
    }


def _tcp_pose_from_state_values(values: np.ndarray) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64).reshape(-1)
    if state.shape != (10,):
        raise ValueError("tcp_pose_world_with_hand_alpha state segment must have length 10")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = state[:3]
    pose[:3, :3] = rotation_matrix_from_6d(state[3:9])
    return pose


def _actual_tcp_pose_m(mapper: TianjiTcpPoseMapper, state) -> np.ndarray:
    if state.arm_joint is not None:
        return _tcp_pose_m(mapper, state.arm_joint.q_rad)
    if state.tianji is not None:
        return _tcp_pose_m(mapper, state.tianji.joint_pos)
    raise RuntimeError(f"RobotState missing arm joint state for side {state.side}")


def _target_tcp_pose_m(mapper: TianjiTcpPoseMapper, action) -> np.ndarray:
    if action.arm_q_rad is None:
        raise RuntimeError(f"RobotAction missing arm_q_rad for side {action.side}")
    return _tcp_pose_m(mapper, action.arm_q_rad)


def _trace_xyz_m(values) -> np.ndarray:
    return np.asarray(tuple(float(value) for value in values), dtype=np.float64)


def _trace_pose_m(values) -> np.ndarray:
    pose = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if pose.shape != (16,):
        raise ValueError("delta ee trace pose must have length 16")
    return pose.reshape(4, 4)


def _tcp_pose_m(mapper: TianjiTcpPoseMapper, q_rad) -> np.ndarray:
    return np.asarray(mapper.tcp_pose_world_from_q_rad(tuple(float(value) for value in q_rad)), dtype=np.float64)


def _pose_xyz_m(pose: np.ndarray) -> np.ndarray:
    return np.asarray((pose[0, 3], pose[1, 3], pose[2, 3]), dtype=np.float64)


def _nan_xyz() -> np.ndarray:
    return np.asarray((float("nan"), float("nan"), float("nan")), dtype=np.float64)


def _nan_pose() -> np.ndarray:
    return np.full((4, 4), float("nan"), dtype=np.float64)


def _error_cm(actual_xyz: np.ndarray, expected_xyz: np.ndarray) -> np.ndarray:
    return (actual_xyz - expected_xyz) * 100.0


def _rotation_error_deg(actual_pose: np.ndarray, expected_pose: np.ndarray) -> float:
    if np.any(np.isnan(actual_pose)) or np.any(np.isnan(expected_pose)):
        return float("nan")
    rotation_delta = actual_pose[:3, :3] @ expected_pose[:3, :3].T
    cos_angle = (float(np.trace(rotation_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _xyz_columns(prefix: str, xyz: np.ndarray) -> dict:
    return {
        f"{prefix}_x_m": float(xyz[0]),
        f"{prefix}_y_m": float(xyz[1]),
        f"{prefix}_z_m": float(xyz[2]),
    }


def _error_columns(prefix: str, error_cm: np.ndarray) -> dict:
    return {
        f"{prefix}_err_x_cm": float(error_cm[0]),
        f"{prefix}_err_y_cm": float(error_cm[1]),
        f"{prefix}_err_z_cm": float(error_cm[2]),
        f"{prefix}_err_norm_cm": float(np.linalg.norm(error_cm)) if not np.any(np.isnan(error_cm)) else float("nan"),
    }


def _rotation_error_column(prefix: str, error_deg: float) -> dict:
    return {f"{prefix}_rot_err_deg": float(error_deg)}


def _add_error_columns(
    row: dict,
    prefix: str,
    actual_xyz: np.ndarray,
    expected_xyz: np.ndarray,
    actual_pose: np.ndarray,
    expected_pose: np.ndarray,
) -> None:
    row.update(_error_columns(prefix, _error_cm(actual_xyz, expected_xyz)))
    row.update(_rotation_error_column(prefix, _rotation_error_deg(actual_pose, expected_pose)))


def _trajectory_summary(df, sides: tuple[str, ...]):
    import pandas as pd

    rows = []
    for side in sides:
        side_df = df[df["side"] == side]
        row = {"side": side, "frames": int(len(side_df))}
        for prefix in (
            "delta_base_vs_gt_current",
            "desired_vs_gt_target",
            "target_vs_desired",
            "debug_target_vs_target",
            "actual_vs_gt_target",
            "target_vs_gt_target",
            "actual_vs_target",
        ):
            _add_summary_columns(row, side_df, prefix, "action_index")
        rows.append(row)
    return pd.DataFrame(rows)


def _reset_summary(df, sides: tuple[str, ...]):
    import pandas as pd

    rows = []
    for side in sides:
        side_df = df[df["side"] == side]
        if len(side_df) != 1:
            raise ValueError("reset pose compare must contain one row per side")
        row_data = side_df.iloc[0]
        row = {"side": side}
        for prefix in ("reset_actual_vs_gt_frame0", "reset_target_vs_gt_frame0", "reset_actual_vs_target"):
            row[f"{prefix}_x_cm"] = float(row_data[f"{prefix}_err_x_cm"])
            row[f"{prefix}_y_cm"] = float(row_data[f"{prefix}_err_y_cm"])
            row[f"{prefix}_z_cm"] = float(row_data[f"{prefix}_err_z_cm"])
            row[f"{prefix}_norm_cm"] = float(row_data[f"{prefix}_err_norm_cm"])
            row[f"{prefix}_rot_deg"] = float(row_data[f"{prefix}_rot_err_deg"])
        rows.append(row)
    return pd.DataFrame(rows)


def _chunk_end_summary(df, sides: tuple[str, ...]):
    import pandas as pd

    rows = []
    for side in sides:
        side_df = df[df["side"] == side]
        row = {"side": side, "chunks": int(len(side_df))}
        for prefix in (
            "target_vs_gt_target",
            "actual_after_wait_vs_gt_target",
            "actual_after_wait_vs_target",
        ):
            _add_summary_columns(row, side_df, prefix, "chunk_end_action_index")
        rows.append(row)
    return pd.DataFrame(rows)


def _add_summary_columns(row: dict, side_df, prefix: str, index_column: str) -> None:
    norm = side_df[f"{prefix}_err_norm_cm"].dropna()
    if len(norm) == 0:
        row[f"{prefix}_mean_cm"] = float("nan")
        row[f"{prefix}_max_cm"] = float("nan")
        row[f"{prefix}_max_{index_column}"] = -1
        row[f"{prefix}_rot_mean_deg"] = float("nan")
        row[f"{prefix}_rot_max_deg"] = float("nan")
        row[f"{prefix}_rot_max_{index_column}"] = -1
        return
    max_position = int(norm.to_numpy().argmax())
    max_label = norm.index[max_position]
    row[f"{prefix}_mean_cm"] = float(norm.mean())
    row[f"{prefix}_max_cm"] = float(norm.iloc[max_position])
    row[f"{prefix}_max_{index_column}"] = int(side_df.loc[max_label, index_column])
    rot = side_df[f"{prefix}_rot_err_deg"].dropna()
    if len(rot) == 0:
        row[f"{prefix}_rot_mean_deg"] = float("nan")
        row[f"{prefix}_rot_max_deg"] = float("nan")
        row[f"{prefix}_rot_max_{index_column}"] = -1
        return
    rot_max_position = int(rot.to_numpy().argmax())
    rot_max_label = rot.index[rot_max_position]
    row[f"{prefix}_rot_mean_deg"] = float(rot.mean())
    row[f"{prefix}_rot_max_deg"] = float(rot.iloc[rot_max_position])
    row[f"{prefix}_rot_max_{index_column}"] = int(side_df.loc[rot_max_label, index_column])


def _plot_actual_trajectory(df, plot_path: Path, sides: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(sides), 4, figsize=(24, 4.5 * len(sides)), constrained_layout=True, squeeze=False)
    for row_index, side in enumerate(sides):
        side_df = df[df["side"] == side]
        action_index = side_df["action_index"].to_numpy()

        ax = axes[row_index, 0]
        ax.plot(side_df["gt_target_y_m"], side_df["gt_target_x_m"], label="GT target", linewidth=2)
        ax.plot(side_df["desired_y_m"], side_df["desired_x_m"], label="Delta desired", linestyle="-.")
        ax.plot(side_df["target_y_m"], side_df["target_x_m"], label="Command target", linestyle="--")
        ax.plot(side_df["actual_y_m"], side_df["actual_x_m"], label="Actual", linestyle=":")
        ax.set_title(f"{side} TCP XY")
        ax.set_xlabel("world y (m)")
        ax.set_ylabel("world x (m)")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

        ax = axes[row_index, 1]
        ax.plot(action_index, side_df["actual_vs_gt_target_err_x_cm"], label="x")
        ax.plot(action_index, side_df["actual_vs_gt_target_err_y_cm"], label="y")
        ax.plot(action_index, side_df["actual_vs_gt_target_err_z_cm"], label="z")
        ax.set_title(f"{side} actual - GT target")
        ax.set_xlabel("action index")
        ax.set_ylabel("cm")
        ax.grid(True)
        ax.legend()

        ax = axes[row_index, 2]
        ax.plot(action_index, side_df["desired_vs_gt_target_err_norm_cm"], label="desired vs GT target")
        ax.plot(action_index, side_df["target_vs_desired_err_norm_cm"], label="command target vs desired")
        ax.plot(action_index, side_df["actual_vs_gt_target_err_norm_cm"], label="actual vs GT target")
        ax.plot(action_index, side_df["target_vs_gt_target_err_norm_cm"], label="command target vs GT target")
        ax.plot(action_index, side_df["actual_vs_target_err_norm_cm"], label="actual vs command target")
        ax.set_title(f"{side} TCP position error norm")
        ax.set_xlabel("action index")
        ax.set_ylabel("cm")
        ax.grid(True)
        ax.legend()

        ax = axes[row_index, 3]
        ax.plot(action_index, side_df["desired_vs_gt_target_rot_err_deg"], label="desired vs GT target")
        ax.plot(action_index, side_df["target_vs_desired_rot_err_deg"], label="command target vs desired")
        ax.plot(action_index, side_df["actual_vs_gt_target_rot_err_deg"], label="actual vs GT target")
        ax.plot(action_index, side_df["target_vs_gt_target_rot_err_deg"], label="command target vs GT target")
        ax.plot(action_index, side_df["actual_vs_target_rot_err_deg"], label="actual vs command target")
        ax.set_title(f"{side} TCP rotation error")
        ax.set_xlabel("action index")
        ax.set_ylabel("deg")
        ax.grid(True)
        ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _plot_chunk_end_trajectory(df, plot_path: Path, sides: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(sides), 2, figsize=(14, 4.0 * len(sides)), constrained_layout=True, squeeze=False)
    for row_index, side in enumerate(sides):
        side_df = df[df["side"] == side]
        action_index = side_df["chunk_end_action_index"].to_numpy()

        ax = axes[row_index, 0]
        ax.plot(action_index, side_df["target_vs_gt_target_err_norm_cm"], label="command target vs GT target")
        ax.plot(
            action_index,
            side_df["actual_after_wait_vs_gt_target_err_norm_cm"],
            label="actual after wait vs GT target",
        )
        ax.plot(
            action_index,
            side_df["actual_after_wait_vs_target_err_norm_cm"],
            label="actual after wait vs command target",
        )
        ax.set_title(f"{side} chunk-end TCP position error")
        ax.set_xlabel("chunk end action index")
        ax.set_ylabel("cm")
        ax.grid(True)
        ax.legend()

        ax = axes[row_index, 1]
        ax.plot(action_index, side_df["target_vs_gt_target_rot_err_deg"], label="command target vs GT target")
        ax.plot(
            action_index,
            side_df["actual_after_wait_vs_gt_target_rot_err_deg"],
            label="actual after wait vs GT target",
        )
        ax.plot(
            action_index,
            side_df["actual_after_wait_vs_target_rot_err_deg"],
            label="actual after wait vs command target",
        )
        ax.set_title(f"{side} chunk-end TCP rotation error")
        ax.set_xlabel("chunk end action index")
        ax.set_ylabel("deg")
        ax.grid(True)
        ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
