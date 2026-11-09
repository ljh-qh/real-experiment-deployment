"""Validate, dry-run, or explicitly replay an Object episode on arms and hands."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import h5py
import numpy as np

from .episode_check import EpisodeError, inspect_episode, write_report
from .sharpa_backend import LOWER_RAD, UPPER_RAD, SharpaReplayBackend, discover, load_sharpa
from .tianji_backend import DEFAULT_JOINT_LIMITS_DEG, TianjiJointBackend, smoothstep


@dataclass
class Track:
    time: np.ndarray
    joints: np.ndarray

    def at(self, stamp):
        return np.array([np.interp(stamp, self.time, self.joints[:, j])
                         for j in range(self.joints.shape[1])])


def load_tracks(path, devices="both", arm_source="measured"):
    tracks = {}
    path = Path(path)
    if path.suffix == ".npz":
        # External 58-column datasets (Ex2in joint_commands / CAP action):
        # cols 0-13 arms (7L+7R), cols 14-57 hands (22L+22R), time in seconds.
        with np.load(path) as z:
            time_s = np.asarray(z["time_s"], dtype=float)
            key = "position_rad" if "position_rad" in z else "joint_target"
            joints = np.asarray(z[key], dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 58:
            raise EpisodeError(f"npz 关节列数 {joints.shape} 不符（应为 N×58：14 臂 + 44 手）")
        if len(time_s) != len(joints) or np.any(np.diff(time_s) <= 0):
            raise EpisodeError("npz time_s 与关节行数不符或非递增")
        if devices in ("both", "arms"):
            tracks["arms"] = Track(time_s.copy(), joints[:, :14].copy())
        if devices in ("both", "hands"):
            tracks["hands"] = Track(time_s.copy(), joints[:, 14:].copy())
        return tracks
    with h5py.File(path / "robot_states.h5", "r") as h5:
        if devices in ("both", "hands"):
            hand = h5["robot_state/sharpa_joints"]
            tracks["hands"] = Track(np.asarray(hand["sys_time_ns"], dtype=np.int64),
                                    np.concatenate([hand["left_rad"][:], hand["right_rad"][:]], axis=1))
        if devices in ("both", "arms"):
            arm = h5["robot_state/pico_teleop"]
            joints = (np.deg2rad(np.concatenate([arm["real_left_deg"][:], arm["real_right_deg"][:]], axis=1))
                      if arm_source == "measured" else arm["command_rad"][:])
            tracks["arms"] = Track(np.asarray(arm["sys_time_ns"], dtype=np.int64), joints)
    # Subtract in int64 before conversion, preserving nanosecond precision.
    origin = min(int(t.time[0]) for t in tracks.values())
    for track in tracks.values():
        track.time = (track.time - origin).astype(float) / 1e9
    return tracks


def arm_sag_deg(path):
    """Mean/max (measured - command) per arm in degrees: how far the arm hangs below target.

    Impedance control leaves a steady-state error, so replaying the measured
    trajectory makes the arm sag a second time (~3 cm at the TCP). Reported so
    the operator can see the size of that offset for this episode.
    """
    with h5py.File(Path(path) / "robot_states.h5", "r") as h5:
        arm = h5["robot_state/pico_teleop"]
        cmd = np.rad2deg(np.asarray(arm["command_rad"][:], dtype=float))
        out = {}
        for side, key, cols in (("left", "real_left_deg", slice(0, 7)),
                                ("right", "real_right_deg", slice(7, 14))):
            delta = np.asarray(arm[key][:], dtype=float) - cmd[:, cols]
            out[side] = (float(np.mean(np.abs(delta))), float(np.max(np.abs(delta))))
        return out


def prepare_tracks(tracks, speed=.25, arm_speed=30., hand_speed=90., min_motion=1.,
                   require_both_sides=True):
    """Validate native limits and scaled velocity before any SDK import/connect."""
    summary = {}
    start = max(track.time[0] for track in tracks.values())
    end = min(track.time[-1] for track in tracks.values())
    if end - start < .5:
        raise EpisodeError("Less than 0.5s common arm/hand time coverage")
    arm_limits = np.deg2rad(np.concatenate([DEFAULT_JOINT_LIMITS_DEG[s]
                                           for s in ("left", "right")]))
    limits = {"arms": arm_limits,
              "hands": np.stack([np.tile(LOWER_RAD, 2), np.tile(UPPER_RAD, 2)], axis=1)}
    for name, track in tracks.items():
        bound = limits[name]
        # Feedback may overshoot the mechanical boundary by encoder noise.
        # Never change the recording: report every <=1-degree replay clamp.
        # External npz (sim-generated) trajectories may sit exactly on the arm
        # boundary with float rounding dust; allow a small clip margin there.
        tolerance = np.deg2rad(1.) if name == "hands" else (
            np.deg2rad(.5) if not require_both_sides else 0.)
        if (np.any(track.joints < bound[:, 0] - tolerance)
                or np.any(track.joints > bound[:, 1] + tolerance)):
            raise EpisodeError(f"{name}: recorded angles outside joint limits")
        clipped = np.clip(track.joints, bound[:, 0], bound[:, 1])
        adjusted = int(np.count_nonzero(np.abs(clipped - track.joints) > 1e-9))
        track.joints = clipped
        velocity = np.max(np.abs(np.diff(clipped, axis=0)) / np.diff(track.time)[:, None])
        scaled = float(np.rad2deg(velocity) * speed)
        cap = arm_speed if name == "arms" else hand_speed
        if scaled > cap:
            suggested = cap / max(float(np.rad2deg(velocity)), 1e-9) * .95
            raise EpisodeError(f"{name}: requested speed needs {scaled:.1f} deg/s > {cap:.1f}; "
                               f"use --speed <= {suggested:.4f}")
        width = 7 if name == "arms" else 22
        window = np.vstack([track.at(start), clipped[(track.time > start) & (track.time < end)],
                            track.at(end)])
        ranges = [float(np.rad2deg(np.ptp(window[:, i * width:(i + 1) * width], axis=0)).max())
                  for i in (0, 1)]
        if min(ranges) < min_motion:
            if require_both_sides or max(ranges) < min_motion:
                raise EpisodeError(f"{name}: one side moves less than {min_motion:g} deg; "
                                   "record a trial with motion on both sides")
        summary[name] = {"rows": len(clipped), "max_scaled_speed_deg_s": scaled,
                         "left_motion_deg": ranges[0], "right_motion_deg": ranges[1],
                         "boundary_noise_values_clamped": adjusted}
    summary["timeline"] = {"start_s": float(start), "stop_s": float(end),
                           "source_duration_s": float(end - start),
                           "replay_duration_s": float((end - start) / speed),
                           "trimmed_edges_s": {k: [float(start - t.time[0]), float(t.time[-1] - end)]
                                               for k, t in tracks.items()}}
    return float(start), float(end), summary


def competing_writers():
    matches = []
    names = {"mujoco_marvin.cli", "retargeting_manus_demo_multiprocess.py",
             "wave_ros_server.py", "sharpa_wave_safe_server.py"}
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        if any(Path(arg).name in names or arg in names for arg in args):
            matches.append(int(proc.name))
    return matches


def arm_feedback(backend):
    values = np.deg2rad(np.concatenate(backend.read_joints_deg()))
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Invalid live dual-arm feedback")
    limits = np.deg2rad(np.concatenate([DEFAULT_JOINT_LIMITS_DEG[side]
                                       for side in ("left", "right")]))
    tolerance = np.deg2rad(1.)
    if np.any(values < limits[:, 0] - tolerance) or np.any(values > limits[:, 1] + tolerance):
        raise RuntimeError("Live dual-arm feedback outside joint limits")
    return values


def sleep_until(deadline):
    delay = deadline - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    if time.monotonic() - deadline > .15:
        raise RuntimeError("Replay scheduler late by >150ms; stopped instead of catching up")


def verify_motion(expected_min, expected_max, actual_min, actual_max, squared_error, stats):
    """A small target error alone cannot prove that the hardware moved."""
    motion = stats["motion"] = {}
    failures = []
    for name in expected_min:
        width = 7 if name == "arms" else 22
        target_range = np.rad2deg(expected_max[name] - expected_min[name])
        measured_range = np.rad2deg(actual_max[name] - actual_min[name])
        rms = np.rad2deg(np.sqrt(squared_error[name] / stats["ticks"]))
        stats.setdefault("rms_feedback_error_deg", {})[name] = rms.tolist()
        motion[name] = {}
        for i, side in enumerate(("left", "right")):
            indices = slice(i * width, (i + 1) * width)
            joint = i * width + int(np.argmax(target_range[indices]))
            expected, measured = float(target_range[joint]), float(measured_range[joint])
            motion[name][side] = {
                "reference_joint_index": joint - i * width,
                "expected_motion_deg": expected,
                "measured_motion_deg": measured,
                "rms_error_deg": float(rms[joint]),
            }
            if expected < .5 or measured < .5 * expected:
                failures.append(f"{name}/{side}: expected {expected:.2f} deg motion, "
                                f"measured {measured:.2f}; feedback may be frozen")
            elif rms[joint] > max(.5, .5 * expected):
                failures.append(f"{name}/{side}: RMS feedback error {rms[joint]:.2f} deg "
                                f"too large for {expected:.2f} deg trial motion")
    stats["motion_verified"] = not failures
    if failures:
        raise RuntimeError("; ".join(failures))


def run_timeline(tracks, backends, readers, start, end, *, speed, hz,
                 arm_error, hand_error, log, settle=1., progress=None):
    """Use a common clock and retain feedback/statistics even on failed ticks."""
    stats = progress if progress is not None else {}
    stats.update(ticks=0, attempted_ticks=0, motion_verified=False,
                 max_feedback_error_deg={name: 0. for name in tracks})
    previous = {name: track.at(start) for name, track in tracks.items()}
    expected_min = {k: v.copy() for k, v in previous.items()}
    expected_max = {k: v.copy() for k, v in previous.items()}
    actual_min, actual_max = {}, {}
    squared_error = {name: np.zeros_like(track.joints[0], dtype=float)
                     for name, track in tracks.items()}
    t0 = time.monotonic()
    ticks = math.ceil((end - start) / speed * hz)
    total = ticks + math.ceil(settle * hz)
    for i in range(total + 1):
        deadline = t0 + i / hz
        source_time = min(end, start + min(i, ticks) / hz * speed)
        commands = {name: track.at(source_time) for name, track in tracks.items()}
        entry = {"mono": time.monotonic(), "source_time_s": source_time,
                 "command_rad": {k: v.tolist() for k, v in commands.items()},
                 "tracking_target_rad": {k: v.tolist() for k, v in previous.items()},
                 "feedback_rad": {}, "feedback_mono": {}, "sent_devices": []}
        stats["attempted_ticks"] += 1
        try:
            sleep_until(deadline)
            entry["mono"] = time.monotonic()
            for name, read in readers.items():
                actual = np.asarray(read(), dtype=float)
                if actual.shape != previous[name].shape or not np.isfinite(actual).all():
                    raise RuntimeError(f"{name}: invalid live feedback")
                entry["feedback_rad"][name] = actual.tolist()
                entry["feedback_mono"][name] = time.monotonic()
                squared_error[name] += (actual - previous[name]) ** 2
                actual_min[name] = np.minimum(actual_min.get(name, actual), actual)
                actual_max[name] = np.maximum(actual_max.get(name, actual), actual)
                error = float(np.rad2deg(np.max(np.abs(actual - previous[name]))))
                stats["max_feedback_error_deg"][name] = max(stats["max_feedback_error_deg"][name], error)
                limit = arm_error if name == "arms" else hand_error
                if error > limit:
                    raise RuntimeError(f"{name}: feedback error {error:.2f} deg exceeds {limit:.2f}")
            if time.monotonic() - deadline > .15:
                raise RuntimeError("Replay feedback read exceeded the 150ms scheduling deadline")
            for name, backend in backends.items():
                backend.write(commands[name])
                entry["sent_devices"].append(name)
            for name, command in commands.items():
                expected_min[name] = np.minimum(expected_min[name], command)
                expected_max[name] = np.maximum(expected_max[name], command)
            previous = commands
            stats["ticks"] += 1
        except BaseException as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stats["elapsed_s"] = time.monotonic() - t0
            log.write(json.dumps(entry, allow_nan=False) + "\n")
    verify_motion(expected_min, expected_max, actual_min, actual_max, squared_error, stats)
    return stats


def execute(tracks, args, start, end, log, report):
    writers = competing_writers()
    if writers:
        raise EpisodeError(f"Other teleop writers still running (PIDs {writers}); stop teleop before replay")
    with_hands = "hands" in tracks
    serials = {side: os.environ.get(f"SHARPA_{side.upper()}_SN", "").strip()
               for side in ("left", "right")}
    is_npz = getattr(args.episode, "suffix", "") == ".npz"
    if with_hands and not is_npz:
        for side, sn in serials.items():
            recorded = report["data_check"].get(side + "_sn")
            if not recorded:
                raise EpisodeError(f"{side}: recording lacks Sharpa device identity metadata")
            if recorded != sn:
                raise EpisodeError(f"{side}: configured Sharpa identity differs from recording")
        schema = report["data_check"].get("sharpa_joints_schema", {})
        if schema.get("units") != "rad" or not schema.get("native_joint_order"):
            raise EpisodeError("Hardware replay requires explicit Sharpa radians and native joint-order metadata")
    if with_hands and is_npz:
        if not all(serials.values()):
            raise EpisodeError("npz 回放仍需 SHARPA_LEFT_SN/SHARPA_RIGHT_SN 环境变量")
        report["data_check"] = {"ok": True, "scope": "external npz: 无录制身份元数据,"
                                "手部身份以当前环境变量为准"}
    if args.devices in ("both", "arms") and not args.robot_ip:
        raise EpisodeError("TIANJI_ROBOT_IP or --robot-ip is required")
    with ExitStack() as resources:
        hands = None
        hand_delta = None
        if with_hands:
            hands = SharpaReplayBackend(serials)
            resources.callback(hands.close)
            initial_hand = hands.read()
            first_hand = tracks["hands"].at(start)
            hand_delta = float(np.rad2deg(np.max(np.abs(first_hand - initial_hand))))
            if hand_delta > args.max_start_hand_deg:
                raise EpisodeError(f"Initial hand distance {hand_delta:.1f} deg exceeds --max-start-hand-deg")
            report["initial_hand_distance_deg"] = hand_delta
        arm = None
        if "arms" in tracks:
            first_arm = tracks["arms"].at(start)
            home = {"left": np.rad2deg(first_arm[:7]).tolist(),
                    "right": np.rad2deg(first_arm[7:]).tolist()}
            arm = TianjiJointBackend(args.robot_ip, home_deg=home, sdk_dir=args.sdk_dir,
                                     ramp_speed_deg_s=args.ramp_speed, initialize=False)
            resources.callback(arm.close)
            if not arm._verify_connection():
                raise RuntimeError("No advancing Tianji telemetry")
            delta = float(np.rad2deg(np.max(np.abs(first_arm - arm_feedback(arm)))))
            if delta > args.max_start_arm_deg:
                raise EpisodeError(f"Initial arm distance {delta:.1f} deg exceeds --max-start-arm-deg")
            report["initial_arm_distance_deg"] = delta
        if hand_delta is not None:
            print(f"真机预检完成：手起点最大差 {hand_delta:.1f}°；双臂和双手将缓慢到达录制起点。", flush=True)
        else:
            print("真机预检完成：双臂将缓慢到达录制起点（arms-only 回放）。", flush=True)
        for remaining in range(args.countdown, 0, -1):
            print(f"{remaining} 秒后开始真机动作，Ctrl+C 取消", flush=True)
            time.sleep(1)
        if hands:
            hands.check_disabled()
        if arm:
            arm.initialize()
        backends, readers = {}, {}
        if hands:
            hands.enable()
            # Smooth, velocity-limited approach for hands, without a jump to frame 0.
            current = hands.read()
            low, high = np.tile(LOWER_RAD, 2), np.tile(UPPER_RAD, 2)
            if np.any(current < low - np.deg2rad(1)) or np.any(current > high + np.deg2rad(1)):
                raise RuntimeError("Current hand feedback outside native limits")
            current = np.clip(current, low, high)
            delta = float(np.rad2deg(np.max(np.abs(first_hand - current))))
            duration = max(3., 1.5 * delta / args.ramp_speed)
            steps = math.ceil(duration * args.hz)
            t0 = time.monotonic()
            last = current.copy()
            for i in range(steps + 1):
                sleep_until(t0 + i / args.hz)
                measured = hands.read()
                if np.rad2deg(np.max(np.abs(measured - last))) > args.hand_error:
                    raise RuntimeError("Sharpa feedback deviation during approach to start")
                target = current + smoothstep(i / steps) * (first_hand - current)
                hands.write(target)
                last = target
                if arm:
                    if np.rad2deg(np.max(np.abs(arm_feedback(arm) - tracks["arms"].at(start)))) > args.arm_error:
                        raise RuntimeError("Arm feedback deviation during hand approach")
                    arm.write(tracks["arms"].at(start))
            deadline = time.monotonic() + 5.
            while np.rad2deg(np.max(np.abs(hands.read() - first_hand))) > args.hand_error:
                if time.monotonic() > deadline:
                    raise RuntimeError("Sharpa did not reach replay start")
                hands.write(first_hand)
                time.sleep(1 / args.hz)
            backends["hands"], readers["hands"] = hands, hands.read
        if arm:
            backends["arms"], readers["arms"] = arm, lambda: arm_feedback(arm)
        report["tracking"] = {}
        run_timeline(tracks, backends, readers, start, end,
                     speed=args.speed, hz=args.hz, arm_error=args.arm_error,
                     hand_error=args.hand_error, log=log, progress=report["tracking"])
    if arm and not arm.shutdown_ok:
        raise RuntimeError("Tianji shutdown was not confirmed; check controller state")
    # Only mark success once both hardware shutdown paths also completed.
    report["hardware_replay_ok"] = True


def positive(text):
    number = float(text)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    sub.add_parser("probe", help="只读发现 Sharpa 左右手和序列号，不使能设备")
    for action in ("check", "replay"):
        p = sub.add_parser(action)
        p.add_argument("episode", type=Path)
        p.add_argument("--devices", choices=("both", "hands", "arms"), default="both")
        p.add_argument("--cameras", nargs="*", choices=("wrist", "main", "wrist2"),
                       default=["wrist", "main", "wrist2"])
        p.add_argument("--no-tactile", action="store_true", help="明确缩小检查范围，不要求触觉")
        p.add_argument("--skip-video-decode", action="store_true", help="快速预检；不能代替完整视频检查")
        p.add_argument("--max-gap", type=positive, default=.25)
        p.add_argument("--edge-slack", type=positive, default=.5)
        p.add_argument("--report", type=Path)
        if action == "replay":
            p.add_argument("--execute", action="store_true", help="实际控制真机；默认仅离线预演")
            p.add_argument("--speed", type=positive, default=.25)
            p.add_argument("--hz", type=positive, default=20.)
            p.add_argument("--arm-source", choices=("measured", "command"), default="command",
                           help="command=回放遥操下发的指令轨迹（默认；抵掉阻抗稳态下垂，落点≈遥操时的真机位置）；"
                                "measured=回放编码器实测轨迹（会在已有下垂上再垂一次）")
            p.add_argument("--arm-speed", type=positive, default=30.)
            p.add_argument("--hand-speed", type=positive, default=90.)
            p.add_argument("--ramp-speed", type=positive, default=10.)
            p.add_argument("--arm-error", type=positive, default=8.)
            p.add_argument("--hand-error", type=positive, default=15.)
            p.add_argument("--max-start-arm-deg", type=positive, default=45.)
            p.add_argument("--max-start-hand-deg", type=positive, default=45.)
            p.add_argument("--countdown", type=int, default=5)
            p.add_argument("--robot-ip", default=os.environ.get("TIANJI_ROBOT_IP", ""))
            p.add_argument("--sdk-dir", default=os.environ.get("MARVIN_SDK_DIR"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "probe":
        sdk = load_sharpa()
        manager, devices = discover(sdk)
        try:
            for info in devices:
                side = "LEFT" if info.hand_side == sdk.HandSide.LEFT else "RIGHT"
                print(f"SHARPA_{side}_SN={info.sn}")
            return 0 if len(devices) == 2 else 2
        finally:
            manager.disconnect_all()
    report = {"hardware_replay_ok": False, "mode": args.action}
    report_path = args.report
    is_npz = args.episode.suffix == ".npz" if args.action in ("check", "replay") else False
    if is_npz and args.action == "check":
        parser().error("check 仅适用于本地 episode 目录；npz 外部数据集请直接用 replay（离线预演）")
    if args.action == "replay":
        if args.execute and not is_npz:
            import json as _json
            try:
                declared = set(_json.loads((args.episode / "manifest.json"
                                            ).read_text()).get("cameras", {}))
            except Exception:
                declared = {"wrist", "main", "wrist2"}
            if args.devices == "arms":
                if args.cameras:
                    parser().error("arms-only 回放不验收相机；请显式传 --cameras（置空）")
            elif (args.devices != "both" or set(args.cameras) != declared
                    or (declared and args.skip_video_decode)):
                parser().error("真机验收必须包含双臂、双手，并覆盖该 episode "
                               "manifest 声明的全部相机（含完整解码视频）")
        if args.execute and args.no_tactile and not is_npz:
            print("警告：--no-tactile 真机回放——episode 无触觉数据；触觉是反馈记录，"
                  "不参与回放控制，仅影响验收完整性", flush=True)
        if is_npz:
            print("外部 npz 数据集：跳过本地 episode 数据审计（无视频/触觉/实测反馈），"
                  "仅做关节限位与速度校验；真机回放为开环，请低速首跑并值守急停", flush=True)
        if args.speed > 1 or args.countdown < 0 or args.ramp_speed > 30:
            parser().error("speed must be <=1, countdown >=0, ramp-speed <=30 deg/s")
        report_path = report_path or Path('.local/replay-reports') / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8] + '.json')
        # Fail before connecting if the requested report already exists.
        if report_path.exists() or report_path.with_suffix('.jsonl').exists():
            parser().error("Report or feedback log already exists; choose a new path")
    try:
        if not is_npz:
            audit = inspect_episode(args.episode, devices=args.devices, cameras=args.cameras,
                                    require_tactile=not args.no_tactile and args.devices != "arms",
                                    decode_video=not args.skip_video_decode,
                                    max_gap=args.max_gap, slack=args.edge_slack)
            report["data_check"] = audit
            for message in audit["errors"]:
                print("FAIL:", message, file=sys.stderr)
            for message in audit["warnings"]:
                print("WARN:", message)
            if not audit["ok"]:
                raise EpisodeError("数据完整性检查失败；拒绝回放")
            print("数据检查通过（范围见报告 scope；已知未录字段不在完整性声明内）", flush=True)
        if args.action == "replay":
            if not is_npz and args.devices in ("both", "arms"):
                sag = arm_sag_deg(args.episode)
                print(f"臂指令/实测差（回放源 {args.arm_source}）："
                      f"左 均 {sag['left'][0]:.2f}° 峰 {sag['left'][1]:.2f}°  "
                      f"右 均 {sag['right'][0]:.2f}° 峰 {sag['right'][1]:.2f}°", flush=True)
            tracks = load_tracks(args.episode, args.devices, args.arm_source)
            start, end, plan = prepare_tracks(tracks, args.speed, args.arm_speed,
                                              args.hand_speed, require_both_sides=not is_npz)
            source_file = args.episode if is_npz else args.episode / 'robot_states.h5'
            report.update(plan=plan, speed=args.speed, arm_source=args.arm_source,
                          source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest())
            print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
            if args.execute:
                def interrupted(signum, _frame):
                    raise KeyboardInterrupt(f"signal {signum}")
                previous = signal.signal(signal.SIGTERM, interrupted)
                try:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with report_path.with_suffix('.jsonl').open('x', buffering=1) as log:
                        report["feedback_log"] = str(report_path.with_suffix('.jsonl'))
                        execute(tracks, args, start, end, log, report)
                finally:
                    signal.signal(signal.SIGTERM, previous)
                print("真机回放完成，实际关节跟踪误差在阈值内，设备已退出控制。", flush=True)
            else:
                print("离线预演完成；未连接设备。真机执行需显式 --execute。", flush=True)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    if report_path:
        write_report(report_path, report)
        print("报告：", report_path, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
