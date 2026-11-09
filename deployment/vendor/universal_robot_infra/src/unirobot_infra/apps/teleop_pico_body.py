"""Full Body orchestration using the same Marvin IK and joint-stream backend."""

from datetime import datetime
import json
import math
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from unirobot_infra.devices.pico_body import ARM_ROLES, JOINT_NAMES, DemoBodySource, PicoBodySource, ReplayBodySource
from unirobot_infra.devices.pico_body_state import BodyInputState
from unirobot_infra.devices.pico_tracker import TrackerActivity
from unirobot_infra.robots.arm.marvin_model import load_home_joints, resolve_model_urdf
from unirobot_infra.teleop.mapping.pico_body_retarget import BodyRetarget, load_body_config
from unirobot_infra.visualization.body_skeleton import draw_body


def default_recording_dir():
    return Path.cwd() / ".local" / "body-recordings"


class BodyRecording:
    def __init__(self, path, config, controller, source):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.started = time.monotonic()
        self.last_sequence = None
        self.write({"type": "metadata", "version": 1, "source": source,
                    "config": config, "urdf": str(resolve_model_urdf()), "joint_names": JOINT_NAMES,
                    "home_arm_rad": controller.arm_command().tolist(),
                    "model_joints": [controller.model.joint(i).name for i in range(controller.model.njnt)],
                    "pose_order": "x,y,z,qx,qy,qz,qw", "units": "meter,radian",
                    "quality_note": "UNKNOWN is not a measured or predicted validity flag"})
        print(f"Body recording: {self.path}", flush=True)

    def write(self, record):
        record.setdefault("elapsed_s", time.monotonic() - self.started)
        def finite_json(value):
            if isinstance(value, dict):
                return {k: finite_json(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [finite_json(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        self.stream.write(json.dumps(finite_json(record), allow_nan=False) + "\n")

    def tick(self, frame, controller, gate, observation, hardware_feedback=None, hand_states=None):
        if frame is not None and frame.sequence != self.last_sequence:
            self.last_sequence = frame.sequence
            self.write({"type": "body_frame", "frame": frame.record(),
                        "elapsed_s": frame.received_at - self.started})
        arms = {}
        for side, (position, rotation, elbow) in controller.targets.items():
            site = controller.site_ids[side]["tcp"]
            actual_rotation = Rotation.from_matrix(controller.data.site_xmat[site].reshape(3,3))
            q = controller.command[controller.qpos_ids[side]]
            limits = controller.model.jnt_range[controller.joint_ids[side]]
            jp = np.zeros((3, controller.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(controller.model, controller.data, jp, jr, site)
            dofs = controller.dof_ids[side]
            singular = np.linalg.svd(np.vstack((jp[:, dofs], .5 * jr[:, dofs])), compute_uv=False)
            arms[side] = {
                "target_position_m": position.tolist(), "target_quat_xyzw": rotation.as_quat().tolist(),
                "target_elbow_m": elbow.tolist(), "actual_position_m": controller.data.site_xpos[site].tolist(),
                "position_error_m": float(np.linalg.norm(position - controller.data.site_xpos[site])),
                "rotation_error_deg": float(np.degrees((rotation * actual_rotation.inv()).magnitude())),
                "joint_margin_deg": np.degrees(np.minimum(q-limits[:,0],limits[:,1]-q)).tolist(),
                "min_singular_value": float(singular[-1]),
            }
        self.write({"type": "control", "frame_sequence": None if frame is None else frame.sequence,
                    "state": gate.state, "reason": gate.reason, "returning_home": controller.returning_home,
                    "body_to_raw": None if observation is None else observation.body_to_raw.tolist(),
                    "origin_raw_m": None if observation is None else observation.origin.tolist(),
                    "model_qpos_rad": controller.command.tolist(), "arm_command_rad": controller.arm_command().tolist(),
                    "arms": arms, "hardware_feedback_deg": hardware_feedback, "hand_states": hand_states})

    def close(self):
        self.stream.close()


class ControlFile:
    """Append-only local JSONL commands for a pedal/helper or headless session.

    Existing lines are skipped at startup; an old reset/resume is never replayed.
    A final partial line is kept until its newline arrives.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.offset = self.path.stat().st_size
        self.pending = ""

    def poll(self):
        if self.path.stat().st_size < self.offset:
            self.offset = 0
            self.pending = ""
        with self.path.open(encoding="utf-8") as stream:
            stream.seek(self.offset)
            text = stream.read(8192)
            self.offset = stream.tell()
        lines = (self.pending + text).split("\n")
        self.pending = lines.pop()
        actions = []
        for line in lines:
            try:
                value = json.loads(line)
                action = value.get("action")
                if action in ("pause", "resume", "reset", "stop"):
                    actions.append(action)
            except (json.JSONDecodeError, AttributeError):
                pass
        return actions


def run(args):
    # Imported only after --input-source body is selected; Object startup stays independent.
    from unirobot_infra.teleop.marvin_teleop import (
        MarvinTeleop, build_model, dispatch_command, draw_diagnostics,
        default_body_config_path, default_body_reference_path,
        model_joint_limits_deg)
    from unirobot_infra.devices.pico_tracker import ROBOT_TO_SIM

    config = load_body_config(args.body_config or default_body_config_path())
    if args.body_scale is not None:
        if not math.isfinite(args.body_scale) or args.body_scale <= 0:
            raise ValueError("--body-scale must be positive and finite")
        config["scale"] = args.body_scale
    elif args.scale != 1.:
        config["scale"] = args.scale
    hardware = args.backend in ("robot", "both")
    if hardware and (args.demo or args.replay or args.preview_body):
        raise ValueError("Body demo, replay and preview are simulation-only")
    home = load_home_joints(profile=args.home_profile)
    model = build_model(home, movable_hands=args.hand_input == "ros")
    controller = MarvinTeleop(model, {}, ROBOT_TO_SIM, config["scale"], args.elbow_gain)
    controller.freshness_seconds = config["freshness_seconds"]
    gate = BodyInputState(config, hardware=hardware)
    retarget = BodyRetarget(config)
    display_retarget = BodyRetarget(config)
    if hardware and config.get("allow_unknown_quality") is True:
        print("WARNING: allow_unknown_quality=true — the hardware backend will move on "
              "UNKNOWN Body quality; nothing self-reports tracking health. "
              "Keep clear of the arms and your hand on the E-stop.", flush=True)
    activity = TrackerActivity({name: name for name in ARM_ROLES})
    keys = SimpleQueue()
    source = ReplayBodySource(args.replay) if args.replay else DemoBodySource() if args.demo else PicoBodySource()
    recording_path = args.record or default_recording_dir() / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
    recorder = None
    robot = None
    hands = None
    feedback = None
    feedback_at = float("-inf")
    observation = None
    display_observation = None
    display_frame = None
    frame = None
    last_sequence = None
    begin_at = None
    live_started = None
    reset_wait_at = None
    reset_required = False
    frames = 0
    states = {}
    started = time.monotonic()
    previous_tick = started
    rejects = 0
    period = 1. / config["control_hz"]
    default_elbow_gain = args.elbow_gain
    controls = ControlFile(args.control_file) if args.control_file else None
    stopped = False

    def invalidate_reference():
        nonlocal begin_at, observation
        controller.reset_reference()
        retarget.reset()
        observation = None
        begin_at = None

    def tick():
        nonlocal frame, observation, last_sequence, begin_at, live_started, previous_tick, rejects
        nonlocal robot, frames, feedback, feedback_at, reset_wait_at, reset_required
        nonlocal stopped
        nonlocal display_observation, display_frame
        now = time.monotonic()
        dt = max(1e-6, min(now - previous_tick, config["max_dt_seconds"]))
        if now - previous_tick > config["freshness_seconds"] and gate.engaged:
            gate.hold("Control loop stalled; resume explicitly")
            invalidate_reference()
        previous_tick = now
        if controls is not None:
            for action in controls.poll():
                keys.put(action)
        while not keys.empty():
            key = keys.get()
            if key == "stop":
                stopped = True
                gate.pause()
                controller.cancel_reset()
                invalidate_reference()
                recorder.write({"type": "event", "action": "stop"})
                return
            if key in (ord("R"), ord("r"), "reset"):
                invalidate_reference()
                gate.resume()
                controller.reset_home(config["reset_speed_deg_s"])
                reset_required = hardware and robot is not None
                reset_wait_at = None
            elif key in (ord(" "), "pause", "resume"):
                if key == "resume" and gate.enabled:
                    continue
                pause = key == "pause" or (key == ord(" ") and gate.enabled)
                if pause:
                    gate.pause()
                    controller.cancel_reset()
                    reset_required = False
                elif not gate.enabled:
                    gate.resume()
                invalidate_reference()
            elif key in (ord("E"), ord("e")):
                controller.elbow_gain = 0. if controller.elbow_gain else (default_elbow_gain or .35)

        frame = source.snapshot()
        now = time.monotonic()
        admitted = gate.evaluate(frame, now)
        new_frame = frame is not None and frame.sequence != last_sequence
        if new_frame:
            last_sequence = frame.sequence
            poses = {name: frame.poses[index] for name, index in ARM_ROLES.items() if frame.valid[index]}
            activity.observe(poses, frame.sequence, frame.received_at)
            if frame.complete:
                try:
                    display_observation = display_retarget.observe(frame)
                    display_frame = frame
                except ValueError:
                    display_retarget.reset()
        if not admitted:
            invalidate_reference()
        elif new_frame:
            try:
                observation = retarget.observe(frame)
                rejects = 0
            except ValueError as exc:
                # Single-frame tracking glitches (tracker reacquisition snaps,
                # degenerate skeleton solves) should skip the frame, not stop
                # teleop; only a sustained bad stream indicates a real problem.
                rejects += 1
                if rejects >= config["max_consecutive_rejects"]:
                    gate.hold(f"{exc} (x{rejects})")
                    invalidate_reference()
                    admitted = False

        # Hardware connection and startup home are deferred until source quality
        # is explicit and a complete skeleton has passed the mapping checks.
        if hardware and robot is None and admitted and observation is not None:
            from unirobot_infra.robots.arm.tianji_marvin_backend import TianjiJointBackend
            robot = TianjiJointBackend(args.robot_ip, home, limits_deg=model_joint_limits_deg(model),
                                       ramp_speed_deg_s=config["reset_speed_deg_s"],
                                       sdk_dir=getattr(args, "sdk_dir", None))
            invalidate_reference()
            admitted = False
            last_sequence = None
            previous_tick = time.monotonic()
            now = previous_tick

        if robot is not None and now - feedback_at >= .05:
            measured = np.array(robot.read_joints_deg(), dtype=float)
            if measured.shape != (2,7) or not np.isfinite(measured).all():
                raise RuntimeError("Invalid arm feedback")
            feedback = measured.tolist()
            feedback_at = now
        if reset_required and not controller.returning_home and feedback is not None:
            error = float(np.max(np.abs(np.array(feedback) - np.array([home['left'],home['right']]))))
            if error <= 5.:
                reset_required = False
            elif reset_wait_at is None:
                reset_wait_at = now
            elif now - reset_wait_at > 10.:
                raise RuntimeError(f"Reset did not reach home: feedback error {error:.1f} degrees")
        if admitted and observation is not None and not controller.returning_home and not reset_required:
            if retarget.reference is None:
                if begin_at is None:
                    begin_at = now + args.ref_delay
                    print(f"Body ready for reference: hold posture for {args.ref_delay:g}s.", flush=True)
                if now >= begin_at and new_frame:
                    retarget.capture(observation, controller,
                                     args.body_reference or default_body_reference_path())
                    gate.engaged = True
                    print("Body reference recorded: X forward, Y left, Z up; both arms.", flush=True)
                    if live_started is None:
                        live_started = now
                        if args.live_duration:
                            print(f"Operation recording started: {args.live_duration:g}s.", flush=True)
            if retarget.reference is not None and new_frame:
                if not args.preview_body:
                    controller.set_targets(retarget.targets(observation), frame.received_at)
                frames += 1

        # Pause/quality gates affect finger output as well as arm following.
        if frame is not None and time.monotonic() - frame.received_at > config["freshness_seconds"]:
            gate.hold("Body frame aged during processing")
            invalidate_reference()
            admitted = False
        if hands is not None:
            hands.tick(admitted and gate.engaged and not controller.returning_home and not reset_required,
                       dt, output_active=not hardware or robot is not None)
        saved_gain = controller.elbow_gain
        if gate.state == "PREDICTED":
            controller.elbow_gain = 0.
        dispatch_command(controller, robot, dt=dt, max_speed_deg_s=config["max_joint_speed_deg_s"])
        controller.elbow_gain = saved_gain
        recorder.tick(frame, controller, gate, observation, feedback, None if hands is None else hands.state)
        status = (gate.state, gate.reason)
        if states.get("input") != status:
            states["input"] = status
            print(f"Body {gate.state}: {gate.reason}", flush=True)
        for role in ARM_ROLES:
            status = activity.status(role, now)[0]
            if states.get(role) != status:
                states[role] = status
                print(f"Body {role}: {status}", flush=True)

    def keep_running():
        now = time.monotonic()
        return ((args.duration <= 0 or now - started < args.duration) and
                (args.live_duration <= 0 or live_started is None or now-live_started < args.live_duration) and
                not getattr(source, "done", False) and not stopped)

    def sleep_until(deadline):
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    try:
        if args.hand_input == "ros":
            raise RuntimeError("ROS2 hand bridge 不纳入 unirobot_infra，见 docs/tianji_merge_manifest.md")
        recorder = BodyRecording(recording_path, config, controller, type(source).__name__)
        source.start()
        print(f"Full Body | backend={args.backend} | home={args.home_profile} | preview={args.preview_body}", flush=True)
        print("R: reset arms / recapture; SPACE: pause/resume with fresh reference; E: elbow guide.", flush=True)
        if args.headless or args.backend == "robot":
            while keep_running():
                deadline = time.monotonic() + period
                tick()
                sleep_until(deadline)
        else:
            from mujoco import viewer
            if args.preview_body:
                model.geom_rgba[:,3] = .12
                model.geom_rgba[model.geom("floor").id,3] = .8
            with viewer.launch_passive(model, controller.data, key_callback=keys.put) as window:
                forward = ROBOT_TO_SIM[:,0]
                window.cam.azimuth = math.degrees(math.atan2(forward[1],forward[0]))
                window.cam.elevation = -15.
                window.cam.distance = 3.1
                window.cam.lookat[:] = controller.data.xpos[model.body("dual_arm_mount").id] + ROBOT_TO_SIM @ [0.,.45,-.25]
                while window.is_running() and keep_running():
                    deadline = time.monotonic() + period
                    tick()
                    draw_diagnostics(window, controller)
                    fresh = (display_frame is not None and frame is not None and
                             display_frame.sequence == frame.sequence and
                             time.monotonic()-frame.received_at <= config["freshness_seconds"])
                    draw_body(window, display_frame, display_observation, controller, args.preview_body, fresh)
                    info = source.diagnostics()
                    valid = int(frame.valid.sum()) if frame is not None else 0
                    countdown = max(0.,begin_at-time.monotonic()) if begin_at is not None else args.ref_delay
                    lines = [f"FULL BODY | {args.backend.upper()} | {'PREVIEW' if args.preview_body else 'TELEOP'}",
                             f"{gate.state}: {gate.reason}",
                             f"Body nodes {valid}/24 | payload {info.get('body_hz',0.):.1f}Hz | quality: {frame.quality if frame else 'NO DATA'}",
                             "Operator X forward / Y left / Z up. Head turns do not set heading.",
                             "R: home + reference | SPACE: pause/resume | E: elbow guide"]
                    if args.demo or args.replay:
                        lines.insert(1, "SYNTHETIC DEMO" if args.demo else f"RECORDED INPUT: {args.replay.name}")
                    if not valid:
                        lines.append("Headset: Full Body + body calibration + Send (Object data cannot drive Body mode)")
                        if info.get("motion_count", 0):
                            lines.append("Object tracker packets detected; switch to Full Body.")
                    if retarget.reference is None:
                        lines.append("Returning home / waiting feedback" if controller.returning_home or reset_required else
                                     f"Reference preparation: {countdown:.1f}s")
                    if live_started is not None and args.live_duration:
                        lines.insert(0,f"RECORDING: {max(0.,args.live_duration-(time.monotonic()-live_started)):.1f}s remaining")
                    lines.extend(controller.diagnostic_lines())
                    lines.extend(activity.diagnostic_lines())
                    if hands is not None:
                        lines.extend(f"{side} fingers: {state}" for side,state in hands.state.items())
                    if info.get("error"):
                        lines.append(info["error"])
                    window.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      "\n".join(lines), ""))
                    window.sync()
                    sleep_until(deadline)
        print(f"Body finished: {frames} mapped frames; record={recording_path}", flush=True)
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                source.close()
            finally:
                try:
                    if hands is not None:
                        hands.close()
                finally:
                    if recorder is not None:
                        recorder.close()
