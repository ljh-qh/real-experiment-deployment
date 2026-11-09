"""Experimental RTC and confirmed-feedback execution on the existing infra layer.

Model action = next measured state (EEF pose + hand joints). Execution:
predicted EEF -> FastLeftIK (initial seed measured, then previous command) -> joint position targets,
sent through unirobot_infra's TianjiRobot in position mode; hands through
SharpaHand. No sag compensation, no smoothing: position mode tracks the
predicted state directly. Safety watchdog records and aborts on divergence.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from deployment_paths import ROOT
U = ROOT / "vendor/universal_robot_infra"
C = ROOT / "vendor/bridge"
sys.path[:0] = [str(U / "src"), str(C), str(ROOT / "assets/robot"), str(ROOT / "vendor/mujoco_marvin")]

import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from kinematics import URDFKinematics, canonical80  # noqa: E402
from adapter import CAMERAS, decode_action, observation  # noqa: E402
from async_policy import FastLeftIK  # noqa: E402
from tianji_command_adapter import ReliableTianjiRobot as TianjiRobot  # noqa: E402
from control_clock import advance_deadline  # noqa: E402
from observation_timing import validate_observation_timing, acquire_synchronized, ObservationTimingError  # noqa: E402
from execution_cleanup import close_execution_resources  # noqa: E402
from rtc_queue import RTCQueue, ThreadOwnedClient  # noqa: E402
from feedback_guard import confirm_tracking  # noqa: E402
from unirobot_infra.robots.end_effector.sharpa import SharpaHand  # noqa: E402
from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths  # noqa: E402

robot = URDFKinematics(ROOT / "assets/robot/marvin_dual_arm.urdf")
ARM_CFG = OmegaConf.create(yaml.safe_load((U / "configs/robot_hardware/arm/tianji.yaml").read_text()))
EE_CFG = OmegaConf.create(yaml.safe_load((U / "configs/robot_hardware/end_effector/sharpa.yaml").read_text()))
LIMITS = np.array([[j.min_deg, j.max_deg] for j in ARM_CFG.joints], float)
PLANNERS = {s: FastLeftIK(robot, LIMITS, s) for s in ("left", "right")}
# The hardware config lists one hand's 22 joints with a side prefix; both
# hands are mechanically symmetric, so the same limit table applies.
_HAND_ROWS = np.asarray([[j.min_rad, j.max_rad] for j in EE_CFG.joints], float)
HAND_LIMITS = {"left": _HAND_ROWS, "right": _HAND_ROWS}


def hands_state(hands):
    return np.r_[np.asarray(hands["left"].read_state().joint_pos, float),
                 np.asarray(hands["right"].read_state().joint_pos, float)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-ready", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Seconds; <=0 runs until the task finishes or the process is stopped")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--sync", action="store_true", help="Disable async prefetch: blocking infer at each boundary")
    ap.add_argument("--prefetch-delay", type=int, default=16,
                    help="Start the next chunk when this many actions remain (official async default: horizon//2)")
    ap.add_argument("--max-hand-step-deg", type=float, default=3.0,
                    help="Robot-layer hand slew limit per tick (90 deg/s at 30 Hz)")
    ap.add_argument("--max-joint-step-deg", type=float, default=3.0,
                    help="Robot-layer slew limit per tick (90 deg/s at 30 Hz), applied to every model")
    ap.add_argument("--rtc-commit-steps",type=int,default=14)
    ap.add_argument("--rtc-prefix-capacity",type=int,default=20)
    ap.add_argument("--feedback-confirm-ms",type=float,default=100)
    ap.add_argument("--arm-error-deg",type=float,default=8)
    ap.add_argument("--hand-error-deg",type=float,default=15)
    ap.add_argument("--task", choices=("badminton","top","bottle"), default="badminton")
    ap.add_argument("--origin-reference", type=Path, default=ROOT / "assets/references/badminton_demo_origin.json")
    args = ap.parse_args()
    if not 0 <= args.feedback_confirm_ms <= 100:
        ap.error("Feedback confirmation must be within 0..100 ms")
    if not np.isfinite([args.arm_error_deg,args.hand_error_deg,args.hz]).all() or min(args.arm_error_deg,args.hand_error_deg,args.hz)<=0:
        ap.error("Invalid control frequency or feedback limits")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "state_position", "hz": args.hz, "samples": [],
              "feedback_limits": {"arm_deg": args.arm_error_deg, "hand_deg": args.hand_error_deg},
              "feedback_max_arm_deg": 0.0, "feedback_max_hand_deg": 0.0}

    from live_observation_source import CameraFeed
    from zmq_chunk_client import ChunkClient
    from zmq_offline_bridge import REPRESENTATION, load_codec

    metadata = json.loads(args.worker_ready.read_text())
    codec = load_codec(C / "codec.py")
    arm = TianjiRobot(ARM_CFG)
    hands = {}
    cameras = {}
    client = None
    owned = False
    rtc_queue = None
    report["feedback_confirmations"] = []
    try:
        arm.connect()
        paths = sharpa_manus_sdk_paths(Path(EE_CFG.sdk.vendor_root))
        configure_sharpa_runtime(paths)
        for side in ("left", "right"):
            hands[side] = SharpaHand(EE_CFG, side=side, sdk_paths=paths)
            hands[side].connect()
        for role, serial, rotate in (("main", "241122305532", 0),
                                     ("wrist2", "262422271713", 90), ("wrist", "260522271432", 90)):
            cameras[role] = CameraFeed(serial, rotate)
        time.sleep(1.0)
        def client_factory():
            c = ChunkClient(codec, observation_endpoint=metadata["observation_endpoint"])
            c.connect(metadata, "infra-state")
            return c
        client = ThreadOwnedClient(client_factory)
        # position-mode hold at the measured pose; impedance never engaged.
        q_deg = {"A": arm.read_joint_position_deg("A"), "B": arm.read_joint_position_deg("B")}
        origin = np.asarray(json.loads(args.origin_reference.read_text())["target_deg"], float)[:14]
        origin_err = float(np.max(np.abs(np.r_[q_deg["A"], q_deg["B"]] - origin)))
        report["origin_error_deg"] = origin_err
        if origin_err > 1.5:
            raise RuntimeError(f"Not at demo origin ({origin_err:.2f} deg); return first")
        arm.enter_position_modes(("A", "B"))
        arm.send_joint_positions_deg(q_deg, blocking=False)
        owned = True
        seed_deg = np.r_[q_deg["A"], q_deg["B"]]
        period = 1.0 / args.hz
        buffer = []
        pending = None
        prev_cmd = seed_deg.copy()
        prev_hand = None
        seq = 0
        i = 0

        from mujoco_marvin.sharpa_backend import SharpaReplayBackend,load_sharpa
        fault_checker=object.__new__(SharpaReplayBackend)
        fault_checker.sdk=load_sharpa()

        def read_feedback():
            for arm_id in ("A","B"):
                status=arm.read_arm_status(arm_id)
                if status.err_code or status.cur_state != 1:
                    raise RuntimeError(f"Native arm fault/mode change: {arm_id} {status}")
            for side,hand in hands.items():
                faults=hand._hand.get_fault_code()
                if not fault_checker._faults_benign(faults):
                    raise RuntimeError(f"{side} native hand fault: {faults}")
            q = {"A": arm.read_joint_position_deg("A"), "B": arm.read_joint_position_deg("B")}
            fingers = hands_state(hands)
            q_all=np.r_[q["A"],q["B"]]
            if not np.isfinite(q_all).all() or not np.isfinite(fingers).all():
                raise RuntimeError("Non-finite hardware feedback")
            if np.any(q_all < np.tile(LIMITS[:,0],2)) or np.any(q_all > np.tile(LIMITS[:,1],2)):
                raise RuntimeError("Native arm joint limit exceeded")
            for side,values in (("left",fingers[:22]),("right",fingers[22:])):
                limits=HAND_LIMITS[side]
                if np.any(values < limits[:,0]-np.deg2rad(1)) or np.any(values > limits[:,1]+np.deg2rad(1)):
                    raise RuntimeError(f"{side} hand feedback outside native limits")
            state = canonical80(np.r_[np.deg2rad(q["A"]), np.deg2rad(q["B"]), fingers][None], robot)[0]
            return q, fingers, state

        def caps():
            def capture_once():
                state_stamp_ns = time.monotonic_ns()
                q, fingers, state = read_feedback()
                images = {}
                camera_stamps_ns = {}
                for key, role in CAMERAS.items():
                    snap = cameras[role].snapshot()
                    deadline = time.monotonic() + 2.0
                    while snap is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                        snap = cameras[role].snapshot()
                    if snap is None:
                        raise RuntimeError(f"camera {role} missing")
                    images[role], camera_stamps_ns[role] = snap
                timing = validate_observation_timing(camera_stamps_ns, state_stamp_ns)
                return q, fingers, state, images, timing
            return acquire_synchronized(
                capture_once, report.setdefault("observation_timing_events", []), clock=time)

        def msg(seq_i, state, images, capture_timing):
            timing = validate_observation_timing(capture_timing["camera_timestamps_ns"],
                                                 capture_timing["state_timestamp_ns"])
            return {"type": "observation", "session": "infra-state", "sequence": seq_i,
                    **timing,
                    "task": args.task, "representation": REPRESENTATION,
                    "observation": {"state": np.asarray(state, np.float32), "images": images}}

        def plan_one(action, arm_seed, hand_seed):
            arms, hand_targets = decode_action(action)
            q_rad = np.r_[PLANNERS["left"].solve(arms["left"], np.deg2rad(arm_seed[:7])),
                          PLANNERS["right"].solve(arms["right"], np.deg2rad(arm_seed[7:]))]
            q_deg_cmd = np.rad2deg(q_rad)
            q_deg_cmd = np.clip(q_deg_cmd, np.r_[LIMITS[:, 0], LIMITS[:, 0]], np.r_[LIMITS[:, 1], LIMITS[:, 1]])
            if args.max_joint_step_deg > 0:
                q_deg_cmd = arm_seed + np.clip(q_deg_cmd - arm_seed, -args.max_joint_step_deg, args.max_joint_step_deg)
            lt = HAND_LIMITS["left"]
            rt = HAND_LIMITS["right"]
            hand_targets = np.r_[np.clip(hand_targets[:22], lt[:, 0], lt[:, 1]),
                                 np.clip(hand_targets[22:], rt[:, 0], rt[:, 1])]
            if hand_seed is not None and args.max_hand_step_deg > 0:
                step = np.deg2rad(args.max_hand_step_deg)
                hand_targets = hand_seed + np.clip(hand_targets - hand_seed, -step, step)
            physical=canonical80(np.r_[np.deg2rad(q_deg_cmd),hand_targets][None],robot)[0]
            return {"raw":np.asarray(action).copy(),"q":q_deg_cmd,"hand":hand_targets,"physical":physical}

        def plan_from_anchor(actions, anchor):
            arm_seed=anchor["q"].copy()
            hand_seed=None if anchor["hand"] is None else anchor["hand"].copy()
            packets=[]
            for action in actions:
                packet=plan_one(action,arm_seed,hand_seed)
                packets.append(packet);arm_seed=packet["q"];hand_seed=packet["hand"]
            return packets

        def plan_actions(actions):
            return plan_from_anchor(actions,{"q":seed_deg,"hand":prev_hand})

        # Prime the first chunk before the control clock starts: the initial
        # inference must not turn into a burst of catch-up ticks.
        q, fingers, state, images, capture_timing = caps()
        prev_hand = fingers.copy()
        report["initial_hand_rad"] = prev_hand.tolist()
        if args.sync:
            reply = client.submit(msg(seq,state,images,capture_timing)).result()
            buffer=[np.asarray(a,float) for a in reply["action"]]
            if len(buffer)!=32: raise RuntimeError("Expected 32 actions")
        else:
            if metadata["engine"] not in ("openwam_rtc", "easywam_bottle") or metadata["rtc_mode"] not in ("identity_vjp", "full_vjp"):
                raise RuntimeError("RTC worker required")
            rtc_queue=RTCQueue(client.submit,plan_actions,args.rtc_commit_steps,args.rtc_prefix_capacity,args.hz,plan_from_anchor=plan_from_anchor,clock=time.monotonic)
            rtc_queue.prime(msg(seq,state,images,capture_timing))
            report["rtc_events"]=rtc_queue.events
        seq += 1
        t0 = time.monotonic()
        next_tick = t0
        report["clock_resets"] = []
        while args.duration <= 0 or time.monotonic() - t0 < args.duration:
            due = next_tick
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            q, fingers, state, images, capture_timing = caps()
            if prev_cmd is not None:
                arm_err = float(np.max(np.abs(prev_cmd - np.r_[q["A"], q["B"]])))
                hand_err = 0.0 if prev_hand is None else float(np.rad2deg(np.max(np.abs(prev_hand - fingers))))
                report["feedback_max_arm_deg"] = max(report["feedback_max_arm_deg"], arm_err)
                report["feedback_max_hand_deg"] = max(report["feedback_max_hand_deg"], hand_err)
                if arm_err > args.arm_error_deg or hand_err > args.hand_error_deg:
                    def read_errors():
                        qq,ff,_=read_feedback()
                        return (float(np.max(abs(prev_cmd-np.r_[qq["A"],qq["B"]]))),
                                0.0 if prev_hand is None else float(np.rad2deg(np.max(abs(prev_hand-ff)))))
                    checked=confirm_tracking(read_errors,args.arm_error_deg,args.hand_error_deg,
                                             args.feedback_confirm_ms/1000)
                    report["feedback_confirmations"].append(dict(step=i,**checked))
                    if not checked["recovered"]:
                        report["feedback_violation"]=dict(t=time.monotonic()-t0,step=i,arm_deg=checked["readings"][-1]["arm_deg"],hand_deg=checked["readings"][-1]["hand_deg"])
                        raise RuntimeError("Persistent tracking error after bounded confirmation")
                    if args.sync:
                        buffer=[]
                    else:
                        rtc_queue.reset()
                    # Reset can wait for in-flight inference; capture only afterwards.
                    q,fingers,state,images,capture_timing=caps()
                    if not args.sync:
                        rtc_queue.prime(msg(seq,state,images,capture_timing));seq+=1
                        # Priming can outlast the freshness window; the next request needs a new snapshot.
                        q,fingers,state,images,capture_timing=caps()
                    # Recovery always replans; a paused RTC result is never reused.
            request=msg(seq,state,images,capture_timing);seq+=1
            if args.sync:
                if not buffer:
                    reply=client.submit(request).result()
                    buffer=[np.asarray(a,float) for a in reply["action"]]
                    if len(buffer)!=32: raise RuntimeError("Expected 32 actions")
                packet=plan_one(buffer.pop(0),seed_deg,prev_hand)
            else:
                packet=rtc_queue.next(i,time.monotonic(),request)
            q_deg_cmd=packet["q"];hand_targets=packet["hand"]
            arm.send_joint_positions_deg({"A":tuple(q_deg_cmd[:7]),"B":tuple(q_deg_cmd[7:])},blocking=False)
            hands["left"].send_joint_position_rad(tuple(hand_targets[:22]))
            hands["right"].send_joint_position_rad(tuple(hand_targets[22:]))
            finished_at = time.monotonic()
            report["samples"].append(dict(t=finished_at - t0, tick_due=due - t0, command_deg=q_deg_cmd.tolist(),
                                          measured_deg=np.r_[q["A"], q["B"]].tolist(),
                                          hand_cmd_rad=hand_targets.tolist(),
                                          hand_meas_rad=fingers.tolist(), raw_action80=packet["raw"].tolist(), projected_action80=packet["physical"].tolist()))
            prev_cmd = q_deg_cmd.copy()
            prev_hand = hand_targets.copy()
            seed_deg = q_deg_cmd
            next_tick, missed = advance_deadline(due, finished_at, period)
            if missed:
                report["clock_resets"].append(dict(step=i, finished_at=finished_at - t0, next_tick=next_tick - t0))
            i += 1
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="aborted", error=str(exc))
        if isinstance(exc, ObservationTimingError):
            report["observation_timing_failure"] = {"reason": exc.reason, **exc.diagnostics}
    finally:
        if arm.connected:
            try:
                deadline = time.monotonic() + 6.0
                next_request = 0.0
                attempts = 0
                states = None
                while time.monotonic() < deadline:
                    states = [int(arm.read_arm_status(a).cur_state) for a in ("A", "B")]
                    if states == [0, 0]:
                        break
                    now = time.monotonic()
                    if now >= next_request:
                        if arm._robot.clear_set():
                            arm._robot.set_state(arm="A", state=0)
                            arm._robot.set_state(arm="B", state=0)
                            arm._robot.send_cmd()
                        attempts += 1
                        next_request = now + 0.5
                    time.sleep(0.02)
                report["idle_states"] = states
                report["idle_request_attempts"] = attempts
                if states != [0, 0]:
                    raise RuntimeError("Idle not acknowledged after repeated requests")
            except BaseException as exc:
                report["shutdown_error"] = str(exc)
        close_execution_resources(report, hands, cameras, client, arm)
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)

    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
