"""State-supervised OpenWAM deployment on the infra robot layer.

Model action = next measured state (EEF pose + hand joints). Execution:
predicted EEF -> FastLeftIK (seed = measured) -> joint position targets,
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

R = Path("/home/admin2026/guchenyang/Project/wuji-hand-teleop")
D = R / ".local/openwam-deploy"
U = Path("/home/admin2026/guchenyang/Project/Universal_Robot_infra")
C = D / "bridge_candidate_20261030_pipeline"
sys.path.insert(0, str(U / "src"))
sys.path.insert(0, str(C))
sys.path.insert(0, str(D / "configs/openwam_deployment"))
sys.path.insert(0, str(R / "MuJoCo-Marvin"))

import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from kinematics import URDFKinematics, canonical80  # noqa: E402
from adapter import CAMERAS, decode_action, observation  # noqa: E402
from async_policy import FastLeftIK  # noqa: E402
from unirobot_infra.robots.arm.tianji import TianjiRobot  # noqa: E402
from unirobot_infra.robots.end_effector.sharpa import SharpaHand  # noqa: E402
from unirobot_infra.utils.sdk_paths import configure_sharpa_runtime, sharpa_manus_sdk_paths  # noqa: E402

robot = URDFKinematics(D / "configs/openwam_deployment/marvin_dual_arm.urdf")
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
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "state_position", "hz": args.hz, "samples": [],
              "feedback_limits": {"arm_deg": 8.0, "hand_deg": 15.0},
              "feedback_max_arm_deg": 0.0, "feedback_max_hand_deg": 0.0}

    from live_observation_source import CameraFeed
    from zmq_chunk_client import ChunkClient
    from zmq_offline_bridge import REPRESENTATION, load_codec

    metadata = json.loads(args.worker_ready.read_text())
    codec = load_codec(C / "review_dev_tianji/src/unirobot_infra/policy/codec.py")
    arm = TianjiRobot(ARM_CFG)
    hands = {}
    cameras = {}
    client = None
    owned = False
    try:
        arm.connect()
        paths = sharpa_manus_sdk_paths(Path(EE_CFG.sdk.vendor_root))
        configure_sharpa_runtime(paths)
        for side in ("left", "right"):
            hands[side] = SharpaHand(EE_CFG, side=side, sdk_paths=paths)
            hands[side].connect()
        cameras = {r: CameraFeed(sn, rot) for r, sn, rot in
                   (("main", "241122305532", 0), ("wrist2", "262422271713", 90), ("wrist", "260522271432", 90))}
        time.sleep(1.0)
        client = ChunkClient(codec, observation_endpoint=metadata["observation_endpoint"])
        client.connect(metadata, "infra-state")
        # position-mode hold at the measured pose; impedance never engaged.
        q_deg = {"A": arm.read_joint_position_deg("A"), "B": arm.read_joint_position_deg("B")}
        origin = np.asarray(json.loads((C / "references/badminton_demo_origin.json").read_text())["target_deg"], float)[:14]
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

        def caps():
            q = {"A": arm.read_joint_position_deg("A"), "B": arm.read_joint_position_deg("B")}
            fingers = hands_state(hands)
            state = canonical80(np.r_[np.deg2rad(q["A"]), np.deg2rad(q["B"]), fingers][None], robot)[0]
            images = {}
            for key, role in CAMERAS.items():
                snap = cameras[role].snapshot()
                deadline = time.monotonic() + 2.0
                while snap is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                    snap = cameras[role].snapshot()
                if snap is None:
                    raise RuntimeError(f"camera {role} missing")
                images[role] = snap[0]
            return q, fingers, state, images

        def msg(seq_i, state, images):
            return {"type": "observation", "session": "infra-state", "sequence": seq_i,
                    "timestamp_ns": time.monotonic_ns(), "source_timestamp_ns": time.monotonic_ns(),
                    "task": "badminton", "representation": REPRESENTATION,
                    "observation": {"state": np.asarray(state, np.float32), "images": images}}

        # Prime the first chunk before the control clock starts: the initial
        # inference must not turn into a burst of catch-up ticks.
        q, fingers, state, images = caps()
        reply = client.infer(msg(seq, state, images))
        seq += 1
        buffer = [np.asarray(a, float) for a in reply["action"]]
        if len(buffer) != 32:
            raise RuntimeError(f"chunk shape {len(buffer)}")
        t0 = time.monotonic()
        while args.duration <= 0 or time.monotonic() - t0 < args.duration:
            due = t0 + i * period
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            q, fingers, state, images = caps()
            if prev_cmd is not None and prev_hand is not None:
                arm_err = float(np.max(np.abs(prev_cmd - np.r_[q["A"], q["B"]])))
                hand_err = float(np.rad2deg(np.max(np.abs(prev_hand - fingers))))
                report["feedback_max_arm_deg"] = max(report["feedback_max_arm_deg"], arm_err)
                report["feedback_max_hand_deg"] = max(report["feedback_max_hand_deg"], hand_err)
                if arm_err > 8.0 or hand_err > 15.0:
                    report["feedback_violation"] = dict(t=time.monotonic() - t0, arm_deg=arm_err,
                                                        hand_deg=hand_err, step=i)
                    raise RuntimeError(f"Feedback exceeded arm8/hand15 at step {i}: arm={arm_err:.2f} hand={hand_err:.2f}")
            if not buffer:
                if pending is not None:
                    pending["thread"].join()
                    if pending["error"] is not None:
                        raise pending["error"]
                    skip = max(0, i - pending["start_step"])
                    buffer = pending["chunk"][skip:] or pending["chunk"]
                    report.setdefault("chunk_skips", []).append({"step": i, "skip": skip})
                    pending = None
                else:
                    reply = client.infer(msg(seq, state, images)); seq += 1
                    buffer = [np.asarray(a, float) for a in reply["action"]]
                    if len(buffer) != 32:
                        raise RuntimeError(f"chunk shape {len(buffer)}")
            elif (not args.sync) and pending is None and len(buffer) <= args.prefetch_delay:
                slot = {"chunk": None, "error": None, "start_step": i}

                def work(_seq=seq, _state=state, _images=images, _slot=slot):
                    try:
                        reply = client.infer(msg(_seq, _state, _images))
                        _slot["chunk"] = [np.asarray(a, float) for a in reply["action"]]
                    except BaseException as exc:
                        _slot["error"] = exc

                slot["thread"] = threading.Thread(target=work, daemon=True)
                slot["thread"].start()
                pending = slot
                seq += 1
            action = buffer.pop(0)
            arms, hand_targets = decode_action(action)
            q_rad = np.r_[PLANNERS["left"].solve(arms["left"], np.deg2rad(seed_deg[:7])),
                          PLANNERS["right"].solve(arms["right"], np.deg2rad(seed_deg[7:]))]
            q_deg_cmd = np.rad2deg(q_rad)
            q_deg_cmd = np.clip(q_deg_cmd, np.r_[LIMITS[:, 0], LIMITS[:, 0]], np.r_[LIMITS[:, 1], LIMITS[:, 1]])
            if args.max_joint_step_deg > 0:
                q_deg_cmd = prev_cmd + np.clip(q_deg_cmd - prev_cmd, -args.max_joint_step_deg, args.max_joint_step_deg)
            arm.send_joint_positions_deg({"A": tuple(q_deg_cmd[:7]), "B": tuple(q_deg_cmd[7:])}, blocking=False)
            lt = HAND_LIMITS["left"]
            rt = HAND_LIMITS["right"]
            hand_targets = np.r_[np.clip(hand_targets[:22], lt[:, 0], lt[:, 1]),
                                 np.clip(hand_targets[22:], rt[:, 0], rt[:, 1])]
            if prev_hand is not None and args.max_hand_step_deg > 0:
                step = np.deg2rad(args.max_hand_step_deg)
                hand_targets = prev_hand + np.clip(hand_targets - prev_hand, -step, step)
            hands["left"].send_joint_position_rad(tuple(hand_targets[:22]))
            hands["right"].send_joint_position_rad(tuple(hand_targets[22:]))
            report["samples"].append(dict(t=time.monotonic() - t0, command_deg=q_deg_cmd.tolist(),
                                          measured_deg=np.r_[q["A"], q["B"]].tolist(),
                                          hand_cmd_rad=hand_targets.tolist(),
                                          hand_meas_rad=fingers.tolist()))
            prev_cmd = q_deg_cmd.copy()
            prev_hand = hand_targets.copy()
            seed_deg = q_deg_cmd
            i += 1
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="aborted", error=str(exc))
    finally:
        if arm.connected:
            try:
                arm._robot.clear_set()
                arm._robot.set_state(arm="A", state=0)
                arm._robot.set_state(arm="B", state=0)
                arm._robot.send_cmd()
                deadline = time.monotonic() + 6.0
                states = None
                while time.monotonic() < deadline:
                    states = [int(arm.read_arm_status(a).cur_state) for a in ("A", "B")]
                    if states == [0, 0]:
                        break
                    time.sleep(0.2)
                report["idle_states"] = states
            except BaseException as exc:
                report["shutdown_error"] = str(exc)
        for h in hands.values():
            try:
                h.close()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        try:
            arm.close()
        except Exception:
            pass
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
