"""Build bottle references from recorded feedback/video only; no hardware imports."""
from pathlib import Path
import sys, json, hashlib, argparse
import h5py
import numpy as np
from PIL import Image
import cv2

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P / "vendor/bridge"), str(P / "assets/robot"), str(P / "vendor/mujoco_marvin")]
from demo_origin import nearest_indices, load_reference, stable_start
from kinematics import URDFKinematics, canonical80
from mujoco_marvin.tianji_backend import DEFAULT_JOINT_LIMITS_DEG
from mujoco_marvin.sharpa_backend import LOWER_RAD, UPPER_RAD
ep = P / "data/bottle_reference_download/episode_bottle/episode_0002"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=P / "assets/references/bottle_observations")
args = parser.parse_args()
out = args.output.resolve()
if not out.is_relative_to(P):
    parser.error("Output must be inside deployment")
out.mkdir(exist_ok=False)
kin = URDFKinematics(P / "assets/robot/marvin_dual_arm.urdf")
with h5py.File(ep / "robot_states.h5", "r") as f:
    a, h = f["robot_state/pico_teleop"], f["robot_state/sharpa_joints"]
    ts = a["sys_time_ns"][:]
    hi, skew = nearest_indices(h["sys_time_ns"][:], ts)
    joints = np.c_[a["real_left_deg"][:], a["real_right_deg"][:],
                   np.rad2deg(h["left_rad"][:][hi]), np.rad2deg(h["right_rad"][:][hi])]
    serials = {}
    for side in ("left", "right"):
        value = h[side + "_sn"][hi[0]]
        serials[side] = value.decode() if isinstance(value, bytes) else str(value)
    assert serials == {"left": "C65E9038C65F", "right": "CA56933FCA57"}
target = joints[0]
limits = np.vstack([DEFAULT_JOINT_LIMITS_DEG["left"], DEFAULT_JOINT_LIMITS_DEG["right"],
                    np.rad2deg(np.c_[LOWER_RAD, UPPER_RAD]), np.rad2deg(np.c_[LOWER_RAD, UPPER_RAD])])
assert np.isfinite(joints).all()
assert np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1]), "First measured bottle pose outside limits"
try:
    stable = stable_start(ts, joints)
    stable_audit = {"found": True, "row": int(stable[0]), "window": stable[1]}
except ValueError as error:
    stable_audit = {"found": False, "reason": str(error)}
origin_row = stable_audit['row'] if stable_audit['found'] else 0
target = joints[origin_row]
assert np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1])
reference = {
    "schema": "openwam_demo_origin_v1", "units": "deg", "source_kind": "measured_demonstration",
    "joint_order": ["left_arm7", "right_arm7", "left_hand22", "right_hand22"],
    "task": "bottle", "episode": ep.name, "episode_path": str(ep), "arm_row": origin_row, "hand_row": int(hi[origin_row]),
    "timestamp_ns": int(ts[origin_row]), "hand_skew_ms": float(skew[origin_row]), "hand_serials": serials,
    "target_deg": target.tolist(), "selection": "Measured sample from initial stable window" if stable_audit["found"] else "First measured source row; no stable-start claim",
    "stable_start_audit": stable_audit,
    "source_sha256": hashlib.sha256((ep / "robot_states.h5").read_bytes()).hexdigest(),
    "urdf_sha256": hashlib.sha256((P / "assets/robot/marvin_dual_arm.urdf").read_bytes()).hexdigest(),
}
ref_path = P / "assets/references/bottle_demo_origin.json"
ref_path.write_text(json.dumps(reference, indent=2))
load_reference(ref_path, limits)
audit = {"task": "bottle", "prompt": "unscrew the bottle cap and pour water into the cup", "hardware_commands": 0, "samples": []}
# Names match the existing worker warmup interface; each is a recorded bottle frame.
for name, ai in (("recorded_start", origin_row), ("recorded_mid", len(ts)//2), ("live_end_reconstructed", len(ts)-1)):
    folder = out / name
    folder.mkdir()
    state = canonical80(np.deg2rad(joints[ai])[None], kin)[0].astype(np.float32)
    np.save(folder / "state80.npy", state)
    row = {"name": name, "source_kind": "recorded", "arm_row": ai, "hand_row": int(hi[ai]),
           "timestamp_ns": int(ts[ai]), "hand_skew_ms": float(skew[ai]), "cameras": {}}
    # Recorded videos already include the camera mounting rotation.
    for role in ("main", "wrist2", "wrist"):
        stamps = [json.loads(line) for line in (ep / role / "timestamps_take001.jsonl").read_text().splitlines()]
        stamps = [x for x in stamps if "i" in x]
        frame = min(stamps, key=lambda x: abs(round(x["mono"] * 1e9) - int(ts[ai])))
        delta_ms = abs(round(frame["mono"] * 1e9) - int(ts[ai])) / 1e6
        assert delta_ms <= 50, (name, role, delta_ms)
        video = ep / role / ("color_take%03d.mp4" % frame["take"])
        cap = cv2.VideoCapture(str(video))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame["i"] - 1)
            ok, bgr = cap.read()
            assert ok, str(video)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(folder / (role + ".png"))
        finally:
            cap.release()
        row["cameras"][role] = {"video": str(video), "frame_1based": frame["i"], "skew_ms": delta_ms, "rotation_deg": 0, "orientation": "preserved_recorded_frame"}
    (folder / "input.json").write_text(json.dumps(row, indent=2))
    audit["samples"].append(row)
(out / "summary.json").write_text(json.dumps(audit, indent=2))
images = []
for role in ("main", "wrist2", "wrist"):
    im = Image.open(out / "recorded_start" / (role + ".png"))
    im.thumbnail((640, 480))
    images.append(im)
preview = Image.new("RGB", (sum(im.width for im in images), 480), (30, 30, 30))
x = 0
for im in images:
    preview.paste(im, (x, 0)); x += im.width
preview.save(out / "origin_preview.jpg")
print(json.dumps({"reference": str(ref_path), "fixtures": str(out), "stable_start_audit": stable_audit, "hardware_commands": 0}))
