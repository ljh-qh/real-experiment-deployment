# MuJoCo-Marvin

MuJoCo teleoperation and episode recording for Tianji dual **Marvin M6** arms.
The default input is four independent PICO Motion Trackers in **Object mode**.
An independent **5-point Full Body** path (headset + ankle trackers + both
controllers) is also available. One mapping/IK core streams seven joint angles
per arm to simulation, to real arms through the vendor Tianji SDK, or to both
at once.

This package was extracted from the `wuji-hand-teleop` workspace so the
simulation and teleop core can be used without ROS 2. It is a research
integration, not a turnkey or hardware-validated distribution. Vendor SDKs
and machine-local configuration are required for live input/output.

> **Safety:** `arms` and `both` connect to real arms; `both` means simulation
> **and hardware**, not Full Body. Clear the work area, keep an emergency stop
> available, use low speed, and validate calibration and motion in `sim` first.

## Architecture

```text
default: PICO Object                    alternative: 5-point Full Body
4 trackers (wrist+upper arm,            headset + ankles + controllers
or redundant forearm pairs)                  | PicoBodySource
  | read_tracker_frame                       | DemoBodySource / Replay
  | PoseFreezeGuard                          | BodyRetarget
  | ForearmRedundancy (optional)             | quality/jump gates
  | persistent direction calibration         | chest-relative reference
  \_____________________       _____________/
                        v     v
                 MarvinTeleop (teleop.py)
       wrist-priority damped least squares (ik.py)
       + elbow null-space secondary task
                 /                   \
      sim: MuJoCo model        hardware: TianjiJointBackend
      (URDF -> MJCF)           (vendor SDK joint stream)

Object + one or more RealSense cameras
  -> unified episodes: RGB/depth/timestamps + robot/Sharpa HDF5 + manifest
```

- `mujoco_marvin.teleop` — Object input, URDF→MJCF conversion, shared IK core,
  viewer/headless loop, RealSense and episode hooks.
- `mujoco_marvin.body_*` — independent Full Body input/retargeting and preview.
- `mujoco_marvin.tracking` — Object frames, freeze guard, redundant-forearm
  fusion, and direction calibration.
- `mujoco_marvin.episode` / `wrist_camera` — unified recording lifecycle and
  per-camera RGB/depth/timestamp writers.
- `mujoco_marvin.tianji_backend` — hardware backend loaded from the vendor SDK;
  no ROS required.

## Installation

Python ≥ 3.10. From the parent workspace:

```bash
python3 -m venv .local/marvin-sim-venv
.local/marvin-sim-venv/bin/python -m pip install -e ./MuJoCo-Marvin
```

Use editable installation so the model and configuration remain available
from the checkout; a self-contained wheel with these resources is not provided.
The core installs `mujoco==3.3.7`, NumPy, SciPy, and PyYAML. Other optional
extras are `plot` and `ros`; ROS is only needed for the separate Body-mode
Wuji finger bridge, not for the main Object pipeline.

Install RealSense/episode recording with `.local/marvin-sim-venv/bin/python -m
pip install -e './MuJoCo-Marvin[episode]'`. The extra provides `pyrealsense2`,
`imageio`, `imageio-ffmpeg`, `Pillow`, `h5py`, and `pyzmq`. Sharpa feedback also
needs the external Sharpa SDK, Pilot/haptic decoder, and their own
dependencies. The calibration GUI and live camera view use tkinter
(`python3-tk` on Debian/Ubuntu).

## Simulation quickstart

Synthetic Object trackers; no headset or robot connection:

```bash
mujoco-marvin sim --demo --duration 10
# Headless smoke run:
mujoco-marvin sim --demo --headless --duration 2
```

The equivalent explicit module entry is `python -m mujoco_marvin.cli`.

## Default: Object trackers

### Configuration and headset

Copy the template and enter four unique SNs from XRoboToolkit:

```bash
cp config/pico_input.yaml.template config/pico_input.yaml
```

Two layouts are supported:

- **wrist + upper arm**: `tracker_sn_{left,right}_{wrist,arm}`; the upper-arm
  tracker supplies the elbow hint.
- **redundant forearm**: primary and auxiliary trackers about 180° apart on
  each forearm (`*_wrist` and `*_wrist_aux`). Remove/comment the `*_arm` keys in
  this layout. Fusion synthesizes the wrist pose through tracker occlusion and
  extrapolates an elbow hint from the forearm axis.

PICO PC Service and the PICO SDK must be installed separately. For USB, connect
XRoboToolkit to `127.0.0.1` and keep `adb reverse tcp:63901 tcp:63901` active;
the workspace launcher maintains it. For WiFi, connect to the PC's configured
address instead. Select **Motion Tracker → Object → Send**.

### Calibrate and run

```bash
mujoco-marvin calibrate   # save direction calibration, then enter sim teleop
mujoco-marvin sim         # simulation only
```

The first live run also enters calibration if no saved file exists. Follow the
GUI: hold the reference, capture forward/left/up motions, save, then verify in
simulation. Calibration defaults to `./.local/pico-axis-calibration.json`;
override it with `MUJOCO_MARVIN_CALIBRATION` or `--calibration-file`.

A freeze guard holds bit-exact stale poses and ramps back to live tracking.
Use `--freeze-frames`, `--ramp-seconds`, and `--forearm-length` to tune the
corresponding Object behavior.

Keys in the MuJoCo viewer:

- **Space** — pause/hold; resume recaptures the reference after a countdown.
- **R** — return both arms to home and recapture the reference.
- **E** — toggle the elbow null-space guide.
- **C** — with cameras enabled, start/stop one unified episode.

The persistent direction calibration, the startup/resume reference countdown,
and MANUS `.mcal` glove calibration are three different procedures.

The workspace wrapper `scripts/pico-teleop-local.sh` wires up the local venv,
PICO SDK, PC Service libraries, optional `.local/pico-teleop.env`, and the USB
watchdog. Its no-argument mode is Full Body `pico`; pass `sim`, `calibrate`,
`arms`, or `both` explicitly for Object workflows.

## RealSense and unified episodes

Enable any of `--wrist-camera`, `--main-camera`, and `--wrist2-camera`; pin the
corresponding `--*-camera-serial` when multiple devices are attached. With at
least one camera in **Object mode**, `C` or the camera-window button starts and
stops all enabled cameras and HDF5 streams as one episode. Recording is paused
at startup, and stale camera frames prevent a take from starting.

The default root is `<process cwd>/.local/episodes/episode_000N/`:

```text
episode_000N/
├── manifest.json
├── robot_states.h5
└── {wrist,main,wrist2}/       # enabled roles only
    ├── color_take001.mp4
    ├── timestamps_take001.jsonl
    └── depth/depth_*.png
```

Each camera records RGB MP4 (30 fps by default), 16-bit depth PNGs in mm
(10 Hz by default), and wall/monotonic timestamp JSONL. HDF5 contains:

- `robot_state/pico_teleop`: arm commands, tracker poses, simulated TCP, and
  real arm feedback when a hardware backend exists — both joint angles
  (`real_left_deg`/`real_right_deg`) and the real TCP pose FK-reconstructed
  from them (`real_tcp_*_pos`/`real_tcp_*_quat`, xyzw quaternions in the same
  base frame as the simulated TCP);
- `robot_state/sharpa_joints`: left/right 22-D joint **feedback** in radians;
- `robot_state/sharpa_commands`: left/right 22-D retargeted joint **commands**
  in radians (from the retargeting ZMQ PUB; absent in older episodes);
- `robot_state/sharpa_tactile/<key>`: per-sensor tactile stream decoded from the
  Pilot PUB — 3-axis `force` (N), 3-axis `torque` (Nm; together the sensor's
  six-axis wrench), the 240×240 uint8 `depth` deform map (pixel → mm via the
  SDK's `deform_map_value()`), the `contact_point` dot array (flat float32
  xyz triplets, 400 points), plus `side`/`finger`/`serial` labels.
  `torque`/`depth`/`contact_point` exist only in episodes whose manifest
  declares `"tactile_streams": ["force", "torque", "depth", "contact_point"]`;
  older episodes carry `force` only. The Pilot stream does not forward the RAW grayscale image —
  that needs a direct SDK connection (`fetch_tactile_frame`), which competes
  with Pilot for the tactile ports.

The manifest records camera frame counts and stream row counts. Streams share
a software monotonic clock for offline alignment; they are **not hardware
synchronized**. Sharpa joint capture uses a read-only SDK thread and has not
been validated on real hands; set `SHARPA_JOINT_AUTO=0` to disable it.

Episodes do not currently include raw MANUS skeletons, the actual 22-D commands
sent to Sharpa, camera intrinsics/extrinsics, task metadata, or calibration
snapshots. See `mujoco_marvin/episode.py` for exact fields and the workspace
root README for the end-to-end workflow.

`--record PATH.jsonl` is a separate Object tracker/IK diagnostic recorder, not
an episode manifest or a replacement for HDF5/camera data.

## Real arms

```bash
export TIANJI_ROBOT_IP=<controller-address>
export MARVIN_SDK_DIR=<vendor-sdk-directory>
mujoco-marvin arms   # real arms only
mujoco-marvin both   # MuJoCo viewer + real arms
```

`arms`/`both` require an IP from `--robot-ip`, `TIANJI_ROBOT_IP`, or `robot_ip`
in `config/tianji_robot.yaml` (in that priority order). Use the self-contained
SDK shipped with MarvinPlatform (`fx_robot.py` and `libMarvinSDK.so` together)
that matches the cabinet firmware; an older split-layout SDK may connect while
its state commands are ignored.

See `env.example` for model, SDK, and calibration overrides. Do not commit real
addresses, serial numbers, or vendor binaries.

## Alternative: 5-point Full Body

Use headset + left/right ankle trackers + both controllers, complete body
calibration in the headset, then select **Full Body → Send**:

```bash
mujoco-marvin preview              # read-only skeleton preview
mujoco-marvin pico                 # Full Body sim teleop
mujoco-marvin pico --demo          # synthetic skeleton
mujoco-marvin pico --replay recording.jsonl
```

`body` aliases `pico`. This path consumes a 24-node PXREA skeleton, uses its own
chest-relative reference, and does not read Object SNs or direction
calibration. Unified camera episodes are currently managed by the Object loop.
The optional `--hand-input ros` Body path is the only ROS-dependent feature and
is imported lazily.

## Tests

```bash
python -m pytest tests/
```

The tests use synthetic inputs and do not require teleoperation hardware. When
the real CAD is absent, a synthetic dual-arm URDF with matching topology is
used; tests specific to real CAD or Wuji meshes are skipped with a reason.
Passing tests is not evidence of real-robot or real-Sharpa validation.

## Acknowledgements and assets

Portions of the mapping/IK design are derived from the
[XRoboToolkit Teleop Sample](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python)
by XR Robotics, used under the MIT License (see `LICENSE`).

The Marvin/Tianji CAD and Wuji hand description ship in `assets/` for
convenience; see `assets/README.md` for provenance. They and the vendor SDKs
may have independent proprietary restrictions. Confirm redistribution rights
before republishing; the package's MIT license does not by itself grant rights
to vendor CAD, meshes, or SDKs. Vendor SDK binaries are not committed.
