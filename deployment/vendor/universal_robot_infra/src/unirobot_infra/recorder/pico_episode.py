"""Unified per-episode recording: all cameras + robot state + Sharpa tactile.

One toggle (C key or the unified camera view button) starts/stops a whole
episode. Each episode gets its own numbered folder under ``.local/episodes/``:

    episode_000N/
      manifest.json
      robot_states.h5             # robot_state/pico_teleop + robot_state/sharpa_tactile/*
      wrist/  color_take001.mp4, depth/, timestamps_take001.jsonl
      main/   ...
      wrist2/ ...

CameraRecorder still owns the per-camera video/depth/index files; this module
owns the episode lifecycle and the non-camera streams. All timestamps share
the teleop monotonic clock (plus wall time), so streams align offline.
"""

import json
from pathlib import Path
import threading
import time

import numpy as np


def _haptic_module():
    """Reuse the Sharpa haptic bridge's protobuf decoder and key map when available."""
    import importlib.util
    import os
    path = os.environ.get("SHARPA_HAPTIC_PY")
    if not path:
        sharpa_repo = os.environ.get("SHARPA_REPO")
        if not sharpa_repo:
            return None
        path = str(Path(sharpa_repo) / "teleop" / "standalone_haptic_feedback.py")
    if not Path(path).is_file():
        return None
    spec = importlib.util.spec_from_file_location("sharpa_haptic_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Hdf5StateWriter:
    """Universal_Robot_infra 风格的逐条写入：robot_state/<stream> 组、
    sys_time_ns 时间戳、field_order 元数据、gzip 压缩。"""

    def __init__(self, path):
        import h5py
        self._file = h5py.File(path, "w", track_order=True)
        self._string_dtype = h5py.string_dtype(encoding="utf-8")
        self._groups = {}

    def append(self, stream, sys_time_ns, fields):
        if stream not in self._groups:
            self._create_group(stream, fields)
        group = self._groups[stream]
        group["sys_time_ns"].resize(group["sys_time_ns"].shape[0] + 1, axis=0)
        group["sys_time_ns"][-1] = sys_time_ns
        for name, value in fields.items():
            dataset = group[name]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = value

    def _create_group(self, stream, fields):
        import numpy as np
        group = self._file.create_group(f"robot_state/{stream}", track_order=True)
        group.attrs["field_order"] = np.asarray(["sys_time_ns", *fields.keys()],
                                                dtype=self._string_dtype)
        group.create_dataset("sys_time_ns", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(256,),
                             compression="gzip", compression_opts=4)
        for name, value in fields.items():
            if isinstance(value, str):
                group.create_dataset(name, shape=(0,), maxshape=(None,),
                                     dtype=self._string_dtype, chunks=(256,))
            else:
                array = np.asarray(value, dtype=np.float32)
                group.create_dataset(name, shape=(0, *array.shape),
                                     maxshape=(None, *array.shape), dtype=np.float32,
                                     chunks=(256, *array.shape),
                                     compression="gzip", compression_opts=4)
        self._groups[stream] = group

    def close(self):
        self._file.flush()
        self._file.close()


class EpisodeManager:
    def __init__(self):
        self._cameras = []
        self._recording = False
        self._dir = None
        self._episode = 0
        self._hdf5 = None
        self._robot_rows = 0
        self._tactile_rows = 0
        self._tactile_thread = None
        self._tactile_stop = None
        self._tactile_decoder = None
        self._touch_key = {}
        self._finger_names = ()
        self._tactile_warned = False
        self._pilot_warned = False
        self._joint_rows = 0
        self._joint_thread = None
        self._joint_stop = None
        self._joint_warned_at = float("-inf")
        self._base_dir = None
        self._lock = threading.Lock()

    def configure(self, cameras, base_dir=None, tactile_address="tcp://127.0.0.1:48008"):
        import os
        self._cameras = list(cameras)
        self._base_dir = Path(base_dir) if base_dir else Path.cwd() / ".local" / "episodes"
        self._tactile_address = tactile_address
        if (os.environ.get("SHARPA_JOINT_AUTO", "1") != "0"
                and self._joint_thread is None):
            self._joint_stop = threading.Event()
            self._joint_thread = threading.Thread(target=self._sharpa_joint_loop,
                                                  daemon=True, name="episode-sharpa-joints")
            self._joint_thread.start()

    @property
    def active(self):
        return self._base_dir is not None

    @property
    def recording(self):
        return self._recording

    @property
    def episode_dir(self):
        return self._dir

    def toggle(self):
        with self._lock:
            if self._recording:
                self._stop_locked()
            else:
                return self._start_locked()

    def _start_locked(self):
        from unirobot_infra.camera.realsense_d405 import CameraRecorder
        stale = [cam.role for cam in self._cameras if cam.frame_age() > 1.]
        if stale:
            print(f"录制未开始：{', '.join(stale)} 无新帧；请等相机恢复后再按开始",
                  flush=True)
            return False
        self._base_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in self._base_dir.glob("episode_*"))
        last = 0
        for name in existing:
            if name.startswith("episode_"):
                try:
                    last = max(last, int(name.split("_")[1].split(".")[0]))
                except (ValueError, IndexError):
                    pass
        self._episode = last + 1
        self._dir = self._base_dir / f"episode_{self._episode:04d}"
        self._dir.mkdir()
        self._started_wall = time.time()
        self._started_mono = time.monotonic()
        for cam in self._cameras:
            rec = CameraRecorder(self._dir / cam.role, recording=True)
            cam.recorder = rec
        self._hdf5 = _Hdf5StateWriter(self._dir / "robot_states.h5")
        self._robot_rows = 0
        self._tactile_rows = 0
        self._joint_rows = 0
        self._recording = True
        self._ensure_pilot_locked()
        self._start_tactile_locked()
        print(f"Episode {self._episode:04d} 开始录制：{self._dir}", flush=True)
        return True

    def _stop_locked(self):
        self._recording = False
        self._last_camera_frames = {}
        for cam in self._cameras:
            if cam.recorder is not None:
                self._last_camera_frames[cam.role] = cam.recorder.frames
                cam.recorder.close()
                cam.recorder = None
        if self._hdf5 is not None:
            self._hdf5.close()
            self._hdf5 = None
        self._write_manifest_locked()
        print(f"Episode {self._episode:04d} 停止，数据已收尾", flush=True)

    def stop(self):
        with self._lock:
            if self._recording:
                self._stop_locked()

    def _write_manifest_locked(self):
        cameras = {}
        for cam in self._cameras:
            role_dir = self._dir / cam.role
            depth_dir = role_dir / "depth"
            cameras[cam.role] = {
                "frames": self._last_camera_frames.get(cam.role, 0),
                "depth_frames": len(list(depth_dir.glob("*.png"))) if depth_dir.is_dir() else 0,
                "files": sorted(p.name for p in role_dir.glob("*") if p.is_file()),
            }
        manifest = {
            "episode": self._episode,
            "start_wall": self._started_wall,
            "stop_wall": time.time(),
            "start_mono": self._started_mono,
            "stop_mono": time.monotonic(),
            "cameras": cameras,
            "robot_rows": self._robot_rows,
            "tactile_rows": self._tactile_rows,
            "sharpa_joint_rows": self._joint_rows,
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")

    def record_robot(self, controller, robot, poses, frame):
        """One teleop tick of arm/tracker state; only written while recording."""
        from scipy.spatial.transform import Rotation
        fields = {"frame": np.int64(frame),
                  "command_rad": controller.command,
                  "pose_left_wrist": poses.get("left_wrist", np.full(7, np.nan)),
                  "pose_right_wrist": poses.get("right_wrist", np.full(7, np.nan)),
                  "pose_left_arm": poses.get("left_arm", np.full(7, np.nan)),
                  "pose_right_arm": poses.get("right_arm", np.full(7, np.nan))}
        for side in ("left", "right"):
            site = controller.site_ids[side]["tcp"]
            fields[f"sim_tcp_{side}_pos"] = controller.data.site_xpos[site]
            fields[f"sim_tcp_{side}_quat"] = Rotation.from_matrix(
                controller.data.site_xmat[site].reshape(3, 3)).as_quat()
        if robot is not None:
            left, right = robot.read_joints_deg()
            fields["real_left_deg"] = left
            fields["real_right_deg"] = right
        with self._lock:
            if self._recording and self._hdf5 is not None:
                self._hdf5.append("pico_teleop", int(time.monotonic() * 1e9), fields)
                self._robot_rows += 1

    # -------------------- Sharpa joint feedback --------------------

    def _sharpa_joint_warning(self, message):
        now = time.monotonic()
        if now - self._joint_warned_at >= 10.:
            print(f"Sharpa 关节录制：{message}", flush=True)
            self._joint_warned_at = now

    def _sharpa_joint_loop(self):
        import importlib
        import os
        import sys
        sdk_path = os.environ.get("SHARPA_SDK_PY", "/opt/sharpa-wave-sdk/python")
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            manager = importlib.import_module("sharpa").SharpaWaveManager.get_instance()
        except Exception as exc:
            self._sharpa_joint_warning(f"无法加载 SDK（{sdk_path}）：{exc}")
            return
        serials = {
            side: os.environ.get(f"SHARPA_{side.upper()}_SN", "")
            for side in ("left", "right")
        }
        if not all(serials.values()):
            self._sharpa_joint_warning("未配置 SHARPA_LEFT_SN / SHARPA_RIGHT_SN，关节录制已停用")
            return
        hands = None
        try:
            while not self._joint_stop.is_set():
                if hands is None:
                    try:
                        available = set(manager.get_all_device_sn())
                        if all(serial in available for serial in serials.values()):
                            hands = {side: manager.connect(serial)
                                     for side, serial in serials.items()}
                            if not all(hands.values()):
                                raise RuntimeError("SDK connect 未返回双手设备")
                            print("Sharpa 关节录制：左右手反馈已连接（22 维，弧度）", flush=True)
                        else:
                            self._sharpa_joint_warning("等待双手上线；本段可能暂无手指关节数据")
                            self._joint_stop.wait(1.)
                            continue
                    except Exception as exc:
                        hands = None
                        self._sharpa_joint_warning(f"连接失败，1 秒后重试：{exc}")
                        self._joint_stop.wait(1.)
                        continue
                try:
                    fields = {}
                    for side in ("left", "right"):
                        error, angles = hands[side].get_joint_position_degree()
                        values = np.asarray(angles, dtype=np.float64)
                        if error.code != 0:
                            raise RuntimeError(
                                f"{side} 读取失败：{getattr(error, 'message', error)}")
                        if values.shape != (22,) or not np.all(np.isfinite(values)):
                            raise RuntimeError(f"{side} 反馈不是 22 个有限关节角")
                        fields[f"{side}_rad"] = np.deg2rad(values)
                    with self._lock:
                        if self._recording and self._hdf5 is not None:
                            self._hdf5.append("sharpa_joints", int(time.monotonic() * 1e9), fields)
                            self._joint_rows += 1
                except Exception as exc:
                    hands = None
                    self._sharpa_joint_warning(f"反馈中断，正在重连：{exc}")
                    self._joint_stop.wait(1.)
                    continue
                self._joint_stop.wait(.02)
        finally:
            try:
                manager.disconnect_all()
            except Exception:
                pass

    # -------------------- Sharpa tactile (Pilot local PUB) --------------------

    def _ensure_pilot_locked(self):
        """录制开始时自动拉起 Sharpa Pilot，保证 48008 触觉 PUB 有数据源。

        已在运行则不重复启动；启动失败只告警，不阻塞其它流的录制。
        SHARPA_PILOT_AUTO=0 关闭自动启动，SHARPA_PILOT_BIN 覆盖程序路径。
        """
        import os
        import subprocess
        if os.environ.get("SHARPA_PILOT_AUTO", "1") == "0":
            return
        binary = os.environ.get("SHARPA_PILOT_BIN", "/opt/sharpa-pilot/sharpa-pilot")
        if not Path(binary).is_file():
            if not self._pilot_warned:
                print(f"触觉录制：未找到 Sharpa Pilot（{binary}），本段可能无触觉流",
                      flush=True)
                self._pilot_warned = True
            return
        try:
            running = any(
                subprocess.run(["pgrep", *args], capture_output=True).returncode == 0
                for args in (["-x", Path(binary).name], ["-f", binary]))
        except OSError:
            running = True  # 无法检测时宁可不重复拉起
        if running:
            return
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":1")
        try:
            with open(self._base_dir / "sharpa-pilot.log", "ab") as log:
                subprocess.Popen([binary], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            print("触觉录制：已自动启动 Sharpa Pilot，首次加载需等几秒", flush=True)
        except OSError as exc:
            print(f"触觉录制：Sharpa Pilot 启动失败：{exc}", flush=True)

    def _start_tactile_locked(self):
        if self._tactile_decoder is None:
            module = _haptic_module()
            if module is not None:
                self._tactile_decoder = module.decode_tactile
                self._touch_key = module.TOUCH_KEY
                self._finger_names = module.FINGERS
        if self._tactile_decoder is None:
            if not self._tactile_warned:
                print("触觉录制：未找到 Sharpa 解码模块（SHARPA_HAPTIC_PY），本段无触觉流",
                      flush=True)
                self._tactile_warned = True
            return
        if self._tactile_thread is None:
            self._tactile_stop = threading.Event()
            self._tactile_thread = threading.Thread(target=self._tactile_loop,
                                                    daemon=True, name="episode-tactile")
            self._tactile_thread.start()

    def _record_tactile_message(self, msg):
        try:
            key, _serial, force = self._tactile_decoder(msg)
            side, finger_index = self._touch_key[key]
        except Exception:
            return
        # Pilot 消息里的 serial 不区分手（实测右手键也带左手 SN），
        # 手别以键映射为准，序列号取该侧实际设备
        import os
        serial = os.environ.get(f"SHARPA_{side.upper()}_SN") or "unknown"
        with self._lock:
            if self._recording and self._hdf5 is not None:
                self._hdf5.append(f"sharpa_tactile/{key}", int(time.monotonic() * 1e9),
                                  {"force": np.asarray(force, dtype=np.float32),
                                   "serial": serial, "side": side,
                                   "finger": self._finger_names[finger_index]})
                self._tactile_rows += 1

    def _tactile_loop(self):
        import zmq
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self._tactile_address)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVTIMEO, 500)
        while not self._tactile_stop.is_set():
            try:
                msg = sub.recv()
            except zmq.Again:
                continue
            self._record_tactile_message(msg)
        try:
            sub.close()
            ctx.term()
        except Exception:
            pass
        self._tactile_thread = None

    def close(self):
        self.stop()
        if self._tactile_stop is not None:
            self._tactile_stop.set()
        if self._joint_stop is not None:
            self._joint_stop.set()
        for thread in (self._tactile_thread, self._joint_thread):
            if thread is not None:
                thread.join(timeout=2)


EPISODES = EpisodeManager()
