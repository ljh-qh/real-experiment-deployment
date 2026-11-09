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
import sys
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
            dataset = group.get(name)
            if dataset is None:
                continue  # group layout is fixed by the first appended row
            dataset.resize(dataset.shape[0] + 1, axis=0)
            if isinstance(value, str):
                dataset[-1] = value
            else:
                dataset[-1] = np.asarray(value, dtype=dataset.dtype)

    def fields_of(self, stream):
        """Field names of an already-created stream, or None if not created yet."""
        group = self._groups.get(stream)
        if group is None:
            return None
        return set(group.keys()) - {"sys_time_ns"}

    def _create_group(self, stream, fields):
        import numpy as np
        group = self._file.create_group(f"robot_state/{stream}", track_order=True)
        group.attrs["field_order"] = np.asarray(["sys_time_ns", *fields.keys()],
                                                dtype=self._string_dtype)
        if stream in ("sharpa_joints", "sharpa_commands"):
            from .sharpa_backend import JOINT_NAMES
            group.attrs["units"] = "rad"
            group.attrs["source"] = ("measured_feedback" if stream == "sharpa_joints"
                                     else "retargeted_command")
            group.attrs["joint_names"] = np.asarray(JOINT_NAMES, dtype=self._string_dtype)
        group.create_dataset("sys_time_ns", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(256,),
                             compression="gzip", compression_opts=4)
        for name, value in fields.items():
            if isinstance(value, str):
                group.create_dataset(name, shape=(0,), maxshape=(None,),
                                     dtype=self._string_dtype, chunks=(256,))
            else:
                array = np.asarray(value)
                if not (array.dtype == np.uint8 or array.dtype.kind == "i"):
                    array = array.astype(np.float32)
                # Cap chunk size near 1 MB so gzip stays cheap and random
                # reads don't drag the default 1 MB chunk cache around.
                chunk_rows = max(1, min(256, 1_000_000 // max(1, array.nbytes)))
                group.create_dataset(name, shape=(0, *array.shape),
                                     maxshape=(None, *array.shape), dtype=array.dtype,
                                     chunks=(chunk_rows, *array.shape),
                                     compression="gzip",
                                     compression_opts=2 if array.ndim >= 2 else 4)
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
        self._tactile_full = False
        self._touch_key = {}
        self._finger_names = ()
        self._tactile_warned = False
        self._tactile_seen = set()
        self._tactile_stride = 0  # 0 = 未初始化（录制开始时从 env 读取）
        self._tactile_queue = None      # queue.Queue：接收线程 → 写盘线程
        self._tactile_writer = None
        self._tactile_dropped = 0       # 队列满丢弃的帧数（>0 说明写盘跟不上）
        self._fields_parser = None      # haptic 模块的轻量 protobuf 字段遍历器
        self._tactile_proc = None       # 独立触觉录制子进程（tactile_recorder.py）
        self._tactile_proc_log = None   # 子进程 stdout 日志路径（含 TACTILE_STATS 行）
        self._pilot_warned = False
        self._joint_rows = 0
        self._joint_thread = None
        self._joint_stop = None
        self._joint_warned_at = float("-inf")
        self._joint_last_mono = float("-inf")
        self._joint_status = "尚未收到双手关节反馈"
        self._recording_errors = []
        self._base_dir = None
        self._fk_data = None
        self._cmd_thread = None
        self._finalize_thread = None  # 停止收尾在后台线程，遥操主循环不等编码器 flush
        self._cmd_stop = None
        self._cmd_rows = 0
        self._cmd_pb2 = None
        self._cmd_status = "尚未收到手指指令流"
        self._cmd_warned_at = float("-inf")
        self._cmd_latest = {}
        self._lock = threading.RLock()  # 可重入：_stop_locked 持锁排空队列时会进 _append_tactile

    def configure(self, cameras, base_dir=None, tactile_address="tcp://127.0.0.1:48008"):
        import os
        self._cameras = list(cameras)
        base_dir = base_dir or os.environ.get("MUJOCO_MARVIN_EPISODE_DIR")
        self._base_dir = Path(base_dir) if base_dir else Path.cwd() / ".local" / "episodes"
        self._tactile_address = tactile_address
        if (os.environ.get("SHARPA_JOINT_AUTO", "1") != "0"
                and self._joint_thread is None):
            self._joint_stop = threading.Event()
            self._joint_thread = threading.Thread(target=self._sharpa_joint_loop,
                                                  daemon=True, name="episode-sharpa-joints")
            self._joint_thread.start()
        if (os.environ.get("SHARPA_COMMAND_AUTO", "1") != "0"
                and self._cmd_thread is None):
            self._cmd_stop = threading.Event()
            self._cmd_thread = threading.Thread(target=self._sharpa_command_loop,
                                                daemon=True, name="episode-sharpa-commands")
            self._cmd_thread.start()

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
            if not self._recording:
                return self._start_locked()
            self._stop_async_locked()
            return None

    def _stop_async_locked(self):
        """持锁调用：只做快操作（翻标志、打戳、摘相机），慢收尾交给后台线程。

        停止时刻在打戳当下（数据真实结束于此）；之后子进程排空、编码器
        flush 耗时不计入 episode 边界，否则各数据流末端会比 stop_mono
        短出收尾耗时，触发 episode_check 的 coverage 报错。
        """
        self._recording = False  # 先翻标志位，生产者线程立即停止写入
        stopped_wall, stopped_mono = time.time(), time.monotonic()
        detached = self._detach_cameras_locked()
        self._finalize_thread = threading.Thread(
            target=self._finalize_stop,
            args=(stopped_wall, stopped_mono, detached),
            daemon=True, name="episode-finalize")
        self._finalize_thread.start()

    def _finalize_stop(self, stopped_wall, stopped_mono, detached):
        """后台线程：排空触觉写盘/子进程、flush 编码器、关 h5、写 manifest。"""
        try:
            writer = self._tactile_writer
            if writer is not None:
                writer.join(timeout=15)  # 锁外等待，避免与 _append_tactile 的锁互掐
            self._stop_locked(stopped_wall, stopped_mono, detached)
        except Exception as exc:
            print(f"Episode {self._episode:04d} 收尾异常：{type(exc).__name__}: {exc}",
                  flush=True)

    def wait_finalized(self, timeout=30):
        """等待上一次停止的后台收尾完成；返回是否在超时内完成。"""
        thread = self._finalize_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _start_locked(self):
        import os
        from .wrist_camera import CameraRecorder
        pending = [cam.role for cam in self._cameras
                   if getattr(cam, "unfinished_recorder", None) is not None
                   and cam.unfinished_recorder.diagnostics["writer_alive"]]
        if pending:
            print(f"录制未开始：{', '.join(pending)} 的上一写盘线程尚未退出；请等待或重启采集", flush=True)
            return False
        if self._finalize_thread is not None and self._finalize_thread.is_alive():
            # 不能在这里 join：收尾线程会经 _append_tactile 拿同一把锁，持锁等待会死锁
            print("录制未开始：上一条的收尾写盘仍在进行；请稍候再按开始", flush=True)
            return False
        stale = [cam.role for cam in self._cameras if cam.frame_age() > 1.]
        if stale:
            print(f"录制未开始：{', '.join(stale)} 无新帧；请等相机恢复后再按开始",
                  flush=True)
            return False
        self._joint_required = os.environ.get("SHARPA_RECORD_REQUIRED", "0") == "1"
        if self._joint_required and time.monotonic() - self._joint_last_mono > .25:
            print(f"录制未开始：Sharpa 双手关节反馈未就绪（{self._joint_status}）", flush=True)
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
        self._robot_rows = 0
        self._tactile_rows = 0
        self._tactile_seen = set()
        self._tactile_dropped = 0
        self._joint_rows = 0
        self._cmd_rows = 0
        self._recording_errors = []
        attached = []
        try:
            for cam in self._cameras:
                previous, cam.recorder = cam.recorder, None
                if previous is not None:
                    previous.close()
                rec = CameraRecorder(self._dir / cam.role, recording=True,
                                     **getattr(cam, "recorder_settings", {}))
                cam.recorder = rec
                attached.append(cam)
            self._hdf5 = _Hdf5StateWriter(self._dir / "robot_states.h5")
        except Exception as exc:
            self._recording_errors.append(f"录制初始化失败：{type(exc).__name__}: {exc}")
            for cam in attached:
                recorder, cam.recorder = cam.recorder, None
                if recorder is not None:
                    recorder.close()
                    cam.unfinished_recorder = recorder if recorder.diagnostics["writer_alive"] else None
            self._last_camera_frames = {}
            self._last_camera_diagnostics = {}
            self._stopped_wall, self._stopped_mono = time.time(), time.monotonic()
            self._write_manifest_locked()
            print(f"录制未开始：{exc}；相机采集和遥操继续运行", flush=True)
            return False
        self._recording = True
        self._ensure_pilot_locked()
        if not self._start_tactile_recorder_locked():  # 独立进程优先（避免抢 GIL）
            self._start_tactile_locked()               # 拉起失败回退进程内写盘
        print(f"Episode {self._episode:04d} 开始录制：{self._dir}", flush=True)
        return True

    def _detach_cameras_locked(self):
        """停止时刻就摘掉相机录制器，之后捕获的帧不再入片。

        编码器 close() 只 flush 已捕获的帧，视频末端因此贴着 stop_mono，
        不会把子进程排空/收尾耗时录进 episode。
        """
        detached = []
        for cam in self._cameras:
            recorder, cam.recorder = cam.recorder, None
            detached.append((cam, recorder, cam.recording_error))
        return detached

    def _stop_locked(self, stopped_wall=None, stopped_mono=None, detached=None):
        if self._joint_required and (not self._joint_rows
                                    or time.monotonic() - self._joint_last_mono > .5):
            self._recording_errors.append("Sharpa 关节反馈缺失或中断超过 0.5 秒")
        self._recording = False
        if stopped_wall is None or stopped_mono is None:
            stopped_wall, stopped_mono = time.time(), time.monotonic()
        if detached is None:
            detached = self._detach_cameras_locked()
        self._stop_tactile_recorder_locked()  # 停触觉子进程（排空队列、关文件）
        self._last_camera_frames = {}
        self._last_camera_diagnostics = {}
        # Detach every producer before draining any encoder. Otherwise later
        # cameras keep recording throughout the first camera's finalization.
        for cam, recorder, error in detached:
            if recorder is not None:
                recorder.close()
                cam.unfinished_recorder = recorder if recorder.diagnostics["writer_alive"] else None
                self._last_camera_frames[cam.role] = recorder.frames
                error = error or recorder.error
                self._last_camera_diagnostics[cam.role] = {
                    "capture_frames": cam.frames, "recoveries": cam.recoveries,
                    "capture_errors": cam.capture_errors, "writer": recorder.diagnostics}
            if error:
                message = f"{cam.role} 相机录制失败：{error}"
                if message not in self._recording_errors:
                    self._recording_errors.append(message)
        if self._hdf5 is not None:
            # 同步排空进程内触觉写盘队列（回退路径用），最后关文件
            if self._tactile_queue is not None:
                import queue as queue_mod
                while True:
                    try:
                        msg, ts_ns = self._tactile_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    self._handle_tactile_item(msg, ts_ns)
            self._hdf5.close()
            self._hdf5 = None
        self._stopped_wall, self._stopped_mono = stopped_wall, stopped_mono
        self._write_manifest_locked()
        print(f"Episode {self._episode:04d} 停止，数据已收尾", flush=True)

    def stop(self):
        with self._lock:
            if self._recording:
                self._stop_async_locked()
        # 锁外等待：收尾线程会经 _append_tactile 拿同一把锁，持锁 join 会死锁
        self.wait_finalized()  # 程序退出路径：同步等收尾写完

    def _write_manifest_locked(self):
        cameras = {}
        for cam in self._cameras:
            role_dir = self._dir / cam.role
            depth_dir = role_dir / "depth"
            cameras[cam.role] = {
                "frames": self._last_camera_frames.get(cam.role, 0),
                "depth_frames": len(list(depth_dir.glob("*.png"))) if depth_dir.is_dir() else 0,
                "files": sorted(p.name for p in role_dir.glob("*") if p.is_file()),
                "diagnostics": self._last_camera_diagnostics.get(cam.role, {}),
            }
        manifest = {
            "episode": self._episode,
            "start_wall": self._started_wall,
            "stop_wall": self._stopped_wall,
            "start_mono": self._started_mono,
            "stop_mono": self._stopped_mono,
            "cameras": cameras,
            "robot_rows": self._robot_rows,
            "tactile_rows": self._tactile_rows,
            "tactile_dropped": self._tactile_dropped,
            "tactile_streams": sorted(self._tactile_seen | {"force"}),
            "sharpa_joint_rows": self._joint_rows,
            "sharpa_command_rows": self._cmd_rows,
            "sharpa_joint_required": self._joint_required,
            "status": "incomplete" if self._recording_errors else "closed",
            "recording_errors": list(self._recording_errors),
        }
        try:
            (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                    encoding="utf-8")
        except OSError as exc:
            # Episode folder was moved/deleted mid-recording; losing the
            # manifest must not take down the teleop loop.
            print(f"Episode {self._episode:04d} manifest 写入失败（目录已被移动/删除）：{exc}",
                  flush=True)

    def record_robot(self, controller, robot, poses, frame, extra=None):
        """One teleop tick of arm/tracker state; only written while recording."""
        from scipy.spatial.transform import Rotation
        if not self._recording:
            return
        fields = {"frame": np.int64(frame), "command_rad": controller.command}
        if extra:
            fields.update(extra)
        serials = getattr(controller, "tracker_sns", {})
        for role in ("left_wrist", "right_wrist", "left_arm", "right_arm"):
            fields[f"pose_{role}"] = poses.get(serials.get(role, role),
                                              poses.get(role, np.full(7, np.nan)))
        for side in ("left", "right"):
            site = controller.site_ids[side]["tcp"]
            fields[f"sim_tcp_{side}_pos"] = controller.data.site_xpos[site]
            fields[f"sim_tcp_{side}_quat"] = Rotation.from_matrix(
                controller.data.site_xmat[site].reshape(3, 3)).as_quat()
        if robot is not None:
            left, right = robot.read_joints_deg()
            fields["real_left_deg"] = left
            fields["real_right_deg"] = right
            for side, (pos, quat) in self._real_tcp(controller, left, right).items():
                fields[f"real_tcp_{side}_pos"] = pos
                fields[f"real_tcp_{side}_quat"] = quat
        with self._lock:
            if self._recording and self._hdf5 is not None:
                failures = []
                for cam in self._cameras:
                    if cam.recording_error:
                        failures.append(f"{cam.role} 相机录制失败：{cam.recording_error}")
                    elif cam.frame_age() > .5:
                        failures.append(f"{cam.role} 相机采集超过 0.5 秒没有新帧")
                if failures:
                    self._recording_errors.extend(failures)
                    self._stop_async_locked()
                    print("本条 episode 已因相机异常停止并标为 incomplete；采集/预览继续", flush=True)
                    return
                if self._joint_required and time.monotonic() - self._joint_last_mono > .5:
                    self._recording_errors.append("录制期间 Sharpa 关节流中断超过 0.5 秒")
                    self._stop_async_locked()
                    print("本条 episode 已提前停止并标为 incomplete；恢复 Sharpa 反馈后重新录制", flush=True)
                    return
                self._hdf5.append("pico_teleop", int(time.monotonic() * 1e9), fields)
                self._robot_rows += 1

    # -------------------- Sharpa joint feedback --------------------

    def _real_tcp(self, controller, left_deg, right_deg):
        """FK of the real joint feedback through the teleop model.

        The vendor SDK exposes no Cartesian TCP channel, so the real TCP pose
        is reconstructed here with the same URDF/base frame as ``sim_tcp_*``.
        """
        import mujoco
        from scipy.spatial.transform import Rotation
        if self._fk_data is None:
            self._fk_data = mujoco.MjData(controller.model)
        data = self._fk_data
        data.qpos[controller.qpos_ids["left"]] = np.deg2rad(left_deg)
        data.qpos[controller.qpos_ids["right"]] = np.deg2rad(right_deg)
        mujoco.mj_forward(controller.model, data)
        out = {}
        for side in ("left", "right"):
            site = controller.site_ids[side]["tcp"]
            out[side] = (data.site_xpos[site].copy(),
                         Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat())
        return out

    def _sharpa_joint_warning(self, message):
        self._joint_status = message
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
                            hands = {side: manager.connect(serial, skip_tactile=True)
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
                        if int(error.code) != 0:
                            raise RuntimeError(
                                f"{side} 读取失败：{getattr(error, 'message', error)}")
                        if values.shape != (22,) or not np.all(np.isfinite(values)):
                            raise RuntimeError(f"{side} 反馈不是 22 个有限关节角")
                        fields[f"{side}_rad"] = np.deg2rad(values)
                        fields[f"{side}_sn"] = serials[side]
                    measured_mono = time.monotonic()
                    with self._lock:
                        self._joint_last_mono = measured_mono
                        self._joint_status = "双手关节反馈正常"
                        if (self._recording and self._hdf5 is not None
                                and measured_mono >= self._started_mono):
                            self._hdf5.append("sharpa_joints", int(measured_mono * 1e9), fields)
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

    # -------------------- Sharpa joint commands (retargeting ZMQ PUB) --------------------

    def _command_pb2(self):
        """Load the Sharpa HandAction protobuf from the retargeting repo."""
        import importlib.util
        import os
        if self._cmd_pb2 is not None:
            return self._cmd_pb2
        path = os.environ.get("SHARPA_HAND_PB2")
        if not path:
            repo = os.environ.get("SHARPA_REPO")
            if not repo:
                return None
            path = str(Path(repo) / "retargeting" / "include" / "proto_hand" / "sharpa_hand_pb2.py")
        if not Path(path).is_file():
            return None
        spec = importlib.util.spec_from_file_location("sharpa_hand_pb2_episode", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._cmd_pb2 = module
        return module

    def _cmd_warning(self, message):
        now = time.monotonic()
        if now - self._cmd_warned_at > 5.:
            print(f"Sharpa 指令录制：{message}", flush=True)
            self._cmd_warned_at = now
        self._cmd_status = message

    def _sharpa_command_loop(self):
        """Subscribe the retargeting HandAction PUB (tcp://127.0.0.1:6668) and log
        the retargeted joint commands (radians, native SDK joint order) while recording."""
        import os
        pb2 = self._command_pb2()
        if pb2 is None:
            self._cmd_warning("未找到 sharpa_hand_pb2（SHARPA_REPO 未配置？），指令录制已停用")
            return
        try:
            import zmq
        except ImportError:
            self._cmd_warning("缺少 zmq，指令录制已停用")
            return
        from .sharpa_backend import JOINT_NAMES
        address = os.environ.get("SHARPA_COMMAND_ADDRESS", "tcp://127.0.0.1:6668")
        ctx = zmq.Context.instance()
        while not self._cmd_stop.is_set():
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.RCVHWM, 4)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(address)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            try:
                while not self._cmd_stop.is_set():
                    try:
                        payload = sock.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        self._cmd_stop.wait(.01)
                        continue
                    msg = pb2.HandAction()
                    msg.ParseFromString(payload)
                    # Each frame may carry only one side (the retargeting publishes
                    # per-side updates); cache per side and record the combined
                    # latest command once both sides have been seen.
                    updated = False
                    for side in ("left", "right"):
                        joint = msg.joint_left if side == "left" else msg.joint_right
                        if not joint.name:
                            continue
                        by_name = {}
                        for n, p in zip(joint.name, joint.position):
                            if n.startswith(("left_", "right_")):
                                n = n.split("_", 1)[1]  # retargeting names are side-prefixed
                            by_name[n] = p
                        values = np.asarray([by_name.get(n, np.nan) for n in JOINT_NAMES],
                                            dtype=np.float64)
                        if not np.all(np.isfinite(values)):
                            self._cmd_warning(f"{side} 指令帧关节名不全，已忽略该侧")
                            continue
                        self._cmd_latest[side] = values
                        updated = True
                    if not updated:
                        continue
                    if any(side not in self._cmd_latest for side in ("left", "right")):
                        continue  # 等双侧都至少出现过一次
                    received_mono = time.monotonic()
                    fields = {"left_rad": self._cmd_latest["left"],
                              "right_rad": self._cmd_latest["right"]}
                    with self._lock:
                        self._cmd_status = "双手指令流正常"
                        if (self._recording and self._hdf5 is not None
                                and received_mono >= self._started_mono):
                            self._hdf5.append("sharpa_commands", int(received_mono * 1e9), fields)
                            self._cmd_rows += 1
            except Exception as exc:
                self._cmd_warning(f"指令流异常，1 秒后重连：{exc}")
                self._cmd_stop.wait(1.)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    # -------------------- Sharpa tactile (Pilot local PUB) --------------------

    # -------------------- Sharpa tactile (独立子进程录制) --------------------

    def _start_tactile_recorder_locked(self):
        """拉起独立触觉录制子进程（tactile_recorder.py 写 tactile.h5）。

        全量触觉写盘在遥操进程内会和主线程抢 GIL（实测 100→35 帧/s），
        独立进程有自己的 GIL，且 time.monotonic() 跨进程对齐。
        SHARPA_TACTILE_SUBPROCESS=0 可关掉回退到进程内写盘。
        """
        import os
        import subprocess
        if os.environ.get("SHARPA_TACTILE_SUBPROCESS", "1") == "0":
            return False
        if self._tactile_decoder is None and _haptic_module() is None:
            return False  # 没解码模块，子进程也跑不了，让进程内路径报警告
        script = Path(__file__).with_name("tactile_recorder.py")
        self._tactile_proc_log = self._dir / "tactile-recorder.log"
        try:
            log = open(self._tactile_proc_log, "wb")
            self._tactile_proc = subprocess.Popen(
                [sys.executable, str(script), "--dir", str(self._dir),
                 "--address", self._tactile_address],
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        except Exception as exc:
            print(f"触觉录制：子进程拉起失败（{exc}），回退进程内写盘", flush=True)
            self._tactile_proc = None
            return False
        print(f"触觉录制：独立进程 pid={self._tactile_proc.pid}", flush=True)
        return True

    def _stop_tactile_recorder_locked(self):
        proc = self._tactile_proc
        if proc is None:
            return
        self._tactile_proc = None
        try:
            proc.terminate()  # SIGTERM → 子进程排空队列、关文件、打印 TACTILE_STATS
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        # 从子进程日志末尾读统计，写进 manifest
        try:
            for line in reversed(self._tactile_proc_log.read_text().splitlines()):
                if line.startswith("TACTILE_STATS "):
                    stats = json.loads(line[len("TACTILE_STATS "):])
                    self._tactile_rows = stats.get("rows", self._tactile_rows)
                    self._tactile_dropped = stats.get("dropped", 0)
                    for fields in stats.get("streams", {}).values():
                        self._tactile_seen.update(fields)
                    break
        except Exception:
            pass
        if self._tactile_rows == 0:
            self._recording_errors.append("触觉子进程未录到数据（看 tactile-recorder.log）")

    def _ensure_pilot_locked(self):
        """录制开始时确认 SDK 触觉桥在跑（48008 触觉 PUB 的数据源）。

        数据链路已不再经过 Pilot（固件单目的地，SDK 桥与 Pilot 互斥）。
        桥没在跑时自动拉起；SHARPA_TACTILE_BRIDGE 可覆盖脚本路径，
        SHARPA_PILOT_AUTO=0 关闭自动拉起。
        """
        import os
        import subprocess
        if os.environ.get("SHARPA_PILOT_AUTO", "1") == "0":
            return
        running = subprocess.run(["pgrep", "-f", "sdk_tactile_bridge"],
                                 capture_output=True).returncode == 0
        if running:
            return
        repo = os.environ.get("SHARPA_REPO", "")
        script = os.environ.get("SHARPA_TACTILE_BRIDGE",
                                str(Path(repo) / "teleop" / "sdk_tactile_bridge.py"))
        venv_py = Path(repo) / ".venv" / "bin" / "python"
        if not Path(script).is_file() or not venv_py.is_file():
            if not self._pilot_warned:
                print(f"触觉录制：未找到 SDK 触觉桥（{script}），本段可能无触觉流",
                      flush=True)
                self._pilot_warned = True
            return
        try:
            with open(self._base_dir / "tactile-bridge.log", "ab") as log:
                subprocess.Popen([str(venv_py), script], env=dict(os.environ),
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            print("触觉录制：已自动启动 SDK 触觉桥，首次连接需等几秒", flush=True)
        except OSError as exc:
            print(f"触觉录制：SDK 触觉桥启动失败：{exc}", flush=True)

    def _start_tactile_locked(self):
        if self._tactile_decoder is None:
            module = _haptic_module()
            if module is not None:
                full = getattr(module, "decode_tactile_full", None)
                if full is not None:
                    # dict: key/serial/force/torque/depth/contact/frame_id/dist_force/raw
                    self._tactile_decoder = full
                    self._tactile_full = True
                else:
                    base = module.decode_tactile
                    self._tactile_decoder = lambda data: dict(
                        zip(("key", "serial", "force"), base(data)),
                        torque=None, depth=None, contact=None, frame_id=None,
                        dist_force=None, raw=None)
                self._touch_key = module.TOUCH_KEY
                self._finger_names = module.FINGERS
                self._fields_parser = getattr(module, "_fields", None)
        if self._tactile_decoder is None:
            if not self._tactile_warned:
                print("触觉录制：未找到 Sharpa 解码模块（SHARPA_HAPTIC_PY），本段无触觉流",
                      flush=True)
                self._tactile_warned = True
            return
        if self._tactile_queue is None:
            import queue
            # 每帧约 300KB；400 帧缓冲 ≈ 4 秒 backlog ≈ 120MB 上限
            self._tactile_queue = queue.Queue(maxsize=400)
        if self._tactile_writer is None:
            self._tactile_writer = threading.Thread(target=self._tactile_writer_loop,
                                                    daemon=True, name="episode-tactile-writer")
            self._tactile_writer.start()
        if self._tactile_thread is None:
            self._tactile_stop = threading.Event()
            self._tactile_thread = threading.Thread(target=self._tactile_loop,
                                                    daemon=True, name="episode-tactile")
            self._tactile_thread.start()

    def _light_key_fid(self, msg):
        """只解析 key（field 1.2）和 frame_id（field 8），不拷贝图像块。

        全量帧约 300KB，接收线程对抽帧丢弃的消息绝不做完整解码，
        否则解码+memcpy 本身就吃掉大半个 GIL。
        """
        fp = self._fields_parser
        if fp is None:
            return None, None
        key = fid = None
        try:
            for number, wire, value in fp(msg):
                if number == 1 and wire == 2:
                    for h_number, h_wire, h_value in fp(value):
                        if h_number == 2 and h_wire == 2:
                            key = h_value.decode()
                elif number == 8 and wire == 0:
                    fid = int(value)
        except Exception:
            return None, None
        return key, fid

    def _record_tactile_message(self, msg):
        if not self._recording:
            return  # 未录制时不入队，否则停止期间队列会积满旧帧
        if self._tactile_proc is not None:
            return  # 子进程录制中，进程内路径停用
        import os
        if not self._tactile_stride:
            self._tactile_stride = max(1, int(os.environ.get("SHARPA_TACTILE_STRIDE", "3")))
        key, fid = None, None
        if self._tactile_stride > 1:
            key, fid = self._light_key_fid(msg)
            if fid is not None and fid % self._tactile_stride != 0:
                return
        # 接收时间戳在入队前打，队列积压不污染帧对齐
        ts_ns = int(time.monotonic() * 1e9)
        try:
            self._tactile_queue.put_nowait((msg, ts_ns))
        except Exception:
            self._tactile_dropped += 1
            if self._tactile_dropped % 100 == 1:
                print(f"触觉录制：写盘队列已满，累计丢帧 {self._tactile_dropped}", flush=True)

    def _tactile_writer_loop(self):
        import queue as queue_mod
        try:
            while True:
                try:
                    msg, ts_ns = self._tactile_queue.get(timeout=0.5)
                except queue_mod.Empty:
                    if not self._recording:
                        return  # 停止且队列已空：退出，stop() 会 join 本线程
                    continue
                self._handle_tactile_item(msg, ts_ns)
        finally:
            self._tactile_writer = None

    def _handle_tactile_item(self, msg, ts_ns):
        try:
            frame = self._tactile_decoder(msg)
            key = frame["key"]
            side, finger_index = self._touch_key[key]
        except Exception:
            return
        # Pilot/桥消息里的 serial 不区分手（实测右手键也带左手 SN），
        # 手别以键映射为准，序列号取该侧实际设备
        import os
        serial = os.environ.get(f"SHARPA_{side.upper()}_SN") or "unknown"
        fields = {"force": np.asarray(frame["force"], dtype=np.float32),
                  "serial": serial, "side": side,
                  "finger": self._finger_names[finger_index]}
        if frame.get("torque") is not None:
            fields["torque"] = np.asarray(frame["torque"], dtype=np.float32)
        if frame.get("depth") is not None:
            fields["depth"] = frame["depth"]
        if frame.get("contact") is not None:
            fields["contact_point"] = frame["contact"]
        if frame.get("frame_id") is not None:
            fields["frame_id"] = np.int32(frame["frame_id"])
        if frame.get("dist_force") is not None:
            fields["dist_force"] = frame["dist_force"]
        if frame.get("raw") is not None:
            fields["raw"] = frame["raw"]
        self._append_tactile(f"sharpa_tactile/{key}", ts_ns, fields)

    def _append_tactile(self, stream, ts_ns, fields):
        with self._lock:
            if self._hdf5 is None:
                return
            known = self._hdf5.fields_of(stream)
            if known is not None:
                offered = set(fields) - {"force", "serial", "side", "finger"}
                if offered != (known - {"force", "serial", "side", "finger"}):
                    return  # 保持组内各数据集行对齐，丢弃字段结构不一致的帧
            try:
                self._hdf5.append(stream, ts_ns, fields)
            except (ValueError, TypeError):
                return  # 变长块不写入，保住整流
            self._tactile_seen.update(fields)
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
        if self._cmd_stop is not None:
            self._cmd_stop.set()
        for thread in (self._tactile_thread, self._joint_thread, self._cmd_thread):
            if thread is not None:
                thread.join(timeout=2)


EPISODES = EpisodeManager()
