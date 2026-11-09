"""独立进程触觉录制器：订阅 SDK 触觉桥（48008），写 episode 目录下的 tactile.h5。

为什么独立进程：全量触觉（DEFORM3+RAW+DIST_FORCE，每帧约 300KB，抽帧后约
100 帧/s）在本进程内写 HDF5 会和遥操主线程抢 GIL——实测独立进程轻松跑满
100 帧/s，塞进遥操进程只剩 35 帧/s 还伴随数秒级卡顿。time.monotonic() 是
系统级 CLOCK_MONOTONIC，跨进程时间戳天然对齐。

用法（由 episode.py 在录制开始时自动拉起，一般不需要手动跑）：
  python -m mujoco_marvin.tactile_recorder --dir <episode_dir> \
      [--address tcp://127.0.0.1:48008] [--stride 3]

退出：SIGINT/SIGTERM → 排空队列、关闭文件、stdout 打印一行
  TACTILE_STATS {"rows": N, "dropped": M, "streams": {...}}
供父进程写进 manifest。
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

def _load_haptic_module():
    import importlib.util
    candidates = [
        os.environ.get("SHARPA_HAPTIC_PY", ""),
        str(Path(os.environ.get("SHARPA_REPO", "")) / "teleop" / "standalone_haptic_feedback.py"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location("sharpa_haptic_decode", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("找不到 standalone_haptic_feedback.py（SHARPA_HAPTIC_PY/SHARPA_REPO）")


class TactileHdf5Writer:
    """与 _Hdf5StateWriter 相同的组布局：sharpa_tactile/<key>/<field>。

    批量写入：逐行 append 的 HDF5 调用开销（10 个数据集 × 每行一次）是全速率
    触觉的最大瓶颈（实测 9ms/帧）；攒 16 行一次写后降到 0.7ms/帧。
    """

    BATCH = 16

    def __init__(self, path):
        import h5py
        self._file = h5py.File(path, "w", track_order=True)
        self._string_dtype = h5py.string_dtype(encoding="utf-8")
        self._groups = {}
        self._buffers = {}  # stream -> [(sys_time_ns, fields), ...]

    def append(self, stream, sys_time_ns, fields):
        buf = self._buffers.setdefault(stream, [])
        buf.append((sys_time_ns, fields))
        if len(buf) >= self.BATCH:
            self._flush_stream(stream)

    def _flush_stream(self, stream):
        buf = self._buffers.pop(stream, [])  # 先取出清空，避免重复 flush 重写
        if not buf:
            return
        first = buf[0][1]
        # 字段结构不一致的帧丢弃，保住整流（缓冲路径下栈不上会直接抛错）
        buf = [(ts, f) for ts, f in buf if set(f) == set(first)]
        if not buf:
            return
        if stream not in self._groups:
            self._create_group(stream, first)
        group = self._groups[stream]
        n = len(buf)
        ds = group["sys_time_ns"]
        ds.resize(ds.shape[0] + n, axis=0)
        ds[-n:] = [ts for ts, _ in buf]
        for name in first:
            ds = group[name]
            values = [fields[name] for _, fields in buf]
            ds.resize(ds.shape[0] + n, axis=0)
            if isinstance(values[0], str):
                ds[-n:] = values
            else:
                ds[-n:] = np.stack([np.asarray(v) for v in values])

    def flush(self):
        for stream in list(self._buffers):
            self._flush_stream(stream)

    def fields_of(self, stream):
        if stream in self._buffers and stream not in self._groups:
            return None  # 还在缓冲里，组未创建
        group = self._groups.get(stream)
        return None if group is None else set(group.keys()) - {"sys_time_ns"}

    def _create_group(self, stream, fields):
        group = self._file.create_group(f"sharpa_tactile/{stream}", track_order=True)
        group.attrs["field_order"] = np.asarray(["sys_time_ns", *fields.keys()],
                                                dtype=self._string_dtype)
        group.create_dataset("sys_time_ns", shape=(0,), maxshape=(None,),
                             dtype=np.int64, chunks=(1024,),
                             compression="gzip", compression_opts=4)
        for name, value in fields.items():
            self._create_dataset(group, name, value)
        self._groups[stream] = group

    def _create_dataset(self, group, name, value):
        if isinstance(value, str):
            group.create_dataset(name, shape=(0,), maxshape=(None,),
                                 dtype=self._string_dtype, chunks=(256,))
            return
        arr = np.asarray(value)
        shape, dtype = arr.shape, arr.dtype
        big = arr.size >= 4096
        chunk = (16, *shape) if big else (1024, *shape)
        # 大块（图像/力场）用 lzf：比 gzip 快约 4 倍、文件仅大 ~8%，
        # 全速率触觉（313 帧/s ≈ 94MB/s）gzip 顶不住
        group.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
                             dtype=dtype, chunks=chunk,
                             compression="lzf" if big else "gzip",
                             compression_opts=None if big else 4)

    def close(self):
        self.flush()
        self._file.close()


class TactileRecorder:
    def __init__(self, episode_dir, address, stride):
        self._dir = Path(episode_dir)
        self._address = address
        self._stride = max(1, stride)
        self._stop = threading.Event()
        self._rows = 0
        self._dropped = 0
        self._streams = {}
        self._hdf5_lock = threading.Lock()  # 两个写盘线程共享一个文件；
        # h5py 在 H5Dwrite 的 C 层会释放 GIL，压缩实际仍可并行
        module = _load_haptic_module()
        self._decode = getattr(module, "decode_tactile_full", None) or (
            lambda data: dict(zip(("key", "serial", "force"), module.decode_tactile(data)),
                              torque=None, depth=None, contact=None, frame_id=None,
                              dist_force=None, raw=None))
        self._touch_key = module.TOUCH_KEY
        self._finger_names = module.FINGERS
        self._fields_parser = getattr(module, "_fields", None)
        import queue
        # 全速率（stride=1）时 313 msg/s ≈ 94MB/s：左右手各一条队列+写盘线程，
        # 1200 帧缓冲 ≈ 4 秒 backlog ≈ 360MB/手上限
        self._queues = {"left": queue.Queue(maxsize=1200),
                        "right": queue.Queue(maxsize=1200)}
        self._writer = None

    # ---------- 接收线程：轻量解析 + 抽帧 + 分手入队 ----------
    def _light_key_fid(self, msg):
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

    def _recv_loop(self):
        import zmq
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self._address)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.RCVTIMEO, 500)
        while not self._stop.is_set():
            try:
                msg = sub.recv()
            except zmq.Again:
                continue
            key, fid = self._light_key_fid(msg)
            if self._stride > 1 and fid is not None and fid % self._stride != 0:
                continue
            ts_ns = int(time.monotonic() * 1e9)  # 入队前打戳，积压不污染对齐
            side = "left" if key and key.startswith("1001") else "right"
            try:
                self._queues[side].put_nowait((msg, ts_ns))
            except Exception:
                self._dropped += 1
        sub.close()
        ctx.term()

    # ---------- 写盘线程（左右手各一）：完整解码 + HDF5 append ----------
    def _writer_loop(self, hdf5, side):
        import queue as queue_mod
        q = self._queues[side]
        while True:
            try:
                msg, ts_ns = q.get(timeout=0.5)
            except queue_mod.Empty:
                with self._hdf5_lock:
                    hdf5.flush()  # 空闲时把不足一批的尾帧落盘
                if self._stop.is_set():
                    return
                continue
            try:
                frame = self._decode(msg)
                key = frame["key"]
                side, finger_index = self._touch_key[key]
            except Exception:
                continue
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
            known_key = key
            with self._hdf5_lock:
                known = hdf5.fields_of(known_key)
                if known is not None:
                    offered = set(fields) - {"force", "serial", "side", "finger"}
                    if offered != (known - {"force", "serial", "side", "finger"}):
                        continue  # 保持组内行对齐
                try:
                    hdf5.append(known_key, ts_ns, fields)
                except (ValueError, TypeError):
                    continue  # 变长块不写入，保住整流
                self._streams[known_key] = sorted(fields)
                self._rows += 1

    def run(self):
        hdf5 = TactileHdf5Writer(self._dir / "tactile.h5")
        writers = [threading.Thread(target=self._writer_loop, args=(hdf5, side),
                                    daemon=True, name=f"tactile-writer-{side}")
                   for side in ("left", "right")]
        for w in writers:
            w.start()
        self._recv_loop()  # 阻塞直到 stop
        for w in writers:
            w.join(timeout=30)  # 排空 backlog
        hdf5.close()
        stats = {"rows": self._rows, "dropped": self._dropped,
                 "streams": self._streams}
        print(f"TACTILE_STATS {json.dumps(stats)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="episode 目录（tactile.h5 写在这里）")
    parser.add_argument("--address", default="tcp://127.0.0.1:48008")
    parser.add_argument("--stride", type=int,
                        default=int(os.environ.get("SHARPA_TACTILE_STRIDE", "3")))
    args = parser.parse_args()
    recorder = TactileRecorder(args.dir, args.address, args.stride)

    def _term(*_):
        recorder._stop.set()
    signal.signal(signal.SIGINT, _term)
    signal.signal(signal.SIGTERM, _term)
    print(f"[tactile-recorder] {args.address} -> {args.dir}/tactile.h5 "
          f"(stride={args.stride})", flush=True)
    recorder.run()


if __name__ == "__main__":
    main()
