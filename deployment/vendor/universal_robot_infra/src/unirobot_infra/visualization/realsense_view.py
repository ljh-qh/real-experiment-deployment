"""Tk live view for the RealSense wrist cameras.

``_CameraViewManager`` renders every registered camera (``WristCamera``) in
one window with a single record-all toggle. Tk runs entirely on the MAIN
thread: ``update()`` + ``sync()`` are pumped from the teleop viewer/headless
loop, so there is no GUI thread and no cross-thread Tcl deletion on shutdown.

``toggle_cameras`` is the record-all entry point: with the unified episode
manager configured it starts/stops a whole episode, otherwise it toggles each
camera's own recorder (legacy per-camera takes).
"""

import threading


class _CameraViewManager:
    """Single window with all camera views and ONE record toggle.

    Tk runs entirely on the MAIN thread: `update()` + `sync()` are pumped
    from the teleop's viewer/headless loop. No GUI thread -> no cross-thread
    Tcl deletion on shutdown. `shutdown()` clears StringVar/PhotoImage refs
    and destroys the root on the same thread that created it."""

    def __init__(self):
        self._cameras = []
        self._root = None
        self._views = {}
        self._state_text = None
        self._lock = threading.Lock()
        self._closed = False

    def register(self, camera):
        with self._lock:
            self._cameras.append(camera)

    def toggle_all(self):
        with self._lock:
            cams = list(self._cameras)
        toggle_cameras(cams)

    def _ensure_root(self):
        if self._root is not None or self._closed:
            return
        import tkinter as tk
        from PIL import ImageTk
        self._tk = tk
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title("相机视角（主视角 + 双腕）——关窗只关显示")
        self._root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self._grid = tk.Frame(self._root)
        self._grid.pack()
        self._state_text = tk.StringVar()
        tk.Button(self._root, textvariable=self._state_text,
                  font=("Sans", 14, "bold"),
                  command=self.toggle_all).pack(fill="x", padx=10, pady=8)

    def update(self):
        """Pump pending Tk events once (main thread's loop)."""
        if self._closed:
            return
        if self._root is None and self._cameras:
            self._ensure_root()
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:
            self.shutdown()

    def sync(self):
        """Create/refresh camera images (main thread's loop)."""
        if self._closed:
            return
        if self._root is None and self._cameras:
            self._ensure_root()
        if self._root is None:
            return
        try:
            self._sync_views()
        except Exception:
            # 窗口被外部销毁等情况：只停显示，不影响遥操/录制主循环
            self.shutdown()

    def _sync_views(self):
        with self._lock:
            cams = list(self._cameras)
        for index, cam in enumerate(cams):
            if cam not in self._views:
                column = self._tk.Frame(self._grid)
                column.grid(row=0, column=index, padx=4, pady=4)
                title = self._tk.Label(column, text=f"{cam.role} [{cam.serial[-4:] if cam.serial else ''}]",
                                       font=("Sans", 11, "bold"))
                title.pack()
                label = self._tk.Label(column)
                label.pack()
                self._views[cam] = {"title": title, "label": label,
                                    "photo": self._tk.PhotoImage(width=1, height=1)}
        for cam, view in self._views.items():
            item = cam.latest()
            if item is not None:
                # 主线程架构下按帧新建 PhotoImage 也安全（Tcl 同线程 GC）
                view["photo"] = self._ImageTk.PhotoImage(item[0])
                view["label"].configure(image=view["photo"])
            state = " — 无新帧，正在重连" if cam.frame_age() > 2. else ""
            view["title"].configure(
                text=f"{cam.role} [{cam.serial[-4:] if cam.serial else ''}]{state}")
        any_recording = any(cam.recording for cam in cams)
        self._state_text.set("■ 停止录制（全部机位，每段一个文件）" if any_recording
                             else "● 开始录制（全部机位）")

    def shutdown(self):
        """Clear Tk object refs, then destroy the root — all on the caller's thread."""
        self._closed = True
        if self._root is None:
            return
        for view in self._views.values():
            view["photo"] = None
        self._views.clear()
        self._state_text = None
        try:
            self._root.destroy()
        except Exception:
            pass
        self._root = None


def toggle_cameras(cameras):
    """Start/stop recording: the unified episode manager when configured,
    otherwise each camera's own recorder (legacy per-camera takes)."""
    from unirobot_infra.recorder.pico_episode import EPISODES
    if EPISODES.active:
        EPISODES.toggle()
        return
    target = not any(cam.recording for cam in cameras)
    for cam in cameras:
        cam.set_recording(target)


_VIEW = _CameraViewManager()
