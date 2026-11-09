from collections.abc import Callable, Sequence

import customtkinter as ctk
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


class ImageDisplayWrapper(EnvWrapper):
    """把 env 关联的相机画面拼成横排，customtkinter 显示在本地窗口。

    这是 env 的普通 wrapper，挂在 `EnvWrapper` 链上。frame source 由
    `cfg.frame_source_mode` 显式选择：

    - `injected`：app 层把"取最新一帧"的 callable 传进来。
      用于相机所有权在 app 层（如 `DataRecorder.latest_camera_frames`）的场景。
    - `info`：直接读取 `info["camera_frames"]`。env 链里有 `CameraObservationWrapper` 等
      往 info 写 frames 的 wrapper 时走这条。用于相机所有权在 env 链内的场景。

    source 存在但当前没有 frame 时 wrapper 是 no-op。

    显示走主进程 customtkinter：第一次 `_show` 时若 `camera_names` 已经传进来，
    立即用占位 label 懒创建 `CTk` 根窗口（避免在主循环里临时建窗导致
    `_default_root` 状态被 StatusGui 等其它 GUI 干扰）；后续每步
    `configure(image=...)` 替换图，并调 `update_idletasks()` 让 X 按钮等事件
    能响应。窗口 X 按钮关闭后 wrapper 退化为 no-op（不再重建），但 env 继续
    运行；`env.close()` 同步销毁窗口。
    """

    def __init__(
        self,
        env: EnvLike,
        cfg: DictConfig,
        frame_source: Callable[[], dict[str, np.ndarray]] | None = None,
        camera_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(env)
        self.cfg = cfg
        self.frame_source = frame_source
        self._frame_source_mode = str(cfg.frame_source_mode) if bool(cfg.enabled) else ""
        self._camera_names: tuple[str, ...] = tuple(camera_names) if camera_names else ()
        self._window: ctk.CTk | None = None
        self._image_labels: dict[str, ctk.CTkLabel] = {}
        self._user_closed = False
        self._closed = False
        self._geometry_locked = False
        self._validate_cfg()

    def set_frame_source(self, frame_source: Callable[[], dict[str, np.ndarray]]) -> None:
        if self._frame_source_mode != "injected":
            raise ValueError("image_display.frame_source_mode must be injected to set frame_source")
        self.frame_source = frame_source

    def reset(self) -> tuple[tuple[float, ...], dict]:
        obs, info = self.env.reset()
        self._show(info)
        return obs, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._show(info)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        window = self._window
        self._window = None
        self._image_labels = {}
        try:
            if window is not None:
                window.destroy()
        finally:
            self.env.close()

    def _show(self, info: dict) -> None:
        if not bool(self.cfg.enabled):
            return
        if self._user_closed:
            return
        if self._frame_source_mode == "injected":
            if self.frame_source is None:
                raise RuntimeError("image_display.frame_source is required when frame_source_mode=injected")
            images = self.frame_source()
        elif self._frame_source_mode == "info":
            images = info["camera_frames"]
        else:
            raise RuntimeError(f"unsupported image_display.frame_source_mode: {self._frame_source_mode}")

        # 若 caller 给了 camera_names 但此时还没拿到帧，先用占位把窗口建出来。
        if not images and self._camera_names and self._window is None:
            self._ensure_window(tuple(self._camera_names))
            return

        if not images:
            return

        self._ensure_window(tuple(images.keys()))
        self._update_window(images)

    def _ensure_window(self, names: tuple[str, ...]) -> None:
        if self._window is not None:
            return
        if not names:
            return
        panel_width = int(self.cfg.panel_width)
        panel_height = max(1, int(round(panel_width * 0.75)))
        total_width = panel_width * len(names) + 16 * (len(names) + 1)
        total_height = panel_height + 64

        window = ctk.CTk()
        window.title(str(self.cfg.window_name))
        window.geometry(_window_geometry(self.cfg, total_width, total_height))
        window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self._window = window

        font = ctk.CTkFont(size=12, weight="bold")
        for name in names:
            frame = ctk.CTkFrame(window)
            frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)
            title = ctk.CTkLabel(frame, text=name, font=font)
            title.pack(side="top", fill="x", padx=4, pady=(4, 2))
            # 占位 label 不带 image，避免在窗口刚建时触发 CTkImage 内部
            # ImageTk.PhotoImage 创建时的 "NULL main window" 风险。
            image_label = ctk.CTkLabel(frame, text="waiting for camera...")
            image_label.pack(side="top", fill="both", expand=True, padx=4, pady=(0, 4))
            self._image_labels[name] = image_label

    def _update_window(self, images: dict[str, np.ndarray]) -> None:
        panel_width = int(self.cfg.panel_width)
        first_rgb = np.asarray(next(iter(images.values())))
        aspect = first_rgb.shape[0] / float(first_rgb.shape[1])
        panel_height = max(1, int(round(panel_width * aspect)))

        for name, rgb in images.items():
            label = self._image_labels[name]
            pil_image = Image.fromarray(np.asarray(rgb)).resize(
                (panel_width, panel_height), Image.BILINEAR
            )
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(panel_width, panel_height))
            label.configure(image=ctk_image)
            label.configure(text="")

        if self._window is not None:
            if not self._geometry_locked:
                total_width = panel_width * len(images) + 16 * (len(images) + 1)
                total_height = panel_height + 64
                self._window.geometry(_window_geometry(self.cfg, total_width, total_height))
                self._geometry_locked = True
            self._window.update_idletasks()

    def _on_window_close(self) -> None:
        self._user_closed = True
        window = self._window
        self._window = None
        self._image_labels = {}
        if window is not None:
            window.destroy()

    def _validate_cfg(self) -> None:
        if not bool(self.cfg.enabled):
            return
        if int(self.cfg.panel_width) <= 0:
            raise ValueError("image_display.panel_width must be positive")
        if not str(self.cfg.window_name):
            raise ValueError("image_display.window_name must not be empty")
        if self._frame_source_mode not in ("injected", "info"):
            raise ValueError("image_display.frame_source_mode must be injected or info")
        if self._frame_source_mode == "injected" and self.frame_source is None:
            raise ValueError("image_display.frame_source is required when frame_source_mode=injected")
        if self._frame_source_mode == "info" and self.frame_source is not None:
            raise ValueError("image_display.frame_source must be omitted when frame_source_mode=info")


def _window_geometry(cfg: DictConfig, width: int, height: int) -> str:
    position = cfg.position
    return f"{int(width)}x{int(height)}+{int(position.x)}+{int(position.y)}"
