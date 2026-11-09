from dataclasses import dataclass, field
import queue
import threading
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True)
class StatusSnapshot:
    robot: str
    source: str
    mapping: str
    rate: str
    frame: str
    teleop: str
    action: str
    usage: str


@dataclass
class StatusGui:
    enabled: bool
    title: str
    width: int
    height: int
    poll_interval_ms: int
    appearance_mode: str
    color_theme: str
    _queue: queue.SimpleQueue[StatusSnapshot] = field(default_factory=queue.SimpleQueue, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _ready_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _startup_error: Exception | None = field(default=None, init=False)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "StatusGui":
        return cls(
            enabled=bool(cfg.enabled),
            title=str(cfg.title),
            width=int(cfg.width),
            height=int(cfg.height),
            poll_interval_ms=int(cfg.poll_interval_ms),
            appearance_mode=str(cfg.appearance_mode),
            color_theme=str(cfg.color_theme),
        )

    def __enter__(self) -> "StatusGui":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, name="StatusGui", daemon=True)
        self._thread.start()
        self._ready_event.wait()
        if self._startup_error is not None:
            raise self._startup_error
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        return None

    def update(self, snapshot: StatusSnapshot) -> None:
        if not self.enabled:
            return
        self._queue.put(snapshot)

    def _run(self) -> None:
        try:
            customtkinter = _load_customtkinter()
            customtkinter.set_appearance_mode(self.appearance_mode)
            customtkinter.set_default_color_theme(self.color_theme)
            root = customtkinter.CTk()
            root.title(self.title)
            root.geometry(f"{self.width}x{self.height}")
            root.minsize(self.width, self.height)
            widgets = _StatusGuiWidgets(customtkinter, root)
            self._ready_event.set()
            root.after(self.poll_interval_ms, lambda: self._poll(root, widgets))
            root.mainloop()
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()

    def _poll(self, root, widgets: "_StatusGuiWidgets") -> None:
        if self._stop_event.is_set():
            root.destroy()
            return
        latest = None
        while not self._queue.empty():
            latest = self._queue.get()
        if latest is not None:
            widgets.update(latest)
        root.after(self.poll_interval_ms, lambda: self._poll(root, widgets))


class _StatusGuiWidgets:
    def __init__(self, customtkinter, root) -> None:
        self.root = root
        self.labels: dict[str, Any] = {}
        self.teleop_label = None
        self.action_label = None
        self.usage_label = None

        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        outer = customtkinter.CTkFrame(root, corner_radius=0)
        outer.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        outer.grid_columnconfigure(1, weight=1)

        title = customtkinter.CTkLabel(
            outer,
            text="unirobot_infra Teleop",
            font=customtkinter.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(16, 8))

        row = 1
        for key, label in (
            ("robot", "robot"),
            ("source", "source"),
            ("mapping", "mapping"),
            ("rate", "rate"),
            ("frame", "frame"),
        ):
            self._add_row(customtkinter, outer, row, key, label)
            row += 1

        teleop_name = customtkinter.CTkLabel(outer, text="teleop", anchor="w", width=92)
        teleop_name.grid(row=row, column=0, sticky="w", padx=(18, 8), pady=4)
        self.teleop_label = customtkinter.CTkLabel(outer, text="", anchor="w")
        self.teleop_label.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=4)
        row += 1

        action_name = customtkinter.CTkLabel(outer, text="action", anchor="w", width=92)
        action_name.grid(row=row, column=0, sticky="w", padx=(18, 8), pady=4)
        self.action_label = customtkinter.CTkLabel(outer, text="", anchor="w")
        self.action_label.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=4)
        row += 1

        usage_title = customtkinter.CTkLabel(
            outer,
            text="使用说明",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        usage_title.grid(row=row, column=0, columnspan=2, sticky="ew", padx=18, pady=(16, 4))
        row += 1

        self.usage_label = customtkinter.CTkLabel(outer, text="", justify="left", anchor="nw", wraplength=460)
        self.usage_label.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=18, pady=(0, 16))
        outer.grid_rowconfigure(row, weight=1)

    def update(self, snapshot: StatusSnapshot) -> None:
        self.labels["robot"].configure(text=snapshot.robot)
        self.labels["source"].configure(text=snapshot.source)
        self.labels["mapping"].configure(text=snapshot.mapping)
        self.labels["rate"].configure(text=snapshot.rate)
        self.labels["frame"].configure(text=snapshot.frame)
        self.teleop_label.configure(text=snapshot.teleop, text_color=_teleop_color(snapshot.teleop))
        self.action_label.configure(text=snapshot.action)
        self.usage_label.configure(text=snapshot.usage, wraplength=max(260, self.root.winfo_width() - 44))

    def _add_row(self, customtkinter, outer, row: int, key: str, label: str) -> None:
        name = customtkinter.CTkLabel(outer, text=label, anchor="w", width=92)
        name.grid(row=row, column=0, sticky="w", padx=(18, 8), pady=4)
        value = customtkinter.CTkLabel(outer, text="", anchor="w")
        value.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=4)
        self.labels[key] = value


def _teleop_color(value: str) -> str:
    if value == "active":
        return "#2da44e"
    return "#8c959f"


def _load_customtkinter():
    import customtkinter

    return customtkinter
