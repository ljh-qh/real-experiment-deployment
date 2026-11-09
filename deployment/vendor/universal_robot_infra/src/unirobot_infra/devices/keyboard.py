from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame


@dataclass(frozen=True)
class KeyboardFrame:
    timestamp_ns: int
    active_commands: tuple[str, ...]

    def to_device_frame(self, side: str) -> DeviceFrame:
        return DeviceFrame(
            device_name="keyboard",
            side=side,
            timestamp_ns=self.timestamp_ns,
            scalars={command: 1.0 for command in self.active_commands},
        )


@dataclass
class KeyboardDevice:
    cfg: DictConfig
    connected: bool = False
    _lock: Lock = field(default_factory=Lock, init=False)
    _pressed_commands: set[str] = field(default_factory=set, init=False)
    _key_to_command: dict[str, str] = field(default_factory=dict, init=False)
    _listener: Any = field(default=None, init=False)

    def connect(self) -> None:
        from pynput import keyboard

        self._key_to_command = _key_to_command(self.cfg.key_bindings)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self.connected = True

    def read_frame(self) -> KeyboardFrame:
        if not self.connected:
            raise RuntimeError("KeyboardDevice is not connected")
        with self._lock:
            active_commands = tuple(sorted(self._pressed_commands))
        return KeyboardFrame(timestamp_ns=now_ns(), active_commands=active_commands)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        self._listener = None
        with self._lock:
            self._pressed_commands.clear()
        self.connected = False

    def _on_press(self, key: Any) -> None:
        key_name = _key_name(key)
        if key_name in self._key_to_command:
            with self._lock:
                self._pressed_commands.add(self._key_to_command[key_name])

    def _on_release(self, key: Any) -> None:
        key_name = _key_name(key)
        if key_name in self._key_to_command:
            with self._lock:
                self._pressed_commands.discard(self._key_to_command[key_name])


def _key_to_command(key_bindings: DictConfig) -> dict[str, str]:
    output = {}
    for command in key_bindings:
        output[_normalize_key_name(str(key_bindings[command]))] = str(command)
    return output


def _key_name(key: Any) -> str:
    if hasattr(key, "char") and key.char is not None:
        return _normalize_key_name(str(key.char))
    return _normalize_key_name(str(key))


def _normalize_key_name(value: str) -> str:
    key_name = value.strip().lower()
    if key_name.startswith("key."):
        return key_name[4:]
    return key_name
