from contextlib import nullcontext
import ctypes
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import threading
from types import TracebackType
from typing import TextIO

from omegaconf import DictConfig


STDOUT_FD = 1
STDERR_FD = 2


@dataclass(frozen=True)
class SdkOutputRule:
    name: str
    pattern: re.Pattern[str]
    file: Path
    tee_terminal: bool

    def matches(self, line: str) -> bool:
        return self.pattern.search(line) is not None


@dataclass
class SdkOutputRouter:
    root_dir: Path
    run_name: str
    rules: tuple[SdkOutputRule, ...]
    tee_unmatched: bool
    raw_log_enabled: bool
    pipe_thread_join_timeout_s: float = 0.5
    run_dir: Path = field(init=False)
    _original_stdout_fd: int | None = field(default=None, init=False)
    _original_stderr_fd: int | None = field(default=None, init=False)
    _threads: list[threading.Thread] = field(default_factory=list, init=False)
    _rule_files: dict[str, TextIO] = field(default_factory=dict, init=False)
    _raw_file: TextIO | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _started: bool = field(default=False, init=False)
    _dropping_output: bool = field(default=False, init=False)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SdkOutputRouter":
        rules = []
        rule_files = set()
        for rule_cfg in cfg.rules:
            file_path = Path(str(rule_cfg.file))
            if file_path.is_absolute() or len(file_path.parts) != 1:
                raise ValueError("app.sdk_output.rules[].file must be a relative file name")
            if file_path in rule_files:
                raise ValueError(f"duplicate app.sdk_output rule file: {file_path}")
            rule_files.add(file_path)
            rules.append(
                SdkOutputRule(
                    name=str(rule_cfg.name),
                    pattern=re.compile(str(rule_cfg.pattern)),
                    file=file_path,
                    tee_terminal=bool(rule_cfg.tee_terminal),
                )
            )
        return cls(
            root_dir=Path(str(cfg.root_dir)),
            run_name=str(cfg.run_name),
            rules=tuple(rules),
            tee_unmatched=bool(cfg.tee_unmatched),
            raw_log_enabled=bool(cfg.raw_log_enabled),
        )

    def __enter__(self) -> "SdkOutputRouter":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("SdkOutputRouter is already started")

        self.run_dir = self.root_dir / _run_dir_name(self.run_name)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for rule in self.rules:
            self._rule_files[rule.name] = (self.run_dir / rule.file).open("w", encoding="utf-8", buffering=1)
        if self.raw_log_enabled:
            self._raw_file = (self.run_dir / "raw.log").open("w", encoding="utf-8", buffering=1)

        _flush_c_stdio()
        sys.stdout.flush()
        sys.stderr.flush()
        self._original_stdout_fd = os.dup(STDOUT_FD)
        self._original_stderr_fd = os.dup(STDERR_FD)

        stdout_read_fd, stdout_write_fd = os.pipe()
        stderr_read_fd, stderr_write_fd = os.pipe()
        os.dup2(stdout_write_fd, STDOUT_FD)
        os.dup2(stderr_write_fd, STDERR_FD)
        os.close(stdout_write_fd)
        os.close(stderr_write_fd)

        self._threads = [
            threading.Thread(target=self._read_pipe, args=(stdout_read_fd, "stdout"), daemon=True),
            threading.Thread(target=self._read_pipe, args=(stderr_read_fd, "stderr"), daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._dropping_output = False
        self._started = True

    def close(self) -> None:
        if not self._started:
            return

        _flush_c_stdio()
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._original_stdout_fd, STDOUT_FD)
        os.dup2(self._original_stderr_fd, STDERR_FD)

        for thread in self._threads:
            thread.join(timeout=self.pipe_thread_join_timeout_s)
        if any(thread.is_alive() for thread in self._threads):
            with self._lock:
                self._dropping_output = True
        self._threads = []

        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
        self._original_stdout_fd = None
        self._original_stderr_fd = None

        for file in self._rule_files.values():
            file.flush()
            file.close()
        self._rule_files = {}
        if self._raw_file is not None:
            self._raw_file.flush()
            self._raw_file.close()
            self._raw_file = None
        self._started = False

    def _read_pipe(self, read_fd: int, fd_name: str) -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            for raw_line in pipe:
                self._route_line(fd_name, raw_line)

    def _route_line(self, fd_name: str, raw_line: bytes) -> None:
        line = raw_line.decode("utf-8")
        matched_rule = _match_rule(self.rules, line)
        with self._lock:
            if self._dropping_output:
                return
            if self._raw_file is not None:
                self._raw_file.write(_format_log_line(fd_name, "raw", line))
            if matched_rule is not None:
                self._rule_files[matched_rule.name].write(_format_log_line(fd_name, matched_rule.name, line))
                if matched_rule.tee_terminal:
                    self._write_original(fd_name, raw_line)
                return
            if self.tee_unmatched:
                self._write_original(fd_name, raw_line)

    def _write_original(self, fd_name: str, raw_line: bytes) -> None:
        if fd_name == "stdout":
            os.write(self._original_stdout_fd, raw_line)
            return
        if fd_name == "stderr":
            os.write(self._original_stderr_fd, raw_line)
            return
        raise ValueError("fd_name must be stdout or stderr")


def start_sdk_output_router_from_config(cfg: DictConfig):
    if not bool(cfg.enabled):
        return nullcontext()
    return SdkOutputRouter.from_config(cfg)


def _match_rule(rules: tuple[SdkOutputRule, ...], line: str) -> SdkOutputRule | None:
    for rule in rules:
        if rule.matches(line):
            return rule
    return None


def _format_log_line(fd_name: str, rule_name: str, line: str) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="microseconds")
    output = f"{timestamp} fd={fd_name} rule={rule_name} {line}"
    if output.endswith("\n"):
        return output
    return f"{output}\n"


def _run_dir_name(run_name: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{run_name}"


def _flush_c_stdio() -> None:
    libc = ctypes.CDLL(None)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int
    libc.fflush(None)
