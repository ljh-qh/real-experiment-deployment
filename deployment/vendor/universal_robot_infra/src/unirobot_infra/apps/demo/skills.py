from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import time
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class SkillTask(Generic[T]):
    name: str
    run: Callable[[], T]


def skill_task(name: str, run: Callable[[], T]) -> SkillTask[T]:
    return SkillTask(name=name, run=run)


class Together:
    def __init__(self, label: str, logger, stop_event: threading.Event | None) -> None:
        self.label = label
        self.logger = logger
        self.stop_event = stop_event

    def run(self, tasks: Sequence[SkillTask[T]]) -> dict[str, T]:
        if not tasks:
            self.logger.info("demo together skipped label=%s", self.label)
            return {}
        task_names = tuple(task.name for task in tasks)
        if len(set(task_names)) != len(task_names):
            raise ValueError(f"duplicate task names in together group label={self.label}: {task_names}")
        self.logger.info("demo together start label=%s tasks=%s", self.label, ",".join(task_names))
        started_s = time.monotonic()
        results: dict[str, T] = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_by_name = {executor.submit(task.run): task.name for task in tasks}
            for future in as_completed(future_by_name):
                task_name = future_by_name[future]
                try:
                    results[task_name] = future.result()
                except Exception:
                    if self.stop_event is not None:
                        self.stop_event.set()
                    self.logger.exception("demo together failed label=%s task=%s", self.label, task_name)
                    raise
                self.logger.info(
                    "demo together completed label=%s task=%s elapsed_s=%.3f",
                    self.label,
                    task_name,
                    time.monotonic() - started_s,
                )
        self.logger.info("demo together done label=%s elapsed_s=%.3f", self.label, time.monotonic() - started_s)
        return {task.name: results[task.name] for task in tasks}
