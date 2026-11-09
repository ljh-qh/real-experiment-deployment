from collections.abc import Sequence

import numpy as np


class ActionChunkBuffer:
    def __init__(self, action_dim: int, execute_steps: int) -> None:
        if action_dim <= 0:
            raise ValueError("policy action_dim must be positive")
        if execute_steps <= 0:
            raise ValueError("policy execute_steps must be positive")
        self.action_dim = int(action_dim)
        self.execute_steps = int(execute_steps)
        self._chunk: tuple[tuple[float, ...], ...] = ()
        self._cursor = 0

    def replace(self, action: Sequence[float] | np.ndarray) -> None:
        array = np.asarray(action, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("policy action must have shape [action_dim] or [horizon, action_dim]")
        if array.shape[1] != self.action_dim:
            raise ValueError("policy action_dim does not match env action layout")
        limit = min(int(array.shape[0]), self.execute_steps)
        self._chunk = tuple(
            tuple(float(value) for value in array[index])
            for index in range(limit)
        )
        self._cursor = 0

    def pop_next(self) -> tuple[float, ...] | None:
        if self._cursor >= len(self._chunk):
            return None
        action = self._chunk[self._cursor]
        self._cursor += 1
        return action

    def clear(self) -> None:
        self._chunk = ()
        self._cursor = 0

    @property
    def remaining(self) -> int:
        return max(0, len(self._chunk) - self._cursor)
