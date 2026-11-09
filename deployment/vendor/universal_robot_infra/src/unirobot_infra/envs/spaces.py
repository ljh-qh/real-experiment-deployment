from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class BoxSpace:
    low: tuple[float, ...]
    high: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.low),)

    def contains(self, values: Sequence[float]) -> bool:
        if len(values) != len(self.low):
            return False
        return all(
            self.low[index] <= float(values[index]) <= self.high[index]
            for index in range(len(self.low))
        )

    def clip(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != len(self.low):
            raise ValueError("BoxSpace value length does not match shape")
        return tuple(
            min(max(float(values[index]), self.low[index]), self.high[index])
            for index in range(len(self.low))
        )
