from dataclasses import asdict, dataclass
from typing import Any, Literal

from unirobot_infra.core.types import Snapshot


MessageType = Literal["snapshot"]


@dataclass(frozen=True)
class RuntimeMessage:
    type: MessageType
    timestamp_ns: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_message(snapshot: Snapshot) -> RuntimeMessage:
    return RuntimeMessage(
        type="snapshot",
        timestamp_ns=snapshot.timestamp_ns,
        payload=snapshot.to_dict(),
    )
