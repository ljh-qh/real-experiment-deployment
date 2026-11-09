from typing import Any

import msgpack
import numpy as np


def encode_message(message: dict[str, Any]) -> list[bytes]:
    items: list[dict[str, Any]] = []
    frames: list[bytes] = []
    inline: dict[str, Any] = {}
    for path, value in _flatten_leaves(message):
        if isinstance(value, np.ndarray):
            arr = _contiguous_array(value)
            items.append(
                {
                    "path": path,
                    "kind": "ndarray",
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                }
            )
            frames.append(arr.tobytes())
        elif isinstance(value, str):
            items.append({"path": path, "kind": "str"})
            frames.append(value.encode("utf-8"))
        else:
            _assign_path(inline, path, value)
    header = {
        "format": "unirobot_policy_raw",
        "items": items,
        "inline": inline,
    }
    return [msgpack.packb(header, use_bin_type=True)] + frames


def decode_message(frames: list[bytes]) -> dict[str, Any]:
    if not frames:
        raise ValueError("empty ZMQ policy message")
    header = msgpack.unpackb(frames[0], raw=False)
    if not isinstance(header, dict) or header["format"] != "unirobot_policy_raw":
        raise ValueError("invalid ZMQ policy message header")
    items = header["items"]
    inline = header["inline"]
    message: dict[str, Any] = dict(inline)
    if len(frames) - 1 != len(items):
        raise ValueError("ZMQ policy message frame count does not match header")
    for index, item in enumerate(items):
        path = item["path"]
        kind = item["kind"]
        if not isinstance(path, list):
            raise TypeError("ZMQ policy message item path must be a list")
        data = frames[index + 1]
        if kind == "ndarray":
            dtype = np.dtype(item["dtype"])
            shape = tuple(int(value) for value in item["shape"])
            array = np.frombuffer(data, dtype=dtype)
            if shape:
                array = array.reshape(shape)
            _assign_path(message, path, array)
        elif kind == "str":
            _assign_path(message, path, data.decode("utf-8"))
        else:
            raise ValueError(f"unsupported ZMQ policy item kind: {kind}")
    return message


def _contiguous_array(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.uint8:
        return np.ascontiguousarray(value, dtype=np.uint8)
    return np.ascontiguousarray(value, dtype=np.float32)


def _flatten_leaves(data: dict[str, Any], prefix: list[str] | None = None) -> list[tuple[list[str], Any]]:
    if prefix is None:
        prefix = []
    leaves: list[tuple[list[str], Any]] = []
    for key, value in data.items():
        path_key = str(key)
        if isinstance(value, dict):
            leaves.extend(_flatten_leaves(value, prefix + [path_key]))
        else:
            leaves.append((prefix + [path_key], value))
    return leaves


def _assign_path(data: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        if key not in cursor:
            cursor[key] = {}
        if not isinstance(cursor[key], dict):
            raise TypeError(f"ZMQ policy message path conflicts at {key}")
        cursor = cursor[key]
    cursor[path[-1]] = value
