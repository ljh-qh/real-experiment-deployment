from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
import time
from typing import Any, Protocol

import msgpack
import numpy as np
from omegaconf import DictConfig

from unirobot_infra.utils.logging import get_logger


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class PolicyClient(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class OpenPiWebsocketPolicyClient:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger(__name__)
        self.ws = _MinimalWebsocket(
            host=str(cfg.host),
            port=int(cfg.port),
            timeout_s=float(cfg.connect_timeout_s),
            path=str(cfg.path),
        )
        self.metadata = self._connect()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.ws.send_binary(_packb(observation))
        response = self.ws.recv_message()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server returned an error:\n{response}")
        result = _unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"OpenPI server returned non-dict response: {type(result)}")
        return result

    def close(self) -> None:
        self.ws.close()

    def _connect(self) -> dict[str, Any]:
        deadline = time.monotonic() + float(self.cfg.connect_wait_s)
        retry_sleep_s = float(self.cfg.retry_sleep_s)
        last_error: BaseException | None = None
        while True:
            try:
                self.ws.connect()
                metadata_msg = self.ws.recv_message()
                if isinstance(metadata_msg, str):
                    raise RuntimeError(f"OpenPI server returned text metadata: {metadata_msg}")
                metadata = _unpackb(metadata_msg)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"OpenPI server returned non-dict metadata: {type(metadata)}")
                self.logger.info(
                    "OpenPI server 已连接 host=%s port=%s metadata_keys=%s",
                    self.cfg.host,
                    self.cfg.port,
                    sorted(metadata),
                )
                return metadata
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Could not connect OpenPI server {self.cfg.host}:{self.cfg.port}: {last_error}") from exc
                self.logger.info("等待 OpenPI server %s:%s (%s)", self.cfg.host, self.cfg.port, exc)
                time.sleep(retry_sleep_s)


class _MinimalWebsocket:
    def __init__(self, host: str, port: int, timeout_s: float, path: str) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.path = path
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        sock.sendall(request)
        response = _read_http_response(sock)
        status_line, headers = _parse_http_response(response)
        if not status_line.startswith("HTTP/1.1 101"):
            raise RuntimeError(f"Websocket upgrade failed: {status_line}")
        expected_accept = base64.b64encode(hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()).decode("ascii")
        if headers["sec-websocket-accept"] != expected_accept:
            raise RuntimeError("Websocket upgrade failed: invalid Sec-WebSocket-Accept")
        self.sock = sock

    def send_binary(self, payload: bytes) -> None:
        self._send_frame(0x2, payload)

    def recv_message(self) -> bytes | str:
        parts: list[bytes] = []
        message_opcode: int | None = None
        while True:
            fin, opcode, payload = self._recv_frame()
            if opcode == 0x8:
                raise RuntimeError("Websocket connection closed by server")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                message_opcode = opcode
                parts = [payload]
            elif opcode == 0x0 and message_opcode is not None:
                parts.append(payload)
            else:
                raise RuntimeError(f"unsupported websocket opcode: {opcode}")
            if fin:
                data = b"".join(parts)
                if message_opcode == 0x1:
                    return data.decode("utf-8", errors="replace")
                return data

    def close(self) -> None:
        if self.sock is None:
            return
        self.sock.close()
        self.sock = None

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self.sock is None:
            raise RuntimeError("Websocket is not connected")
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(0x80 | length)
        elif length < (1 << 16):
            header.extend(struct.pack("!BH", 0x80 | 126, length))
        else:
            header.extend(struct.pack("!BQ", 0x80 | 127, length))
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_frame(self) -> tuple[bool, int, bytes]:
        if self.sock is None:
            raise RuntimeError("Websocket is not connected")
        first, second = self._read_exact(2)
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        return fin, opcode, payload

    def _read_exact(self, nbytes: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("Websocket is not connected")
        chunks = []
        remaining = nbytes
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError("Socket closed while reading websocket frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def build_policy_client(cfg: DictConfig) -> PolicyClient:
    if str(cfg.type) == "openpi_websocket":
        return OpenPiWebsocketPolicyClient(cfg)
    raise ValueError(f"unsupported policy server type: {cfg.type}")


def _read_http_response(sock: socket.socket) -> bytes:
    chunks = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("Socket closed during websocket upgrade")
        chunks.append(chunk)
        data = b"".join(chunks)
        if b"\r\n\r\n" in data:
            return data.split(b"\r\n\r\n", 1)[0]


def _parse_http_response(response: bytes) -> tuple[str, dict[str, str]]:
    lines = response.decode("iso-8859-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return lines[0], headers


def _packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_array)


def _unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_unpack_array, strict_map_key=False)


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value
