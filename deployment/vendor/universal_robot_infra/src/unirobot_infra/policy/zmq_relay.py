from typing import Any

from omegaconf import DictConfig
import zmq

from unirobot_infra.policy.codec import decode_message, encode_message


class ZmqPolicyRelay:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._context = zmq.Context.instance()
        self._obs_socket = self._context.socket(zmq.PUB)
        self._action_socket = self._context.socket(zmq.SUB)
        self._closed = False
        self._configure_socket(self._obs_socket)
        self._configure_socket(self._action_socket)
        if cfg.observation.mode == "bind":
            self._obs_socket.bind(str(cfg.observation.endpoint))
        elif cfg.observation.mode == "connect":
            self._obs_socket.connect(str(cfg.observation.endpoint))
        else:
            raise ValueError("policy.zmq.observation.mode must be bind or connect")
        self._action_socket.setsockopt_string(zmq.SUBSCRIBE, str(cfg.action.topic))
        if cfg.action.mode == "bind":
            self._action_socket.bind(str(cfg.action.endpoint))
        elif cfg.action.mode == "connect":
            self._action_socket.connect(str(cfg.action.endpoint))
        else:
            raise ValueError("policy.zmq.action.mode must be bind or connect")

    def publish_observation(self, payload: dict[str, Any]) -> None:
        topic = str(self.cfg.observation.topic).encode("utf-8")
        self._obs_socket.send_multipart([topic] + encode_message(payload))

    def recv_action(self) -> dict[str, Any] | None:
        poll_timeout_ms = int(self.cfg.action.poll_timeout_ms)
        if not self._action_socket.poll(timeout=poll_timeout_ms) & zmq.POLLIN:
            return None
        frames = self._action_socket.recv_multipart()
        if not frames:
            raise ValueError("empty ZMQ action message")
        topic = frames[0].decode("utf-8")
        if topic != str(self.cfg.action.topic):
            return None
        return decode_message(frames[1:])

    def close(self) -> None:
        if self._closed:
            return
        self._obs_socket.close()
        self._action_socket.close()
        self._closed = True

    def __enter__(self) -> "ZmqPolicyRelay":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _configure_socket(self, socket) -> None:
        socket.setsockopt(zmq.LINGER, int(self.cfg.linger_ms))
        socket.setsockopt(zmq.SNDHWM, int(self.cfg.snd_hwm))
        socket.setsockopt(zmq.RCVHWM, int(self.cfg.rcv_hwm))
