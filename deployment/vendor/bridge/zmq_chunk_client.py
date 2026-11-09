"""Data-only ZMQ chunk client. This module cannot send hardware commands."""
import time

import zmq

from zmq_offline_bridge import validate_chunk, wait_readable, wait_subscriber


class ChunkClient:
    def __init__(self, codec, timeout_ms=10000, observation_endpoint='tcp://127.0.0.1:*'):
        self.codec = codec
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.XPUB)
        self.receiver = self.context.socket(zmq.SUB)
        self.control = self.context.socket(zmq.REQ)
        self.sockets = (self.publisher, self.receiver, self.control)
        for socket in self.sockets:
            socket.setsockopt(zmq.LINGER, 0)
        self.publisher.bind(observation_endpoint)
        self.observation_endpoint = self.publisher.getsockopt_string(zmq.LAST_ENDPOINT)
        self.receiver.setsockopt(zmq.SUBSCRIBE, b'action')
        self.session = None
        self.chunk_id = 0

    def connect(self, metadata, session):
        self.receiver.connect(metadata['action_endpoint'])
        self.control.connect(metadata['control_endpoint'])
        wait_subscriber(self.publisher, b'observation', self.timeout_ms)
        deadline = time.monotonic() + self.timeout_ms / 1000
        while not self.command({'type': 'ping'})['action_subscribed']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Action subscription not acknowledged')
        self.reset(session)

    def reset(self, session):
        reply = self.command({'type': 'reset', 'session': session})
        if reply['type'] != 'reset_ack' or reply['session'] != session:
            raise RuntimeError('Missing reset acknowledgement')
        self.session, self.chunk_id = session, 0

    def command(self, payload):
        self.control.send_json(payload)
        wait_readable(self.control, self.timeout_ms)
        return self.control.recv_json()

    def infer(self, message):
        start = time.monotonic()
        self.publisher.send_multipart([b'observation'] + self.codec.encode_message(message))
        wait_readable(self.receiver, self.timeout_ms)
        frames = self.receiver.recv_multipart()
        if frames[0] != b'action':
            raise ValueError('Unexpected response topic')
        reply = self.codec.decode_message(frames[1:])
        validate_chunk(reply, self.session, message['sequence'])
        if reply['chunk_id'] != self.chunk_id:
            raise ValueError('Unexpected chunk id')
        self.chunk_id += 1
        reply['roundtrip_seconds'] = time.monotonic() - start
        return reply

    def close(self):
        for socket in self.sockets:
            socket.close()
        self.context.term()
