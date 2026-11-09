"""Recorded-input ZMQ contract probe. No hardware imports or command sink."""
import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from PIL import Image
import zmq

from adapter import CAMERAS, decode_action


REPRESENTATION = 'openwam_canonical80_mount'
TASK_PROMPTS = {
    'badminton': 'badminton',
    'top': 'pull the string to spin the top on the table',
    'bottle': 'unscrew the bottle cap and pour water into the cup',
}


def load_codec(path):
    """Load the audited Universal codec without package/hardware initialization."""
    spec = importlib.util.spec_from_file_location('universal_policy_codec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_openwam(message, encode_images=True):
    if message['type'] != 'observation' or message['representation'] != REPRESENTATION:
        raise ValueError('Expected OpenWAM canonical80 observation')
    if message['task'] not in TASK_PROMPTS:
        raise ValueError('Unsupported task')
    state = np.asarray(message['observation']['state'], dtype=np.float32)
    if state.shape != (80,) or not np.isfinite(state).all():
        raise ValueError('Expected finite state80')
    active = np.zeros(80, dtype=bool)
    active[0:9] = active[10:32] = active[34:43] = active[44:66] = True
    if np.any(state[~active] != 0):
        raise ValueError('Canonical80 padding must be zero')
    decode_action(state)
    images = {}
    for key, role in CAMERAS.items():
        array = np.asarray(message['observation']['images'][role])
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f'{role}: expected uint8 RGB HWC')
        picture = Image.fromarray(array)
        if encode_images:
            buffer = io.BytesIO()
            picture.save(buffer, format='PNG')
            images[key] = base64.b64encode(buffer.getvalue()).decode('ascii')
        else:
            images[key] = picture
    return {'type': 'obs', 'images': images, 'state': state.tolist(), 'prompt': TASK_PROMPTS[message['task']]}


def validate_chunk(message, session, sequence):
    if message['type'] == 'error':
        raise RuntimeError(message['message'])
    if message['type'] != 'action' or message['representation'] != REPRESENTATION:
        raise ValueError('Unexpected action representation')
    if message['session'] != session or message['observation_sequence'] != sequence:
        raise ValueError('Stale session or observation sequence')
    if message['hardware_commands_sent'] != 0:
        raise ValueError('Offline worker must not send hardware commands')
    chunk = np.asarray(message['action'], dtype=np.float32)
    if chunk.shape != (32, 80):
        raise ValueError('Expected a complete (32,80) chunk')
    for action in chunk:
        decode_action(action)
    return chunk


def wait_readable(socket, timeout_ms):
    if not socket.poll(timeout_ms, zmq.POLLIN):
        raise TimeoutError('ZMQ probe timed out')


def wait_subscriber(socket, topic, timeout_ms):
    wait_readable(socket, timeout_ms)
    if socket.recv() != b'\x01' + topic:
        raise RuntimeError('Unexpected ZMQ subscription')


def websocket_chunk(payload, url):
    """Compatibility probe only: 32 existing requests, not a real-time player."""
    from websockets.sync.client import connect
    actions, timings, steps = [], [], []
    with connect(url, open_timeout=20, max_size=32 * 1024 * 1024) as ws:
        ws.send(json.dumps({'type': 'reset'}))
        if json.loads(ws.recv(timeout=30))['type'] != 'reset_ack':
            raise RuntimeError('Missing OpenWAM reset acknowledgement')
        for _ in range(32):
            start = time.monotonic()
            ws.send(json.dumps(payload))
            result = json.loads(ws.recv(timeout=600))
            if result['type'] != 'action':
                raise RuntimeError(result)
            decode_action(result['action'])
            actions.append(result['action'])
            steps.append(result['step'])
            timings.append(time.monotonic() - start)
    if steps != list(range(1, 33)):
        raise RuntimeError('Unexpected server action order after reset')
    return np.asarray(actions, dtype=np.float32), timings


def run_probe(message, codec, backend, timeout_ms=10000):
    """Use ephemeral loopback ports and subscription events, never fixed sleeps."""
    context = zmq.Context()
    publisher = context.socket(zmq.XPUB)
    receiver = context.socket(zmq.SUB)
    publisher.setsockopt(zmq.LINGER, 0)
    receiver.setsockopt(zmq.LINGER, 0)
    publisher.bind('tcp://127.0.0.1:*')
    observation_endpoint = publisher.getsockopt_string(zmq.LAST_ENDPOINT)
    receiver.setsockopt(zmq.SUBSCRIBE, b'action')
    ready = threading.Event()
    worker_state = {}

    def worker():
        subscriber = context.socket(zmq.SUB)
        output = context.socket(zmq.XPUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        output.setsockopt(zmq.LINGER, 0)
        try:
            subscriber.setsockopt(zmq.SUBSCRIBE, b'observation')
            subscriber.connect(observation_endpoint)
            output.bind('tcp://127.0.0.1:*')
            worker_state['endpoint'] = output.getsockopt_string(zmq.LAST_ENDPOINT)
            ready.set()
            wait_subscriber(output, b'action', timeout_ms)
            wait_readable(subscriber, timeout_ms)
            frames = subscriber.recv_multipart()
            if frames[0] != b'observation':
                raise ValueError('Unexpected observation topic')
            received = codec.decode_message(frames[1:])
            start = time.monotonic()
            payload = to_openwam(received)
            # Record exactly what reaches the backend for equivalence checks.
            worker_state['payload'] = payload
            chunk, request_seconds = backend(payload)
            reply = {
                'type': 'action', 'session': received['session'],
                'observation_sequence': received['sequence'],
                'representation': REPRESENTATION, 'chunk_id': 0,
                'observation_timestamp_ns': received['timestamp_ns'],
                'generated_timestamp_ns': time.monotonic_ns(),
                'action_period_s': 1 / 30, 'action': chunk,
                'worker_seconds': time.monotonic() - start,
                'request_seconds': request_seconds, 'hardware_commands_sent': 0,
            }
            validate_chunk(reply, received['session'], received['sequence'])
            output.send_multipart([b'action'] + codec.encode_message(reply))
        except Exception as exc:
            worker_state['error'] = exc
            ready.set()
        finally:
            # Keep the PUB alive until the consumer has received the chunk.
            worker_state['finished'].wait(timeout_ms / 1000)
            subscriber.close()
            output.close()

    worker_state['finished'] = threading.Event()
    thread = threading.Thread(target=worker, name='offline-policy-worker', daemon=True)
    thread.start()
    try:
        if not ready.wait(timeout_ms / 1000):
            raise TimeoutError('Worker startup timed out')
        if 'error' in worker_state:
            raise worker_state['error']
        receiver.connect(worker_state['endpoint'])
        wait_subscriber(publisher, b'observation', timeout_ms)
        start = time.monotonic()
        publisher.send_multipart([b'observation'] + codec.encode_message(message))
        # Periodic checks expose worker failures without waiting for the deadline.
        deadline = time.monotonic() + timeout_ms / 1000
        while not receiver.poll(100, zmq.POLLIN):
            if 'error' in worker_state:
                raise worker_state['error']
            if time.monotonic() >= deadline:
                raise TimeoutError('Action response timed out')
        frames = receiver.recv_multipart()
        if frames[0] != b'action':
            raise ValueError('Unexpected action topic')
        reply = codec.decode_message(frames[1:])
        validate_chunk(reply, message['session'], message['sequence'])
        reply['roundtrip_seconds'] = time.monotonic() - start
        return reply, worker_state['payload']
    finally:
        worker_state['finished'].set()
        thread.join(timeout=2)
        publisher.close()
        receiver.close()
        # A failed, still-running backend owns its sockets until it exits.
        if not thread.is_alive():
            context.term()


def fixture_backend(payload):
    """Deterministic transport fixture; does not represent model inference."""
    chunk = np.repeat(np.asarray(payload['state'], dtype=np.float32)[None], 32, axis=0)
    chunk[:, 0] += np.arange(32, dtype=np.float32) * 0.00001
    return chunk, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--row', type=int, default=50)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--backend', choices=['fixture', 'openwam'], required=True)
    parser.add_argument('--server-url', default='ws://127.0.0.1:8848')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from kinematics import URDFKinematics
    from offline_probe import prepare_recorded_observation
    direct, images, skews, stamp = prepare_recorded_observation(
        args.episode, args.row, URDFKinematics(args.urdf))
    session = str(uuid.uuid4())
    message = {
        'type': 'observation', 'session': session, 'sequence': 0,
        'timestamp_ns': time.monotonic_ns(), 'source_timestamp_ns': stamp,
        'task': 'badminton', 'representation': REPRESENTATION,
        'observation': {'state': np.asarray(direct['state'], dtype=np.float32), 'images': images},
    }
    backend = fixture_backend if args.backend == 'fixture' else (
        lambda payload: websocket_chunk(payload, args.server_url))
    reply, converted = run_probe(message, load_codec(args.codec), backend,
                                 timeout_ms=10000 if args.backend == 'fixture' else 660000)
    if converted != direct:
        raise RuntimeError('ZMQ backend payload differs from direct offline probe')
    report = {
        'backend': args.backend, 'episode': str(args.episode), 'row': args.row,
        'session': session, 'skew_seconds': skews, 'state80': direct['state'],
        'payload_identical_to_direct_probe': True, 'all_finite': True,
        'hardware_commands_sent': 0, 'action_shape': list(reply['action'].shape),
        'action': reply['action'].tolist(), 'request_seconds': reply['request_seconds'],
        'worker_seconds': reply['worker_seconds'], 'roundtrip_seconds': reply['roundtrip_seconds'],
        'note': 'Recorded fixed observation. No hardware connections or paced execution. '
                'Fixture results are transport validation only; OpenWAM mode drains 32 single-action requests.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ('action', 'state80')}))


if __name__ == '__main__':
    main()
