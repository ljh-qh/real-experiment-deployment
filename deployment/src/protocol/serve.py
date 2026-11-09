"""Stable ZMQ policy protocol; no simulator or hardware imports."""
import importlib.util, json, os
from pathlib import Path
import time
import numpy as np
import zmq

REPRESENTATION = 'openwam_canonical80_mount'


def load_codec(path):
    spec = importlib.util.spec_from_file_location('policy_codec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loopback_endpoint(value):
    if not value.startswith('tcp://127.0.0.1:'):
        raise ValueError('Use explicit loopback TCP endpoints')
    return value


def serve_worker(backend, codec, observation_endpoint, action_endpoint, control_endpoint,
                 ready_file, max_age_ms=1000):
    context = zmq.Context()
    observation = context.socket(zmq.SUB)
    action = context.socket(zmq.XPUB)
    control = context.socket(zmq.REP)
    sockets = (observation, action, control)
    for socket in sockets:
        socket.setsockopt(zmq.LINGER, 0)
    observation.setsockopt(zmq.RCVHWM, 2)
    action.setsockopt(zmq.SNDHWM, 2)
    observation.setsockopt(zmq.SUBSCRIBE, b'observation')
    observation.connect(observation_endpoint)
    action.bind(action_endpoint)
    control.bind(control_endpoint)
    metadata = {'pid': os.getpid(), 'observation_endpoint': observation_endpoint,
                'action_endpoint': action.getsockopt_string(zmq.LAST_ENDPOINT),
                'control_endpoint': control.getsockopt_string(zmq.LAST_ENDPOINT),
                'representation': REPRESENTATION, 'action_period_s': 1 / 30,
                'hardware_commands_sent': 0}
    metadata.update(getattr(backend, 'deployment_metadata', {}))
    poller = zmq.Poller()
    for socket in sockets:
        poller.register(socket, zmq.POLLIN)
    session, sequence, chunk_id = None, -1, 0
    subscribers = 0
    path = Path(ready_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata))
    temporary.replace(path)
    print(json.dumps({'type': 'ready', **metadata}), flush=True)
    try:
        while True:
            events = dict(poller.poll(1000))
            if action in events:
                subscription = action.recv()
                if subscription[1:] == b'action':
                    subscribers += 1 if subscription[0] == 1 else -1
            if control in events:
                request = control.recv_json()
                command = request['type']
                if command == 'reset':
                    session = str(request['session'])
                    sequence, chunk_id = -1, 0
                    if hasattr(backend, 'reset'):
                        backend.reset()
                    control.send_json({'type': 'reset_ack', 'session': session,
                                       'action_subscribed': subscribers > 0})
                elif command == 'ping':
                    control.send_json({'type': 'pong', 'session': session,
                                       'action_subscribed': subscribers > 0})
                elif command == 'shutdown':
                    control.send_json({'type': 'shutdown_ack'})
                    break
                else:
                    control.send_json({'type': 'error', 'message': 'Unknown control command'})
            if observation not in events:
                continue
            frames = observation.recv_multipart()
            while observation.poll(0, zmq.POLLIN):
                frames = observation.recv_multipart()
            received = None
            try:
                if frames[0] != b'observation':
                    raise ValueError('Unexpected observation topic')
                received = codec.decode_message(frames[1:])
                if session is None or received['session'] != session:
                    raise ValueError('Observation session has not been reset/acknowledged')
                if received['sequence'] <= sequence:
                    raise ValueError('Observation sequence is stale or duplicated')
                age_ns = time.monotonic_ns() - received['timestamp_ns']
                if age_ns < 0 or age_ns > max_age_ms * 1_000_000:
                    raise ValueError('Observation timestamp is stale or in the future')
                if subscribers <= 0:
                    raise ValueError('No action subscriber')
                chunk, timings = backend(received)
                sequence = received['sequence']
                reply = {'type': 'action', 'session': session, 'observation_sequence': sequence,
                         'representation': REPRESENTATION, 'chunk_id': chunk_id,
                         'observation_timestamp_ns': received['timestamp_ns'],
                         'source_timestamp_ns': received['source_timestamp_ns'],
                         'generated_timestamp_ns': time.monotonic_ns(), 'action_period_s': 1 / 30,
                         'action': chunk, 'timings': timings, 'hardware_commands_sent': 0}
                chunk_id += 1
            except Exception as exc:
                reply = {'type': 'error', 'message': str(exc), 'session': session,
                         'hardware_commands_sent': 0}
                if received is not None:
                    reply['observation_sequence'] = received['sequence']
            action.send_multipart([b'action'] + codec.encode_message(reply))
    finally:
        for socket in sockets:
            socket.close()
        context.term()
