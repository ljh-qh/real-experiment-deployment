"""ZMQ policy process owning the OpenWAM engine; no hardware connections."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import zmq

from adapter import decode_action
from image_resize import prepare_resize_input
from zmq_offline_bridge import REPRESENTATION, load_codec, to_openwam


class DirectChunkModel:
    def __init__(self, engine, preprocessor, binary_dims=()):
        self.engine = engine
        self.preprocessor = preprocessor
        self.binary_dims = tuple(binary_dims)
        self.image_resize = 'bilinear'

    @classmethod
    def load(cls, config, checkpoint, checkpoint_name, device):
        from openwam.deploy.server import (
            _load_deploy_yaml, _validate_inference_config, merge_deploy_cfg,
        )
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.engine import JointInferenceEngine
        from openwam.deploy.obs_preprocess import ObsPreprocessor

        deploy = _load_deploy_yaml(str(config))
        _validate_inference_config(deploy)
        if deploy.inference.inference_mode != 'sync' or deploy.inference.num_frames != 33:
            raise ValueError('Direct chunk worker requires sync generation of 32 actions')
        training, architecture = load_from_checkpoint_dir(
            str(checkpoint), device=device, ckpt_name=checkpoint_name)
        cfg = merge_deploy_cfg(training, deploy)
        engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
        dims = getattr(architecture, 'binary_command_dims', ()) or ()
        return cls(engine, ObsPreprocessor.from_cfg(cfg, engine), dims)

    def __call__(self, message):
        start = time.monotonic()
        # The stock preprocessor accepts PIL RGB images directly: no PNG/base64.
        obs = self.preprocessor.preprocess(prepare_resize_input(
            to_openwam(message, encode_images=False), self.preprocessor, self.image_resize))
        prepared = time.monotonic()
        conditions = {
            'observation': obs, 'first_frame_image': [obs['image']],
            'prompt': obs['prompt'], 'proprio': obs['state'],
        }
        result = self.engine.generate(conditions)
        actions = result['actions']
        if hasattr(actions, 'detach'):
            actions = actions.detach().cpu().numpy()
        chunk = np.array(actions, dtype=np.float32, copy=True)
        if chunk.shape != (32, 80) or not np.isfinite(chunk).all():
            raise ValueError('Engine must return finite physical-unit (32,80) actions')
        # Same final command projection as the stock WAMPolicy boundary.
        for dimension in self.binary_dims:
            if dimension < 0 or dimension >= 80:
                raise ValueError('Binary command dimension outside action80')
            chunk[:, dimension] = np.where(chunk[:, dimension] > .5, 1., -1.)
        for action in chunk:
            decode_action(action)
        finished = time.monotonic()
        return chunk, {'preprocess_seconds': prepared - start,
                       'generate_validate_seconds': finished - prepared,
                       'model_seconds': finished - start}


def serve_worker(backend, codec, observation_endpoint, action_endpoint, control_endpoint,
                 ready_file, max_age_ms):
    """One process, one session, one model call per accepted observation."""
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
    metadata.update(getattr(backend,'deployment_metadata',{}))
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
            # Discard queued older observations before starting a generation.
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
                if 'task' in metadata and received['task'] != metadata['task']:
                    raise ValueError('Observation task does not match loaded worker')
                chunk, timings = backend(received)
                sequence = received['sequence']
                reply = {
                    'type': 'action', 'session': session, 'observation_sequence': sequence,
                    'representation': REPRESENTATION, 'chunk_id': chunk_id,
                    'observation_timestamp_ns': received['timestamp_ns'],
                    'source_timestamp_ns': received['source_timestamp_ns'],
                    'generated_timestamp_ns': time.monotonic_ns(), 'action_period_s': 1 / 30,
                    'action': chunk, 'timings': timings, 'hardware_commands_sent': 0,
                }
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


def loopback_endpoint(value):
    if not value.startswith('tcp://127.0.0.1:'):
        raise argparse.ArgumentTypeError('Use explicit loopback TCP endpoints')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-name', default='checkpoint_step_30000.safetensors')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--codec', type=Path, required=True)
    parser.add_argument('--observation-endpoint', type=loopback_endpoint, required=True)
    parser.add_argument('--action-endpoint', type=loopback_endpoint, required=True)
    parser.add_argument('--control-endpoint', type=loopback_endpoint, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--max-age-ms', type=int, default=1000)
    parser.add_argument('--context-cache', action='store_true',
                        help='Audited request-local KV reuse with compiled joint loop')
    parser.add_argument('--warmup-fixtures', type=Path,
                        help='Saved fixed_observation_diagnosis directory; required for context-cache')
    parser.add_argument('--task', choices=('badminton', 'top', 'bottle'), default='badminton')
    parser.add_argument('--image-resize', choices=('bilinear', 'lanczos'), default='bilinear')
    args = parser.parse_args()
    if args.max_age_ms <= 0:
        parser.error('--max-age-ms must be positive')
    if args.context_cache and args.warmup_fixtures is None:
        parser.error('--context-cache requires --warmup-fixtures before publishing ready')
    model = DirectChunkModel.load(args.config, args.checkpoint, args.checkpoint_name, args.device)
    model.image_resize = args.image_resize
    if args.context_cache:
        from deploy_context_cache import install_context_cache
        from PIL import Image
        install_context_cache(model,verify_context=True)
        for name in ('recorded_start','recorded_mid','live_end_reconstructed','recorded_start'):
            folder=args.warmup_fixtures/name
            message=dict(type='observation',representation=REPRESENTATION,task=args.task,
                observation=dict(state=np.load(folder/'state80.npy'),images={role:
                    np.asarray(Image.open(folder/(role+'.png')).convert('RGB'))
                    for role in ('main','wrist','wrist2')}))
            _,timing=model(message)
            print(json.dumps({'type':'warmup','input':name,**timing}),flush=True)
    model.deployment_metadata={'context_cache':args.context_cache,'config':str(args.config),
                               'warmup_completed':bool(args.context_cache),'task':args.task,'image_resize':args.image_resize}
    serve_worker(model, load_codec(args.codec), args.observation_endpoint, args.action_endpoint,
                 args.control_endpoint, args.ready_file, args.max_age_ms)


if __name__ == '__main__':
    main()
