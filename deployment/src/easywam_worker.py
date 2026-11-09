"""ZMQ EasyWAM policy worker; no hardware imports or command sink."""
import argparse, json, time, os
from pathlib import Path
import numpy as np
import torch
from easywam_adapter import EasyWAMPolicy
from deployment_paths import ROOT
from protocol.serve import serve_worker, loopback_endpoint, REPRESENTATION, load_codec


class EasyWAMChunkModel:
    def __init__(self, version, device, warmup_fixtures=None, image_resize='bilinear', rtc=False):
        self.policy = EasyWAMPolicy(version, device=device, image_resize=image_resize)
        self.guidance = None
        if rtc:
            from easywam_rtc import EasyWAMRTC
            self.guidance = EasyWAMRTC(self.policy)
        self.task = 'bottle' if version == 'bottle' else 'badminton'
        self._first_request = True
        self.deployment_metadata = {'engine': f'easywam_{version}', 'context_cache': True,
                                    'warmup_completed': False, 'execution_profile': os.environ.get('EASYWAM_EXECUTION_PROFILE', 'baseline'), 'task': self.task, 'image_resize': image_resize}
        self.deployment_metadata.update(rtc_mode='identity_vjp' if rtc else 'disabled', rtc_is_approximation=rtc)
        if warmup_fixtures is not None and os.environ.get('EASYWAM_SKIP_WARMUP', '0') != '1':
            self.warmup(Path(warmup_fixtures))

    def warmup(self, fixtures):
        import numpy as np
        from PIL import Image
        folder = Path(fixtures) / 'recorded_start'
        state = np.load(folder / 'state80.npy')
        images = {k: np.asarray(Image.open(folder / f'{k}.png').convert('RGB'))
                  for k in ('main', 'wrist', 'wrist2')}
        self.policy.infer(state, images)
        self.policy.infer(state, images)
        self.deployment_metadata['warmup_completed'] = True

    def reset(self):
        self._first_request = True
        if self.guidance is not None:
            self.guidance.clear()

    def __call__(self, message):
        start = time.monotonic()
        if message['task'] != self.task:
            raise ValueError('Observation task does not match loaded worker')
        obs = message['observation'] if 'observation' in message else message
        state = np.asarray(obs['state'], dtype=np.float32)
        imgs = obs['images']
        if 'main' in imgs:
            images = {'main': imgs['main'], 'wrist2': imgs['wrist2'], 'wrist': imgs['wrist']}
        else:
            images = {'main': imgs['head_camera'],
                      'wrist2': imgs['left_wrist_camera'], 'wrist': imgs['right_wrist_camera']}
        prepared = time.monotonic()
        soft = self._first_request and os.environ.get('EASYWAM_SOFT_START', '0') == '1'
        anchor = int(os.environ.get('EASYWAM_ANCHOR_STEPS', '10'))
        rtc = message.get('rtc')
        if rtc is not None and self.guidance is None:
            raise ValueError('RTC request requires an RTC-enabled worker')
        guided_steps = 0
        try:
            if rtc is not None:
                self.guidance.set_request(rtc['prefix'], rtc['commit_steps'])
            chunk = self.policy.infer(state, images, soft_start=soft, anchor_steps=anchor)
            if rtc is not None:
                guided_steps = self.guidance.steps
                if guided_steps == 0:
                    raise RuntimeError('RTC request bypassed the action denoiser')
        finally:
            if self.guidance is not None:
                self.guidance.clear()
        self._first_request = False
        if chunk.shape != (32, 80) or not np.isfinite(chunk).all():
            raise ValueError('EasyWAM must return finite (32,80)')
        finished = time.monotonic()
        return chunk, {'rtc_guided_steps': guided_steps, 'preprocess_seconds': prepared - start,
                       'generate_validate_seconds': finished - prepared,
                       'model_seconds': finished - start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', choices=('v1','bottle'), default='v1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--observation-endpoint', type=loopback_endpoint, required=True)
    p.add_argument('--action-endpoint', type=loopback_endpoint, required=True)
    p.add_argument('--control-endpoint', type=loopback_endpoint, required=True)
    p.add_argument('--ready-file', type=Path, required=True)
    p.add_argument('--codec', type=Path, required=True)
    p.add_argument('--max-age-ms', type=int, default=1000)
    p.add_argument('--warmup-fixtures', type=Path,
                   default=ROOT / "assets/references/fixed_observation_diagnosis")
    p.add_argument('--image-resize', choices=('bilinear','lanczos'), default='bilinear')
    p.add_argument('--rtc', action='store_true')
    args = p.parse_args()
    backend = EasyWAMChunkModel(args.version, args.device, args.warmup_fixtures, args.image_resize, args.rtc)
    serve_worker(backend, load_codec(args.codec), args.observation_endpoint,
                 args.action_endpoint, args.control_endpoint, args.ready_file,
                 args.max_age_ms)


if __name__ == '__main__':
    main()
