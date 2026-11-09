from __future__ import annotations
from pathlib import Path
import base64
import io
import json
import os
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate

from deployment_paths import ROOT as DEPLOY
ROOT = DEPLOY / "assets"
ACTIVE = np.zeros(80, dtype=bool)
ACTIVE[0:9] = ACTIVE[10:32] = ACTIVE[34:43] = ACTIVE[44:66] = True


def _as_rgb(value) -> Image.Image:
    if isinstance(value, np.ndarray) and value.dtype.kind in 'US':
        value = value.item()
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        if isinstance(value, (np.str_, np.bytes_)):
            value = value.item()
        if isinstance(value, str):
            value = base64.b64decode(value)
        return Image.open(io.BytesIO(value)).convert('RGB')
    return Image.fromarray(np.asarray(value, dtype=np.uint8))


def compose_image(images: dict, size=(384, 320), image_resize="bilinear") -> Image.Image:
    from multiview import assemble_multiview_layout
    frames = {
        'head_camera': _as_rgb(images['main']),
        'left_camera': _as_rgb(images['wrist2']),
        'right_camera': _as_rgb(images['wrist']),
    }
    if image_resize == "lanczos":
        if size != (384, 320):
            raise ValueError("LANCZOS requires audited 384x320 geometry")
        frames = {k: v.resize((320,256) if k == "head_camera" else (160,128), Image.Resampling.LANCZOS) for k,v in frames.items()}
    elif image_resize != "bilinear":
        raise ValueError("Unknown image resize mode")
    return assemble_multiview_layout(frames, ('head_camera', 'left_camera', 'right_camera'), size[0], size[1])


class EasyWAMPolicy:
    def __init__(self, version: str = 'v1', device: str = 'cuda:0', image_resize: str = 'bilinear'):
        self.version = version
        self.device = device
        self.image_resize = image_resize
        ckpt_dir = ROOT / 'checkpoints' / {'v1': 'easywam_badminton', 'bottle': 'easywam_bottle'}[version]
        cfg = OmegaConf.load(ckpt_dir / ('deployment_model.yaml' if version == 'bottle' else 'config.yaml'))
        cfg.model.backbone.model_id = str(ROOT / 'base_models/Cosmos-Predict2.5-2B')
        cfg.model.backbone.reason_model_id = str(ROOT / 'base_models/Cosmos-Predict2.5-2B')
        cfg.model.action_dit_pretrained_path = None
        cfg.model.skip_dit_load_from_pretrain = True
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        self.model.load_checkpoint(str(ckpt_dir / ('step_008205.pt' if version == 'bottle' else 'step_004465.pt')))
        execution = os.environ.get('EASYWAM_EXECUTION_PROFILE', 'baseline')
        if execution == 'cuda_graph_fa2':
            from easywam_cuda_graph import install_cuda_graph_execution
            self.cuda_graph = install_cuda_graph_execution(self.model, device)
        elif execution != 'baseline':
            raise ValueError('Unknown EasyWAM execution profile')
        if execution == 'baseline' and os.environ.get('EASYWAM_COMPILE', '0') == '1':
            from model.helpers.inference import configure_inference_compile
            configure_inference_compile(self.model, enabled=True, mode='reduce-overhead',
                                        fullgraph=os.environ.get('EASYWAM_FULLGRAPH', '0') == '1',
                                        dynamic=False)
        # v1 (state[t+1]): one measured 62-dim min/max block; the text context
        # comes from the common badminton context file.
        stats_path = ckpt_dir / 'normalization_stats_62.npy' if version == 'bottle' else ROOT / 'checkpoints/easywam_common/normalization_stats_62.npy'
        stats = np.load(stats_path,
                        allow_pickle=True).item()
        self.mn = np.asarray(stats['min'], dtype=np.float32)
        self.mx = np.asarray(stats['max'], dtype=np.float32)
        self.mn_state = self.mn
        self.mx_state = self.mx
        ctx_path = ckpt_dir / 'bottle_easywam_context.pt' if version == 'bottle' else ROOT / 'checkpoints/easywam_common/badminton_context.pt'
        ctx = torch.load(ctx_path,
                         map_location='cpu', weights_only=False)
        self.prompt = ctx['prompt']
        if version == 'bottle' and self.prompt != 'unscrew the bottle cap and pour water into the cup':
            raise ValueError('Bottle context prompt mismatch')
        self.context = ctx['context']
        self.context_mask = ctx['context_mask']
        self.hand_lead = None
        if os.environ.get('EASYWAM_HAND_LEAD', '0') == '1':
            hp = Path(__file__).resolve().parents[1] / 'configs' / 'hand_lead.json'
            if hp.exists():
                hd = json.loads(hp.read_text())
                scale = float(os.environ.get('EASYWAM_HAND_LEAD_SCALE', '1.0'))
                self.hand_lead = np.r_[np.asarray(hd['left_rad'], dtype=np.float32), np.asarray(hd['right_rad'], dtype=np.float32)] * scale

    def normalize_state(self, state80: np.ndarray) -> np.ndarray:
        x = np.asarray(state80, dtype=np.float32)[ACTIVE]
        return np.clip((x - self.mn_state) / np.maximum(self.mx_state - self.mn_state, 1e-6) * 2.0 - 1.0,
                       -1.0, 1.0)

    def unnormalize_action(self, action62: np.ndarray) -> np.ndarray:
        a = np.asarray(action62, dtype=np.float32)
        return (a + 1.0) * 0.5 * (self.mx - self.mn) + self.mn

    def limit_chunk(self, action62: np.ndarray, state62: np.ndarray) -> np.ndarray:
        """Keep per-tick EEF/hand deltas inside deployment limits."""
        xyz = list(range(0, 3)) + list(range(31, 34))
        rot = list(range(3, 9)) + list(range(34, 40))
        hand = list(range(9, 31)) + list(range(40, 62))
        out = np.asarray(action62, dtype=np.float32).copy()
        prev = np.asarray(state62, dtype=np.float32).copy()
        for t in range(out.shape[0]):
            delta = out[t] - prev
            delta[xyz] = np.clip(delta[xyz], -0.004, 0.004)
            delta[rot] = np.clip(delta[rot], -0.008, 0.008)
            delta[hand] = np.clip(delta[hand], -0.03, 0.03)
            out[t] = prev + delta
            prev = out[t]
        return out

    def infer(self, state80: np.ndarray, images: dict, prompt: str | None = None,
              num_inference_steps: int = 10, soft_start: bool = False, anchor_steps: int = 0) -> np.ndarray:
        num_inference_steps = int(os.environ.get('EASYWAM_STEPS', num_inference_steps))
        image = compose_image(images, image_resize=self.image_resize)
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        image_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.bfloat16)
        proprio = torch.from_numpy(self.normalize_state(state80))[None].to(self.device, torch.bfloat16)
        kwargs = dict(input_image=image_t, action_horizon=32, num_video_frames=9,
                      proprio=proprio, num_inference_steps=num_inference_steps)
        kwargs['prompt'] = None
        kwargs['context'] = self.context.unsqueeze(0).to(self.device, torch.bfloat16)
        kwargs['context_mask'] = self.context_mask.unsqueeze(0).to(self.device)
        seed_env = os.environ.get("EASYWAM_SEED", "")
        if seed_env:
            kwargs["seed"] = int(seed_env)
        out = self.model.infer_action(**kwargs)['action']
        if hasattr(out, 'detach'):
            out = out.detach().float().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (32, 62):
            raise ValueError(f'expected (32,62), got {out.shape}')
        out = self.unnormalize_action(out)
        if out.shape[0] >= 3 and os.environ.get('EASYWAM_SMOOTH', '0') == '1':
            smoothed = out.copy()
            smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
            out = smoothed
        compact_state = np.asarray(state80, dtype=np.float32)[ACTIVE]
        if soft_start:
            n = min(8, out.shape[0])
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            out[:n] = w * out[:n] + (1.0 - w) * compact_state
        arm = list(range(0, 9)) + list(range(31, 40))
        hand = list(range(9, 31)) + list(range(40, 62))
        if os.environ.get('EASYWAM_HOLD_HANDS') == '1':
            out[:, hand] = compact_state[hand]
        if os.environ.get('EASYWAM_HOLD_ARMS') == '1':
            out[:, arm] = compact_state[arm]
        result = np.zeros((32, 80), dtype=np.float32)
        result[:, ACTIVE] = out
        return result
