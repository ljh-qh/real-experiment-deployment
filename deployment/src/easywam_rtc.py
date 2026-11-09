"""Deployment-only identity-VJP RTC for EasyWAM continuous action flow."""
from types import MethodType
import numpy as np
import torch
from rtc_guidance import guided_velocity, prefix_weights
from easywam_adapter import ACTIVE

class EasyWAMRTC:
    def __init__(self, policy, max_guidance=10.0):
        if not np.isfinite(max_guidance) or max_guidance <= 0:
            raise ValueError("Invalid RTC guidance cap")
        self.policy = policy
        self.cap = float(max_guidance)
        self.request = None
        self.steps = 0
        self.original = policy.model._predict_joint_noise
        owner = self
        def predict(model, **kwargs):
            if owner.request is None:
                return owner.original(**kwargs)
            x = kwargs["latents_action"]
            if tuple(x.shape) != (1, 32, 62):
                raise ValueError("RTC requires a single 32x62 action batch")
            ts = kwargs["timestep_action"].detach().float().reshape(-1)
            if ts.numel() != 1:
                raise ValueError("RTC requires one action timestep")
            sigma = float(ts[0]) / model.infer_action_scheduler.num_train_timesteps
            target, weights = owner.request
            def denoise(value):
                return owner.original(**dict(kwargs, latents_action=value))
            owner.steps += 1
            return guided_velocity(denoise, x, sigma, target, weights,
                                   "identity_vjp", owner.cap)
        policy.model._predict_joint_noise = MethodType(predict, policy.model)

    def set_request(self, prefix, commit):
        prefix = np.asarray(prefix, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 80 or not np.isfinite(prefix).all():
            raise ValueError("RTC prefix must be finite physical [N,80]")
        if not isinstance(commit, int) or not 0 < commit <= len(prefix) <= 32:
            raise ValueError("Invalid RTC commitment")
        scale = (self.policy.mx - self.policy.mn) / 2
        if scale.shape != (62,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid action normalization scale")
        target = np.zeros((1,32,62), np.float32)
        target[0,:len(prefix)] = (prefix[:,ACTIVE] - self.policy.mn) / scale - 1
        # Do not clip: the target is an executed physical prefix, not an observation.
        self.request = (torch.from_numpy(target).to(self.policy.device),
                        prefix_weights(commit,len(prefix),32,self.policy.device)[None,:,None])
        self.steps = 0

    def clear(self):
        self.request = None
        self.steps = 0
