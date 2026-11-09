"""Inference-time RTC prefix guidance for OpenWAM's descending-sigma flow.

Equations follow Black et al., arXiv:2506.07339 and the authors' Kinetix implementation.
full_vjp differentiates the denoised estimate. identity_vjp is an explicitly
labelled approximation; it does not differentiate the network.
"""
from types import MethodType
import numpy as np
import torch

ACTIVE = np.zeros(80, dtype=bool)
ACTIVE[0:9] = ACTIVE[10:32] = ACTIVE[34:43] = ACTIVE[44:66] = True


def prefix_weights(commit, end, horizon, device=None):
    if not 0 <= commit <= end <= horizon:
        raise ValueError("Require 0 <= commit <= prefix end <= horizon")
    w = torch.zeros(horizon, dtype=torch.float32, device=device)
    w[:commit] = 1
    if end > commit:
        w[commit:end] = torch.arange(end-commit, 0, -1, device=device) / (end-commit+1)
    return w


def guidance_gain(sigma, cap):
    if not np.isfinite(sigma) or not 0 < sigma <= 1 or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Invalid RTC sigma/guidance cap")
    if sigma == 1:
        return float(cap)
    return min(float(cap), (sigma*sigma + (1-sigma)**2) / (sigma*(1-sigma)))


def guided_velocity(denoise, x, sigma, target, weights, mode, cap):
    if mode not in ("full_vjp", "identity_vjp"):
        raise ValueError("Unsupported RTC guidance mode")
    if target is None or not bool(torch.any(weights)):
        video, velocity = denoise(x)
        return video, velocity
    if mode == "full_vjp":
        with torch.enable_grad():
            leaf = x.detach().clone().requires_grad_(True)
            video, velocity = denoise(leaf)
            if not velocity.requires_grad:
                raise RuntimeError("Full RTC VJP requires a differentiable action denoiser")
            clean = leaf.float() - sigma * velocity.float()
            error = ((target - clean) * weights).detach()
            correction = torch.autograd.grad(clean, leaf, grad_outputs=error, create_graph=False)[0].float()
    else:
        with torch.no_grad():
            video, velocity = denoise(x)
            clean = x.float() - sigma * velocity.float()
            correction = (target - clean) * weights
    guided = velocity.float() - guidance_gain(sigma, cap) * correction
    if not torch.isfinite(guided).all():
        raise RuntimeError("RTC guidance produced non-finite velocity")
    return video.detach(), guided.to(velocity.dtype).detach()


class RTCGuidance:
    def __init__(self, architecture, mode="identity_vjp", max_guidance=10.0):
        self.architecture = architecture
        self.mode = mode
        self.max_guidance = float(max_guidance)
        self.request = None
        self.steps = 0
        self.original_forward = architecture.forward
        # Derive the OUTPUT inverse affine transform, never proprio normalization.
        norm = architecture.normalizer
        if norm is None:
            raise ValueError("RTC requires an explicit action normalizer")
        z = np.zeros((1,80), np.float32)
        self.offset = np.asarray(norm.unnormalize(z), np.float32)[0]
        self.scale = np.asarray(norm.unnormalize(z+1), np.float32)[0] - self.offset
        if self.offset.shape != (80,) or np.any(~np.isfinite(self.scale)) or np.any(self.scale[ACTIVE] <= 0):
            raise ValueError("Unexpected RTC action normalizer")
        probe = np.linspace(-.7,.7,80,dtype=np.float32)[None]
        np.testing.assert_allclose(norm.unnormalize(probe), probe*self.scale+self.offset, atol=2e-6)
        owner = self
        def forward(_architecture, noisy_actions, action_timestep, **kwargs):
            def denoise(x):
                return owner.original_forward(x, action_timestep, **kwargs)
            if owner.request is None:
                return denoise(noisy_actions)
            sigma = float(action_timestep.detach().float().reshape(-1)[0]) / float(_architecture.action_scheduler.num_train_timesteps)
            target, weights = owner.request
            owner.steps += 1
            return guided_velocity(denoise, noisy_actions, sigma, target, weights, owner.mode, owner.max_guidance)
        architecture.forward = MethodType(forward, architecture)

    def set_request(self, physical_prefix, commit):
        prefix = np.asarray(physical_prefix,dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 80 or not np.isfinite(prefix).all():
            raise ValueError("RTC prefix must be finite physical-unit [N,80]")
        n = len(prefix)
        if not 0 < commit <= n <= 32:
            raise ValueError("Invalid RTC committed prefix length")
        padded = np.zeros((1,32,80),np.float32)
        padded[0,:n,ACTIVE] = ((prefix[:,ACTIVE]-self.offset[ACTIVE])/self.scale[ACTIVE]).T
        # Assignment above uses NumPy mixed indexing [N_active,N_prefix].
        device = self.architecture.device
        target = torch.from_numpy(padded).to(device=device)
        active = torch.from_numpy(ACTIVE).to(device=device)
        weights = prefix_weights(commit,n,32,device=device)[None,:,None] * active[None,None,:]
        self.request = (target,weights)
        self.steps = 0

    def clear(self):
        self.request = None
