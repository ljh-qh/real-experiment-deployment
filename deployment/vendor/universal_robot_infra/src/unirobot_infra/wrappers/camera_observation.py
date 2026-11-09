from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from unirobot_infra.camera import CameraManager
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


class CameraObservationWrapper(EnvWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig, cameras: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self.camera_manager = _build_camera_manager(cfg, cameras)

    def reset(self) -> tuple[tuple[float, ...], dict]:
        observation, info = self.env.reset()
        self._attach_camera_frames(info)
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._attach_camera_frames(info)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.camera_manager is not None:
            self.camera_manager.close()
        self.env.close()

    def _attach_camera_frames(self, info: dict) -> None:
        if not bool(self.cfg.enabled):
            return
        if self.camera_manager is None:
            raise RuntimeError("camera_observation.enabled requires a camera manager")
        info["camera_frames"] = self.camera_manager.read(timeout_s=float(self.cfg.read_timeout_s))

    def attach_camera_frames(self, info: dict) -> None:
        self._attach_camera_frames(info)


def _build_camera_manager(cfg: DictConfig, cameras: DictConfig) -> CameraManager | None:
    if not bool(cfg.enabled):
        return None
    camera_names = tuple(str(camera_name) for camera_name in cfg.camera_names)
    if not camera_names:
        raise ValueError("camera_observation.camera_names must not be empty when enabled")
    if float(cfg.read_timeout_s) <= 0.0:
        raise ValueError("camera_observation.read_timeout_s must be positive")
    return CameraManager(OmegaConf.create({camera_name: cameras[camera_name] for camera_name in camera_names}))
