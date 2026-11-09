from __future__ import annotations

import base64
import json
import os
from urllib import request

import cv2
import numpy as np
from omegaconf import DictConfig

from unirobot_infra.perception.types import BBox2D, Detection2D


class DinoXProDetector:
    def __init__(self, cfg: DictConfig) -> None:
        if str(cfg.type) != "dino_x_pro":
            raise ValueError("DinoXProDetector requires detector.type=dino_x_pro")
        self.api_url = str(cfg.api_url)
        self.api_key_env = str(cfg.api_key_env)
        if self.api_key_env not in os.environ:
            raise RuntimeError(f"environment variable {self.api_key_env} is required for DINO-X Pro")
        self.api_key = os.environ[self.api_key_env]
        self.model = str(cfg.model)
        self.prompts = _prompt_payload(cfg.prompts)
        self.box_threshold = float(cfg.box_threshold)
        self.timeout_s = float(cfg.timeout_s)

    def detect(self, image_rgb: np.ndarray) -> tuple[Detection2D, ...]:
        payload = {
            "model": self.model,
            "image": {
                "encoding": "jpeg_base64",
                "data": base64.b64encode(encode_image_jpeg(image_rgb)).decode("ascii"),
            },
            "prompts": self.prompts,
            "box_threshold": self.box_threshold,
            "bbox_format": "xyxy_pixels",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return parse_dinox_response(response_payload)


def encode_image_jpeg(image_rgb: np.ndarray) -> bytes:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    ok, encoded = cv2.imencode(".jpg", image_rgb[..., ::-1])
    if not ok:
        raise RuntimeError("failed to encode image as jpeg")
    return encoded.tobytes()


def parse_dinox_response(payload: dict) -> tuple[Detection2D, ...]:
    detections = []
    for item in payload["detections"]:
        bbox = item["bbox"]
        if len(bbox) != 4:
            raise ValueError("DINO-X detection bbox must have length 4")
        detections.append(
            Detection2D(
                label=str(item["label"]),
                score=float(item["score"]),
                bbox=BBox2D(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                ),
            )
        )
    return tuple(detections)


def _prompt_payload(prompts) -> list[dict[str, str]]:
    payload = []
    for prompt in prompts:
        payload.append({"label": str(prompt.label), "text": str(prompt.text)})
    return payload
