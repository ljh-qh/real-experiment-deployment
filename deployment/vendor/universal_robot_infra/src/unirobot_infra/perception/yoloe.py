from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.perception.types import BBox2D, Detection2D


class YoloEDetector:
    def __init__(self, cfg: DictConfig) -> None:
        if str(cfg.type) != "yoloe":
            raise ValueError("YoloEDetector requires detector.type=yoloe")
        yoloe_cls = _import_yoloe()
        self.text_model_path = Path(str(cfg.text_model_path))
        if not self.text_model_path.is_file():
            raise FileNotFoundError(f"YOLOE text model not found: {self.text_model_path}")
        self.model = yoloe_cls(str(cfg.model_path), verbose=bool(cfg.verbose))
        self.classes = tuple(str(name) for name in cfg.classes)
        if len(self.classes) == 0:
            raise ValueError("YOLOE detector classes must not be empty")
        previous_weights_dir = _set_ultralytics_weights_dir(self.text_model_path.parent)
        try:
            self.model.set_classes(list(self.classes))
        finally:
            _set_ultralytics_weights_dir(previous_weights_dir)
        self.device = str(cfg.device)
        self.image_size = int(cfg.image_size)
        self.conf_threshold = float(cfg.conf_threshold)
        self.iou_threshold = float(cfg.iou_threshold)
        self.max_det = int(cfg.max_det)

    def detect(self, image_rgb: np.ndarray) -> tuple[Detection2D, ...]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        image_bgr = image_rgb[..., ::-1].copy()
        results = self.model.predict(
            image_bgr,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("YOLOE detector expects one prediction result for one input image")
        return detections_from_yoloe_result(results[0])


def detections_from_yoloe_result(result) -> tuple[Detection2D, ...]:
    boxes = result.boxes
    if boxes is None:
        return tuple()
    xyxy = _to_numpy(boxes.xyxy)
    scores = _to_numpy(boxes.conf)
    class_ids = _to_numpy(boxes.cls).astype(np.int64)
    detections = []
    for bbox, score, class_id in zip(xyxy, scores, class_ids, strict=True):
        detections.append(
            Detection2D(
                label=str(result.names[int(class_id)]),
                score=float(score),
                bbox=BBox2D(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                ),
            )
        )
    return tuple(detections)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _import_yoloe():
    import logging

    from ultralytics import YOLOE
    from ultralytics.utils import LOGGER as ultralytics_logger

    ultralytics_logger.setLevel(logging.WARNING)
    return YOLOE


def _set_ultralytics_weights_dir(weights_dir: str | Path) -> str:
    from ultralytics.utils import SETTINGS

    previous_weights_dir = str(SETTINGS["weights_dir"])
    SETTINGS["weights_dir"] = str(weights_dir)
    return previous_weights_dir
