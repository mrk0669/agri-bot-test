"""Tier 2 perception: the learned detector (Section 5.6).

A YOLOv8-nano / YOLO11-nano object detector trained on the agricultural
datasets listed in the Annexure and executed with TensorRT on the Jetson Orin
Nano. It generalises to real plant appearance - leaf shape and texture rather
than marker colour - demonstrates learning-based robustness, and provides
redundancy when a marker is partially occluded.

**Availability is not assumed.** ``ultralytics`` may be absent on a dev laptop
and the ``.engine`` file certainly does not exist until it has been exported on
the target device. This module therefore loads lazily and reports
``available == False`` instead of raising, so the runtime degrades to the
deterministic colour tier rather than failing to start. That is the same
layering the proposal describes: the colour baseline is the guaranteed scorer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import BBox, Detection, DetectionSource, TargetClass
from ..utils.logging_setup import get_logger

__all__ = ["UltralyticsBase", "YoloDetector"]

log = get_logger("vision.yolo")


class UltralyticsBase:
    """Shared lazy-loading and result-parsing for ultralytics-backed detectors."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.45,
        iou: float = 0.50,
        imgsz: int = 640,
        device: Any = 0,
        half: bool = True,
        max_det: int = 20,
        source: DetectionSource = DetectionSource.YOLO,
    ):
        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.half = bool(half)
        self.max_det = int(max_det)
        self.source = source

        self._model = None
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self.last_inference_ms: float = 0.0
        self.inference_count: int = 0

    # -- loading ------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True once a model is loaded, attempting the load exactly once."""
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._try_load()
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _resolve_weights(self) -> Optional[str]:
        """Find the weights file, or return None with the reason logged."""
        candidate = Path(self.weights)
        if candidate.is_file():
            return str(candidate)
        # Ultralytics resolves bare model names (e.g. "yolov8n.pt") from its own
        # cache or by download, so a non-existent path is not automatically an
        # error when it has no directory component.
        if candidate.parent in (Path("."), Path("")):
            return self.weights
        return None

    def _try_load(self) -> None:
        self._load_attempted = True
        try:
            from ultralytics import YOLO  # noqa: WPS433 - deliberate lazy import
        except Exception as exc:  # pragma: no cover - depends on environment
            self._load_error = f"ultralytics not importable: {exc}"
            log.warning("Learned detector unavailable: %s", self._load_error)
            return

        resolved = self._resolve_weights()
        if resolved is None:
            self._load_error = f"weights not found: {self.weights}"
            log.warning("Learned detector unavailable: %s", self._load_error)
            return

        try:
            self._model = self._construct(YOLO, resolved)
            log.info("Loaded learned detector from %s", resolved)
        except Exception as exc:  # pragma: no cover - depends on environment
            self._model = None
            self._load_error = f"failed to load {resolved}: {exc}"
            log.warning("Learned detector unavailable: %s", self._load_error)

    def _construct(self, yolo_cls, resolved: str):
        return yolo_cls(resolved)

    # -- inference ----------------------------------------------------------
    def _predict(self, frame: np.ndarray):
        return self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )

    def _parse(self, results, name_to_class) -> List[Detection]:
        """Convert ultralytics results into :class:`Detection` objects."""
        out: List[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(result, "names", {}) or {}
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

            for box, conf, cls_idx in zip(xyxy, confs, clses):
                label = names.get(int(cls_idx), str(int(cls_idx)))
                target = name_to_class(int(cls_idx), label)
                if target is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox = BBox(x1, y1, x2, y2)
                out.append(Detection(
                    cls=target,
                    bbox=bbox,
                    confidence=float(conf),
                    source=self.source,
                    area_px=bbox.area,
                    meta={"label": label, "cls_idx": int(cls_idx)},
                ))
        return out

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self._model is not None,
            "weights": self.weights,
            "load_error": self._load_error,
            "inference_count": self.inference_count,
            "last_inference_ms": round(self.last_inference_ms, 2),
        }


class YoloDetector(UltralyticsBase):
    """Trained crop/weed detector.

    ``class_map`` maps the trained model's class indices onto the two classes
    the mission cares about. Any index not in the map is dropped, so a model
    trained on a 16-crop / 58-weed dataset such as CropAndWeed can be used
    without retraining simply by writing the mapping into the config.
    """

    def __init__(
        self,
        weights: str,
        class_map: Optional[Dict[int, str]] = None,
        fallback_weights: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(weights, source=DetectionSource.YOLO, **kwargs)
        self.class_map = {int(k): str(v) for k, v in (class_map or {}).items()}
        self.fallback_weights = fallback_weights

    @classmethod
    def from_config(cls, cfg) -> "YoloDetector":
        """Build from the ``perception.yolo`` config section."""
        raw_map = cfg.get("class_map", {}) or {}
        return cls(
            weights=cfg.get("weights", "yolov8n.pt"),
            class_map=dict(raw_map.items()) if hasattr(raw_map, "items") else raw_map,
            fallback_weights=cfg.get("fallback_weights"),
            conf=cfg.get("conf", 0.45),
            iou=cfg.get("iou", 0.50),
            imgsz=cfg.get("imgsz", 640),
            device=cfg.get("device", 0),
            half=cfg.get("half", True),
            max_det=cfg.get("max_det", 20),
        )

    def _resolve_weights(self) -> Optional[str]:
        """Prefer the TensorRT engine, fall back to the PyTorch checkpoint.

        On the Jetson the ``.engine`` is the fast path but it is device- and
        JetPack-specific; on a laptop only the ``.pt`` exists. Trying both means
        one config file works on both machines.
        """
        primary = super()._resolve_weights()
        if primary is not None and Path(primary).is_file():
            return primary
        if self.fallback_weights:
            fallback = Path(self.fallback_weights)
            if fallback.is_file():
                log.info("Engine %s absent, falling back to %s",
                         self.weights, self.fallback_weights)
                return str(fallback)
        return primary

    def _map_class(self, idx: int, label: str) -> Optional[TargetClass]:
        name = self.class_map.get(idx, label).strip().lower()
        if name in ("weed", "weeds"):
            return TargetClass.WEED
        if name in ("crop", "crops", "plant", "seedling"):
            return TargetClass.CROP
        return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference. Returns ``[]`` when the model is unavailable."""
        if frame is None or frame.size == 0 or not self.available:
            return []
        start = time.perf_counter()
        try:
            results = self._predict(frame)
        except Exception as exc:  # pragma: no cover - runtime robustness
            log.error("YOLO inference failed, dropping to colour tier: %s", exc)
            return []
        self.last_inference_ms = (time.perf_counter() - start) * 1e3
        self.inference_count += 1
        return self._parse(results, self._map_class)
