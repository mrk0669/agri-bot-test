"""Tier 3 perception: open-vocabulary zero-shot detection (Section 5.6).

If the time available to collect and label a custom dataset proves short, an
open-vocabulary detector removes the need for a training set entirely.
YOLO-World accepts text prompts such as "red marker", "green plant" or "weed"
and detects them without any task-specific training.

This is the AI layer with no dataset dependency: it is slower and less precise
than a fine-tuned nano model, so it sits behind the colour baseline in the
fusion rule rather than in front of it. Like Tier 2 it degrades to unavailable
rather than raising when the model cannot be loaded.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import Detection, DetectionSource, TargetClass
from ..utils.logging_setup import get_logger
from .yolo_detector import UltralyticsBase

__all__ = ["ZeroShotDetector"]

log = get_logger("vision.zeroshot")


class ZeroShotDetector(UltralyticsBase):
    """Open-vocabulary detector driven by text prompts.

    The prompt list is flattened into a single vocabulary handed to the model,
    and a reverse index maps each returned label back onto ``crop`` or ``weed``.
    Prompt order therefore defines class indices, so the reverse map is built
    from the same flattening rather than assumed.
    """

    def __init__(
        self,
        model: str = "yolov8s-world.pt",
        prompts: Optional[Dict[str, Sequence[str]]] = None,
        conf: float = 0.30,
        **kwargs,
    ):
        super().__init__(model, conf=conf, source=DetectionSource.ZEROSHOT, **kwargs)
        self.prompts: Dict[str, List[str]] = {
            "weed": list((prompts or {}).get("weed", ["weed", "red marker"])),
            "crop": list((prompts or {}).get("crop", ["green plant", "crop seedling"])),
        }
        self._vocabulary: List[str] = []
        self._label_to_class: Dict[str, TargetClass] = {}
        self._build_vocabulary()

    @classmethod
    def from_config(cls, cfg) -> "ZeroShotDetector":
        """Build from the ``perception.zeroshot`` config section."""
        raw = cfg.get("prompts", {}) or {}
        prompts = {
            "weed": list(raw.get("weed", []) or []),
            "crop": list(raw.get("crop", []) or []),
        }
        return cls(
            model=cfg.get("model", "yolov8s-world.pt"),
            prompts=prompts,
            conf=cfg.get("conf", 0.30),
            imgsz=cfg.get("imgsz", 640),
            device=cfg.get("device", 0),
            half=cfg.get("half", False),
        )

    def _build_vocabulary(self) -> None:
        self._vocabulary = []
        self._label_to_class = {}
        for cls_name, target in (("weed", TargetClass.WEED), ("crop", TargetClass.CROP)):
            for prompt in self.prompts.get(cls_name, []):
                text = str(prompt).strip()
                if not text or text.lower() in self._label_to_class:
                    continue
                self._vocabulary.append(text)
                self._label_to_class[text.lower()] = target
        if not self._vocabulary:
            raise ValueError("ZeroShotDetector requires at least one prompt")

    @property
    def vocabulary(self) -> List[str]:
        return list(self._vocabulary)

    def set_prompts(self, weed: Sequence[str], crop: Sequence[str]) -> None:
        """Re-point the detector at a new vocabulary at runtime.

        Useful in the arena: if the markers turn out to be orange rather than
        red, the operator changes the prompt instead of retraining anything.
        """
        self.prompts = {"weed": list(weed), "crop": list(crop)}
        self._build_vocabulary()
        if self._model is not None:
            self._install_vocabulary(self._model)

    def _construct(self, yolo_cls, resolved: str):
        """Load a world model and immediately install the prompt vocabulary."""
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(resolved)
        except Exception:
            # Recent ultralytics exposes world models through the plain YOLO
            # entry point too; fall back rather than declaring unavailable.
            model = yolo_cls(resolved)
        if not hasattr(model, "set_classes"):
            raise RuntimeError(
                f"{resolved} is not an open-vocabulary model (no set_classes)"
            )
        self._install_vocabulary(model)
        return model

    def _install_vocabulary(self, model) -> bool:
        """Encode the prompt vocabulary into the model.

        ``set_classes`` runs a CLIP text encoder. Once the detector has run an
        inference the weights have been moved to the GPU, and the text encoder
        then trips over a CPU/GPU tensor mismatch. That only bites when prompts
        are changed *after* the first frame - which is exactly the arena
        workflow this tier exists to support ("the markers are orange, not
        red"). Retrying on CPU keeps re-prompting available at runtime; the
        next predict moves the weights back.
        """
        try:
            model.set_classes(self._vocabulary)
            return True
        except RuntimeError as exc:
            if "device" not in str(exc).lower():
                log.warning("set_classes failed: %s", exc)
                return False
            try:
                model.to("cpu")
                model.set_classes(self._vocabulary)
                log.debug("re-prompted on CPU after a device mismatch")
                return True
            except Exception as retry_exc:  # pragma: no cover
                log.warning("set_classes failed after CPU retry: %s", retry_exc)
                return False
        except Exception as exc:  # pragma: no cover
            log.warning("set_classes failed: %s", exc)
            return False

    def _map_class(self, idx: int, label: str) -> Optional[TargetClass]:
        key = str(label).strip().lower()
        if key in self._label_to_class:
            return self._label_to_class[key]
        # Ultralytics may return the index rather than the text when the model
        # was re-prompted after load; fall back to positional lookup.
        if 0 <= idx < len(self._vocabulary):
            return self._label_to_class.get(self._vocabulary[idx].lower())
        return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run open-vocabulary inference. Returns ``[]`` when unavailable."""
        if frame is None or frame.size == 0 or not self.available:
            return []
        start = time.perf_counter()
        try:
            results = self._predict(frame)
        except Exception as exc:  # pragma: no cover
            log.error("Zero-shot inference failed: %s", exc)
            return []
        self.last_inference_ms = (time.perf_counter() - start) * 1e3
        self.inference_count += 1
        return self._parse(results, self._map_class)
