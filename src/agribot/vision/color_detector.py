"""Tier 1 perception: deterministic colour and geometry (Section 5.6).

This is the primary point-scoring path. It recognises the challenge's green
crop markers and red weed markers by HSV thresholding followed by area and
shape gating. It requires no training, runs at very high frame rate, and
returns each target's pixel coordinates directly for aiming - which is why the
system remains robust in an unfamiliar arena.

Two implementation points matter:

* **Red wraps the hue circle.** Red occupies both ends of the OpenCV H range
  (0-10 and 170-180), so it needs two thresholds OR-ed together. A single range
  silently misses half the red markers depending on lighting.
* **Geometry gating is what separates a marker from a red object.** Area alone
  admits a stray glove or a competitor's chassis. Extent (contour area over
  bounding-rect area), aspect ratio and solidity (contour area over convex-hull
  area) together reject shapes that are not compact quadrilateral-ish markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import BBox, Detection, DetectionSource, TargetClass

__all__ = [
    "ColorClassSpec", "ColorDetector", "RejectedRegion", "build_mask_from_ranges",
]


@dataclass
class RejectedRegion:
    """A region that matched on colour but failed a geometry gate.

    Colour alone cannot tell a marker from a red shirt; the geometry gates can,
    and this records which gate did it and by how much. That turns "the robot
    did not see my marker" from a guessing game into a lookup during venue
    calibration, and it is what lets the detector *show* its reasoning rather
    than only its conclusions.
    """

    cls: TargetClass
    bbox: BBox
    reason: str
    area_px: float
    extent: float
    aspect: float
    solidity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cls": self.cls.value,
            "bbox": list(self.bbox.as_tuple()),
            "reason": self.reason,
            "area_px": round(self.area_px, 1),
            "extent": round(self.extent, 3),
            "aspect": round(self.aspect, 3),
            "solidity": round(self.solidity, 3),
        }


@dataclass
class ColorClassSpec:
    """HSV ranges plus geometry gates for one marker class."""

    hsv_ranges: List[Tuple[Sequence[int], Sequence[int]]]
    min_area_px: float = 600.0
    max_area_px: float = 120000.0
    min_extent: float = 0.35
    min_aspect: float = 0.35
    max_aspect: float = 3.0
    min_solidity: float = 0.70

    @classmethod
    def from_config(cls, cfg) -> "ColorClassSpec":
        ranges = [(tuple(r["lower"]), tuple(r["upper"])) for r in cfg.hsv_ranges]
        return cls(
            hsv_ranges=ranges,
            min_area_px=cfg.get("min_area_px", 600),
            max_area_px=cfg.get("max_area_px", 120000),
            min_extent=cfg.get("min_extent", 0.35),
            min_aspect=cfg.get("min_aspect", 0.35),
            max_aspect=cfg.get("max_aspect", 3.0),
            min_solidity=cfg.get("min_solidity", 0.70),
        )


def build_mask_from_ranges(
    hsv: np.ndarray,
    ranges: Sequence[Tuple[Sequence[int], Sequence[int]]],
) -> np.ndarray:
    """OR together several HSV ranges into one binary mask.

    Required for red, which straddles the H=0/180 seam.
    """
    if not ranges:
        raise ValueError("at least one HSV range is required")
    mask = None
    for lower, upper in ranges:
        part = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


@dataclass
class ColorDetector:
    """Deterministic HSV + geometry detector for crop and weed markers."""

    weed: ColorClassSpec
    crop: ColorClassSpec
    morph_kernel: int = 5
    blur_kernel: int = 5
    max_detections_per_class: int = 10

    @classmethod
    def from_config(cls, cfg) -> "ColorDetector":
        """Build from the ``perception.color`` config section."""
        return cls(
            weed=ColorClassSpec.from_config(cfg.weed),
            crop=ColorClassSpec.from_config(cfg.crop),
            morph_kernel=cfg.get("morph_kernel", 5),
            blur_kernel=cfg.get("blur_kernel", 5),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect all crop and weed markers in one BGR frame."""
        return self.detect_with_diagnostics(frame)[0]

    def detect_with_diagnostics(
        self, frame: np.ndarray
    ) -> Tuple[List[Detection], List[RejectedRegion]]:
        """Detect, and also report the colour matches the geometry gates threw out.

        Used by the calibration tooling and by the figures in the proposal:
        being able to point at a rejected region and name the gate that
        rejected it is what makes the geometry stage inspectable.
        """
        if frame is None or frame.size == 0:
            return [], []

        work = frame
        if self.blur_kernel and self.blur_kernel >= 3:
            k = self.blur_kernel | 1
            work = cv2.GaussianBlur(work, (k, k), 0)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        detections: List[Detection] = []
        rejected: List[RejectedRegion] = []
        detections.extend(self._detect_class(hsv, self.weed, TargetClass.WEED, rejected))
        detections.extend(self._detect_class(hsv, self.crop, TargetClass.CROP, rejected))
        return detections, rejected

    def class_mask(self, frame: np.ndarray, cls: TargetClass) -> np.ndarray:
        """Binary mask for one class - used by the HSV calibration tool."""
        spec = self.weed if cls is TargetClass.WEED else self.crop
        work = frame
        if self.blur_kernel and self.blur_kernel >= 3:
            k = self.blur_kernel | 1
            work = cv2.GaussianBlur(work, (k, k), 0)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        return self._clean(build_mask_from_ranges(hsv, spec.hsv_ranges))

    # -- internals ----------------------------------------------------------
    def _clean(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_kernel and self.morph_kernel >= 3:
            k = self.morph_kernel | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _detect_class(
        self,
        hsv: np.ndarray,
        spec: ColorClassSpec,
        cls: TargetClass,
        rejected: Optional[List[RejectedRegion]] = None,
    ) -> List[Detection]:
        mask = self._clean(build_mask_from_ranges(hsv, spec.hsv_ranges))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def reject(contour, area, extent, aspect, solidity, reason) -> None:
            if rejected is None:
                return
            x, y, w, h = cv2.boundingRect(contour)
            rejected.append(RejectedRegion(
                cls=cls, bbox=BBox.from_xywh(float(x), float(y), float(w), float(h)),
                reason=reason, area_px=area, extent=extent,
                aspect=aspect, solidity=solidity,
            ))

        out: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            extent = area / float(w * h)
            aspect = w / float(h)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0

            if area < spec.min_area_px:
                # Too small to be a marker at any working distance - this is
                # speckle, and reporting it would drown the useful rejections.
                continue
            if area > spec.max_area_px:
                reject(contour, area, extent, aspect, solidity,
                       f"area {area:.0f} px > max {spec.max_area_px:.0f}")
                continue
            if extent < spec.min_extent:
                reject(contour, area, extent, aspect, solidity,
                       f"extent {extent:.2f} < {spec.min_extent:.2f}")
                continue
            if not (spec.min_aspect <= aspect <= spec.max_aspect):
                reject(contour, area, extent, aspect, solidity,
                       f"aspect {aspect:.2f} outside "
                       f"[{spec.min_aspect:.2f}, {spec.max_aspect:.2f}]")
                continue
            if solidity < spec.min_solidity:
                reject(contour, area, extent, aspect, solidity,
                       f"solidity {solidity:.2f} < {spec.min_solidity:.2f}")
                continue

            out.append(Detection(
                cls=cls,
                bbox=BBox.from_xywh(float(x), float(y), float(w), float(h)),
                confidence=self._confidence(extent, solidity),
                source=DetectionSource.COLOR,
                area_px=area,
                meta={
                    "extent": round(extent, 3),
                    "aspect": round(aspect, 3),
                    "solidity": round(solidity, 3),
                },
            ))

        # Largest first: the nearest marker is the one the robot must act on.
        out.sort(key=lambda d: d.area_px, reverse=True)
        return out[: self.max_detections_per_class]

    @staticmethod
    def _confidence(extent: float, solidity: float) -> float:
        """Map shape quality onto a confidence comparable with the learned tier.

        A perfectly filled, perfectly convex blob scores 1.0. The floor of 0.5
        reflects that a detection which already passed every geometry gate is
        never weak evidence - the gates, not this score, do the rejecting.
        """
        quality = 0.5 * (min(extent, 1.0) + min(solidity, 1.0))
        return float(np.clip(0.5 + 0.5 * quality, 0.0, 1.0))
