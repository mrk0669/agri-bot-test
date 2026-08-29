"""Shared value types passed between AgriBot subsystems.

These are deliberately plain dataclasses with no behaviour beyond simple
derived properties and serialisation. Every boundary in the stack
(perception -> fusion -> targeting -> actuation -> telemetry) exchanges one of
these, which keeps the modules independently testable.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "TargetClass",
    "DetectionSource",
    "BBox",
    "Detection",
    "Track",
    "LineObservation",
    "DriveCommand",
    "AimSolution",
    "SprayEvent",
    "ImuSample",
    "EncoderSample",
    "RangeSample",
    "McuTelemetry",
    "FusedState",
]


class TargetClass(str, Enum):
    """What a detector believes it is looking at."""

    CROP = "crop"
    WEED = "weed"
    UNKNOWN = "unknown"


class DetectionSource(str, Enum):
    """Which perception tier produced a detection (Section 5.6)."""

    COLOR = "color"          # Tier 1 - deterministic HSV + geometry
    YOLO = "yolo"            # Tier 2 - trained nano detector
    ZEROSHOT = "zeroshot"    # Tier 3 - open-vocabulary, no training data
    FUSED = "fused"          # output of the late-fusion rule


@dataclass(frozen=True)
class BBox:
    """Axis-aligned pixel box, ``(x1, y1)`` top-left, ``(x2, y2)`` bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"degenerate bbox: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def iou(self, other: "BBox") -> float:
        """Intersection over union with another box; 0.0 if disjoint."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def distance_to(self, other: "BBox") -> float:
        """Euclidean distance between centroids, in pixels."""
        return math.hypot(self.cx - other.cx, self.cy - other.cy)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(x, y, x + w, y + h)


@dataclass
class Detection:
    """A single classified region in one frame."""

    cls: TargetClass
    bbox: BBox
    confidence: float
    source: DetectionSource
    area_px: float = 0.0
    # Free-form per-source diagnostics (extent, solidity, hsv range index...).
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.bbox.centroid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cls": self.cls.value,
            "bbox": list(self.bbox.as_tuple()),
            "confidence": round(float(self.confidence), 4),
            "source": self.source.value,
            "area_px": round(float(self.area_px), 1),
            "meta": self.meta,
        }


@dataclass
class Track:
    """A detection persisted across frames, used to confirm before acting.

    Acting on a single frame is how a system sprays a glint of sunlight. A
    target must be seen ``confirm_frames`` times in a row before the mission
    state machine will interrupt travel for it.
    """

    track_id: int
    cls: TargetClass
    bbox: BBox
    confidence: float
    source: DetectionSource
    hits: int = 1
    age: int = 0            # frames since last seen
    sprayed: bool = False
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.bbox.centroid

    def is_confirmed(self, confirm_frames: int) -> bool:
        return self.hits >= confirm_frames

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "cls": self.cls.value,
            "bbox": list(self.bbox.as_tuple()),
            "confidence": round(float(self.confidence), 4),
            "source": self.source.value,
            "hits": self.hits,
            "age": self.age,
            "sprayed": self.sprayed,
        }


@dataclass
class LineObservation:
    """Output of the guidance-line extractor for one frame (Section 5.2)."""

    found: bool
    # Normalised horizontal error in [-1, +1]; negative = line left of centre.
    error: float = 0.0
    centroid_px: Optional[Tuple[float, float]] = None
    mask_area_px: float = 0.0
    roi_shape: Tuple[int, int] = (0, 0)   # (height, width) of the ROI
    confidence: float = 0.0               # mask_area / roi_area, clipped
    # Which sensor produced this: "vision" (primary, rules-compliant) or "ir"
    # (the silent fail-safe). Recorded so a run can always be audited for what
    # the robot was actually steering on.
    source: str = "vision"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "error": round(float(self.error), 4),
            "centroid_px": list(self.centroid_px) if self.centroid_px else None,
            "mask_area_px": round(float(self.mask_area_px), 1),
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class DriveCommand:
    """Normalised wheel commands in [-1, +1]; the MCU scales these to PWM."""

    left: float
    right: float

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            v = getattr(self, name)
            if not -1.0001 <= v <= 1.0001:
                raise ValueError(f"DriveCommand.{name}={v} outside [-1, 1]")

    @property
    def linear(self) -> float:
        """Mean of the two sides - proportional to forward speed."""
        return 0.5 * (self.left + self.right)

    @property
    def differential(self) -> float:
        """Right minus left - proportional to yaw rate."""
        return self.right - self.left

    @staticmethod
    def stopped() -> "DriveCommand":
        return DriveCommand(0.0, 0.0)

    def to_dict(self) -> Dict[str, float]:
        return {"left": round(self.left, 4), "right": round(self.right, 4)}


@dataclass(frozen=True)
class AimSolution:
    """Pan/tilt angles solved from a target's pixel coordinates (Section 5.7)."""

    pan_deg: float
    tilt_deg: float
    in_range: bool                     # False if the raw solution was clamped
    clamped: bool = False
    range_m: Optional[float] = None    # from depth camera, when available
    source_px: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pan_deg": round(self.pan_deg, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "in_range": self.in_range,
            "clamped": self.clamped,
            "range_m": round(self.range_m, 4) if self.range_m is not None else None,
            "source_px": list(self.source_px) if self.source_px else None,
        }


@dataclass
class SprayEvent:
    """One logged intervention - the unit of the sustainability metric.

    ``volume_ml`` is measured by the in-line flow sensor where one is fitted,
    and falls back to the nominal per-burst dose otherwise; ``measured`` says
    which, so that reported ml/weed is never silently an estimate.
    """

    event_id: int
    track_id: int
    timestamp: float
    aim: AimSolution
    burst_ms: float
    volume_ml: float
    measured: bool
    mode: str = "spray"                # "spray" | "mark"
    distance_m: float = 0.0            # odometry at the moment of the burst
    detector_source: str = "fused"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["aim"] = self.aim.to_dict()
        d["timestamp"] = round(self.timestamp, 4)
        d["volume_ml"] = round(self.volume_ml, 4)
        return d


# ---------------------------------------------------------------------------
# Raw sensor samples arriving from the MCU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImuSample:
    """Body-frame IMU sample. Rates in rad/s, accelerations in m/s^2."""

    t: float
    gyro_z: float
    accel_x: float
    mag_heading_rad: Optional[float] = None
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class EncoderSample:
    """Wheel-encoder derived velocities, metres per second."""

    t: float
    left_mps: float
    right_mps: float

    @property
    def linear_mps(self) -> float:
        return 0.5 * (self.left_mps + self.right_mps)


@dataclass(frozen=True)
class RangeSample:
    """Ultrasonic range readings in metres; ``inf`` for no echo."""

    t: float
    front_m: float
    front_left_m: float = math.inf

    @property
    def min_m(self) -> float:
        return min(self.front_m, self.front_left_m)


@dataclass
class McuTelemetry:
    """One decoded telemetry frame from the microcontroller."""

    t: float
    imu: Optional[ImuSample] = None
    encoders: Optional[EncoderSample] = None
    ranges: Optional[RangeSample] = None
    flow_ticks: int = 0
    battery_v: float = 0.0
    ir_array: Tuple[int, ...] = ()
    seq: int = 0
    ok: bool = True


@dataclass
class FusedState:
    """Output of the Kalman stack - the robot's belief about itself."""

    t: float
    heading_rad: float
    heading_var: float
    gyro_bias_rad_s: float
    distance_m: float
    velocity_mps: float
    accel_bias_mps2: float
    distance_var: float
    encoder_rejected: bool = False

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.heading_rad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 4),
            "heading_deg": round(self.heading_deg, 3),
            "gyro_bias_deg_s": round(math.degrees(self.gyro_bias_rad_s), 4),
            "distance_m": round(self.distance_m, 4),
            "velocity_mps": round(self.velocity_mps, 4),
            "accel_bias_mps2": round(self.accel_bias_mps2, 4),
            "encoder_rejected": self.encoder_rejected,
        }
