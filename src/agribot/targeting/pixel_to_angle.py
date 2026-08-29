"""Pixel-to-actuator mapping for the pan/tilt spray head (Section 5.7).

Once a weed target's pixel coordinates are confirmed by the fusion rule they
are converted into actuator commands. Following the approach validated in
published edge-device precision-spraying work, the pixel position is mapped
linearly onto the aiming actuator: horizontal pixel offset maps to pan angle,
vertical pixel offset maps to tilt angle.

The linear model is not an approximation of convenience - it is what a fixed,
rigid camera and a fixed nozzle geometry actually produce over the small
working envelope of this robot, and its two coefficients are directly
measurable by ``tools/calibrate_spray.py`` without needing full camera
intrinsics.

**Elevated markers.** The rules allow signs on the floor or on a surface of
about fifteen centimetres. A pure image-to-angle map implicitly assumes every
target lies on the ground plane, and that assumption is what breaks on a raised
marker: the same pixel row corresponds to a different physical range, so the
tilt solution is wrong. When a depth camera supplies a true range, the tilt is
re-solved from the geometry instead. Where no depth is available the
ground-plane solution is used and flagged, so the log records which markers
were engaged on an assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..types import AimSolution
from ..utils.geometry import clamp, ground_range_from_pixel

__all__ = ["AxisCalibration", "PixelToAngleSolver"]


@dataclass
class AxisCalibration:
    """Linear pixel-offset -> servo-angle model for one axis."""

    centre_deg: float
    deg_per_px: float
    min_deg: float
    max_deg: float
    invert: bool = False
    servo_channel: int = 0

    def __post_init__(self) -> None:
        if self.min_deg >= self.max_deg:
            raise ValueError(
                f"axis limits inverted: min={self.min_deg} max={self.max_deg}"
            )
        if not self.min_deg <= self.centre_deg <= self.max_deg:
            raise ValueError(
                f"centre_deg={self.centre_deg} outside [{self.min_deg}, {self.max_deg}]"
            )

    @classmethod
    def from_config(cls, cfg) -> "AxisCalibration":
        return cls(
            centre_deg=cfg.centre_deg,
            deg_per_px=cfg.deg_per_px,
            min_deg=cfg.min_deg,
            max_deg=cfg.max_deg,
            invert=cfg.get("invert", False),
            servo_channel=cfg.get("servo_channel", 0),
        )

    def solve(self, pixel_offset: float) -> Tuple[float, bool]:
        """Map a signed pixel offset to an angle. Returns ``(deg, clamped)``."""
        sign = -1.0 if self.invert else 1.0
        raw = self.centre_deg + sign * self.deg_per_px * pixel_offset
        limited = clamp(raw, self.min_deg, self.max_deg)
        return limited, (abs(limited - raw) > 1e-9)


@dataclass
class PixelToAngleSolver:
    """Solves pan/tilt angles for a target at a given pixel coordinate."""

    pan: AxisCalibration
    tilt: AxisCalibration
    frame_width: int = 640
    frame_height: int = 480
    # Camera geometry, used only for the depth-corrected tilt path.
    camera_height_m: float = 0.22
    mount_pitch_deg: float = 35.0
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    use_depth_when_available: bool = True
    marker_height_max_m: float = 0.15
    # Nozzle pivot offset below the camera optical centre, metres.
    nozzle_drop_m: float = 0.06

    @classmethod
    def from_config(cls, targeting_cfg, camera_cfg) -> "PixelToAngleSolver":
        """Build from the ``targeting`` and ``camera`` config sections."""
        return cls(
            pan=AxisCalibration.from_config(targeting_cfg.pan),
            tilt=AxisCalibration.from_config(targeting_cfg.tilt),
            frame_width=camera_cfg.get("width", 640),
            frame_height=camera_cfg.get("height", 480),
            camera_height_m=camera_cfg.get("mount_height_m", 0.22),
            mount_pitch_deg=camera_cfg.get("mount_pitch_deg", 35.0),
            fx=camera_cfg.get("fx", 615.0),
            fy=camera_cfg.get("fy", 615.0),
            cx=camera_cfg.get("cx", 320.0),
            cy=camera_cfg.get("cy", 240.0),
            use_depth_when_available=targeting_cfg.tilt.get(
                "use_depth_when_available", True),
            marker_height_max_m=targeting_cfg.tilt.get("marker_height_max_m", 0.15),
        )

    # -- main entry point ---------------------------------------------------
    def solve(
        self,
        pixel_x: float,
        pixel_y: float,
        range_m: Optional[float] = None,
    ) -> AimSolution:
        """Solve the aiming angles for a target centroid.

        Args:
            pixel_x, pixel_y: target centroid in full-frame pixel coordinates.
            range_m: true range from a depth camera, if fitted.
        """
        dx = pixel_x - (self.frame_width / 2.0)
        dy = pixel_y - (self.frame_height / 2.0)

        pan_deg, pan_clamped = self.pan.solve(dx)

        if range_m is not None and self.use_depth_when_available and range_m > 0:
            tilt_deg, tilt_clamped = self._tilt_from_range(pixel_x, pixel_y, range_m)
        else:
            tilt_deg, tilt_clamped = self.tilt.solve(dy)

        clamped = pan_clamped or tilt_clamped
        return AimSolution(
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            in_range=not clamped,
            clamped=clamped,
            range_m=range_m,
            source_px=(pixel_x, pixel_y),
        )

    # -- depth-corrected tilt ----------------------------------------------
    def _tilt_from_range(
        self, pixel_x: float, pixel_y: float, range_m: float
    ) -> Tuple[float, bool]:
        """Re-solve tilt from a measured range instead of the ground plane.

        The measured range gives the target's true distance along the camera
        ray. Combined with the ray's angle below the optical axis, that fixes
        the target's height, which is the quantity the ground-plane assumption
        gets wrong for an elevated marker.
        """
        ray_below_axis = math.atan2(pixel_y - self.cy, self.fy)
        depression_from_horizontal = math.radians(self.mount_pitch_deg) + ray_below_axis

        # Decompose the range into horizontal reach and drop below the camera.
        horizontal_m = range_m * math.cos(depression_from_horizontal)
        drop_m = range_m * math.sin(depression_from_horizontal)

        target_height_m = self.camera_height_m - drop_m
        # Physically bound the result: a marker cannot be below the floor, nor
        # above the maximum elevated surface the rules permit.
        target_height_m = clamp(target_height_m, 0.0, self.marker_height_max_m)

        nozzle_height_m = self.camera_height_m - self.nozzle_drop_m
        vertical_m = nozzle_height_m - target_height_m

        if horizontal_m <= 1e-6:
            nozzle_depression = math.pi / 2.0
        else:
            nozzle_depression = math.atan2(vertical_m, horizontal_m)

        # Express the geometric depression on the tilt servo's own scale: the
        # servo centre corresponds to the mount pitch, and one degree of
        # geometry is one degree of servo.
        delta_deg = math.degrees(nozzle_depression) - self.mount_pitch_deg
        sign = -1.0 if self.tilt.invert else 1.0
        raw = self.tilt.centre_deg + sign * delta_deg
        limited = clamp(raw, self.tilt.min_deg, self.tilt.max_deg)
        return limited, (abs(limited - raw) > 1e-9)

    # -- diagnostics --------------------------------------------------------
    def estimate_ground_range(self, pixel_y: float) -> float:
        """Ground-plane range to the pixel row - the fallback depth estimate."""
        return ground_range_from_pixel(
            pixel_y,
            self.frame_height,
            self.camera_height_m,
            self.mount_pitch_deg,
            self.fy,
            self.cy,
        )

    def centre_solution(self) -> AimSolution:
        """The stowed / home aim, used between targets."""
        return AimSolution(
            pan_deg=self.pan.centre_deg,
            tilt_deg=self.tilt.centre_deg,
            in_range=True,
            clamped=False,
        )
