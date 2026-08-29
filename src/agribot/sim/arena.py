"""Software-in-the-loop arena.

Couples the :class:`~agribot.hal.mock_mcu.MockMcu` physics to the synthetic
field renderer through a proper ground-plane camera projection, producing the
frames the robot would actually see from its current pose.

That closes the loop: the runtime steers on frames rendered from a pose that
its own drive commands produced. A test can therefore assert that the robot
*converged onto the line* and *sprayed the weed but not the crop*, which no
amount of unit testing of the parts can establish.

The projection is the standard pinhole model over a flat ground plane, and is
the exact inverse of :func:`agribot.utils.geometry.ground_range_from_pixel`,
so the targeting code and the simulator agree on where a marker at a given
range appears in the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import TargetClass
from ..vision.camera import CameraBase
from .field import ArenaStyle, Marker, render_frame

__all__ = ["ArenaMarker", "ArenaLayout", "SimulatedArena", "SimulatedCamera"]


@dataclass
class ArenaMarker:
    """A marker placed in world coordinates along the row.

    ``along_m`` is the distance from the row start; ``lateral_m`` is the offset
    from the guidance line, positive to the right of the direction of travel.
    """

    cls: TargetClass
    along_m: float
    lateral_m: float = 0.0
    height_m: float = 0.0        # 0 = on the floor, up to 0.15 = elevated
    size_m: float = 0.05


@dataclass
class ArenaLayout:
    """The full arena: one row of markers plus rendering nuisance parameters."""

    row_length_m: float = 6.0
    markers: List[ArenaMarker] = field(default_factory=list)
    style: ArenaStyle = field(default_factory=ArenaStyle)
    line_width_m: float = 0.02
    # Distance past the row end at which the guidance line stops being drawn.
    line_ends_at_m: Optional[float] = None

    @staticmethod
    def default_demo() -> "ArenaLayout":
        """A short demonstration row: two weeds, two crops, one elevated weed."""
        return ArenaLayout(
            row_length_m=3.0,
            markers=[
                ArenaMarker(TargetClass.CROP, along_m=0.60, lateral_m=-0.07),
                ArenaMarker(TargetClass.WEED, along_m=1.00, lateral_m=0.06),
                ArenaMarker(TargetClass.CROP, along_m=1.45, lateral_m=0.07),
                ArenaMarker(TargetClass.WEED, along_m=1.90, lateral_m=-0.06),
                ArenaMarker(TargetClass.WEED, along_m=2.40, lateral_m=0.05,
                            height_m=0.15),
            ],
            style=ArenaStyle(soil_speckle=16, brightness_gradient=0.25, seed=11),
        )


class SimulatedArena:
    """Renders the camera view for a given robot pose."""

    def __init__(
        self,
        layout: ArenaLayout,
        width: int = 640,
        height: int = 480,
        camera_height_m: float = 0.22,
        mount_pitch_deg: float = 35.0,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ):
        self.layout = layout
        self.width = int(width)
        self.height = int(height)
        self.camera_height_m = camera_height_m
        self.mount_pitch_rad = math.radians(mount_pitch_deg)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_config(cls, cfg, layout: ArenaLayout) -> "SimulatedArena":
        cam = cfg.camera
        return cls(
            layout,
            width=cam.get("width", 640),
            height=cam.get("height", 480),
            camera_height_m=cam.get("mount_height_m", 0.22),
            mount_pitch_deg=cam.get("mount_pitch_deg", 35.0),
            fx=cam.get("fx", 615.0),
            fy=cam.get("fy", 615.0),
            cx=cam.get("cx", 320.0),
            cy=cam.get("cy", 240.0),
        )

    # -- projection ---------------------------------------------------------
    def project(
        self, forward_m: float, lateral_m: float, height_m: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        """Project a world point onto the image. None if behind or out of view.

        ``forward_m`` is distance ahead of the camera, ``lateral_m`` is offset
        to the right, ``height_m`` is height above the ground plane.
        """
        if forward_m <= 1e-3:
            return None

        drop = self.camera_height_m - height_m
        # Angle of the point below the horizontal through the optical centre.
        depression = math.atan2(drop, forward_m)
        ray_below_axis = depression - self.mount_pitch_rad
        # Beyond +/-80 deg from the axis the tangent explodes and the point is
        # far outside any real lens field of view.
        if abs(ray_below_axis) > math.radians(80.0):
            return None

        v = self.cy + self.fy * math.tan(ray_below_axis)
        slant = math.hypot(forward_m, drop)
        u = self.cx + self.fx * lateral_m / slant

        if not (-self.width <= u <= 2 * self.width):
            return None
        return u, v

    def pixels_per_metre_at(self, forward_m: float) -> float:
        """Lateral image scale at a given forward distance."""
        drop = self.camera_height_m
        slant = math.hypot(max(forward_m, 1e-3), drop)
        return self.fx / slant

    # -- rendering ----------------------------------------------------------
    def render(
        self,
        along_m: float,
        lateral_m: float,
        heading_rad: float,
    ) -> np.ndarray:
        """Render the view from a robot at the given pose.

        Args:
            along_m: distance travelled down the row.
            lateral_m: displacement from the guidance line, positive = robot
                is to the right of the line.
            heading_rad: heading relative to the row, positive = yawed left.
        """
        # -- guidance line --------------------------------------------------
        bottom_range = self._range_at_row(self.height)
        scale = self.pixels_per_metre_at(bottom_range)

        line_end = self.layout.line_ends_at_m
        if line_end is None:
            line_end = self.layout.row_length_m
        draw_line = along_m < line_end

        line_width_px = max(6, int(self.layout.line_width_m * scale))
        line_points = self._project_row(along_m, lateral_m, heading_rad, line_end)

        # -- markers in view ------------------------------------------------
        markers: List[Marker] = []
        for arena_marker in self.layout.markers:
            forward = arena_marker.along_m - along_m
            if forward <= 0.02 or forward > 1.2:
                continue
            # Express the marker in the robot's frame.
            rel_lateral = arena_marker.lateral_m - lateral_m
            rotated_forward = (forward * math.cos(heading_rad)
                               + rel_lateral * math.sin(heading_rad))
            rotated_lateral = (-forward * math.sin(heading_rad)
                               + rel_lateral * math.cos(heading_rad))
            projected = self.project(rotated_forward, rotated_lateral,
                                     arena_marker.height_m)
            if projected is None:
                continue
            u, v = projected
            size_px = arena_marker.size_m * self.pixels_per_metre_at(rotated_forward)
            if size_px < 24 or not (0 <= v <= self.height):
                continue
            markers.append(Marker(arena_marker.cls, u, v, size_px,
                                  elevated=arena_marker.height_m > 0.05))

        return render_frame(
            width=self.width,
            height=self.height,
            line_width_px=line_width_px,
            markers=markers,
            style=self.layout.style,
            draw_line=draw_line and len(line_points) >= 2,
            line_points=line_points,
        )

    def _project_row(
        self,
        along_m: float,
        lateral_m: float,
        heading_rad: float,
        line_end_m: float,
    ) -> List[Tuple[float, float]]:
        """Project the guidance line into the image as a perspective polyline.

        For a point at forward distance ``f`` along the robot's own axis, the
        row (which lies at world lateral zero) sits at robot-frame lateral
        offset ``X = (f*sin(h) - lateral) / cos(h)``. Sampling ``f`` and
        projecting each ``(f, X)`` reproduces both the perspective convergence
        and the correct direction of tilt.

        Getting that tilt direction right matters: when the robot is yawed
        *left*, points further down the row appear further *right* in the
        image. A renderer that leans the line the other way trains the control
        loop to steer into its own error, and the closed loop diverges.
        """
        cos_h = math.cos(heading_rad)
        if abs(cos_h) < 1e-3:
            return []

        points: List[Tuple[float, float]] = []
        # Sample from just in front of the wheels out to the far edge of view.
        for i in range(41):
            forward = 0.05 + i * (1.10 - 0.05) / 40.0
            if along_m + forward > line_end_m:
                break
            lateral_offset = (forward * math.sin(heading_rad) - lateral_m) / cos_h
            projected = self.project(forward, lateral_offset, 0.0)
            if projected is None:
                continue
            u, v = projected
            if -2 * self.width <= u <= 3 * self.width and -self.height <= v <= 2 * self.height:
                points.append((u, v))
        return points

    def _range_at_row(self, pixel_row: float) -> float:
        """Ground range imaged at a pixel row - inverse of :meth:`project`."""
        ray_below_axis = math.atan2(pixel_row - self.cy, self.fy)
        depression = self.mount_pitch_rad + ray_below_axis
        if depression <= 1e-6:
            return math.inf
        return self.camera_height_m / math.tan(depression)


class SimulatedCamera(CameraBase):
    """Camera backend that renders from a live pose source.

    ``pose_fn`` returns ``(along_m, lateral_m, heading_rad)``; it is called on
    every ``read()``, so the frames follow whatever the drive commands did.
    """

    def __init__(self, arena: SimulatedArena, pose_fn, max_frames: Optional[int] = None):
        self.arena = arena
        self.pose_fn = pose_fn
        self.max_frames = max_frames
        self.width = arena.width
        self.height = arena.height
        self._count = 0
        self._open = False

    def open(self) -> bool:
        self._open = True
        self._count = 0
        return True

    @property
    def is_open(self) -> bool:
        return self._open

    def read(self):
        if not self._open:
            return False, None
        if self.max_frames is not None and self._count >= self.max_frames:
            self._open = False
            return False, None
        self._count += 1
        along, lateral, heading = self.pose_fn()
        return True, self.arena.render(along, lateral, heading)

    def release(self) -> None:
        self._open = False
