"""Synthetic arena renderer.

Generates frames that look enough like the competition field for the real
perception code to run on them: a textured soil background, a light guidance
line, and green crop / red weed markers.

This is what lets navigation and perception be verified without hardware. The
frames are rendered with the same colours the config thresholds target, so a
test failure means the *code* broke, not that the test fixture drifted.

The renderer models the two things that actually break field vision:
uneven illumination (a brightness gradient plus a specular glare patch) and
soil speckle. Both are configurable, so a test can assert that the pipeline
still finds the line under conditions a fixed intensity threshold would fail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import TargetClass

__all__ = ["Marker", "ArenaStyle", "render_frame", "render_line_sequence"]

# Colours chosen to sit inside the config's default HSV gates.
SOIL_BGR = (58, 78, 108)        # dark brown: high saturation, low value
LINE_BGR = (235, 238, 240)      # near-white: low saturation, high value
WEED_BGR = (40, 40, 205)        # red marker
CROP_BGR = (62, 178, 72)        # green marker


@dataclass
class Marker:
    """A marker placed in the frame."""

    cls: TargetClass
    cx: float
    cy: float
    size: float = 70.0
    elevated: bool = False       # placed on the ~15 cm raised surface

    def bbox(self) -> Tuple[int, int, int, int]:
        half = self.size / 2.0
        return (
            int(self.cx - half), int(self.cy - half),
            int(self.cx + half), int(self.cy + half),
        )


@dataclass
class ArenaStyle:
    """Rendering nuisance parameters - the things that break naive thresholds."""

    soil_speckle: float = 18.0        # per-pixel noise sigma on the soil
    brightness_gradient: float = 0.35  # fractional falloff left-to-right
    glare_strength: float = 0.0        # 0..1 specular patch on the line
    glare_centre: Tuple[float, float] = (0.5, 0.8)
    blur: float = 0.0                  # motion blur kernel in px, 0 = none
    seed: int = 0


@lru_cache(maxsize=8)
def _soil_base(width: int, height: int, speckle: float, seed: int) -> np.ndarray:
    """Soil background with fixed speckle. Cached; callers must copy."""
    base = np.full((height, width, 3), SOIL_BGR, dtype=np.uint8)
    if speckle > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, speckle, (height, width, 1))
        base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return base


@lru_cache(maxsize=8)
def _brightness_ramp(width: int, gradient: float) -> np.ndarray:
    """Left-to-right illumination falloff, cached per width."""
    return np.linspace(1.0, 1.0 - gradient, width, dtype=np.float32)


@lru_cache(maxsize=8)
def _glare_field(width: int, height: int, gx: int, gy: int, radius: int) -> np.ndarray:
    """Normalised specular glare patch, cached per geometry."""
    overlay = np.zeros((height, width), dtype=np.float32)
    cv2.circle(overlay, (gx, gy), radius, 1.0, -1)
    return cv2.GaussianBlur(overlay, (0, 0), radius / 2.0)


def render_frame(
    width: int = 640,
    height: int = 480,
    line_offset_px: float = 0.0,
    line_angle_deg: float = 0.0,
    line_width_px: int = 46,
    markers: Optional[Sequence[Marker]] = None,
    style: Optional[ArenaStyle] = None,
    draw_line: bool = True,
    line_points: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Render one synthetic arena frame.

    Args:
        line_offset_px: horizontal offset of the line at the *bottom* of the
            frame, relative to the frame centre. Positive = line to the right,
            which the controller must correct by steering right.
        line_angle_deg: tilt of the line; positive leans right going up the
            frame, i.e. the robot is yawed relative to the row.
        markers: crop and weed markers to draw.
        draw_line: set False to simulate a lost line (row end, gap).
        line_points: explicit image-space polyline for the guidance line. When
            supplied it overrides ``line_offset_px``/``line_angle_deg``. The
            arena renderer passes the true perspective projection of the row
            this way; a straight line with a single tilt angle is only a
            small-angle approximation and gets the curvature wrong at the
            large heading errors a recovery manoeuvre produces.
    """
    style = style or ArenaStyle()

    # The soil layer depends only on size, speckle and seed, and a simulated
    # mission renders thousands of frames over the same ground. Building it
    # once and copying turns the renderer from the bottleneck of the SIL test
    # into a rounding error.
    frame = _soil_base(width, height, float(style.soil_speckle), int(style.seed)).copy()

    # -- guidance line ------------------------------------------------------
    if draw_line and line_points:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in line_points],
                       dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, LINE_BGR, line_width_px,
                          lineType=cv2.LINE_AA)
    elif draw_line:
        angle = math.radians(line_angle_deg)
        x_bottom = width / 2.0 + line_offset_px
        # The line recedes up the frame; tilt shifts its top end.
        x_top = x_bottom + math.tan(angle) * height
        cv2.line(
            frame,
            (int(round(x_bottom)), height),
            (int(round(x_top)), 0),
            LINE_BGR,
            line_width_px,
            lineType=cv2.LINE_AA,
        )

    # -- markers ------------------------------------------------------------
    for marker in markers or []:
        colour = WEED_BGR if marker.cls is TargetClass.WEED else CROP_BGR
        x1, y1, x2, y2 = marker.bbox()
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
        # A thin darker border, as a printed marker on card would have.
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      tuple(int(c * 0.75) for c in colour), 2)

    # -- illumination -------------------------------------------------------
    if style.brightness_gradient > 0:
        ramp = _brightness_ramp(width, float(style.brightness_gradient))
        frame = np.clip(frame.astype(np.float32) * ramp[None, :, None], 0, 255)
        frame = frame.astype(np.uint8)

    if style.glare_strength > 0:
        overlay = _glare_field(
            width, height,
            int(style.glare_centre[0] * width),
            int(style.glare_centre[1] * height),
            int(0.18 * min(width, height)),
        )
        boost = (overlay * style.glare_strength * 255.0)[:, :, None]
        frame = np.clip(frame.astype(np.float32) + boost, 0, 255).astype(np.uint8)

    if style.blur and style.blur >= 1:
        k = int(style.blur) | 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k          # horizontal motion blur
        frame = cv2.filter2D(frame, -1, kernel)

    return frame


def render_line_sequence(
    offsets: Sequence[float],
    width: int = 640,
    height: int = 480,
    style: Optional[ArenaStyle] = None,
) -> List[np.ndarray]:
    """Render a sequence of frames with the given line offsets."""
    return [
        render_frame(width=width, height=height, line_offset_px=off, style=style)
        for off in offsets
    ]
