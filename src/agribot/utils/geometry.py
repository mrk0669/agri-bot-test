"""Small geometric and numeric helpers shared across the stack."""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "clamp",
    "deadband",
    "lerp",
    "wrap_pi",
    "wrap_2pi",
    "angle_diff",
    "normalise_error",
    "ground_range_from_pixel",
    "pixel_to_ground_offset",
    "low_pass_alpha",
]


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    if low > high:
        raise ValueError(f"clamp bounds inverted: low={low} > high={high}")
    return low if value < low else high if value > high else value


def deadband(value: float, width: float) -> float:
    """Zero out ``|value| < width``, and remove the step at the edges.

    Used on the PID output so that a robot sitting on the line does not
    dither the motors at a level that only heats the drivers.
    """
    if width <= 0:
        return value
    if abs(value) <= width:
        return 0.0
    return value - math.copysign(width, value)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation; ``t`` is clamped to ``[0, 1]``."""
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to ``(-pi, +pi]``."""
    wrapped = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def wrap_2pi(angle_rad: float) -> float:
    """Wrap an angle to ``[0, 2pi)``."""
    wrapped = math.fmod(angle_rad, 2.0 * math.pi)
    return wrapped + 2.0 * math.pi if wrapped < 0.0 else wrapped


def angle_diff(a_rad: float, b_rad: float) -> float:
    """Shortest signed difference ``a - b``, wrapped to ``(-pi, +pi]``.

    Every heading comparison in the Kalman filter and the turn controller goes
    through this; comparing raw angles across the +/-pi seam is the classic way
    to make a robot spin the long way round.
    """
    return wrap_pi(a_rad - b_rad)


def normalise_error(pixel_x: float, frame_width: int) -> float:
    """Map a pixel column to a normalised error in ``[-1, +1]``.

    ``-1`` is the left edge, ``0`` the optical centre, ``+1`` the right edge.
    Working in normalised units keeps the PID gains independent of the capture
    resolution, so switching 640x480 -> 1280x720 does not require re-tuning.
    """
    if frame_width <= 0:
        raise ValueError(f"frame_width must be positive, got {frame_width}")
    centre = frame_width / 2.0
    return clamp((pixel_x - centre) / centre, -1.0, 1.0)


def ground_range_from_pixel(
    pixel_y: float,
    frame_height: int,
    camera_height_m: float,
    mount_pitch_deg: float,
    fy: float,
    cy: float,
) -> float:
    """Forward ground distance to the point imaged at row ``pixel_y``.

    Assumes a flat ground plane and a camera pitched ``mount_pitch_deg`` below
    horizontal. This is the fallback used when no depth camera is fitted; for
    an elevated marker the assumption breaks, which is exactly why the config
    prefers a true depth reading when one is available (Section 5.5).

    Returns ``inf`` for rays at or above the horizon.
    """
    if camera_height_m <= 0:
        raise ValueError("camera_height_m must be positive")
    if fy <= 0:
        raise ValueError("fy must be positive")
    # Angle of this pixel row below the optical axis.
    ray_below_axis = math.atan2(pixel_y - cy, fy)
    depression = math.radians(mount_pitch_deg) + ray_below_axis
    if depression <= 1e-6:
        return math.inf
    return camera_height_m / math.tan(depression)


def pixel_to_ground_offset(
    pixel_x: float,
    range_m: float,
    fx: float,
    cx: float,
) -> float:
    """Lateral ground offset (m, +right) of a point at ``range_m`` imaged at ``pixel_x``."""
    if fx <= 0:
        raise ValueError("fx must be positive")
    if not math.isfinite(range_m):
        return math.inf
    return (pixel_x - cx) * range_m / fx


def low_pass_alpha(cutoff_hz: float, dt: float) -> float:
    """First-order low-pass smoothing factor for a given cutoff and timestep.

    Returns ``alpha`` for ``y += alpha * (x - y)``. ``cutoff_hz <= 0`` disables
    filtering (alpha = 1, pass-through).
    """
    if cutoff_hz <= 0 or dt <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (tau + dt)
