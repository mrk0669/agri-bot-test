"""Differential-drive mixing and speed shaping (Section 5.1).

Steering is achieved purely by the speed difference between the left and right
sides, so the PID correction maps directly onto a wheel-speed differential with
no steering linkage. This module owns that mapping and the safety shaping that
sits on top of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..types import DriveCommand
from ..utils.geometry import clamp

__all__ = ["DifferentialMixer", "safety_scale", "wheel_speeds_to_body"]


@dataclass
class DifferentialMixer:
    """Turns (base speed, steering correction) into normalised wheel commands.

    ``max_speed_mps`` is the mechanical ceiling used to normalise physical
    speeds into the ``[-1, +1]`` range the MCU expects.

    Two behaviours are worth calling out:

    * **Differential preservation on saturation.** Naively clamping each wheel
      after adding the correction destroys the *difference* between them
      exactly when the correction is largest, i.e. in the sharpest turn - the
      robot straightens up mid-corner. Instead, when either side would exceed
      the limit the pair is scaled down together, which preserves the turn
      radius and only gives up forward speed.
    * **Turn-in-place support.** ``turn()`` produces a pure differential with
      zero mean, used by the row-end manoeuvre.
    """

    max_speed_mps: float
    wheel_base_m: float
    min_output: float = 0.0        # command below which motors merely buzz
    preserve_differential: bool = True

    def __post_init__(self) -> None:
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if self.wheel_base_m <= 0:
            raise ValueError("wheel_base_m must be positive")

    @classmethod
    def from_config(cls, cfg) -> "DifferentialMixer":
        """Build from the ``robot`` config section."""
        return cls(
            max_speed_mps=cfg.max_speed_mps,
            wheel_base_m=cfg.wheel_base_m,
        )

    # -- primary mapping ----------------------------------------------------
    def mix(self, base_speed_mps: float, correction: float) -> DriveCommand:
        """Mix a forward speed (m/s) and a normalised steering correction.

        ``correction`` is the PID output in ``[-1, +1]``. **Positive means turn
        left (counter-clockwise)** - the same convention as :meth:`turn` and as
        the positive yaw rate reported by :func:`wheel_speeds_to_body`. Every
        angular quantity in the stack therefore has one sign convention.

        The closed loop resolves as follows. A line imaged to the *right* of
        centre gives a positive ``LineObservation.error``; the PID computes
        ``setpoint - measurement`` and so produces a *negative* correction;
        that speeds the left wheel and slows the right, turning the robot
        right, towards the line. Which is what a robot sitting left of the row
        needs to do.
        """
        base = clamp(base_speed_mps / self.max_speed_mps, -1.0, 1.0)
        correction = clamp(correction, -1.0, 1.0)

        left = base - correction
        right = base + correction

        if self.preserve_differential:
            peak = max(abs(left), abs(right))
            if peak > 1.0:
                left /= peak
                right /= peak
        else:
            left = clamp(left, -1.0, 1.0)
            right = clamp(right, -1.0, 1.0)

        return DriveCommand(
            self._apply_min(clamp(left, -1.0, 1.0)),
            self._apply_min(clamp(right, -1.0, 1.0)),
        )

    def turn(self, yaw_rate_sign: float, speed_mps: float) -> DriveCommand:
        """Pure turn in place. ``yaw_rate_sign`` > 0 turns left (CCW)."""
        magnitude = clamp(abs(speed_mps) / self.max_speed_mps, 0.0, 1.0)
        direction = 1.0 if yaw_rate_sign >= 0 else -1.0
        # CCW: left wheel reverses, right wheel advances.
        return DriveCommand(
            self._apply_min(-direction * magnitude),
            self._apply_min(direction * magnitude),
        )

    def straight(self, speed_mps: float) -> DriveCommand:
        cmd = clamp(speed_mps / self.max_speed_mps, -1.0, 1.0)
        return DriveCommand(self._apply_min(cmd), self._apply_min(cmd))

    def _apply_min(self, value: float) -> float:
        """Suppress commands too small to move the robot, avoiding motor whine."""
        if self.min_output > 0 and 0 < abs(value) < self.min_output:
            return 0.0
        return value

    # -- inverse ------------------------------------------------------------
    def body_velocity(self, left_mps: float, right_mps: float) -> Tuple[float, float]:
        """Wheel speeds (m/s) -> (linear m/s, angular rad/s)."""
        return wheel_speeds_to_body(left_mps, right_mps, self.wheel_base_m)

    def yaw_rate_for(self, cmd: DriveCommand) -> float:
        """Approximate body yaw rate (rad/s) a normalised command will produce."""
        left_mps = cmd.left * self.max_speed_mps
        right_mps = cmd.right * self.max_speed_mps
        return (right_mps - left_mps) / self.wheel_base_m


def wheel_speeds_to_body(
    left_mps: float, right_mps: float, wheel_base_m: float
) -> Tuple[float, float]:
    """Differential-drive forward kinematics.

    Returns ``(linear_mps, angular_rad_s)``; positive angular is counter-clockwise.
    """
    if wheel_base_m <= 0:
        raise ValueError("wheel_base_m must be positive")
    linear = 0.5 * (left_mps + right_mps)
    angular = (right_mps - left_mps) / wheel_base_m
    return linear, angular


def safety_scale(
    range_m: float,
    stop_m: float,
    slow_m: float,
) -> float:
    """Speed multiplier in ``[0, 1]`` from the nearest obstacle range.

    Ultrasonic sensors run as an independent safety layer that can halt the
    drive irrespective of what the vision pipeline reports (Section 5.2), so
    this is applied to the *output* of the controller, not folded into it.

    Full stop at or inside ``stop_m``, linear ramp up to ``slow_m``, full speed
    beyond. ``inf`` (no echo) means the path is clear.
    """
    if slow_m <= stop_m:
        raise ValueError(f"slow_m ({slow_m}) must exceed stop_m ({stop_m})")
    if not math.isfinite(range_m):
        return 1.0
    if range_m <= stop_m:
        return 0.0
    if range_m >= slow_m:
        return 1.0
    return (range_m - stop_m) / (slow_m - stop_m)
