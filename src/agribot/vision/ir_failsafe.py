"""Infra-red reflectance array fail-safe (Section 5.2).

The competition guidelines require navigation through a computer-vision
technique, so the camera is the primary and only *navigating* sensor. The
Pololu QTR-8A array is retained purely as a **silent fail-safe**: it is
consulted only when the vision pipeline has already reported the line lost,
and the observation it produces is tagged so the log always shows which sensor
the robot was actually steering on.

This is deliberately conservative. The array sits under the chassis and sees a
few centimetres of floor, so it can hold the robot on the line across a glare
patch or a scuffed section, but it has no look-ahead and cannot anticipate a
bend. It buys seconds, not autonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..types import LineObservation
from ..utils.geometry import clamp

__all__ = ["IrLineSensor"]


@dataclass
class IrLineSensor:
    """Turns a QTR-style reflectance bitmask into a normalised line error.

    ``channels`` sensors are assumed evenly spaced across the chassis width,
    left to right, with bit *i* set when sensor *i* sees the line.
    """

    channels: int = 8
    enabled: bool = True
    #: Minimum sensors that must see the line before the reading is trusted.
    min_active: int = 1
    #: Maximum sensors that may see the line. More than this and the array is
    #: over a junction, a wide patch of reflective floor, or a wash-out - none
    #: of which give a meaningful centroid.
    max_active: int = 4

    @classmethod
    def from_config(cls, cfg) -> "IrLineSensor":
        """Build from the ``navigation`` config section."""
        return cls(enabled=cfg.get("ir_failsafe_enabled", True))

    def observe(self, ir_array: Optional[Sequence[int]]) -> LineObservation:
        """Derive a line observation from the reflectance bitmask.

        Returns an observation with ``found=False`` when the array is disabled,
        absent, or reporting a pattern that cannot be a single line.
        """
        if not self.enabled or not ir_array:
            return LineObservation(found=False)

        readings = [1 if bit else 0 for bit in ir_array][: self.channels]
        active = sum(readings)
        if active < self.min_active or active > self.max_active:
            return LineObservation(found=False)

        # Weighted centroid over sensor indices, mapped to [-1, +1] with the
        # same sign convention as the vision extractor: positive means the line
        # is to the right of centre.
        n = len(readings)
        centre = (n - 1) / 2.0
        weighted = sum(i * w for i, w in enumerate(readings))
        centroid = weighted / active
        error = clamp((centroid - centre) / centre, -1.0, 1.0) if centre > 0 else 0.0

        # Confidence falls off as more sensors light up: one sensor on the line
        # is a crisp reading, four is a smear.
        confidence = 1.0 - (active - 1) / float(max(self.max_active, 1))

        return LineObservation(
            found=True,
            error=error,
            centroid_px=None,          # the array has no pixel coordinates
            mask_area_px=float(active),
            confidence=clamp(confidence, 0.0, 1.0),
            source="ir",
        )
