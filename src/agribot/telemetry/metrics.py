"""Mission metrics - the numbers that answer the judging criteria.

The competition scores efficient spraying, crop protection and reliable
navigation. This module turns the run into those figures directly, so that the
claim made in the proposal ("spraying efficiency reported as a measured
millilitres-per-weed figure rather than asserted") is produced by the software
rather than computed by hand afterwards.

The headline comparison is against blanket spraying: a boom covering the row at
a conventional application rate would consume a volume proportional to the
*area covered*, while targeted intervention consumes a volume proportional to
the *number of weeds*. Both are computed from the same run so the saving is a
measurement, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..targeting.spray_controller import SprayStats
from ..types import SprayEvent

__all__ = ["MissionMetrics", "blanket_equivalent_ml"]

#: Conventional blanket application rate for a boom sprayer, millilitres per
#: square metre. 200 L/ha = 20 ml/m^2 is a standard field rate and is the
#: baseline the saving is quoted against.
BLANKET_ML_PER_M2 = 20.0


def blanket_equivalent_ml(
    distance_m: float,
    swath_m: float,
    rate_ml_per_m2: float = BLANKET_ML_PER_M2,
) -> float:
    """Volume a blanket sprayer would have used over the same ground."""
    if distance_m < 0 or swath_m <= 0:
        raise ValueError("distance must be non-negative and swath positive")
    return distance_m * swath_m * rate_ml_per_m2


@dataclass
class MissionMetrics:
    """Everything measured over one run."""

    duration_s: float = 0.0
    distance_m: float = 0.0
    rows_completed: int = 0

    frames_processed: int = 0
    frames_line_found: int = 0
    mean_abs_line_error: float = 0.0
    max_abs_line_error: float = 0.0

    weeds_detected: int = 0
    weeds_treated: int = 0
    crops_seen: int = 0
    crop_vetoes: int = 0

    spray: Optional[SprayStats] = None
    events: List[SprayEvent] = field(default_factory=list)

    encoder_samples: int = 0
    encoder_rejected: int = 0
    state_transitions: int = 0
    faults: List[str] = field(default_factory=list)

    swath_m: float = 0.30

    # -- derived ------------------------------------------------------------
    @property
    def line_lock_rate(self) -> float:
        """Fraction of frames in which the guidance line was found."""
        if not self.frames_processed:
            return 0.0
        return self.frames_line_found / self.frames_processed

    @property
    def ml_per_weed(self) -> float:
        return self.spray.ml_per_weed if self.spray else float("nan")

    @property
    def total_ml(self) -> float:
        return self.spray.total_ml if self.spray else 0.0

    @property
    def blanket_ml(self) -> float:
        return blanket_equivalent_ml(self.distance_m, self.swath_m)

    @property
    def saving_ratio(self) -> float:
        """Blanket volume divided by targeted volume. ``inf`` if nothing sprayed."""
        used = self.total_ml
        if used <= 0:
            return float("inf")
        return self.blanket_ml / used

    @property
    def saving_percent(self) -> float:
        """Percentage of the blanket volume that was *not* used."""
        blanket = self.blanket_ml
        if blanket <= 0:
            return 0.0
        return 100.0 * (1.0 - min(self.total_ml / blanket, 1.0))

    @property
    def crop_protection_rate(self) -> float:
        """Fraction of crop encounters that produced no actuation.

        This is 1.0 for a correct run by construction - the fusion rule makes
        it structural - so a value below 1.0 is a genuine defect signal rather
        than a statistic to be tuned.
        """
        if not self.crops_seen:
            return 1.0
        return 1.0 - (self.crops_sprayed / self.crops_seen)

    #: Set by the runtime if actuation ever fired on a crop-classified target.
    crops_sprayed: int = 0

    @property
    def encoder_reject_rate(self) -> float:
        if not self.encoder_samples:
            return 0.0
        return self.encoder_rejected / self.encoder_samples

    # -- reporting ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 2),
            "distance_m": round(self.distance_m, 3),
            "rows_completed": self.rows_completed,
            "navigation": {
                "frames_processed": self.frames_processed,
                "line_lock_rate": round(self.line_lock_rate, 4),
                "mean_abs_line_error": round(self.mean_abs_line_error, 4),
                "max_abs_line_error": round(self.max_abs_line_error, 4),
                "encoder_reject_rate": round(self.encoder_reject_rate, 4),
                "state_transitions": self.state_transitions,
            },
            "perception": {
                "weeds_detected": self.weeds_detected,
                "weeds_treated": self.weeds_treated,
                "crops_seen": self.crops_seen,
                "crop_vetoes": self.crop_vetoes,
                "crops_sprayed": self.crops_sprayed,
                "crop_protection_rate": round(self.crop_protection_rate, 4),
            },
            "sustainability": {
                "total_ml": round(self.total_ml, 3),
                "ml_per_weed": (round(self.ml_per_weed, 3)
                                if self.spray and self.spray.events else None),
                "measured": self.spray.fully_measured if self.spray else False,
                "blanket_equivalent_ml": round(self.blanket_ml, 1),
                "saving_ratio": (round(self.saving_ratio, 2)
                                 if self.total_ml > 0 else None),
                "saving_percent": round(self.saving_percent, 2),
                "assumed_blanket_rate_ml_per_m2": BLANKET_ML_PER_M2,
            },
            "spray": self.spray.to_dict() if self.spray else None,
            "faults": self.faults,
        }

    def report(self) -> str:
        """Human-readable summary printed at the end of a run."""
        d = self.to_dict()
        lines = [
            "=" * 62,
            f"  MISSION SUMMARY - {self.duration_s:.1f} s, {self.distance_m:.2f} m, "
            f"{self.rows_completed} row(s)",
            "=" * 62,
            "  Navigation",
            f"    line lock rate        : {self.line_lock_rate * 100:.1f}% "
            f"of {self.frames_processed} frames",
            f"    mean |line error|     : {self.mean_abs_line_error:.4f} "
            f"(max {self.max_abs_line_error:.4f})",
            f"    encoder samples gated : {self.encoder_rejected}/{self.encoder_samples} "
            f"({self.encoder_reject_rate * 100:.1f}%)",
            "  Perception",
            f"    weeds treated         : {self.weeds_treated} of "
            f"{self.weeds_detected} detected",
            f"    crops seen / vetoed   : {self.crops_seen} / {self.crop_vetoes}",
            f"    crops sprayed         : {self.crops_sprayed}  "
            f"(protection {self.crop_protection_rate * 100:.1f}%)",
            "  Sustainability",
        ]
        if self.spray and self.spray.events:
            measured = "measured" if self.spray.fully_measured else "part-nominal"
            lines += [
                f"    fluid used            : {self.total_ml:.2f} ml ({measured})",
                f"    per weed              : {self.ml_per_weed:.2f} ml",
                f"    blanket equivalent    : {self.blanket_ml:.1f} ml "
                f"@ {BLANKET_ML_PER_M2:.0f} ml/m2",
                f"    saving                : {self.saving_percent:.1f}% "
                f"({self.saving_ratio:.1f}x less)",
            ]
        else:
            lines.append("    no spray events recorded")
        if self.faults:
            lines += ["  Faults"] + [f"    - {f}" for f in self.faults]
        lines.append("=" * 62)
        return "\n".join(lines)
