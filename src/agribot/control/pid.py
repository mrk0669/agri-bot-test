"""PID controller for the vision line-following loop (Section 5.2).

The controller consumes the normalised horizontal centroid error produced by
the guidance-line extractor and produces a normalised differential wheel-speed
correction. Three details matter on a real field and are handled explicitly:

* **Anti-windup.** If the line is lost or the robot is physically blocked, an
  unbounded integrator charges and the robot lurches when the line returns.
  The integral term is clamped, and additionally back-calculated whenever the
  output saturates.
* **Derivative filtering.** The error signal comes from a pixel centroid and
  is quantised and noisy; raw differentiation of it is amplified noise. The
  derivative path is low-pass filtered at a configurable cutoff.
* **Variable timestep.** Frame intervals jitter under inference load, so dt is
  measured per update rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..utils.geometry import clamp, low_pass_alpha

__all__ = ["PID", "PIDState"]


@dataclass
class PIDState:
    """Introspectable snapshot of the controller, for telemetry and tuning."""

    error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    output: float = 0.0
    saturated: bool = False
    dt: float = 0.0

    def to_dict(self) -> dict:
        return {
            "error": round(self.error, 4),
            "p": round(self.p_term, 4),
            "i": round(self.i_term, 4),
            "d": round(self.d_term, 4),
            "out": round(self.output, 4),
            "sat": self.saturated,
        }


class PID:
    """Discrete PID with clamped, back-calculated integral and filtered derivative."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        output_limit: float = 1.0,
        integral_limit: float = 0.5,
        derivative_filter_hz: float = 12.0,
        setpoint: float = 0.0,
    ):
        if output_limit <= 0:
            raise ValueError("output_limit must be positive")
        if integral_limit < 0:
            raise ValueError("integral_limit must be non-negative")

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.derivative_filter_hz = float(derivative_filter_hz)
        self.setpoint = float(setpoint)

        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._d_filtered = 0.0
        self.state = PIDState()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "PID":
        """Build from the ``navigation.pid`` config section."""
        return cls(
            kp=cfg.kp,
            ki=cfg.ki,
            kd=cfg.kd,
            output_limit=cfg.get("output_limit", 1.0),
            integral_limit=cfg.get("integral_limit", 0.5),
            derivative_filter_hz=cfg.get("derivative_filter_hz", 12.0),
        )

    # -- operation ----------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated state. Call on every state-machine transition
        into a driving state, so a stale integrator cannot kick the robot."""
        self._integral = 0.0
        self._prev_error = None
        self._d_filtered = 0.0
        self.state = PIDState()

    def update(self, measurement: float, dt: float) -> float:
        """Advance the controller one step and return the clamped output.

        Args:
            measurement: current process value (the normalised line error).
            dt: seconds since the previous update; must be positive.
        """
        if dt <= 0:
            # A duplicated frame or a clock glitch. Hold the previous output
            # rather than dividing by zero in the derivative path.
            return self.state.output

        error = self.setpoint - measurement

        p_term = self.kp * error

        # Trapezoidal integration is slightly more accurate than rectangular
        # for a signal that changes within the frame interval.
        if self._prev_error is None:
            self._integral += error * dt
        else:
            self._integral += 0.5 * (error + self._prev_error) * dt
        self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        if self._prev_error is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (error - self._prev_error) / dt
        alpha = low_pass_alpha(self.derivative_filter_hz, dt)
        self._d_filtered += alpha * (raw_derivative - self._d_filtered)
        d_term = self.kd * self._d_filtered

        unclamped = p_term + i_term + d_term
        output = clamp(unclamped, -self.output_limit, self.output_limit)
        saturated = output != unclamped

        # Back-calculation: when saturated, unwind the integral by the excess
        # so it does not keep charging against a limit it cannot overcome.
        if saturated and self.ki != 0.0:
            excess = unclamped - output
            self._integral -= excess / self.ki
            self._integral = clamp(
                self._integral, -self.integral_limit, self.integral_limit
            )
            i_term = self.ki * self._integral

        self._prev_error = error
        self.state = PIDState(
            error=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            output=output,
            saturated=saturated,
            dt=dt,
        )
        return output

    # -- live tuning --------------------------------------------------------
    def set_gains(self, kp: Optional[float] = None, ki: Optional[float] = None,
                  kd: Optional[float] = None) -> None:
        """Update gains in place (used by the interactive tuning tool)."""
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            # Rescale the accumulator so the integral contribution is continuous
            # across the gain change instead of stepping the output.
            if self.ki != 0.0 and ki != 0.0:
                self._integral *= self.ki / float(ki)
            elif ki == 0.0:
                self._integral = 0.0
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)

    @property
    def integral(self) -> float:
        return self._integral

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PID(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
                f"out={self.state.output:.3f})")
