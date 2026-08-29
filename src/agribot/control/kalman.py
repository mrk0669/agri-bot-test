"""Kalman sensor fusion for heading and travelled distance (Section 5.3).

Two independent filters, each fusing sources that fail in complementary ways.

**Heading** - a two-state filter over ``[heading, gyro_bias]``. A gyroscope
integrated over time is extremely smooth in the short term but drifts without
bound, because its zero-rate bias is itself slowly varying. A magnetometer is
absolutely referenced and drift-free but noisy and vulnerable to local magnetic
disturbance. Carrying the bias as an explicit state means the filter *identifies
and subtracts* the drift rather than merely smoothing its effect, which is what
keeps the fused estimate accurate over a long run.

**Distance** - a three-state filter over ``[position, velocity, accel_bias]``.
The accelerometer drives the prediction and the wheel encoder corrects it with
a velocity measurement.

The dominant field failure is not sensor noise but wheel spin: on loose soil the
wheels turn faster than the robot advances, so the encoder over-reads and pure
odometry gains distance that was never travelled. The filter therefore applies a
**fixed physical innovation gate**, rejecting any encoder sample that disagrees
with the predicted velocity by more than a bound set by the maximum acceleration
the rover can achieve in one encoder interval.

That fixed bound is used deliberately in preference to the more common adaptive
N-sigma gate. While the filter coasts through a spin its covariance grows; an
N-sigma gate widens along with it, and a spin sample is eventually admitted with
high confidence. The state is then corrupted and the filter never re-acquires.
This failure was observed directly during development, and the fixed bound
removes it. ``AdaptiveGateDistanceKalman`` reproduces the broken behaviour so
that the regression test can demonstrate the difference rather than assert it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..utils.geometry import wrap_pi

__all__ = [
    "HeadingKalman",
    "DistanceKalman",
    "AdaptiveGateDistanceKalman",
    "FusionStack",
    "joseph_update",
]


def joseph_update(
    P: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Joseph-form covariance update: ``(I-KH)P(I-KH)' + KRK'``.

    Algebraically identical to ``(I-KH)P`` for the optimal gain, but it stays
    symmetric and positive-definite under finite precision. On a long run at
    100 Hz the simple form can lose symmetry and drive the filter unstable, so
    the extra multiply is cheap insurance.
    """
    n = P.shape[0]
    IKH = np.eye(n) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T
    return 0.5 * (P_new + P_new.T)


# ---------------------------------------------------------------------------
# Heading filter
# ---------------------------------------------------------------------------


class HeadingKalman:
    """Two-state heading filter: ``x = [heading_rad, gyro_bias_rad_s]``.

    Call :meth:`predict` at the gyro rate and :meth:`update` at the
    magnetometer rate. Both accept an explicit ``dt`` so a jittery sample
    interval is handled exactly rather than assumed.
    """

    def __init__(
        self,
        gyro_noise_std: float = 0.0035,
        gyro_bias_walk_std: float = 0.00035,
        mag_noise_std_rad: float = 0.0483,   # 2.77 deg
        initial_heading: float = 0.0,
        initial_bias: float = 0.0,
        initial_heading_var: float = 0.05,
        initial_bias_var: float = 0.01,
        innovation_gate_rad: Optional[float] = math.radians(12.0),
    ):
        self.sigma_g = float(gyro_noise_std)
        self.sigma_bw = float(gyro_bias_walk_std)
        self.R = np.array([[float(mag_noise_std_rad) ** 2]])
        self.innovation_gate_rad = innovation_gate_rad

        self.x = np.array([float(initial_heading), float(initial_bias)])
        self.P = np.diag([float(initial_heading_var), float(initial_bias_var)])
        self.H = np.array([[1.0, 0.0]])

        self.n_updates = 0
        self.n_rejected = 0
        self.last_innovation_rad = 0.0

    @classmethod
    def from_config(cls, cfg, initial_heading: float = 0.0) -> "HeadingKalman":
        """Build from the ``fusion.heading`` config section."""
        gate = cfg.get("mag_innovation_gate_deg", None)
        return cls(
            gyro_noise_std=cfg.get("gyro_noise_std", 0.0035),
            gyro_bias_walk_std=cfg.get("gyro_bias_walk_std", 0.00035),
            mag_noise_std_rad=math.radians(cfg.get("mag_noise_std_deg", 2.77)),
            initial_heading=initial_heading,
            initial_heading_var=cfg.get("initial_heading_var", 0.05),
            initial_bias_var=cfg.get("initial_bias_var", 0.01),
            innovation_gate_rad=math.radians(gate) if gate else None,
        )

    # -- prediction ---------------------------------------------------------
    def predict(self, gyro_z_rad_s: float, dt: float) -> None:
        """Propagate with a raw gyro sample. The bias state is subtracted here."""
        if dt <= 0:
            return

        # heading += (omega_measured - bias) * dt ; bias is a random walk.
        self.x[0] = wrap_pi(self.x[0] + (gyro_z_rad_s - self.x[1]) * dt)

        F = np.array([[1.0, -dt], [0.0, 1.0]])

        # Continuous-discrete process noise: gyro white noise enters the angle,
        # bias random walk enters the bias and couples into the angle.
        q_g = self.sigma_g ** 2
        q_b = self.sigma_bw ** 2
        Q = np.array([
            [q_g * dt + q_b * dt ** 3 / 3.0, -q_b * dt ** 2 / 2.0],
            [-q_b * dt ** 2 / 2.0, q_b * dt],
        ])

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -- correction ---------------------------------------------------------
    def update(self, mag_heading_rad: float) -> bool:
        """Correct with an absolute heading measurement.

        Returns True if the measurement was accepted, False if the innovation
        gate rejected it as a magnetic disturbance.
        """
        # The innovation MUST be wrapped: comparing raw angles across the
        # +/-pi seam produces a ~2pi innovation and a violent, wrong correction.
        innovation = wrap_pi(mag_heading_rad - self.x[0])
        self.last_innovation_rad = innovation

        if (
            self.innovation_gate_rad is not None
            and abs(innovation) > self.innovation_gate_rad
        ):
            self.n_rejected += 1
            return False

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ np.array([innovation])).ravel()
        self.x[0] = wrap_pi(self.x[0])
        self.P = joseph_update(self.P, K, self.H, self.R)

        self.n_updates += 1
        return True

    # -- accessors ----------------------------------------------------------
    @property
    def heading_rad(self) -> float:
        return float(self.x[0])

    @property
    def heading_deg(self) -> float:
        return math.degrees(float(self.x[0]))

    @property
    def bias_rad_s(self) -> float:
        return float(self.x[1])

    @property
    def bias_deg_s(self) -> float:
        return math.degrees(float(self.x[1]))

    @property
    def heading_var(self) -> float:
        return float(self.P[0, 0])

    def __repr__(self) -> str:  # pragma: no cover
        return (f"HeadingKalman(heading={self.heading_deg:.2f}deg, "
                f"bias={self.bias_deg_s:.3f}deg/s)")


# ---------------------------------------------------------------------------
# Distance filter
# ---------------------------------------------------------------------------


class DistanceKalman:
    """Three-state distance filter: ``x = [pos_m, vel_mps, accel_bias_mps2]``.

    :meth:`predict` takes a raw longitudinal accelerometer sample; :meth:`update`
    takes an encoder-derived velocity and applies the fixed innovation gate.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.06,
        accel_bias_walk_std: float = 0.002,
        encoder_noise_std: float = 0.006,
        innovation_gate_mps: float = 0.05,
        initial_pos: float = 0.0,
        initial_vel: float = 0.0,
        initial_bias: float = 0.0,
        initial_pos_var: float = 0.001,
        initial_vel_var: float = 0.01,
        initial_bias_var: float = 0.05,
        max_consecutive_rejects: int = 100,
    ):
        self.sigma_a = float(accel_noise_std)
        self.sigma_ba = float(accel_bias_walk_std)
        self.R = np.array([[float(encoder_noise_std) ** 2]])
        self.innovation_gate_mps = float(innovation_gate_mps)
        self.max_consecutive_rejects = int(max_consecutive_rejects)

        self.x = np.array([float(initial_pos), float(initial_vel), float(initial_bias)])
        self.P = np.diag([
            float(initial_pos_var), float(initial_vel_var), float(initial_bias_var)
        ])
        self.H = np.array([[0.0, 1.0, 0.0]])

        self.n_updates = 0
        self.n_rejected = 0
        self.consecutive_rejects = 0
        self.last_innovation_mps = 0.0
        self.last_rejected = False
        self._initialised = False

    @classmethod
    def from_config(cls, cfg) -> "DistanceKalman":
        """Build from the ``fusion.distance`` config section."""
        return cls(
            accel_noise_std=cfg.get("accel_noise_std", 0.06),
            accel_bias_walk_std=cfg.get("accel_bias_walk_std", 0.002),
            encoder_noise_std=cfg.get("encoder_noise_std", 0.006),
            innovation_gate_mps=cfg.get("encoder_innovation_gate_mps", 0.05),
            initial_pos_var=cfg.get("initial_pos_var", 0.001),
            initial_vel_var=cfg.get("initial_vel_var", 0.01),
            initial_bias_var=cfg.get("initial_bias_var", 0.05),
            max_consecutive_rejects=cfg.get("max_consecutive_rejects", 100),
        )

    # -- prediction ---------------------------------------------------------
    def predict(self, accel_x_mps2: float, dt: float) -> None:
        if dt <= 0:
            return

        a_true = accel_x_mps2 - self.x[2]
        self.x[0] = self.x[0] + self.x[1] * dt + 0.5 * a_true * dt ** 2
        self.x[1] = self.x[1] + a_true * dt
        # bias unchanged (random walk)

        F = np.array([
            [1.0, dt, -0.5 * dt ** 2],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0],
        ])

        # Accel white noise enters through the double integrator; bias walk is
        # an independent driving term on the third state.
        q_a = self.sigma_a ** 2
        q_b = self.sigma_ba ** 2
        Q = np.array([
            [q_a * dt ** 4 / 4.0, q_a * dt ** 3 / 2.0, 0.0],
            [q_a * dt ** 3 / 2.0, q_a * dt ** 2, 0.0],
            [0.0, 0.0, q_b * dt],
        ])

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -- correction ---------------------------------------------------------
    def _gate_threshold(self) -> float:
        """Fixed physical bound - deliberately independent of the covariance."""
        return self.innovation_gate_mps

    def update(self, encoder_vel_mps: float) -> bool:
        """Correct with an encoder velocity. Returns True if accepted."""
        innovation = float(encoder_vel_mps) - self.x[1]
        self.last_innovation_mps = innovation

        # The gate defends a *prediction*. On the very first sample there is no
        # prediction worth defending - the filter was seeded at zero velocity,
        # which is an assumption, not a measurement. Without this the gate locks
        # a filter started mid-motion out of ever acquiring: every sample looks
        # like a spin relative to a velocity the robot left behind long ago.
        if not self._initialised:
            self._initialised = True
            self.x[1] = float(encoder_vel_mps)
            self.P[1, 1] = float(self.R[0, 0])
            self.n_updates += 1
            self.last_rejected = False
            return True

        threshold = self._gate_threshold()
        rejected = abs(innovation) > threshold

        if rejected:
            self.consecutive_rejects += 1
            # If the encoder disagrees for a long time it is more likely that
            # the filter has drifted than that the wheels have spun for two
            # solid seconds, so force a re-acquire rather than coast forever.
            if self.consecutive_rejects >= self.max_consecutive_rejects:
                self.consecutive_rejects = 0
                self._apply_correction(innovation, inflate=4.0)
                self.n_updates += 1
                self.last_rejected = False
                return True
            self.n_rejected += 1
            self.last_rejected = True
            return False

        self.consecutive_rejects = 0
        self._apply_correction(innovation)
        self.n_updates += 1
        self.last_rejected = False
        return True

    def _apply_correction(self, innovation: float, inflate: float = 1.0) -> None:
        R = self.R * inflate
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ np.array([innovation])).ravel()
        self.P = joseph_update(self.P, K, self.H, R)

    # -- accessors ----------------------------------------------------------
    @property
    def position_m(self) -> float:
        return float(self.x[0])

    @property
    def velocity_mps(self) -> float:
        return float(self.x[1])

    @property
    def accel_bias(self) -> float:
        return float(self.x[2])

    @property
    def position_var(self) -> float:
        return float(self.P[0, 0])

    def reset_position(self, value: float = 0.0) -> None:
        """Zero the odometer at a row start without disturbing velocity/bias."""
        self.x[0] = float(value)
        self.P[0, 0] = 1e-4

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DistanceKalman(pos={self.position_m:.3f}m, "
                f"vel={self.velocity_mps:.3f}m/s, rejects={self.n_rejected})")


class AdaptiveGateDistanceKalman(DistanceKalman):
    """Identical filter with the *adaptive* N-sigma gate, kept as a counterexample.

    This is the textbook gate ``|innovation| > N * sqrt(S)``. It is not used in
    flight. It exists so that ``tests/test_kalman_distance.py`` can demonstrate
    the failure described in Section 5.3 - the gate widening with the growing
    covariance until a wheel-spin sample is admitted - rather than merely
    asserting that it happens.
    """

    def __init__(self, *args, n_sigma: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sigma = float(n_sigma)

    def _gate_threshold(self) -> float:
        S = float((self.H @ self.P @ self.H.T + self.R)[0, 0])
        return self.n_sigma * math.sqrt(max(S, 1e-12))


# ---------------------------------------------------------------------------
# Combined stack
# ---------------------------------------------------------------------------


@dataclass
class FusionStack:
    """Both filters behind one interface, driven by MCU telemetry.

    The runtime holds one of these and feeds it every telemetry frame; the
    mission state machine reads ``heading_deg`` and ``distance_m`` from it.
    """

    heading: HeadingKalman
    distance: DistanceKalman
    _last_predict_t: Optional[float] = field(default=None, repr=False)
    _last_mag_t: Optional[float] = field(default=None, repr=False)
    _last_enc_t: Optional[float] = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg, initial_heading: float = 0.0) -> "FusionStack":
        """Build both filters from the ``fusion`` config section."""
        return cls(
            heading=HeadingKalman.from_config(cfg.heading, initial_heading),
            distance=DistanceKalman.from_config(cfg.distance),
        )

    def step(
        self,
        t: float,
        gyro_z: float,
        accel_x: float,
        mag_heading: Optional[float] = None,
        encoder_vel: Optional[float] = None,
    ) -> Tuple[bool, bool]:
        """Advance both filters one sample.

        Returns ``(mag_accepted, encoder_accepted)``; either is False when the
        corresponding measurement was absent or gated out.
        """
        dt = 0.0 if self._last_predict_t is None else t - self._last_predict_t
        self._last_predict_t = t

        if dt > 0:
            self.heading.predict(gyro_z, dt)
            self.distance.predict(accel_x, dt)

        mag_ok = False
        if mag_heading is not None:
            mag_ok = self.heading.update(mag_heading)
            self._last_mag_t = t

        enc_ok = False
        if encoder_vel is not None:
            enc_ok = self.distance.update(encoder_vel)
            self._last_enc_t = t

        return mag_ok, enc_ok

    @property
    def heading_deg(self) -> float:
        return self.heading.heading_deg

    @property
    def distance_m(self) -> float:
        return self.distance.position_m

    @property
    def velocity_mps(self) -> float:
        return self.distance.velocity_mps

    def snapshot(self, t: float):
        """Package the current belief as a :class:`~agribot.types.FusedState`."""
        from ..types import FusedState

        return FusedState(
            t=t,
            heading_rad=self.heading.heading_rad,
            heading_var=self.heading.heading_var,
            gyro_bias_rad_s=self.heading.bias_rad_s,
            distance_m=self.distance.position_m,
            velocity_mps=self.distance.velocity_mps,
            accel_bias_mps2=self.distance.accel_bias,
            distance_var=self.distance.position_var,
            encoder_rejected=self.distance.last_rejected,
        )
