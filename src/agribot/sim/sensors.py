"""Ground-truth trajectory and sensor models for the sensor-fusion study.

This module generates the forty-second in-row run described in Section 5.3:
two row-end turns, a magnetometer disturbance, and two wheel-spin events. It
produces ground truth alongside the corrupted signals a real MEMS IMU and a
quantised encoder would deliver, so the Kalman implementation can be scored
against a known answer.

The same data structure is produced by :func:`load_run_from_csv`, which reads a
logged hardware run. The analysis in ``tools/kalman_sim.py`` is therefore
identical whether the input is simulated or recorded on the robot - the claim
in the proposal that "the accompanying script accepts logged CSV data in place
of the simulated signals" is implemented here rather than asserted.

Sensor noise figures default to values representative of the BNO055-class IMU
and the 1440 CPR quadrature encoder specified in Section 4.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TurnEvent",
    "SpinEvent",
    "DisturbanceEvent",
    "RunProfile",
    "SimulatedRun",
    "simulate_run",
    "load_run_from_csv",
    "save_run_to_csv",
    "DEFAULT_PROFILE",
]


@dataclass(frozen=True)
class TurnEvent:
    """A row-end turn: constant yaw rate between ``t_start`` and ``t_end``."""

    t_start: float
    t_end: float
    total_deg: float

    @property
    def rate_deg_s(self) -> float:
        span = self.t_end - self.t_start
        return self.total_deg / span if span > 0 else 0.0


@dataclass(frozen=True)
class SpinEvent:
    """A wheel-spin episode.

    On loose soil the wheels turn faster than the robot advances, so the
    encoder reports ``excess_mps`` more than the true body velocity for the
    duration of the event. The integral of that excess is distance the robot
    never travelled - the error the fixed innovation gate exists to reject.
    """

    t_start: float
    t_end: float
    excess_mps: float

    @property
    def distance_gained_m(self) -> float:
        return self.excess_mps * max(0.0, self.t_end - self.t_start)


@dataclass(frozen=True)
class DisturbanceEvent:
    """A local magnetic disturbance biasing the magnetometer heading."""

    t_start: float
    t_end: float
    offset_deg: float


@dataclass
class RunProfile:
    """Everything that defines one simulated run.

    The defaults reproduce the run reported in Section 5.3 of the proposal.
    """

    duration_s: float = 40.0
    dt: float = 0.01                      # 100 Hz IMU / prediction rate
    cruise_mps: float = 0.18
    accel_ramp_s: float = 1.0             # smooth start/stop, no step in accel

    turns: Sequence[TurnEvent] = field(default_factory=lambda: (
        TurnEvent(14.0, 17.0, 180.0),
        TurnEvent(31.0, 34.0, -180.0),
    ))
    spins: Sequence[SpinEvent] = field(default_factory=lambda: (
        SpinEvent(8.0, 9.2, 0.2492),
        SpinEvent(36.0, 37.2, 0.2233),
    ))
    disturbances: Sequence[DisturbanceEvent] = field(default_factory=lambda: (
        DisturbanceEvent(20.0, 21.5, 12.0),
    ))

    # -- IMU error model ----------------------------------------------------
    gyro_bias_deg_s: float = 0.780        # the bias the filter must identify
    gyro_noise_std: float = 0.0035        # rad/s
    gyro_bias_walk_std: float = 0.00006   # rad/s/sqrt(s)
    accel_bias_mps2: float = 0.0432
    accel_noise_std: float = 0.060        # m/s^2
    accel_bias_walk_std: float = 0.0003

    # -- Magnetometer -------------------------------------------------------
    mag_noise_std_deg: float = 1.50
    mag_rate_hz: float = 20.0

    # -- Encoder ------------------------------------------------------------
    encoder_noise_std: float = 0.0060     # m/s
    encoder_rate_hz: float = 50.0
    encoder_quantum_mps: float = 0.0      # set >0 to model tick quantisation

    # Seed 12 is the run reported in Section 5.3 and reproduced by the
    # regression test; tools/kalman_sim.py --sweep re-checks 8 further seeds.
    seed: int = 12

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")


DEFAULT_PROFILE = RunProfile()


@dataclass
class SimulatedRun:
    """Ground truth plus the corrupted sensor streams for one run.

    All arrays share the same length and time base. ``mag_valid`` and
    ``enc_valid`` mark the samples on which a magnetometer or encoder reading
    is actually available (they run slower than the IMU).
    """

    t: np.ndarray
    # -- ground truth -------------------------------------------------------
    true_heading_rad: np.ndarray
    true_velocity_mps: np.ndarray
    true_distance_m: np.ndarray
    # -- sensor streams -----------------------------------------------------
    gyro_z: np.ndarray            # rad/s, biased and noisy
    accel_x: np.ndarray           # m/s^2, biased and noisy
    mag_heading_rad: np.ndarray   # absolute heading, noisy, disturbed
    mag_valid: np.ndarray         # bool
    encoder_mps: np.ndarray       # m/s, over-reads during spin
    enc_valid: np.ndarray         # bool
    spin_active: np.ndarray       # bool, ground truth (for scoring the gate)
    disturbance_active: np.ndarray
    profile: Optional[RunProfile] = None

    def __len__(self) -> int:
        return len(self.t)

    @property
    def gyro_bias_true_deg_s(self) -> float:
        return self.profile.gyro_bias_deg_s if self.profile else float("nan")

    @property
    def final_true_distance_m(self) -> float:
        return float(self.true_distance_m[-1])


def _smoothstep(x: float) -> float:
    """C1-continuous ramp on [0, 1] - avoids an impulsive accelerometer step."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _speed_profile(t: float, prof: RunProfile) -> float:
    """True body speed at time ``t``: cruise, zero during turns, ramped edges."""
    for turn in prof.turns:
        if turn.t_start <= t < turn.t_end:
            return 0.0
        # Decelerate into the turn and accelerate out of it.
        if turn.t_start - prof.accel_ramp_s <= t < turn.t_start:
            return prof.cruise_mps * (1.0 - _smoothstep(
                (t - (turn.t_start - prof.accel_ramp_s)) / prof.accel_ramp_s))
        if turn.t_end <= t < turn.t_end + prof.accel_ramp_s:
            return prof.cruise_mps * _smoothstep((t - turn.t_end) / prof.accel_ramp_s)

    if t < prof.accel_ramp_s:
        return prof.cruise_mps * _smoothstep(t / prof.accel_ramp_s)
    return prof.cruise_mps


def _yaw_rate(t: float, prof: RunProfile) -> float:
    """True yaw rate (rad/s) at time ``t``."""
    for turn in prof.turns:
        if turn.t_start <= t < turn.t_end:
            return math.radians(turn.rate_deg_s)
    return 0.0


def simulate_run(profile: Optional[RunProfile] = None) -> SimulatedRun:
    """Generate one complete run: ground truth and corrupted sensor streams."""
    prof = profile or DEFAULT_PROFILE
    rng = np.random.default_rng(prof.seed)

    n = int(round(prof.duration_s / prof.dt)) + 1
    t = np.arange(n) * prof.dt

    true_heading = np.zeros(n)
    true_vel = np.zeros(n)
    true_dist = np.zeros(n)
    true_yaw_rate = np.zeros(n)
    true_accel = np.zeros(n)

    # -- integrate ground truth --------------------------------------------
    for i in range(n):
        true_vel[i] = _speed_profile(t[i], prof)
        true_yaw_rate[i] = _yaw_rate(t[i], prof)
    # Central-difference the speed profile to get true longitudinal accel.
    true_accel[1:-1] = (true_vel[2:] - true_vel[:-2]) / (2.0 * prof.dt)
    if n > 1:
        true_accel[0] = (true_vel[1] - true_vel[0]) / prof.dt
        true_accel[-1] = (true_vel[-1] - true_vel[-2]) / prof.dt

    for i in range(1, n):
        # Trapezoidal so the truth itself is not a source of integration error.
        true_dist[i] = true_dist[i - 1] + 0.5 * (true_vel[i] + true_vel[i - 1]) * prof.dt
        true_heading[i] = true_heading[i - 1] + 0.5 * (
            true_yaw_rate[i] + true_yaw_rate[i - 1]) * prof.dt

    # -- gyroscope: constant bias + slow random walk + white noise ----------
    gyro_bias = np.empty(n)
    b = math.radians(prof.gyro_bias_deg_s)
    walk_g = rng.normal(0.0, prof.gyro_bias_walk_std * math.sqrt(prof.dt), n)
    for i in range(n):
        gyro_bias[i] = b
        b += walk_g[i]
    gyro_z = true_yaw_rate + gyro_bias + rng.normal(0.0, prof.gyro_noise_std, n)

    # -- accelerometer ------------------------------------------------------
    accel_bias = np.empty(n)
    ab = prof.accel_bias_mps2
    walk_a = rng.normal(0.0, prof.accel_bias_walk_std * math.sqrt(prof.dt), n)
    for i in range(n):
        accel_bias[i] = ab
        ab += walk_a[i]
    accel_x = true_accel + accel_bias + rng.normal(0.0, prof.accel_noise_std, n)

    # -- magnetometer: absolute, noisy, occasionally disturbed --------------
    mag_step = max(1, int(round((1.0 / prof.mag_rate_hz) / prof.dt)))
    mag_valid = np.zeros(n, dtype=bool)
    mag_valid[::mag_step] = True

    disturbance = np.zeros(n)
    disturbance_active = np.zeros(n, dtype=bool)
    for dist_ev in prof.disturbances:
        mask = (t >= dist_ev.t_start) & (t < dist_ev.t_end)
        disturbance[mask] += math.radians(dist_ev.offset_deg)
        disturbance_active |= mask

    mag_heading = (
        true_heading
        + disturbance
        + rng.normal(0.0, math.radians(prof.mag_noise_std_deg), n)
    )

    # -- encoder: true velocity plus spin over-read -------------------------
    enc_step = max(1, int(round((1.0 / prof.encoder_rate_hz) / prof.dt)))
    enc_valid = np.zeros(n, dtype=bool)
    enc_valid[::enc_step] = True

    spin_excess = np.zeros(n)
    spin_active = np.zeros(n, dtype=bool)
    for spin in prof.spins:
        mask = (t >= spin.t_start) & (t < spin.t_end)
        spin_excess[mask] += spin.excess_mps
        spin_active |= mask

    encoder = true_vel + spin_excess + rng.normal(0.0, prof.encoder_noise_std, n)
    if prof.encoder_quantum_mps > 0:
        encoder = np.round(encoder / prof.encoder_quantum_mps) * prof.encoder_quantum_mps

    return SimulatedRun(
        t=t,
        true_heading_rad=true_heading,
        true_velocity_mps=true_vel,
        true_distance_m=true_dist,
        gyro_z=gyro_z,
        accel_x=accel_x,
        mag_heading_rad=mag_heading,
        mag_valid=mag_valid,
        encoder_mps=encoder,
        enc_valid=enc_valid,
        spin_active=spin_active,
        disturbance_active=disturbance_active,
        profile=prof,
    )


# ---------------------------------------------------------------------------
# CSV interchange - the same analysis runs on logged hardware data
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "t", "gyro_z", "accel_x", "mag_heading_rad", "mag_valid",
    "encoder_mps", "enc_valid", "true_heading_rad", "true_velocity_mps",
    "true_distance_m",
]


def save_run_to_csv(run: SimulatedRun, path: Path) -> Path:
    """Write a run to CSV in the schema :func:`load_run_from_csv` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for i in range(len(run)):
            writer.writerow([
                f"{run.t[i]:.6f}",
                f"{run.gyro_z[i]:.9f}",
                f"{run.accel_x[i]:.9f}",
                f"{run.mag_heading_rad[i]:.9f}",
                int(run.mag_valid[i]),
                f"{run.encoder_mps[i]:.9f}",
                int(run.enc_valid[i]),
                f"{run.true_heading_rad[i]:.9f}",
                f"{run.true_velocity_mps[i]:.9f}",
                f"{run.true_distance_m[i]:.9f}",
            ])
    return path


def load_run_from_csv(path: Path) -> SimulatedRun:
    """Load a logged run recorded on the robot.

    Ground-truth columns are optional: when absent (as they are on hardware,
    where there is no external reference) the arrays are filled with NaN and
    the RMSE columns of the report are reported as unavailable rather than
    silently computed against zeros.
    """
    path = Path(path)
    rows: List[dict] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no data rows")

    def col(name: str, default: Optional[float] = None) -> np.ndarray:
        if name not in rows[0]:
            if default is None:
                return np.full(len(rows), np.nan)
            return np.full(len(rows), default)
        out = np.empty(len(rows))
        for i, r in enumerate(rows):
            value = r.get(name, "")
            out[i] = float(value) if value not in ("", None) else np.nan
        return out

    def bool_col(name: str) -> np.ndarray:
        if name not in rows[0]:
            return np.ones(len(rows), dtype=bool)
        return np.array([str(r[name]).strip() in ("1", "True", "true") for r in rows])

    n = len(rows)
    return SimulatedRun(
        t=col("t"),
        true_heading_rad=col("true_heading_rad"),
        true_velocity_mps=col("true_velocity_mps"),
        true_distance_m=col("true_distance_m"),
        gyro_z=col("gyro_z"),
        accel_x=col("accel_x"),
        mag_heading_rad=col("mag_heading_rad"),
        mag_valid=bool_col("mag_valid"),
        encoder_mps=col("encoder_mps"),
        enc_valid=bool_col("enc_valid"),
        spin_active=np.zeros(n, dtype=bool),
        disturbance_active=np.zeros(n, dtype=bool),
        profile=None,
    )
