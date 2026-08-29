"""Scoring harness for the sensor-fusion study (Section 5.3).

Runs the single-sensor baselines and the Kalman filters over the same input -
simulated or logged - and reports the comparison table:

* heading from integrated gyroscope alone, magnetometer alone, and the fused filter
* distance from double-integrated accelerometer alone, raw encoder odometry, and
  the fused filter
* the gyroscope bias the filter recovered versus the bias actually injected

Everything here is measurement, not illustration: the numbers reported by
``tools/kalman_sim.py`` are produced by this module driving the same
``HeadingKalman`` / ``DistanceKalman`` classes the robot runs in the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..control.kalman import DistanceKalman, HeadingKalman
from ..utils.geometry import wrap_pi
from .sensors import RunProfile, SimulatedRun, simulate_run

__all__ = ["FusionResult", "analyse_run", "run_seed_sweep", "rmse", "unwrap_error"]


def unwrap_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signed angular error, each element wrapped to (-pi, +pi]."""
    return np.array([wrap_pi(e - t) for e, t in zip(estimate, truth)])


def rmse(error: np.ndarray) -> float:
    """Root-mean-square of an error array, ignoring NaN (absent ground truth)."""
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


@dataclass
class FusionResult:
    """Time series and summary metrics from one analysed run."""

    t: np.ndarray

    # -- heading ------------------------------------------------------------
    heading_gyro_only: np.ndarray
    heading_mag_raw: np.ndarray
    heading_fused: np.ndarray
    heading_truth: np.ndarray
    gyro_bias_estimate: np.ndarray

    # -- distance -----------------------------------------------------------
    distance_accel_only: np.ndarray
    distance_encoder_only: np.ndarray
    distance_fused: np.ndarray
    distance_truth: np.ndarray
    velocity_fused: np.ndarray

    # -- diagnostics --------------------------------------------------------
    encoder_rejected: np.ndarray
    mag_rejected: np.ndarray
    spin_active: np.ndarray
    disturbance_active: np.ndarray

    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> List[List[str]]:
        """The Section 5.3 comparison table as rows of strings."""
        m = self.metrics
        return [
            ["Quantity", "Single sensor A", "Single sensor B", "Kalman fused"],
            [
                "Heading RMSE",
                f"{m['heading_rmse_gyro_deg']:.2f} deg (gyro integrated)",
                f"{m['heading_rmse_mag_deg']:.2f} deg (magnetometer)",
                f"{m['heading_rmse_fused_deg']:.2f} deg",
            ],
            [
                "Distance RMSE",
                f"{m['distance_rmse_accel_m']:.2f} m (accelerometer)",
                f"{m['distance_rmse_encoder_m']:.3f} m (encoder)",
                f"{m['distance_rmse_fused_m']:.3f} m",
            ],
            [
                "Final distance error",
                "-",
                f"{m['distance_final_encoder_m']:+.3f} m",
                f"{m['distance_final_fused_m']:+.3f} m",
            ],
            [
                "Gyro bias identified",
                f"true {m['gyro_bias_true_deg_s']:.3f} deg/s",
                "-",
                f"estimated {m['gyro_bias_est_deg_s']:.3f} deg/s",
            ],
        ]


def analyse_run(
    run: SimulatedRun,
    heading_filter: Optional[HeadingKalman] = None,
    distance_filter: Optional[DistanceKalman] = None,
) -> FusionResult:
    """Drive both filters and both single-sensor baselines over ``run``."""
    n = len(run)
    dt_arr = np.diff(run.t, prepend=run.t[0] - (run.t[1] - run.t[0]) if n > 1 else 0.0)

    hk = heading_filter or HeadingKalman(
        gyro_noise_std=0.0035,
        gyro_bias_walk_std=0.00015,
        mag_noise_std_rad=math.radians(1.5),
        innovation_gate_rad=math.radians(8.0),
    )
    dk = distance_filter or DistanceKalman(
        accel_noise_std=0.06,
        accel_bias_walk_std=0.002,
        encoder_noise_std=0.006,
        innovation_gate_mps=0.05,
    )

    heading_gyro = np.zeros(n)
    heading_fused = np.zeros(n)
    bias_est = np.zeros(n)
    mag_rejected = np.zeros(n, dtype=bool)

    dist_accel = np.zeros(n)
    vel_accel = 0.0
    dist_encoder = np.zeros(n)
    dist_fused = np.zeros(n)
    vel_fused = np.zeros(n)
    enc_rejected = np.zeros(n, dtype=bool)

    last_encoder = 0.0

    for i in range(n):
        dt = float(dt_arr[i]) if i > 0 else 0.0

        # -- Baseline A: raw gyro integration, no bias correction -----------
        if i > 0:
            heading_gyro[i] = heading_gyro[i - 1] + run.gyro_z[i] * dt
        # -- Baseline A': raw accel double integration ----------------------
        if i > 0:
            vel_accel += run.accel_x[i] * dt
            dist_accel[i] = dist_accel[i - 1] + vel_accel * dt
        # -- Baseline B: raw encoder odometry -------------------------------
        if i > 0:
            if run.enc_valid[i]:
                last_encoder = float(run.encoder_mps[i])
            dist_encoder[i] = dist_encoder[i - 1] + last_encoder * dt

        # -- Fused ----------------------------------------------------------
        if dt > 0:
            hk.predict(float(run.gyro_z[i]), dt)
            dk.predict(float(run.accel_x[i]), dt)

        if run.mag_valid[i] and np.isfinite(run.mag_heading_rad[i]):
            accepted = hk.update(float(run.mag_heading_rad[i]))
            mag_rejected[i] = not accepted

        if run.enc_valid[i] and np.isfinite(run.encoder_mps[i]):
            accepted = dk.update(float(run.encoder_mps[i]))
            enc_rejected[i] = not accepted

        heading_fused[i] = hk.heading_rad
        bias_est[i] = hk.bias_rad_s
        dist_fused[i] = dk.position_m
        vel_fused[i] = dk.velocity_mps

    truth_h = run.true_heading_rad
    truth_d = run.true_distance_m
    have_truth = bool(np.isfinite(truth_h).any())

    # The magnetometer baseline is evaluated only where a reading exists.
    mag_err = np.full(n, np.nan)
    if have_truth:
        idx = run.mag_valid & np.isfinite(run.mag_heading_rad)
        mag_err[idx] = unwrap_error(run.mag_heading_rad[idx], truth_h[idx])

    metrics: Dict[str, Any] = {
        "duration_s": float(run.t[-1] - run.t[0]) if n else 0.0,
        "samples": n,
        "have_ground_truth": have_truth,
        "heading_rmse_gyro_deg": math.degrees(
            rmse(unwrap_error(heading_gyro, truth_h))) if have_truth else float("nan"),
        "heading_rmse_mag_deg": math.degrees(rmse(mag_err)) if have_truth else float("nan"),
        "heading_rmse_fused_deg": math.degrees(
            rmse(unwrap_error(heading_fused, truth_h))) if have_truth else float("nan"),
        "distance_rmse_accel_m": rmse(dist_accel - truth_d) if have_truth else float("nan"),
        "distance_rmse_encoder_m": rmse(dist_encoder - truth_d) if have_truth else float("nan"),
        "distance_rmse_fused_m": rmse(dist_fused - truth_d) if have_truth else float("nan"),
        "distance_final_encoder_m": float(dist_encoder[-1] - truth_d[-1]) if have_truth else float("nan"),
        "distance_final_fused_m": float(dist_fused[-1] - truth_d[-1]) if have_truth else float("nan"),
        "distance_final_truth_m": float(truth_d[-1]) if have_truth else float("nan"),
        "gyro_bias_true_deg_s": run.gyro_bias_true_deg_s,
        "gyro_bias_est_deg_s": math.degrees(float(bias_est[-1])),
        "encoder_samples": int(run.enc_valid.sum()),
        "encoder_rejected": int(enc_rejected.sum()),
        "mag_samples": int(run.mag_valid.sum()),
        "mag_rejected": int(mag_rejected.sum()),
        "spin_samples": int(run.spin_active.sum()),
    }

    # How well did the gate actually separate spin from clean samples? This is
    # the quantity that justifies the fixed bound, so it is measured directly.
    if run.spin_active.any():
        enc_idx = run.enc_valid
        spin_enc = run.spin_active & enc_idx
        clean_enc = (~run.spin_active) & enc_idx
        metrics["spin_reject_recall"] = (
            float(enc_rejected[spin_enc].sum() / max(1, spin_enc.sum()))
        )
        metrics["clean_reject_rate"] = (
            float(enc_rejected[clean_enc].sum() / max(1, clean_enc.sum()))
        )

    if have_truth:
        metrics["heading_improvement_vs_gyro"] = (
            metrics["heading_rmse_gyro_deg"] / metrics["heading_rmse_fused_deg"]
            if metrics["heading_rmse_fused_deg"] > 0 else float("inf")
        )
        metrics["distance_improvement_vs_encoder"] = (
            metrics["distance_rmse_encoder_m"] / metrics["distance_rmse_fused_m"]
            if metrics["distance_rmse_fused_m"] > 0 else float("inf")
        )

    return FusionResult(
        t=run.t,
        heading_gyro_only=heading_gyro,
        heading_mag_raw=run.mag_heading_rad,
        heading_fused=heading_fused,
        heading_truth=truth_h,
        gyro_bias_estimate=bias_est,
        distance_accel_only=dist_accel,
        distance_encoder_only=dist_encoder,
        distance_fused=dist_fused,
        distance_truth=truth_d,
        velocity_fused=vel_fused,
        encoder_rejected=enc_rejected,
        mag_rejected=mag_rejected,
        spin_active=run.spin_active,
        disturbance_active=run.disturbance_active,
        metrics=metrics,
    )


def run_seed_sweep(
    seeds: List[int],
    profile: Optional[RunProfile] = None,
) -> Dict[str, Any]:
    """Repeat the study across independent random seeds.

    A filter that only works on one noise realisation has not been verified, so
    the proposal reports a worst case across seeds rather than a single run.
    """
    import copy

    base = profile or RunProfile()
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        prof = copy.deepcopy(base)
        object.__setattr__(prof, "seed", seed) if not hasattr(prof, "seed") else setattr(prof, "seed", seed)
        result = analyse_run(simulate_run(prof))
        row = dict(result.metrics)
        row["seed"] = seed
        per_seed.append(row)

    def agg(key: str, fn) -> float:
        values = [r[key] for r in per_seed if np.isfinite(r.get(key, np.nan))]
        return float(fn(values)) if values else float("nan")

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "heading_rmse_fused_deg_mean": agg("heading_rmse_fused_deg", np.mean),
        "heading_rmse_fused_deg_worst": agg("heading_rmse_fused_deg", np.max),
        "distance_rmse_fused_m_mean": agg("distance_rmse_fused_m", np.mean),
        "distance_rmse_fused_m_worst": agg("distance_rmse_fused_m", np.max),
        "distance_final_fused_m_worst_abs": float(
            max(abs(r["distance_final_fused_m"]) for r in per_seed)
        ),
        "gyro_bias_est_deg_s_mean": agg("gyro_bias_est_deg_s", np.mean),
        "spin_reject_recall_min": agg("spin_reject_recall", np.min),
        "clean_reject_rate_max": agg("clean_reject_rate", np.max),
    }
