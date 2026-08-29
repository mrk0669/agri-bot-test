#!/usr/bin/env python3
"""Reproduce the sensor-fusion study of proposal Section 5.3.

Runs the Kalman filters the robot actually flies over a forty-second in-row
run, scores them against the single-sensor baselines, and emits the comparison
table and the four-panel figure.

    python tools/kalman_sim.py                    # table + figure
    python tools/kalman_sim.py --sweep 8          # 8-seed robustness check
    python tools/kalman_sim.py --csv run.csv      # analyse a logged hardware run
    python tools/kalman_sim.py --export run.csv   # write the simulated run out

The ``--csv`` path is the point of the exercise: the same analysis runs on
logged robot data as on the simulator, so the filter is verified on hardware
with one command rather than re-implemented for the bench.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agribot.config import load_config
from agribot.control.kalman import DistanceKalman, HeadingKalman
from agribot.sim.fusion_study import FusionResult, analyse_run, run_seed_sweep
from agribot.sim.sensors import (
    RunProfile,
    load_run_from_csv,
    save_run_to_csv,
    simulate_run,
)

#: Values published in the proposal, for side-by-side comparison.
PUBLISHED = {
    "heading_rmse_gyro_deg": 18.27,
    "heading_rmse_mag_deg": 2.77,
    "heading_rmse_fused_deg": 0.60,
    "distance_rmse_accel_m": 16.44,
    "distance_rmse_encoder_m": 0.300,
    "distance_rmse_fused_m": 0.004,
    "distance_final_encoder_m": 0.567,
    "distance_final_fused_m": -0.006,
    "gyro_bias_est_deg_s": 0.790,
}


def filters_from_config(cfg):
    """Build the filters exactly as the robot builds them."""
    return (
        HeadingKalman.from_config(cfg.fusion.heading),
        DistanceKalman.from_config(cfg.fusion.distance),
    )


def print_table(result: FusionResult, compare: bool = True) -> None:
    rows = result.summary_table()
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = "-+-".join("-" * w for w in widths)
    for i, row in enumerate(rows):
        print(" | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print(line)

    m = result.metrics
    print()
    print(f"  heading improvement vs integrated gyro : "
          f"{m.get('heading_improvement_vs_gyro', float('nan')):.1f}x")
    print(f"  distance improvement vs raw odometry   : "
          f"{m.get('distance_improvement_vs_encoder', float('nan')):.1f}x")
    if "spin_reject_recall" in m:
        print(f"  wheel-spin samples rejected by the gate: "
              f"{m['spin_reject_recall'] * 100:.1f}%")
        print(f"  clean samples wrongly rejected         : "
              f"{m['clean_reject_rate'] * 100:.2f}%")
    print(f"  magnetometer updates gated out         : "
          f"{m['mag_rejected']} of {m['mag_samples']}")

    if compare and m.get("have_ground_truth"):
        print()
        print("  Against the values published in Section 5.3:")
        print(f"    {'metric':<32}{'this run':>12}{'published':>12}{'delta':>10}")
        for key, published in PUBLISHED.items():
            actual = m.get(key, float("nan"))
            if not np.isfinite(actual):
                continue
            delta = actual - published
            print(f"    {key:<32}{actual:>12.4f}{published:>12.4f}{delta:>+10.4f}")


def make_figure(result: FusionResult, path: Path) -> Optional[Path]:
    """Render the four-panel Figure 5. Returns None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (figure skipped: matplotlib unavailable - {exc})")
        return None

    t = result.t
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Fig. 5 - Kalman filter results on a simulated in-row run",
        fontsize=13, fontweight="bold",
    )

    # (a) heading traces.
    # The filter state is wrapped to (-pi, +pi], so a 180 deg row-end turn sits
    # exactly on the seam and the raw trace flickers between +180 and -180.
    # That is a plotting artefact, not filter behaviour - panel (b) shows the
    # wrapped error is flat there - so the traces are unwrapped for display.
    ax = axes[0][0]
    ax.plot(t, np.degrees(np.unwrap(result.heading_truth)), "k-", lw=2.0,
            label="truth", zorder=5)
    ax.plot(t, np.degrees(np.unwrap(result.heading_gyro_only)), color="tab:red",
            lw=1.2, label="gyro integrated")
    mag_idx = np.isfinite(result.heading_mag_raw)
    ax.plot(t[mag_idx], np.degrees(np.unwrap(result.heading_mag_raw[mag_idx])), ".",
            color="tab:orange", ms=1.6, alpha=0.45, label="magnetometer")
    ax.plot(t, np.degrees(np.unwrap(result.heading_fused)), color="tab:blue", lw=1.6,
            label="Kalman fused")
    if result.disturbance_active.any():
        ax.axvspan(t[result.disturbance_active][0], t[result.disturbance_active][-1],
                   color="orange", alpha=0.15, label="mag disturbance")
    ax.set_title("(a) Heading")
    ax.set_xlabel("time (s)"); ax.set_ylabel("heading (deg)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # (b) heading error
    ax = axes[0][1]
    gyro_err = np.degrees([_wrap(a - b) for a, b in
                           zip(result.heading_gyro_only, result.heading_truth)])
    fused_err = np.degrees([_wrap(a - b) for a, b in
                            zip(result.heading_fused, result.heading_truth)])
    ax.plot(t, gyro_err, color="tab:red", lw=1.2, label="gyro integrated")
    ax.plot(t, fused_err, color="tab:blue", lw=1.6, label="Kalman fused")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"(b) Heading error  (fused RMSE "
                 f"{result.metrics['heading_rmse_fused_deg']:.2f} deg)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("error (deg)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) distance
    ax = axes[1][0]
    ax.plot(t, result.distance_truth, "k-", lw=2.0, label="truth", zorder=5)
    ax.plot(t, result.distance_encoder_only, color="tab:green", lw=1.3,
            label="encoder odometry")
    ax.plot(t, result.distance_fused, color="tab:blue", lw=1.6, label="Kalman fused")
    _shade_spins(ax, t, result.spin_active, "wheel spin")
    ax.set_title("(c) Travelled distance")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # (d) distance error
    ax = axes[1][1]
    ax.plot(t, result.distance_encoder_only - result.distance_truth,
            color="tab:green", lw=1.3, label="encoder odometry")
    ax.plot(t, result.distance_fused - result.distance_truth,
            color="tab:blue", lw=1.6, label="Kalman fused")
    ax.axhline(0, color="k", lw=0.8)
    _shade_spins(ax, t, result.spin_active, None)
    ax.set_title(
        f"(d) Distance error  (final: odometry "
        f"{result.metrics['distance_final_encoder_m']:+.3f} m, fused "
        f"{result.metrics['distance_final_fused_m']:+.3f} m)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("error (m)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _wrap(angle: float) -> float:
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def _shade_spins(ax, t, spin_active, label) -> None:
    if not spin_active.any():
        return
    edges = np.diff(spin_active.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(t[min(s, len(t) - 1)], t[min(e, len(t) - 1)],
                   color="red", alpha=0.12,
                   label=label if (label and i == 0) else None)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None,
                        help="analyse a logged run instead of simulating")
    parser.add_argument("--export", type=Path, default=None,
                        help="write the simulated run to CSV and exit")
    parser.add_argument("--figure", type=Path,
                        default=Path("reports/fig5_kalman.png"))
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--sweep", type=int, default=0,
                        help="repeat across N seeds and report the worst case")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the metrics as JSON")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    profile = RunProfile()
    if args.seed is not None:
        profile.seed = args.seed

    if args.export:
        path = save_run_to_csv(simulate_run(profile), args.export)
        print(f"wrote simulated run to {path}")
        return 0

    if args.csv:
        run = load_run_from_csv(args.csv)
        print(f"Analysing logged run: {args.csv}  ({len(run)} samples)")
        if not np.isfinite(run.true_heading_rad).any():
            print("  note: no ground-truth columns present, so RMSE against truth")
            print("        cannot be computed. Filter output is still produced.")
    else:
        run = simulate_run(profile)
        print(f"Simulated run: {profile.duration_s:.0f} s, seed {profile.seed}, "
              f"{len(run)} samples")
        print(f"  {len(profile.turns)} row-end turn(s), {len(profile.spins)} wheel-spin "
              f"event(s), {len(profile.disturbances)} magnetometer disturbance(s)")
    print()

    heading_filter, distance_filter = filters_from_config(cfg)
    result = analyse_run(run, heading_filter, distance_filter)
    print_table(result, compare=not args.csv)

    if args.sweep:
        print()
        seeds = [profile.seed] + [profile.seed + 7 * i for i in range(1, args.sweep)]
        sweep = run_seed_sweep(seeds, profile)
        print(f"  Robustness across {len(seeds)} independent seeds:")
        print(f"    fused heading RMSE  : mean "
              f"{sweep['heading_rmse_fused_deg_mean']:.3f} deg, worst "
              f"{sweep['heading_rmse_fused_deg_worst']:.3f} deg")
        print(f"    fused distance RMSE : mean "
              f"{sweep['distance_rmse_fused_m_mean']:.5f} m, worst "
              f"{sweep['distance_rmse_fused_m_worst']:.5f} m")
        print(f"    worst |final error| : "
              f"{sweep['distance_final_fused_m_worst_abs']:.5f} m")
        print(f"    spin rejection      : min "
              f"{sweep['spin_reject_recall_min'] * 100:.1f}% recall, max "
              f"{sweep['clean_reject_rate_max'] * 100:.2f}% false rejects")

    if not args.no_figure:
        path = make_figure(result, args.figure)
        if path:
            print(f"\n  figure written to {path}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        print(f"  metrics written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
