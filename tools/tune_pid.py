#!/usr/bin/env python3
"""Closed-loop PID tuning for the line-following controller.

Sweeps gains against the *real* extractor driving the *real* mixer over
rendered frames from the arena model, and reports the settled lateral error.
It is not a formula - it runs the same code the robot runs and measures what
the robot would do.

    python tools/tune_pid.py                    # sweep and rank
    python tools/tune_pid.py --check            # score the configured gains
    python tools/tune_pid.py --apply            # write the best gains to a local override

Scoring is the RMS lateral displacement over the second half of the run, from
a deliberately bad start (4 cm off the row, yawed 4 degrees). A configuration
that diverges, or that loses the line, is discarded rather than ranked.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agribot.config import load_config
from agribot.control.differential import DifferentialMixer
from agribot.control.pid import PID
from agribot.sim.arena import ArenaLayout, SimulatedArena
from agribot.vision.line_follow import LineFollower


def evaluate(
    cfg,
    arena: SimulatedArena,
    follower: LineFollower,
    mixer: DifferentialMixer,
    kp: float,
    ki: float,
    kd: float,
    output_limit: float,
    lat0_m: float = 0.04,
    head0_deg: float = -4.0,
    duration_s: float = 8.0,
    dt: float = 1.0 / 30.0,
) -> Optional[Dict[str, float]]:
    """Run one closed-loop trial. Returns None if the robot diverged."""
    pid = PID(kp, ki, kd, output_limit=output_limit,
              integral_limit=cfg.navigation.pid.get("integral_limit", 0.25),
              derivative_filter_hz=cfg.navigation.pid.get("derivative_filter_hz", 12.0))
    top = cfg.robot.max_speed_mps
    cruise = cfg.robot.cruise_speed_mps

    # World frame: +y to the left, so a robot right of the row has y < 0.
    x, y, heading = 0.0, -lat0_m, math.radians(head0_deg)
    laterals: List[float] = []
    peak_heading = 0.0
    lost = 0

    for _ in range(int(duration_s / dt)):
        observation = follower.process(arena.render(x, -y, heading))
        if not observation.found:
            lost += 1
            correction = pid.state.output
        else:
            correction = pid.update(observation.error, dt)

        command = mixer.mix(cruise, correction)
        left, right = command.left * top, command.right * top
        linear = 0.5 * (left + right)
        angular = (right - left) / cfg.robot.wheel_base_m

        heading += angular * dt
        x += linear * math.cos(heading) * dt
        y += linear * math.sin(heading) * dt

        laterals.append(-y)
        peak_heading = max(peak_heading, abs(heading))
        if abs(y) > 0.30:
            return None

    tail = laterals[len(laterals) // 2:]
    return {
        "rms_mm": math.sqrt(sum(v * v for v in tail) / len(tail)) * 1000.0,
        "final_mm": laterals[-1] * 1000.0,
        "peak_heading_deg": math.degrees(peak_heading),
        "frames_lost": lost,
        "distance_m": x,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="score the gains currently in the config and exit")
    parser.add_argument("--apply", action="store_true",
                        help="write the winning gains to config/robot.local.yaml")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    arena = SimulatedArena.from_config(cfg, ArenaLayout(row_length_m=30.0, markers=[]))
    follower = LineFollower.from_config(cfg.navigation.line)
    mixer = DifferentialMixer.from_config(cfg.robot)

    if args.check:
        p = cfg.navigation.pid
        result = evaluate(cfg, arena, follower, mixer, p.kp, p.ki, p.kd,
                          p.get("output_limit", 0.5), duration_s=args.duration)
        print(f"Configured gains: kp={p.kp} ki={p.ki} kd={p.kd} "
              f"output_limit={p.get('output_limit', 0.5)}")
        if result is None:
            print("  DIVERGED - the robot left the row.")
            return 1
        print(f"  settled RMS lateral error : {result['rms_mm']:.2f} mm")
        print(f"  final lateral error       : {result['final_mm']:+.2f} mm")
        print(f"  peak heading excursion    : {result['peak_heading_deg']:.1f} deg")
        print(f"  frames with no line       : {result['frames_lost']}")
        return 0

    grid = list(itertools.product(
        [0.35, 0.5, 0.7, 0.85, 1.1],      # kp
        [0.0, 0.02, 0.04],                # ki
        [0.0, 0.05, 0.10],                # kd
        [0.35, 0.50],                     # output limit
    ))
    print(f"Sweeping {len(grid)} configurations over {args.duration:.0f} s each "
          f"(this renders every frame, so it takes a minute)...\n")

    scored: List[Tuple[float, Tuple[float, float, float, float], Dict[str, float]]] = []
    diverged = 0
    for kp, ki, kd, olim in grid:
        result = evaluate(cfg, arena, follower, mixer, kp, ki, kd, olim,
                          duration_s=args.duration)
        if result is None:
            diverged += 1
            continue
        scored.append((result["rms_mm"], (kp, ki, kd, olim), result))

    scored.sort(key=lambda row: row[0])
    print(f"{'rms_mm':>8} {'kp':>5} {'ki':>5} {'kd':>5} {'olim':>5} "
          f"{'final_mm':>9} {'peakHead':>9} {'lost':>5}")
    print("-" * 60)
    for rms, (kp, ki, kd, olim), result in scored[: args.top]:
        print(f"{rms:8.2f} {kp:5.2f} {ki:5.2f} {kd:5.2f} {olim:5.2f} "
              f"{result['final_mm']:9.2f} {result['peak_heading_deg']:9.1f} "
              f"{result['frames_lost']:5d}")
    print(f"\n{diverged} configuration(s) diverged and were discarded.")

    if scored and args.apply:
        _rms, (kp, ki, kd, olim), _result = scored[0]
        path = Path(cfg.get("_meta", {}).get("config_path", "config/robot.yaml"))
        local = path.with_name(path.stem + ".local" + path.suffix)
        local.write_text(
            "# Written by tools/tune_pid.py --apply. Overrides robot.yaml.\n"
            "navigation:\n"
            "  pid:\n"
            f"    kp: {kp}\n    ki: {ki}\n    kd: {kd}\n"
            f"    output_limit: {olim}\n",
            encoding="utf-8",
        )
        print(f"\nwrote winning gains to {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
