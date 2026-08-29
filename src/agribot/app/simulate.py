"""Software-in-the-loop mission simulation.

Runs the complete, unmodified runtime against the mock MCU and the synthetic
arena. The robot steers on frames rendered from a pose its own drive commands
produced, so what this exercises is the *system*, not the parts.

Use it to check a change end to end before touching hardware::

    python -m agribot.app.simulate --seconds 40 --report
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..hal.mcu_link import McuLink
from ..hal.mock_mcu import MockMcu, MockMcuConfig
from ..sim.arena import ArenaLayout, SimulatedArena, SimulatedCamera
from ..telemetry.logger import RunLogger
from ..telemetry.metrics import MissionMetrics
from ..utils.logging_setup import get_logger, setup_logging
from .runtime import AgriBotRuntime

__all__ = ["SimulationHarness", "main"]

log = get_logger("app.simulate")


class SimulationHarness:
    """Drives the runtime and the mock robot in lockstep on a virtual clock.

    Time is advanced explicitly rather than slept, so a forty-second mission
    runs in well under a second and the result is deterministic - both
    necessary for the simulation to be usable as a regression test.
    """

    def __init__(
        self,
        cfg: Config,
        layout: Optional[ArenaLayout] = None,
        dt: float = 1.0 / 30.0,
        mcu_dt: float = 0.01,
        start_lateral_m: float = 0.0,
        start_heading_deg: float = 0.0,
        seed: int = 7,
        logger: Optional[RunLogger] = None,
        dry_run: bool = False,
    ):
        self.cfg = cfg
        self.dt = float(dt)
        self.mcu_dt = float(mcu_dt)
        self.layout = layout or ArenaLayout.default_demo()

        self.mock = MockMcu(
            MockMcuConfig(
                wheel_base_m=cfg.robot.wheel_base_m,
                max_speed_mps=cfg.robot.max_speed_mps,
                seed=seed,
            ),
            telemetry_hz=cfg.fusion.distance.get("predict_rate_hz", 100.0),
        )
        # Seed the pose so the robot starts off the line and has to converge.
        self.mock.y = -start_lateral_m
        self.mock.heading_rad = math.radians(start_heading_deg)

        self.link = McuLink(self.mock, timeout_s=cfg.safety.get("mcu_timeout_s", 0.5),
                            clock=lambda: self.mock.t)
        self.arena = SimulatedArena.from_config(cfg, self.layout)
        self.camera = SimulatedCamera(self.arena, self._pose)
        self.runtime = AgriBotRuntime(
            cfg, self.camera, self.link, logger=logger,
            clock=lambda: self.mock.t, dry_run=dry_run,
        )
        self.pose_history: List[Dict[str, float]] = []

    def _pose(self):
        """Robot pose expressed for the arena renderer."""
        # Mock world frame: +x forward along the row, +y to the left.
        # Arena frame: lateral positive means the robot is right of the line.
        return self.mock.x, -self.mock.y, self.mock.heading_rad

    def run(self, seconds: float = 40.0) -> MissionMetrics:
        """Run the mission for at most ``seconds`` of simulated time."""
        if not self.runtime.setup():
            return self.runtime.shutdown("setup failed")

        reason = "time limit"
        steps_per_tick = max(1, int(round(self.dt / self.mcu_dt)))
        end_t = seconds

        try:
            while self.mock.t < end_t:
                for _ in range(steps_per_tick):
                    self.mock.step(self.mcu_dt)
                output = self.runtime.tick()
                self.pose_history.append({
                    "t": self.mock.t,
                    "x": self.mock.x,
                    "y": self.mock.y,
                    "heading_deg": math.degrees(self.mock.heading_rad),
                    "state": output.state.value if output else "none",
                })
                if output is not None and output.state.value in (
                    "MISSION_COMPLETE", "ESTOP"
                ):
                    reason = f"terminal state {output.state.value}"
                    break
        except KeyboardInterrupt:  # pragma: no cover
            reason = "interrupted"

        metrics = self.runtime.shutdown(reason)
        return metrics

    # -- analysis helpers ---------------------------------------------------
    @property
    def final_lateral_error_m(self) -> float:
        """Signed lateral displacement from the row at the end of the run."""
        return -self.mock.y

    def lateral_rms_after(self, t_start: float) -> float:
        """RMS lateral error after ``t_start`` - convergence, not transient."""
        values = [p["y"] for p in self.pose_history if p["t"] >= t_start]
        if not values:
            return float("nan")
        return math.sqrt(sum(v * v for v in values) / len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agribot-sim",
        description="Software-in-the-loop AgriBot mission simulation.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to robot.yaml (default: config/robot.yaml)")
    parser.add_argument("--seconds", type=float, default=40.0,
                        help="simulated seconds to run")
    parser.add_argument("--start-lateral", type=float, default=0.04,
                        help="initial displacement from the line, metres")
    parser.add_argument("--start-heading", type=float, default=-4.0,
                        help="initial heading error, degrees")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rows", type=int, default=None,
                        help="override mission.rows")
    parser.add_argument("--log", action="store_true",
                        help="write a run log under data/logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="disable actuation, perception and navigation only")
    parser.add_argument("--report", action="store_true",
                        help="print the full mission summary")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(console_level="WARNING" if args.quiet else "INFO", force=True)

    overrides: Dict[str, Any] = {}
    if args.rows is not None:
        overrides["mission"] = {"rows": args.rows}
    cfg = load_config(args.config, overrides=overrides or None)

    logger = RunLogger.from_config(cfg.telemetry) if args.log else None
    harness = SimulationHarness(
        cfg,
        start_lateral_m=args.start_lateral,
        start_heading_deg=args.start_heading,
        seed=args.seed,
        logger=logger,
        dry_run=args.dry_run,
    )
    metrics = harness.run(seconds=args.seconds)

    if args.report or not args.quiet:
        print(metrics.report())
        print(f"  final lateral error   : {harness.final_lateral_error_m * 100:+.2f} cm")
        print(f"  lateral RMS after 3 s : "
              f"{harness.lateral_rms_after(3.0) * 100:.2f} cm")

    # A run that sprayed a crop is a failure regardless of anything else.
    return 1 if metrics.crops_sprayed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
