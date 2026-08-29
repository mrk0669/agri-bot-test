"""Live mission entry point.

    agribot                      # run the mission with config/robot.yaml
    agribot --dry-run            # navigate and perceive, never actuate the nozzle
    agribot --mode mark          # servo felt marker instead of wet spray
    agribot --preflight          # run the checks and exit

Signals are handled so that ``systemctl stop`` or Ctrl-C brings the robot to a
controlled halt - motors off, valve shut, pump off - rather than leaving the
nozzle open when the process dies.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import load_config
from ..hal.mcu_link import McuLink
from ..telemetry.logger import RunLogger
from ..utils.logging_setup import get_logger, setup_logging
from ..vision.camera import Camera
from .runtime import AgriBotRuntime

__all__ = ["main"]

log = get_logger("app.main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agribot",
        description="Autonomous precision agriculture robot - Robofest Gujarat 6.0.",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to robot.yaml (default: config/robot.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="navigate and perceive but never open the valve")
    parser.add_argument("--mode", choices=["spray", "mark"], default=None,
                        help="override spray.mode")
    parser.add_argument("--rows", type=int, default=None, help="override mission.rows")
    parser.add_argument("--port", default=None, help="override mcu.port")
    parser.add_argument("--camera", default=None,
                        help="override camera.source (index or video path)")
    parser.add_argument("--no-log", action="store_true", help="do not write a run log")
    parser.add_argument("--preflight", action="store_true",
                        help="run the preflight checks and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _overrides(args) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if args.mode:
        out.setdefault("spray", {})["mode"] = args.mode
    if args.rows is not None:
        out.setdefault("mission", {})["rows"] = args.rows
    if args.port:
        out.setdefault("mcu", {})["port"] = args.port
    if args.camera is not None:
        source: Any = args.camera
        try:
            source = int(args.camera)
        except ValueError:
            pass
        out.setdefault("camera", {})["source"] = source
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=_overrides(args) or None)

    setup_logging(
        log_dir=Path(cfg.telemetry.get("log_dir", "data/logs")),
        console_level="DEBUG" if args.verbose else cfg.telemetry.get("console_level", "INFO"),
        file_level=cfg.telemetry.get("file_level", "DEBUG"),
        force=True,
    )

    if args.preflight:
        from .preflight import main as preflight_main
        return preflight_main(["--config", str(args.config)] if args.config else [])

    logger = None if args.no_log else RunLogger.from_config(cfg.telemetry)
    runtime = AgriBotRuntime(
        cfg,
        camera=Camera.from_config(cfg.camera),
        mcu_link=McuLink.from_config(cfg.mcu),
        logger=logger,
        dry_run=args.dry_run,
    )

    # A mission that is killed must still shut the valve. Both signals raise
    # KeyboardInterrupt, which run() already funnels into an orderly shutdown.
    def _handle(signum, _frame):
        log.warning("signal %s received, stopping", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle)

    metrics = runtime.run()
    print(metrics.report())

    # A run that actuated on a crop is a failure whatever else happened.
    if metrics.crops_sprayed:
        log.error("mission sprayed %d crop target(s)", metrics.crops_sprayed)
        return 2
    return 0 if not metrics.faults else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
