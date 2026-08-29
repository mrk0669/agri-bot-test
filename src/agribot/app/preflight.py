"""Pre-run checks.

Run this before every competition attempt. It verifies the things that are
cheap to check on the bench and expensive to discover in the arena: that the
config parses and is self-consistent, that the camera delivers frames at the
resolution the spray calibration assumes, that the MCU answers, and that the
detectors the config claims to enable can actually load.

Exit code is 0 only if every REQUIRED check passed. Optional checks (the
learned detector, the depth camera) report status without failing the run,
because the design degrades to the deterministic colour tier by intent.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..config import Config, load_config
from ..utils.logging_setup import setup_logging

__all__ = ["Check", "run_checks", "main"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status != FAIL


def _check_config(cfg: Config) -> List[Check]:
    """Validate cross-field consistency that a schema alone cannot express."""
    out: List[Check] = []

    box = cfg.robot.bounding_box_cm
    biggest = max(box.length, box.width, box.height)
    out.append(Check(
        "bounding box <= 30 cm",
        PASS if biggest <= 30.0 + 1e-9 else FAIL,
        f"largest dimension {biggest:.1f} cm (rules allow 30)",
    ))

    cruise, top = cfg.robot.cruise_speed_mps, cfg.robot.max_speed_mps
    out.append(Check(
        "cruise below max speed",
        PASS if 0 < cruise <= top else FAIL,
        f"cruise {cruise} / max {top} m/s",
    ))

    stop, slow = cfg.safety.ultrasonic_stop_m, cfg.safety.ultrasonic_slow_m
    out.append(Check(
        "ultrasonic bands ordered",
        PASS if slow > stop > 0 else FAIL,
        f"stop {stop} m, slow {slow} m",
    ))

    # The stall watchdog must allow enough time to creep the row-end distance,
    # or the machine drops into RECOVER before it can ever declare a row end.
    creep = cfg.robot.cruise_speed_mps * cfg.navigation.get("blind_creep_scale", 0.6)
    needed = cfg.mission.row_end_detect_m / creep if creep > 0 else math.inf
    limit = cfg.navigation.get("line_lost_stop_s", 5.0)
    out.append(Check(
        "line-loss watchdog exceeds row-end creep",
        PASS if limit > needed else FAIL,
        f"need > {needed:.1f} s to creep {cfg.mission.row_end_detect_m} m, "
        f"watchdog is {limit} s",
    ))

    olim = cfg.navigation.pid.output_limit
    base = cruise / top
    out.append(Check(
        "PID output limit will not reverse a wheel",
        PASS if olim <= base else WARN,
        f"limit {olim:.2f} vs normalised cruise {base:.2f}",
        required=False,
    ))

    pan, tilt = cfg.targeting.pan, cfg.targeting.tilt
    out.append(Check(
        "servo limits bracket centre",
        PASS if (pan.min_deg <= pan.centre_deg <= pan.max_deg
                 and tilt.min_deg <= tilt.centre_deg <= tilt.max_deg) else FAIL,
        f"pan {pan.min_deg}-{pan.max_deg} c{pan.centre_deg}, "
        f"tilt {tilt.min_deg}-{tilt.max_deg} c{tilt.centre_deg}",
    ))

    dose = cfg.spray.ml_per_burst_nominal
    tank = cfg.spray.reservoir_ml
    out.append(Check(
        "reservoir holds a useful number of doses",
        PASS if tank / max(dose, 1e-6) >= 20 else WARN,
        f"{tank / max(dose, 1e-6):.0f} doses at {dose} ml",
        required=False,
    ))

    gate = cfg.fusion.distance.encoder_innovation_gate_mps
    out.append(Check(
        "encoder innovation gate is a fixed physical bound",
        PASS if 0 < gate < 0.5 else FAIL,
        f"{gate} m/s",
    ))
    return out


def _check_camera(cfg: Config) -> List[Check]:
    from ..vision.camera import Camera

    cam = Camera.from_config(cfg.camera)
    if not cam.open():
        return [Check("camera opens", FAIL, f"source {cfg.camera.get('source')!r}")]
    try:
        ok, frame = cam.read()
        if not ok or frame is None:
            return [Check("camera delivers frames", FAIL, "read() returned nothing")]
        h, w = frame.shape[:2]
        want = (cfg.camera.get("width", 640), cfg.camera.get("height", 480))
        checks = [Check("camera delivers frames", PASS, f"{w}x{h}")]
        checks.append(Check(
            "resolution matches spray calibration",
            PASS if (w, h) == want else FAIL,
            f"got {w}x{h}, calibration assumes {want[0]}x{want[1]}",
        ))
        return checks
    finally:
        cam.release()


def _check_mcu(cfg: Config) -> List[Check]:
    import time

    from ..hal.mcu_link import McuLink

    link = McuLink.from_config(cfg.mcu)
    if not link.open():
        return [Check("MCU link opens", FAIL, f"port {cfg.mcu.get('port')!r}")]
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if link.poll() is not None:
                telemetry = link.telemetry
                checks = [Check("MCU telemetry received", PASS,
                                f"seq {telemetry.seq}, {link.frames_received} frames")]
                battery = telemetry.battery_v
                checks.append(Check(
                    "battery above 11.0 V",
                    PASS if battery >= 11.0 else WARN,
                    f"{battery:.2f} V", required=False,
                ))
                checks.append(Check(
                    "frame error rate below 1%",
                    PASS if link.frames_dropped <= max(1, link.frames_received // 100)
                    else WARN,
                    f"{link.frames_dropped} dropped of {link.frames_received}",
                    required=False,
                ))
                return checks
            time.sleep(0.02)
        return [Check("MCU telemetry received", FAIL, "no valid frame within 2 s")]
    finally:
        link.close()


def _check_detectors(cfg: Config) -> List[Check]:
    out: List[Check] = []
    if cfg.perception.yolo.get("enabled", False):
        from ..vision.yolo_detector import YoloDetector
        det = YoloDetector.from_config(cfg.perception.yolo)
        out.append(Check(
            "learned detector loads",
            PASS if det.available else WARN,
            det.load_error or cfg.perception.yolo.get("weights", ""),
            required=False,
        ))
    if cfg.perception.zeroshot.get("enabled", False):
        from ..vision.zeroshot_detector import ZeroShotDetector
        det = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        out.append(Check(
            "zero-shot detector loads",
            PASS if det.available else WARN,
            det.load_error or cfg.perception.zeroshot.get("model", ""),
            required=False,
        ))
    if not out:
        out.append(Check(
            "learned tiers disabled",
            PASS,
            "running colour tier only - the guaranteed scorer",
            required=False,
        ))
    return out


def run_checks(cfg: Config, skip_hardware: bool = False) -> List[Check]:
    """Run every check, tolerating exceptions from any single one."""
    checks: List[Check] = []
    stages: List[Tuple[str, Callable[[Config], List[Check]]]] = [
        ("config", _check_config),
        ("detectors", _check_detectors),
    ]
    if not skip_hardware:
        stages += [("camera", _check_camera), ("mcu", _check_mcu)]

    for name, fn in stages:
        try:
            checks.extend(fn(cfg))
        except Exception as exc:
            checks.append(Check(f"{name} check", FAIL, f"raised: {exc}"))
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agribot-preflight", description="Pre-run checks for the AgriBot.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-hardware", action="store_true",
                        help="config and detector checks only")
    args = parser.parse_args(argv)
    setup_logging(console_level="WARNING", force=True)

    cfg = load_config(args.config)
    checks = run_checks(cfg, skip_hardware=args.skip_hardware)

    width = max(len(c.name) for c in checks) + 2
    print("=" * (width + 46))
    print(f"  AGRIBOT PREFLIGHT - {cfg.robot.name}")
    print("=" * (width + 46))
    for check in checks:
        marker = {PASS: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}[check.status]
        print(f"  {marker} {check.name:<{width}} {check.detail}")

    failures = [c for c in checks if c.status == FAIL and c.required]
    warnings = [c for c in checks if c.status == WARN]
    print("=" * (width + 46))
    print(f"  {len(checks)} checks, {len(failures)} failed, {len(warnings)} warnings")
    if failures:
        print("  NOT READY - fix the failures above before running.")
    else:
        print("  READY")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
