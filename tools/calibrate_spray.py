#!/usr/bin/env python3
"""Fit the pixel-to-actuator mapping for the spray head (Section 5.7).

Procedure, on the bench with the reservoir filled with water:

1. Place a target where the camera can see it and where the nozzle can reach.
2. The tool shows the camera view. Click the target -> its pixel coordinates.
3. Drive the pan/tilt servos with the arrow keys until the jet lands on it.
4. Press SPACE to record the (pixel, angle) pair.
5. Repeat for at least six well-spread targets, then press F to fit.

The fit is a least-squares linear model per axis, which is what the geometry
gives over this small working envelope. The tool reports the residual in
degrees; if the residual exceeds roughly one degree the mount has flexed or
the camera has moved, and re-fitting will not fix it - the mount must be made
rigid first, because the whole mapping assumes a fixed camera-to-nozzle
transform.

    python tools/calibrate_spray.py                 # interactive
    python tools/calibrate_spray.py --fit pairs.csv # fit from a recorded CSV

Keys: arrows = jog servos   SPACE = record   F = fit   T = test shot   Q = quit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agribot.config import load_config

Pair = Tuple[float, float, float, float]     # px, py, pan_deg, tilt_deg


def fit_axis(offsets: Sequence[float], angles: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares fit ``angle = centre + slope * offset``.

    Returns ``(centre_deg, deg_per_px, rms_residual_deg)``.
    """
    x = np.asarray(offsets, dtype=float)
    y = np.asarray(angles, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to fit an axis")
    if np.allclose(x, x[0]):
        raise ValueError("all samples share one pixel offset - spread the targets out")

    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    centre, slope = float(coefficients[0]), float(coefficients[1])
    residual = float(np.sqrt(np.mean((y - (centre + slope * x)) ** 2)))
    return centre, slope, residual


def fit_pairs(pairs: Sequence[Pair], frame_w: int, frame_h: int) -> dict:
    """Fit both axes and report the model plus its residuals."""
    if len(pairs) < 3:
        raise ValueError(f"need at least 3 samples, got {len(pairs)}")

    dx = [p[0] - frame_w / 2.0 for p in pairs]
    dy = [p[1] - frame_h / 2.0 for p in pairs]
    pan = [p[2] for p in pairs]
    tilt = [p[3] for p in pairs]

    pan_centre, pan_slope, pan_residual = fit_axis(dx, pan)
    tilt_centre, tilt_slope, tilt_residual = fit_axis(dy, tilt)

    return {
        "samples": len(pairs),
        "pan": {"centre_deg": round(pan_centre, 3),
                "deg_per_px": round(pan_slope, 6),
                "invert": pan_slope < 0,
                "rms_residual_deg": round(pan_residual, 3)},
        "tilt": {"centre_deg": round(tilt_centre, 3),
                 "deg_per_px": round(tilt_slope, 6),
                 "invert": tilt_slope < 0,
                 "rms_residual_deg": round(tilt_residual, 3)},
    }


def emit_yaml(fit: dict) -> None:
    pan, tilt = fit["pan"], fit["tilt"]
    print("\n" + "=" * 62)
    print("Paste into config/robot.yaml under targeting:")
    print("  pan:")
    print(f"    centre_deg: {pan['centre_deg']}")
    print(f"    deg_per_px: {abs(pan['deg_per_px']):.6f}")
    print(f"    invert: {str(pan['invert']).lower()}")
    print("  tilt:")
    print(f"    centre_deg: {tilt['centre_deg']}")
    print(f"    deg_per_px: {abs(tilt['deg_per_px']):.6f}")
    print(f"    invert: {str(tilt['invert']).lower()}")
    print("=" * 62)
    print(f"  fitted from {fit['samples']} samples")
    print(f"  pan residual  : {pan['rms_residual_deg']:.3f} deg")
    print(f"  tilt residual : {tilt['rms_residual_deg']:.3f} deg")
    worst = max(pan["rms_residual_deg"], tilt["rms_residual_deg"])
    if worst > 1.0:
        print(f"  WARNING: residual {worst:.2f} deg exceeds 1 deg. The mapping assumes")
        print("           a rigid camera-to-nozzle transform; check the mount for flex")
        print("           before trusting this fit.")
    print()


def load_pairs(path: Path) -> List[Pair]:
    pairs: List[Pair] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pairs.append((
                float(row["pixel_x"]), float(row["pixel_y"]),
                float(row["pan_deg"]), float(row["tilt_deg"]),
            ))
    return pairs


def save_pairs(pairs: Sequence[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pixel_x", "pixel_y", "pan_deg", "tilt_deg"])
        writer.writerows(pairs)


def interactive(cfg, out_csv: Path) -> int:  # pragma: no cover - needs hardware
    import cv2

    from agribot.hal.mcu_link import McuLink
    from agribot.vision.camera import Camera

    camera = Camera.from_config(cfg.camera)
    link = McuLink.from_config(cfg.mcu)
    if not camera.open():
        print("camera did not open")
        return 1
    if not link.open():
        print("MCU did not open")
        camera.release()
        return 1

    pan = float(cfg.targeting.pan.centre_deg)
    tilt = float(cfg.targeting.tilt.centre_deg)
    target_px: Optional[Tuple[int, int]] = None
    pairs: List[Pair] = []
    window = "AgriBot spray calibration"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal target_px
        if event == cv2.EVENT_LBUTTONDOWN:
            target_px = (x, y)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    link.send_aim(pan, tilt)
    print(__doc__.split("Keys:")[1].strip())

    try:
        while True:
            link.send_heartbeat()
            link.poll()
            ok, frame = camera.read()
            if not ok or frame is None:
                break

            if target_px:
                cv2.drawMarker(frame, target_px, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(frame,
                        f"pan={pan:.1f} tilt={tilt:.1f}  samples={len(pairs)}"
                        f"  target={target_px}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow(window, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 81 or key == ord("a"):
                pan = max(cfg.targeting.pan.min_deg, pan - 0.5)
            if key == 83 or key == ord("d"):
                pan = min(cfg.targeting.pan.max_deg, pan + 0.5)
            if key == 82 or key == ord("w"):
                tilt = min(cfg.targeting.tilt.max_deg, tilt + 0.5)
            if key == 84 or key == ord("s"):
                tilt = max(cfg.targeting.tilt.min_deg, tilt - 0.5)
            if key in (81, 82, 83, 84, ord("a"), ord("d"), ord("w"), ord("s")):
                link.send_aim(pan, tilt)
            if key == ord("t"):
                link.send_pump(True)
                cv2.waitKey(150)
                link.send_valve(True)
                cv2.waitKey(int(cfg.spray.burst_ms))
                link.send_valve(False)
                link.send_pump(False)
            if key == ord(" ") and target_px:
                pairs.append((float(target_px[0]), float(target_px[1]), pan, tilt))
                print(f"  recorded {len(pairs)}: px={target_px} pan={pan:.1f} "
                      f"tilt={tilt:.1f}")
                target_px = None
            if key == ord("f"):
                if len(pairs) < 3:
                    print("  need at least 3 samples")
                    continue
                save_pairs(pairs, out_csv)
                emit_yaml(fit_pairs(pairs, cfg.camera.width, cfg.camera.height))
    finally:
        link.stop()
        link.close()
        camera.release()
        cv2.destroyAllWindows()

    if pairs:
        save_pairs(pairs, out_csv)
        print(f"saved {len(pairs)} samples to {out_csv}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fit", type=Path, default=None,
                        help="fit from a recorded CSV instead of running interactively")
    parser.add_argument("--out", type=Path, default=Path("data/spray_calibration.csv"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.fit:
        pairs = load_pairs(args.fit)
        print(f"Fitting {len(pairs)} samples from {args.fit}")
        emit_yaml(fit_pairs(pairs, cfg.camera.width, cfg.camera.height))
        return 0

    return interactive(cfg, args.out)


if __name__ == "__main__":
    sys.exit(main())
