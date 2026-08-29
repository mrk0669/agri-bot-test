#!/usr/bin/env python3
"""Interactive HSV threshold calibration for the line and the markers.

The single most valuable twenty minutes at the venue. Arena lighting is never
the lighting the thresholds were tuned under, and every colour gate in the
system reads from ``config/robot.yaml``, so re-tuning here and pasting the
result is the whole field-adaptation procedure.

    python tools/calibrate_hsv.py --target line
    python tools/calibrate_hsv.py --target weed --source 0
    python tools/calibrate_hsv.py --target crop --image shot.jpg

Keys:  s = print YAML for the current values   r = reset   q = quit
       p = pick: click a pixel to seed the range around it

The ``--sample`` mode needs no GUI at all: it takes the median HSV inside a
central box over several frames and prints a range around it, which is the
fallback when the Jetson is running headless over SSH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from agribot.config import load_config
from agribot.vision.camera import Camera, VideoFileCamera

WINDOW = "AgriBot HSV calibration"
TRACKBARS = [
    ("H min", 0, 179), ("H max", 179, 179),
    ("S min", 0, 255), ("S max", 255, 255),
    ("V min", 0, 255), ("V max", 255, 255),
]


def _initial_bounds(cfg, target: str) -> Tuple[List[int], List[int]]:
    """Seed the sliders from whatever the config currently holds."""
    if target == "line":
        return list(cfg.navigation.line.hsv_lower), list(cfg.navigation.line.hsv_upper)
    spec = cfg.perception.color.weed if target == "weed" else cfg.perception.color.crop
    first = spec.hsv_ranges[0]
    return list(first["lower"]), list(first["upper"])


def _emit_yaml(target: str, lower: List[int], upper: List[int]) -> None:
    print("\n" + "=" * 58)
    if target == "line":
        print("Paste into config/robot.yaml under navigation.line:")
        print(f"    hsv_lower: {lower}")
        print(f"    hsv_upper: {upper}")
    else:
        section = "weed" if target == "weed" else "crop"
        print(f"Paste into config/robot.yaml under perception.color.{section}:")
        print("      hsv_ranges:")
        print(f"        - {{ lower: {lower}, upper: {upper} }}")
        if target == "weed":
            print("      # Red wraps the hue circle: keep BOTH ranges. If the hue you")
            print("      # measured is near 0, add the mirrored range near 180 as well:")
            hi_lo = [max(0, 180 - upper[0]), lower[1], lower[2]]
            print(f"        - {{ lower: {hi_lo}, upper: [180, {upper[1]}, {upper[2]}] }}")
    print("=" * 58 + "\n")


def sample_mode(camera, frames: int, box_frac: float, target: str) -> int:
    """Headless calibration: median HSV of a central box over several frames."""
    samples = []
    for _ in range(frames):
        ok, frame = camera.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        bw, bh = int(w * box_frac), int(h * box_frac)
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        patch = cv2.cvtColor(frame[y0:y0 + bh, x0:x0 + bw], cv2.COLOR_BGR2HSV)
        samples.append(patch.reshape(-1, 3))

    if not samples:
        print("No frames captured.")
        return 1

    stacked = np.vstack(samples)
    median = np.median(stacked, axis=0)
    # Percentile spread rather than a fixed margin, so a uniformly lit marker
    # gets a tight gate and a shadowed one gets a loose one.
    low = np.percentile(stacked, 5, axis=0)
    high = np.percentile(stacked, 95, axis=0)

    lower = [int(max(0, low[0] - 5)), int(max(0, low[1] - 30)), int(max(0, low[2] - 30))]
    upper = [int(min(179, high[0] + 5)), int(min(255, high[1] + 30)),
             int(min(255, high[2] + 30))]

    print(f"Sampled {len(stacked)} pixels over {len(samples)} frames.")
    print(f"  median HSV : {median.astype(int).tolist()}")
    print(f"  5-95 pct   : {low.astype(int).tolist()} .. {high.astype(int).tolist()}")
    _emit_yaml(target, lower, upper)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--target", choices=["line", "weed", "crop"], default="line")
    parser.add_argument("--source", default=None,
                        help="camera index or video path (default: config)")
    parser.add_argument("--image", type=Path, default=None,
                        help="calibrate on a still image instead of a camera")
    parser.add_argument("--sample", action="store_true",
                        help="headless: sample a central box, no GUI needed")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--box", type=float, default=0.15,
                        help="central sample box, as a fraction of the frame")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    lower, upper = _initial_bounds(cfg, args.target)

    # -- acquire a source ---------------------------------------------------
    still = None
    camera = None
    if args.image:
        still = cv2.imread(str(args.image))
        if still is None:
            print(f"cannot read {args.image}")
            return 1
    else:
        source = args.source if args.source is not None else cfg.camera.get("source", 0)
        try:
            source = int(source)
        except (TypeError, ValueError):
            pass
        camera = (VideoFileCamera(source, loop=True) if isinstance(source, str)
                  else Camera.from_config(cfg.camera.merged({"source": source})))
        if not camera.open():
            print(f"cannot open camera source {source!r}")
            return 1

    if args.sample:
        if still is not None:
            from agribot.vision.camera import FrameListCamera
            camera = FrameListCamera([still], loop=True)
            camera.open()
        try:
            return sample_mode(camera, args.frames, args.box, args.target)
        finally:
            camera.release()

    # -- interactive --------------------------------------------------------
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(f"No GUI available ({exc}). Re-run with --sample for headless mode.")
        if camera:
            camera.release()
        return 1

    for name, _default, maximum in TRACKBARS:
        cv2.createTrackbar(name, WINDOW, 0, maximum, lambda _v: None)
    order = [("H min", lower[0]), ("H max", upper[0]),
             ("S min", lower[1]), ("S max", upper[1]),
             ("V min", lower[2]), ("V max", upper[2])]
    for name, value in order:
        cv2.setTrackbarPos(name, WINDOW, int(value))

    print(__doc__.split("Keys:")[1].split("The ``--sample``")[0].strip())

    while True:
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = camera.read()
            if not ok or frame is None:
                break

        lo = [cv2.getTrackbarPos(n, WINDOW) for n in ("H min", "S min", "V min")]
        hi = [cv2.getTrackbarPos(n, WINDOW) for n in ("H max", "S max", "V max")]

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        overlay = cv2.bitwise_and(frame, frame, mask=mask)

        coverage = 100.0 * cv2.countNonZero(mask) / mask.size
        cv2.putText(overlay, f"{args.target}  lower={lo} upper={hi}  {coverage:.1f}%",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(WINDOW, np.hstack([frame, overlay]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            _emit_yaml(args.target, lo, hi)
        if key == ord("r"):
            for name, value in order:
                cv2.setTrackbarPos(name, WINDOW, int(value))

    cv2.destroyAllWindows()
    if camera:
        camera.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
