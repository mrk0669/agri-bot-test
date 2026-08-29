#!/usr/bin/env python3
"""Export trained weights to a TensorRT engine for the Jetson Orin Nano.

    python tools/export_tensorrt.py --weights data/runs/weeds/weights/best.pt

**Run this ON the Jetson.** A TensorRT engine is specialised to the exact GPU
architecture, TensorRT version and JetPack build it was created on; an engine
exported on a laptop will not load on the Orin, and one exported on a different
JetPack will fail at load time with an unhelpful deserialisation error. The
runtime falls back to the ``.pt`` checkpoint when the engine is missing, so a
failed export degrades performance rather than the mission.

INT8 needs a calibration set of representative images; FP16 needs nothing and
is usually within a point or two of mAP, which is why it is the default here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def on_jetson() -> bool:
    """True if this looks like a Jetson (device-tree model string)."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            text = Path(path).read_bytes().decode("utf-8", "ignore").lower()
            if "jetson" in text or "orin" in text or "tegra" in text:
                return True
        except Exception:
            continue
    return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", default=True,
                        help="FP16 (default)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 - needs --calib-data")
    parser.add_argument("--calib-data", type=Path, default=None,
                        help="dataset YAML used for INT8 calibration")
    parser.add_argument("--workspace", type=int, default=4, help="GiB")
    parser.add_argument("--out", type=Path, default=Path("models"),
                        help="where to copy the finished engine")
    parser.add_argument("--force", action="store_true",
                        help="export even when this does not look like a Jetson")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"weights not found: {args.weights}")
        return 1

    if not on_jetson() and not args.force:
        print("This does not look like a Jetson.")
        print("A TensorRT engine is tied to the GPU architecture, TensorRT version")
        print("and JetPack build it was exported on, so an engine built here will")
        print("not load on the Orin. Run this on the robot, or pass --force if you")
        print("really mean to export locally.")
        return 1

    if args.int8 and not args.calib_data:
        print("--int8 requires --calib-data (a dataset YAML of representative images)")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. pip install -r requirements-ml.txt")
        return 1

    print(f"Exporting {args.weights} -> TensorRT "
          f"({'INT8' if args.int8 else 'FP16'}, imgsz={args.imgsz})")
    print("  this takes several minutes and a lot of RAM; do not run it with the")
    print("  mission running.")

    model = YOLO(str(args.weights))
    export_kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        half=not args.int8 and args.half,
        int8=args.int8,
        workspace=args.workspace,
        verbose=True,
    )
    if args.int8:
        export_kwargs["data"] = str(args.calib_data)

    start = time.perf_counter()
    engine_path = model.export(**export_kwargs)
    elapsed = time.perf_counter() - start

    engine = Path(engine_path)
    if not engine.is_file():
        print(f"export reported {engine_path} but no file was written")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "weeds_yolo11n.engine"
    shutil.copy2(engine, destination)

    print(f"\nExported in {elapsed:.0f} s")
    print(f"  engine : {destination}  ({destination.stat().st_size / 1e6:.1f} MB)")
    print("\nNow set in config/robot.yaml:")
    print("  perception:")
    print("    yolo:")
    print("      enabled: true")
    print(f"      weights: \"{destination.as_posix()}\"")
    print("\nThen verify with:")
    print("  python tools/bench_perception.py --frames 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
