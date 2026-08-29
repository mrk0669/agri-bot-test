#!/usr/bin/env python3
"""Benchmark the perception stack, tier by tier.

Answers the question the whole two-tier design turns on: does the pipeline
keep up at the speed the robot travels? At 0.18 m/s a marker crossing a 0.4 m
working envelope is in view for about two seconds, and the tracker needs three
consecutive frames to confirm, so anything above ~10 fps end to end has margin.

    python tools/bench_perception.py                  # synthetic frames
    python tools/bench_perception.py --source 0       # live camera
    python tools/bench_perception.py --frames 300

Reports per-stage latency so it is obvious *what* is slow: the colour tier
should be sub-millisecond, and the learned tier is the one that costs.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from agribot.config import load_config
from agribot.sim.arena import ArenaLayout, SimulatedArena
from agribot.types import TargetClass
from agribot.vision.color_detector import ColorDetector
from agribot.vision.fusion import PerceptionFusion, TargetTracker
from agribot.vision.line_follow import LineFollower
from agribot.vision.yolo_detector import YoloDetector
from agribot.vision.zeroshot_detector import ZeroShotDetector


def time_stage(fn: Callable, frames: List[np.ndarray], warmup: int = 3) -> Dict[str, float]:
    """Time a callable over the frames, discarding warm-up iterations."""
    for frame in frames[:warmup]:
        fn(frame)
    samples: List[float] = []
    for frame in frames:
        start = time.perf_counter()
        fn(frame)
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "max_ms": samples[-1],
        "fps": 1000.0 / statistics.fmean(samples) if samples else 0.0,
    }


def synth_frames(cfg, count: int) -> List[np.ndarray]:
    arena = SimulatedArena.from_config(cfg, ArenaLayout.default_demo())
    return [arena.render(0.2 + i * 0.01, 0.01, 0.0) for i in range(count)]


def live_frames(cfg, source, count: int) -> List[np.ndarray]:
    from agribot.vision.camera import Camera
    camera = Camera.from_config(cfg.camera.merged({"source": source}))
    if not camera.open():
        raise RuntimeError(f"cannot open camera {source!r}")
    try:
        frames = []
        for _ in range(count):
            ok, frame = camera.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        return frames
    finally:
        camera.release()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--source", default=None,
                        help="camera index for live frames (default: synthetic)")
    parser.add_argument("--force-yolo", action="store_true",
                        help="benchmark the learned tier even if disabled in config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        frames = live_frames(cfg, source, args.frames)
        origin = f"live camera {source!r}"
    else:
        frames = synth_frames(cfg, args.frames)
        origin = "synthetic arena"

    if not frames:
        print("no frames captured")
        return 1
    h, w = frames[0].shape[:2]
    print(f"Benchmarking {len(frames)} frames at {w}x{h} from {origin}\n")

    follower = LineFollower.from_config(cfg.navigation.line)
    colour = ColorDetector.from_config(cfg.perception.color)
    fusion = PerceptionFusion.from_config(cfg.perception.fusion)
    tracker = TargetTracker.from_config(cfg.perception.fusion)

    stages: List[tuple] = [
        ("line following (HSV + moments)", follower.process),
        ("colour tier (Tier 1)", colour.detect),
    ]

    yolo = None
    if cfg.perception.yolo.get("enabled", False) or args.force_yolo:
        yolo = YoloDetector.from_config(cfg.perception.yolo)
        if yolo.available:
            stages.append(("learned tier (Tier 2)", yolo.detect))
        else:
            print(f"  Tier 2 unavailable: {yolo.load_error}\n")

    zeroshot = None
    if cfg.perception.zeroshot.get("enabled", False):
        zeroshot = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        if zeroshot.available:
            stages.append(("zero-shot tier (Tier 3)", zeroshot.detect))
        else:
            print(f"  Tier 3 unavailable: {zeroshot.load_error}\n")

    def full_pipeline(frame):
        follower.process(frame)
        sets = [colour.detect(frame)]
        if yolo is not None and yolo.available:
            sets.append(yolo.detect(frame))
        if zeroshot is not None and zeroshot.available:
            sets.append(zeroshot.detect(frame))
        decision = fusion.fuse(*sets)
        tracker.update(decision.actionable)
        return tracker.confirmed_targets()

    stages.append(("FULL PIPELINE", full_pipeline))

    header = f"{'stage':<34}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'fps':>9}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, fn in stages:
        stats = time_stage(fn, frames)
        results[name] = stats
        print(f"{name:<34}{stats['mean_ms']:>8.2f}m{stats['median_ms']:>8.2f}m"
              f"{stats['p95_ms']:>8.2f}m{stats['max_ms']:>8.2f}m{stats['fps']:>9.1f}")

    print()
    full = results["FULL PIPELINE"]
    cruise = cfg.robot.cruise_speed_mps
    confirm = cfg.perception.fusion.get("confirm_frames", 3)
    envelope_m = 0.40
    dwell_s = envelope_m / cruise if cruise > 0 else float("inf")
    frames_available = dwell_s * full["fps"]

    print(f"  At {cruise} m/s a marker stays in a {envelope_m:.2f} m envelope for "
          f"{dwell_s:.1f} s,")
    print(f"  giving {frames_available:.0f} frames; the tracker needs {confirm} to "
          f"confirm.")
    if frames_available >= confirm * 3:
        print(f"  VERDICT: comfortable margin ({frames_available / confirm:.0f}x).")
        return 0
    if frames_available >= confirm:
        print("  VERDICT: adequate but tight - consider a lower cruise speed.")
        return 0
    print("  VERDICT: TOO SLOW. Reduce cruise speed, shrink imgsz, or disable a tier.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
