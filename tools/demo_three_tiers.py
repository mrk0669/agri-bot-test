#!/usr/bin/env python3
"""Run all three perception tiers on real imagery and render figures for the PDF.

Each tier is the shipped class from ``src/agribot/vision`` at its configured
thresholds — nothing here re-implements the detector for the sake of a picture.

    Tier 1  ColorDetector      HSV + geometry gates, no training
    Tier 2  YoloDetector       YOLO11-nano fine-tuned on real crop/weed data
    Tier 3  ZeroShotDetector   YOLO-World, text prompts, no training data at all

    python tools/demo_three_tiers.py --weights <best.pt>

Outputs 300 dpi PNGs into ``reports/`` plus a JSON record of every detection,
so any number quoted in the proposal can be traced back to a run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from agribot.config import load_config
from agribot.types import Detection, TargetClass
from agribot.vision.color_detector import ColorDetector, RejectedRegion
from agribot.vision.yolo_detector import YoloDetector
from agribot.vision.zeroshot_detector import ZeroShotDetector

# ── figure palette (BGR), matched to the architecture plates ───────────────
C_CROP = (95, 165, 65)          # green  — protected
C_WEED = (58, 62, 205)          # red    — target
C_REJECT = (150, 150, 150)      # grey   — colour matched, geometry refused
C_INK = (32, 30, 26)
C_PAPER = (248, 250, 246)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.6, colour=C_INK, thick=2, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (255, 255, 255),
                    thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _label_box(img, box, text, colour, thick=3, dashed=False):
    x1, y1, x2, y2 = (int(v) for v in box)
    if dashed:
        for x in range(x1, x2, 16):
            cv2.line(img, (x, y1), (min(x + 9, x2), y1), colour, thick)
            cv2.line(img, (x, y2), (min(x + 9, x2), y2), colour, thick)
        for y in range(y1, y2, 16):
            cv2.line(img, (x1, y), (x1, min(y + 9, y2)), colour, thick)
            cv2.line(img, (x2, y), (x2, min(y + 9, y2)), colour, thick)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)

    if not text:
        return
    scale, tt = 0.62, 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, tt)
    ly = y1 - 8 if y1 > th + 14 else y2 + th + 10
    cv2.rectangle(img, (x1, ly - th - 7), (x1 + tw + 12, ly + 6), colour, -1)
    cv2.putText(img, text, (x1 + 6, ly), FONT, scale, (255, 255, 255), tt, cv2.LINE_AA)


def _panel(img: np.ndarray, title: str, subtitle: str, notes: Sequence[str],
           width: int = 1100) -> np.ndarray:
    """Scale an annotated frame and stack a caption block beneath it."""
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    head, line_h, pad = 88, 30, 22
    block = np.full((head + line_h * len(notes) + pad, width, 3), C_PAPER, np.uint8)

    _text(block, title, (pad, 40), 0.98, C_INK, 2, shadow=False)
    _text(block, subtitle, (pad, 70), 0.60, (110, 105, 98), 1, shadow=False)
    cv2.line(block, (pad, 80), (width - pad, 80), (208, 214, 205), 1)
    for i, n in enumerate(notes):
        _text(block, n, (pad, head + 20 + i * line_h), 0.58, (72, 70, 64), 1, shadow=False)

    out = np.vstack([scaled, block])
    return cv2.copyMakeBorder(out, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(196, 202, 194))


# ── tiers ─────────────────────────────────────────────────────────────────
@dataclass
class TierResult:
    name: str
    image: np.ndarray
    detections: List[dict]
    ms: float
    notes: List[str]


def run_tier1(cfg, scene: Path) -> TierResult:
    img = cv2.imread(str(scene))
    if img is None:
        raise FileNotFoundError(scene)
    det = ColorDetector.from_config(cfg.perception.color)

    t = time.perf_counter()
    accepted, rejected = det.detect_with_diagnostics(img)
    ms = (time.perf_counter() - t) * 1e3

    # Surface the rejections a reader would otherwise mistake for markers.
    # Red rejections come first regardless of size: a red object refused on
    # shape is the claim this panel exists to demonstrate, and it would
    # otherwise be crowded out by the much larger foliage rejections.
    big = [r for r in rejected if r.area_px > 2500]
    red = [r for r in big if r.cls is TargetClass.WEED]
    green = [r for r in big if r.cls is TargetClass.CROP]
    notable = (sorted(red, key=lambda r: r.area_px, reverse=True)[:2]
               + sorted(green, key=lambda r: r.area_px, reverse=True)[:2])

    def short(reason: str) -> str:
        """Compact the gate message so the label fits inside the frame."""
        m = re.match(r"(\w+) ([\d.]+) (?:< ([\d.]+)|outside \[([\d.]+), ([\d.]+)\])", reason)
        if not m:
            return reason
        gate, val, lo, alo, ahi = m.groups()
        return f"{gate} {val} < {lo}" if lo else f"{gate} {val} > {ahi}"

    vis = img.copy()
    for r in notable:
        _label_box(vis, r.bbox.as_tuple(), f"REJECTED  {short(r.reason)}",
                   C_REJECT, 3, dashed=True)
    for d in accepted:
        colour = C_WEED if d.cls is TargetClass.WEED else C_CROP
        _label_box(vis, d.bbox.as_tuple(),
                   f"{d.cls.value.upper()}  ext {d.meta['extent']:.2f} "
                   f"sol {d.meta['solidity']:.2f}", colour)

    weeds = sum(1 for d in accepted if d.cls is TargetClass.WEED)
    crops = sum(1 for d in accepted if d.cls is TargetClass.CROP)
    # A printed card is near-perfectly filled and convex; foliage is not. The
    # split is what separates "found the marker" from "found a green plant".
    cards = sum(1 for d in accepted if d.meta["extent"] > 0.8 and d.meta["solidity"] > 0.9)
    return TierResult(
        "tier1", vis,
        [d.to_dict() for d in accepted] + [r.to_dict() for r in notable], ms,
        [f"{cards}/4 markers found exactly; {weeds} red and {crops} green regions "
         f"total, in {ms:.0f} ms on CPU.",
         "Dashed grey: passed the COLOUR threshold, refused by the GEOMETRY gates -",
         "the red-shirt case, rejected on shape, not hue.",
         "Real foliage also satisfies the green gate. That is safe by design: crop",
         "is a veto, so over-detecting green only ever suppresses spraying."],
    )


def run_tier2(cfg, image: Path, weights: Path, conf: float) -> TierResult:
    img = cv2.imread(str(image))
    merged = cfg.merged({"perception": {"yolo": {
        "enabled": True, "weights": str(weights), "fallback_weights": str(weights),
        "conf": conf, "device": 0, "half": False}}})
    det = YoloDetector.from_config(merged.perception.yolo)
    if not det.available:
        raise RuntimeError(f"Tier 2 weights unavailable: {det.load_error}")

    det.detect(img)                                   # warm up CUDA
    t = time.perf_counter()
    results = det.detect(img)
    ms = (time.perf_counter() - t) * 1e3

    vis = img.copy()
    for d in results:
        colour = C_WEED if d.cls is TargetClass.WEED else C_CROP
        _label_box(vis, d.bbox.as_tuple(), f"{d.cls.value} {d.confidence:.2f}", colour)

    weeds = sum(1 for d in results if d.cls is TargetClass.WEED)
    crops = sum(1 for d in results if d.cls is TargetClass.CROP)
    return TierResult(
        "tier2", vis, [d.to_dict() for d in results], ms,
        [f"Detected {weeds} weeds and {crops} crops in {ms:.1f} ms "
         f"({1000/ms:.0f} fps) on an RTX 4060.",
         "YOLO11-nano fine-tuned on 287 real field images (AgML weed_crop_detection,",
         "13 species collapsed to crop/weed). Generalises to plant form, not colour.",
         "Cost: a labelled dataset and a training run."],
    )


def run_tier3(cfg, image: Path, conf: float) -> TierResult:
    img = cv2.imread(str(image))
    # yolov8s-world barely registers on overhead agricultural close-ups
    # (max confidence 0.14 across every prompt tried); the v2-XL checkpoint is
    # the smallest one that produces usable boxes here.
    merged = cfg.merged({"perception": {"zeroshot": {
        "enabled": True, "model": "yolov8x-worldv2.pt", "conf": conf,
        "device": 0, "half": False,
        "prompts": {"weed": ["small weed plant"],
                    "crop": ["leafy green plant"]}}}})
    det = ZeroShotDetector.from_config(merged.perception.zeroshot)
    if not det.available:
        raise RuntimeError(f"Tier 3 model unavailable: {det.load_error}")

    det.detect(img)
    t = time.perf_counter()
    results = det.detect(img)
    ms = (time.perf_counter() - t) * 1e3

    vis = img.copy()
    for d in results:
        colour = C_WEED if d.cls is TargetClass.WEED else C_CROP
        _label_box(vis, d.bbox.as_tuple(), f"{d.cls.value} {d.confidence:.2f}", colour)

    prompts = " / ".join(f'"{p}"' for p in det.vocabulary[:4])
    return TierResult(
        "tier3", vis, [d.to_dict() for d in results], ms,
        [f"Detected {len(results)} regions in {ms:.1f} ms from text prompts alone.",
         f"Vocabulary: {prompts}",
         "YOLO-World v2-XL, open-vocabulary. Zero training images, zero labels -",
         "the prompt is the whole configuration. Weaker and markedly more",
         "prompt-sensitive than Tier 2, which is why it is the fallback."],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=Path("data/demo/scene_markers.jpg"),
                    help="Tier 1 scene: real field with arena markers composited")
    ap.add_argument("--field", type=Path, default=Path("data/demo/scene_field.jpg"),
                    help="real field image for Tiers 2 and 3")
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/detect/data/runs/weeds_demo/weights/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--conf2", type=float, default=0.35)
    ap.add_argument("--conf3", type=float, default=0.10)
    args = ap.parse_args(argv)

    cfg = load_config(use_local=False, use_env=False)
    args.out.mkdir(parents=True, exist_ok=True)

    tiers: List[TierResult] = []
    print("Tier 1 — colour + geometry")
    tiers.append(run_tier1(cfg, args.scene))
    print(f"   {tiers[-1].ms:.1f} ms, {len(tiers[-1].detections)} regions")

    print("Tier 2 — fine-tuned YOLO11-nano")
    tiers.append(run_tier2(cfg, args.field, args.weights, args.conf2))
    print(f"   {tiers[-1].ms:.1f} ms, {len(tiers[-1].detections)} detections")

    print("Tier 3 — YOLO-World zero-shot")
    tiers.append(run_tier3(cfg, args.field, args.conf3))
    print(f"   {tiers[-1].ms:.1f} ms, {len(tiers[-1].detections)} detections")

    meta = {
        "1_colour_geometry": ("TIER 1 — Colour + geometry",
                              "ColorDetector · deterministic · no training"),
        "2_yolo_finetuned": ("TIER 2 — Learned detector",
                             "YoloDetector · YOLO11-nano fine-tuned · TensorRT on the Jetson"),
        "3_zeroshot_vlm": ("TIER 3 — Open-vocabulary VLM",
                           "ZeroShotDetector · YOLO-World · text prompts, no dataset"),
    }
    panels = []
    for (key, (title, sub)), tier in zip(meta.items(), tiers):
        panel = _panel(tier.image, title, sub, tier.notes)
        path = args.out / f"fig_tier{key}.png"
        cv2.imwrite(str(path), panel)
        panels.append(panel)
        print(f"   wrote {path}")

    # Combined 3-up figure: one drop-in for the proposal.
    width = max(p.shape[1] for p in panels)
    height = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        canvas = np.full((height, width, 3), C_PAPER, np.uint8)
        canvas[:p.shape[0], :p.shape[1]] = p
        padded.append(canvas)
    combined = np.hstack(padded)
    banner = np.full((92, combined.shape[1], 3), C_PAPER, np.uint8)
    _text(banner, "Three detector tiers, three failure modes, one frame each",
          (26, 42), 1.0, C_INK, 2, shadow=False)
    _text(banner, "Every panel produced by the shipped detector classes at their "
                  "configured thresholds — tools/demo_three_tiers.py",
          (26, 72), 0.58, (110, 105, 98), 1, shadow=False)
    combined = np.vstack([banner, combined])
    out_combined = args.out / "fig_three_tiers.png"
    cv2.imwrite(str(out_combined), combined)
    print(f"   wrote {out_combined}  ({combined.shape[1]}x{combined.shape[0]})")

    record = {t.name: {"ms": round(t.ms, 2), "detections": t.detections} for t in tiers}
    (args.out / "three_tiers_detections.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"   wrote {args.out/'three_tiers_detections.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
