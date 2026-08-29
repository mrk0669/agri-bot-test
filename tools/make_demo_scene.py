#!/usr/bin/env python3
"""Build the Tier-1 demonstration scene.

Takes a real, permissively-licensed field photograph and composites the
competition's own markers onto it: green crop markers, red weed markers, and a
deliberate red **distractor** — an irregular piece of red cloth standing in for
the "red shirt" case the geometry gate exists to reject.

This is a composite, and the figure says so. It is also exactly what the arena
is: printed markers placed on a real field. The point of the demonstration is
not that the markers are real, it is that the *detector* is — the same
``ColorDetector`` the robot runs, at its shipped thresholds, deciding on real
soil, real foliage and real outdoor illumination.

    python tools/make_demo_scene.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

# Marker colours, matched to the arena stock the team will print on.
CROP_BGR = (58, 150, 62)
WEED_BGR = (46, 44, 196)
DISTRACTOR_BGR = (52, 48, 168)      # a duller, cloth-like red


def _shade(img: np.ndarray, box: Tuple[int, int, int, int], strength: float = 0.45) -> None:
    """Drop a soft contact shadow under a marker so it sits in the scene."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    pad = int((x2 - x1) * 0.16)
    sx1, sy1 = max(0, x1 + pad // 2), max(0, y1 + pad)
    sx2, sy2 = min(w, x2 + pad), min(h, y2 + pad)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    patch = img[sy1:sy2, sx1:sx2].astype(np.float32)
    img[sy1:sy2, sx1:sx2] = np.clip(patch * (1.0 - strength), 0, 255).astype(np.uint8)


def _texture(size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Per-pixel noise so a marker is not a flat synthetic rectangle."""
    return rng.normal(0.0, 6.0, (size[1], size[0], 1))


def place_marker(
    img: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    colour: Tuple[int, int, int],
    rng: np.random.Generator,
    tilt_deg: float = 0.0,
) -> Tuple[int, int, int, int]:
    """Composite one square card marker, lightly tilted and textured."""
    half = size // 2
    card = np.full((size, size, 3), colour, dtype=np.float32)
    card += _texture((size, size), rng)
    # A thin darker border, as a printed card on stiff board would show.
    cv2.rectangle(card, (0, 0), (size - 1, size - 1),
                  tuple(c * 0.7 for c in colour), max(2, size // 22))

    mask = np.full((size, size), 255, np.uint8)
    if abs(tilt_deg) > 0.1:
        m = cv2.getRotationMatrix2D((half, half), tilt_deg, 1.0)
        card = cv2.warpAffine(card, m, (size, size), borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, m, (size, size), borderValue=0)

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size
    h, w = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"marker at ({cx},{cy}) size {size} falls outside the frame")

    _shade(img, (x1, y1, x2, y2))
    roi = img[y1:y2, x1:x2].astype(np.float32)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    # Let a little of the scene's own luminance through, so the card picks up
    # the ambient light of the photograph instead of looking pasted on.
    ambient = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lift = ((ambient - ambient.mean()) * 0.22)[:, :, None]
    blended = np.clip(card + lift, 0, 255)
    img[y1:y2, x1:x2] = (roi * (1 - alpha) + blended * alpha).astype(np.uint8)
    return x1, y1, x2, y2


def place_distractor(
    img: np.ndarray, cx: int, cy: int, size: int, rng: np.random.Generator
) -> Tuple[int, int, int, int]:
    """Composite an irregular red cloth — the object geometry must reject.

    Deliberately red enough to pass any hue threshold, and deliberately the
    wrong *shape*: a crumpled, elongated, non-convex blob. Colour alone cannot
    tell it from a marker; extent, aspect and solidity can.
    """
    w = int(size * 3.4)
    h = int(size * 0.95)
    layer = np.zeros((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.uint8)

    # A discarded garment: a long crumpled body with a sleeve thrown out to one
    # side. Both properties are what actually distinguishes cloth from a card —
    # it is elongated, and it is deeply non-convex where the folds and the
    # sleeve cut into the outline.
    body = []
    for i in range(18):
        ang = 2 * np.pi * i / 18
        rx = (w / 2) * (0.55 + 0.42 * rng.random())
        ry = (h / 2) * (0.42 + 0.40 * rng.random())
        body.append([int(w / 2 + rx * np.cos(ang)), int(h / 2 + ry * np.sin(ang))])
    cv2.fillPoly(mask, [np.array(body, np.int32)], 255)

    # A sleeve flung out, and deep folds bitten back into the body.
    cv2.fillPoly(mask, [np.array([
        [int(w * 0.20), int(h * 0.52)], [int(w * 0.02), int(h * 0.10)],
        [int(w * 0.14), int(h * 0.04)], [int(w * 0.34), int(h * 0.44)],
    ], np.int32)], 255)
    for fx, fy, fr in ((0.46, 0.02, 0.42), (0.70, 0.98, 0.40), (0.28, 0.94, 0.30)):
        cv2.circle(mask, (int(w * fx), int(h * fy)), int(h * fr), 0, -1)

    layer[:] = DISTRACTOR_BGR
    layer += rng.normal(0.0, 11.0, (h, w, 1))
    # Fabric folds: a few darker streaks.
    for _ in range(7):
        p1 = (int(rng.random() * w), int(rng.random() * h))
        p2 = (int(rng.random() * w), int(rng.random() * h))
        cv2.line(layer, p1, p2, tuple(c * 0.72 for c in DISTRACTOR_BGR),
                 max(2, h // 18))

    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h
    ih, iw = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
        raise ValueError("distractor falls outside the frame")

    _shade(img, (x1, y1, x2, y2), 0.35)
    roi = img[y1:y2, x1:x2].astype(np.float32)
    alpha = (cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0)[:, :, None]
    img[y1:y2, x1:x2] = (roi * (1 - alpha)
                         + np.clip(layer, 0, 255) * alpha).astype(np.uint8)
    return x1, y1, x2, y2


#: Marker layouts, as fractions of the frame: (kind, cx, cy, size, tilt).
#: Fractional so a layout survives a change of source resolution.
LAYOUTS = {
    # Overhead soil-dominated field plot. Markers sit on bare soil between the
    # plants, which is where the arena actually places them - a marker buried
    # in a closed canopy is not separable by colour, because the canopy is the
    # same colour.
    "field": [
        ("weed", 0.20, 0.72, 0.115, -6.0),
        ("crop", 0.63, 0.28, 0.100, 4.0),
        ("weed", 0.80, 0.62, 0.075, 9.0),
        ("crop", 0.36, 0.15, 0.070, -3.0),
        ("distractor", 0.47, 0.88, 0.085, 0.0),
    ],
}


def build(
    src: Path,
    dst: Path,
    layout: str = "field",
    crop: Optional[Tuple[float, float, float, float]] = None,
    width: int = 1400,
    seed: int = 3,
) -> Path:
    """Composite the arena markers onto a real field photograph.

    Args:
        crop: optional ``(x0, y0, x1, y1)`` fractional crop applied first.
        width: output width; the layout is fractional so it scales with this.
    """
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"cannot read {src}")

    if crop:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop
        img = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].copy()

    img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    placed = []
    for kind, fx, fy, fs, tilt in LAYOUTS[layout]:
        cx, cy = int(fx * w), int(fy * h)
        size = int(fs * w)
        if kind == "distractor":
            placed.append((kind, place_distractor(img, cx, cy, size, rng)))
        else:
            colour = WEED_BGR if kind == "weed" else CROP_BGR
            placed.append((kind, place_marker(img, cx, cy, size, colour, rng, tilt)))

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"wrote {dst}  ({w}x{h})")
    for kind, box in placed:
        print(f"   {kind:11s} bbox={box}")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("data/demo/hires/0014-of-00016_0035.jpg"))
    ap.add_argument("--dst", type=Path, default=Path("data/demo/scene_markers.jpg"))
    ap.add_argument("--layout", default="field", choices=sorted(LAYOUTS))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)
    build(args.src, args.dst, layout=args.layout, width=args.width, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
