#!/usr/bin/env python3
"""Convert the AgML weed/crop detection dataset into YOLO format.

Source: ``Project-AgML/weed_crop_detection`` on Hugging Face (CC BY 4.0) —
1,120 field images with 17,693 boxes over 13 species. Cited in the proposal's
Annexure family of crop/weed corpora.

The 13 species are collapsed onto the two classes the mission actually acts on,
in the order ``perception.yolo.class_map`` expects:

    0 = crop   Blackbean · Canola · Corn · Field Pea · Flax · Lentil ·
               Soybean · Sugar beet
    1 = weed   Horseweed · Kochia · Ragweed · Redroot Pigweed · Waterhemp

That collapse is the point. The robot does not need to know it is looking at
kochia rather than waterhemp; it needs to know whether to open the valve. A
detector trained on the two-class problem is smaller, faster and easier to keep
honest than one carrying thirteen species heads it will never use.

Source images are several thousand pixels on a side. They are resized to the
training size on the way out, which keeps the extracted dataset in the tens of
megabytes rather than gigabytes.

    python tools/prepare_weed_dataset.py --shards data/demo/agml --out data/demo/weeds
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SPECIES = {
    0: "Blackbean", 1: "Canola", 2: "Corn", 3: "Field Pea", 4: "Flax",
    5: "Horseweed", 6: "Kochia", 7: "Lentil", 8: "Ragweed",
    9: "Redroot Pigweed", 10: "Soybean", 11: "Sugar beet", 12: "Waterhemp",
}
WEED_IDS = {5, 6, 8, 9, 12}
#: YOLO class indices, matching perception.yolo.class_map in config/robot.yaml.
CROP, WEED = 0, 1


def to_binary(category: int) -> int:
    return WEED if category in WEED_IDS else CROP


def convert_shard(
    path: Path,
    img_dir: Path,
    lbl_dir: Path,
    imgsz: int,
    quality: int,
    stats: Dict[str, int],
) -> List[str]:
    """Extract one parquet shard. Returns the stems written."""
    import pyarrow.parquet as pq
    from PIL import Image

    written: List[str] = []
    pf = pq.ParquetFile(path)
    idx = 0
    for batch in pf.iter_batches(batch_size=4):
        for row in batch.to_pylist():
            idx += 1
            stem = f"{path.stem[-13:]}_{idx:04d}"
            try:
                im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            except Exception as exc:
                stats["decode_errors"] += 1
                print(f"    skip {stem}: {exc}")
                continue

            w0, h0 = im.size
            scale = imgsz / max(w0, h0)
            if scale < 1.0:
                im = im.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                               Image.LANCZOS)

            objs = row["objects"] or {}
            boxes = objs.get("bbox") or []
            cats = objs.get("categories") or []

            lines: List[str] = []
            for box, cat in zip(boxes, cats):
                # Source boxes are COCO absolute [x, y, w, h] on the ORIGINAL
                # image, so they are normalised against the original size and
                # are then resolution-independent.
                x, y, bw, bh = (float(v) for v in box[:4])
                if bw <= 1 or bh <= 1:
                    stats["degenerate_boxes"] += 1
                    continue
                cx, cy = (x + bw / 2) / w0, (y + bh / 2) / h0
                nw, nh = bw / w0, bh / h0
                if not (0 < cx < 1 and 0 < cy < 1):
                    stats["out_of_frame"] += 1
                    continue
                nw, nh = min(nw, 1.0), min(nh, 1.0)
                cls = to_binary(int(cat))
                stats["crop_boxes" if cls == CROP else "weed_boxes"] += 1
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not lines:
                stats["images_without_boxes"] += 1
                continue

            im.save(img_dir / f"{stem}.jpg", quality=quality)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(stem)
            stats["images"] += 1
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=Path("data/demo/agml"))
    ap.add_argument("--out", type=Path, default=Path("data/demo/weeds"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--val-fraction", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delete-shards", action="store_true",
                    help="remove each parquet once extracted (frees disk)")
    args = ap.parse_args(argv)

    shards = sorted(args.shards.glob("*.parquet"))
    if not shards:
        print(f"no parquet shards in {args.shards}")
        return 1

    staging_img = args.out / "_all" / "images"
    staging_lbl = args.out / "_all" / "labels"
    staging_img.mkdir(parents=True, exist_ok=True)
    staging_lbl.mkdir(parents=True, exist_ok=True)

    stats = dict(images=0, crop_boxes=0, weed_boxes=0, decode_errors=0,
                 degenerate_boxes=0, out_of_frame=0, images_without_boxes=0)
    all_stems: List[str] = []
    for shard in shards:
        print(f"  {shard.name} ({shard.stat().st_size/1e6:.0f} MB)")
        all_stems += convert_shard(shard, staging_img, staging_lbl,
                                   args.imgsz, args.quality, stats)
        if args.delete_shards:
            shard.unlink()
            print(f"    extracted and removed")

    if not all_stems:
        print("nothing extracted")
        return 1

    # -- split -------------------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(all_stems)
    n_val = max(1, int(len(all_stems) * args.val_fraction))
    splits = {"val": all_stems[:n_val], "train": all_stems[n_val:]}

    for split, stems in splits.items():
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (staging_img / f"{stem}.jpg").replace(args.out / split / "images" / f"{stem}.jpg")
            (staging_lbl / f"{stem}.txt").replace(args.out / split / "labels" / f"{stem}.txt")

    for leftover in (staging_img, staging_lbl):
        for f in leftover.glob("*"):
            f.unlink()
    (args.out / "_all" / "images").rmdir()
    (args.out / "_all" / "labels").rmdir()
    (args.out / "_all").rmdir()

    # Ultralytics resolves 'path' relative to its datasets root unless absolute.
    data_yaml = args.out / "data.yaml"
    data_yaml.write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: crop\n  1: weed\n",
        encoding="utf-8",
    )

    meta = {
        "source": "Project-AgML/weed_crop_detection (Hugging Face)",
        "license": "CC BY 4.0",
        "citation": "Upadhyay et al., Weed-crop dataset in precision agriculture, "
                    "Data in Brief 2025",
        "species_collapsed": {SPECIES[k]: ("weed" if k in WEED_IDS else "crop")
                              for k in sorted(SPECIES)},
        "counts": {**stats, "train": len(splits["train"]), "val": len(splits["val"])},
        "imgsz": args.imgsz,
    }
    (args.out / "dataset_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  images written : {stats['images']}  "
          f"(train {len(splits['train'])} / val {len(splits['val'])})")
    print(f"  boxes          : {stats['crop_boxes']} crop, {stats['weed_boxes']} weed")
    for k in ("decode_errors", "degenerate_boxes", "out_of_frame", "images_without_boxes"):
        if stats[k]:
            print(f"  {k:22s}: {stats[k]}")
    print(f"  dataset yaml   : {data_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
