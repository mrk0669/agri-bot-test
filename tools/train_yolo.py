#!/usr/bin/env python3
"""Train the Tier-2 crop/weed detector (Section 5.6, Annexure datasets).

The recommended sequence from the proposal:

1. Start with **Sesame Crop & Weed** (Kaggle, Apache-2.0, ~1300 images, already
   in YOLO format) - small, clean, and enough to prove the pipeline.
2. Augment with a **custom set of the actual green and red markers**
   photographed on the practice field and labelled in Roboflow. This is the set
   that matters most: the arena markers are what the robot must actually score
   on, and fifty of your own images beat ten thousand of someone else's.
3. Add **CropAndWeed** or **DeepWeeds** only if greater robustness is needed.

    python tools/train_yolo.py --data datasets/sesame/data.yaml --epochs 80
    python tools/train_yolo.py --data ... --model yolo11n.pt --imgsz 640

Class order must be ``[crop, weed]`` to match ``perception.yolo.class_map`` in
the config. The tool checks the dataset YAML for that and refuses to train on a
mismatch, because a silently transposed class map is a detector that sprays
every crop and spares every weed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

EXPECTED_CLASSES = ["crop", "weed"]


def check_dataset(data_yaml: Path) -> List[str]:
    """Validate the dataset descriptor. Returns a list of problems found."""
    problems: List[str] = []
    if not data_yaml.is_file():
        return [f"dataset YAML not found: {data_yaml}"]

    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        problems.append("dataset YAML has no 'names'")
    else:
        lowered = [str(n).strip().lower() for n in names]
        if lowered != EXPECTED_CLASSES:
            problems.append(
                f"class order is {lowered}, expected {EXPECTED_CLASSES}. "
                "Fix the dataset or update perception.yolo.class_map - a "
                "transposed map sprays crops and spares weeds."
            )

    root = data_yaml.parent
    for split in ("train", "val"):
        rel = spec.get(split)
        if not rel:
            problems.append(f"dataset YAML has no '{split}' split")
            continue
        path = (root / rel).resolve()
        if not path.exists():
            problems.append(f"{split} path does not exist: {path}")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="dataset YAML (ultralytics format)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="base weights: yolo11n.pt or yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", type=Path, default=Path("data/runs"))
    parser.add_argument("--name", default="weeds")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--check-only", action="store_true",
                        help="validate the dataset and exit without training")
    args = parser.parse_args(argv)

    problems = check_dataset(args.data)
    if problems:
        print("Dataset problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Dataset OK: {args.data} (classes {EXPECTED_CLASSES})")
    if args.check_only:
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. pip install -r requirements-ml.txt")
        return 1

    print(f"Training {args.model} for {args.epochs} epochs at {args.imgsz} px")
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        # A nano model on a small agricultural set benefits from aggressive
        # photometric augmentation - arena lighting will not match the dataset.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
        verbose=True,
    )

    # Ultralytics resolves a *relative* project path against its own configured
    # runs_dir, so the finished run is not necessarily at `project/name`.
    # Ask the trainer where it actually saved rather than reconstructing it.
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", "")
                    or (Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"  run directory: {save_dir}")
    if best.is_file():
        print(f"  best weights : {best}")
        print(f"  next step    : python tools/export_tensorrt.py --weights {best}")
        print("  then set perception.yolo.enabled: true in config/robot.yaml")
    try:
        metrics = getattr(results, "results_dict", {}) or {}
        if metrics:
            print("  metrics:")
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                        "metrics/precision(B)", "metrics/recall(B)"):
                if key in metrics:
                    print(f"    {key:<22} {metrics[key]:.4f}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
