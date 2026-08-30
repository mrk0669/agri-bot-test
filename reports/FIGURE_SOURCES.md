# Figure sources, licences and attribution

Everything needed to publish `fig_three_tiers.png` and its three single panels
in the Robofest Gujarat 6.0 proposal. **Copy the attribution lines into the
figure caption or the Annexure** — every source below requires credit.

---

## Images

### Tier 1 panel — `data/demo/scene_markers.jpg`

| | |
|---|---|
| **Base photograph** | AgML *Weed Crop Detection*, image `train-00014-of-00016 #0048` |
| **Source** | https://huggingface.co/datasets/Project-AgML/weed_crop_detection |
| **Licence** | **CC BY 4.0** — attribution required, commercial use allowed, no share-alike |
| **Citation** | Upadhyay, A., Villamil Mahecha, M., Mettler, J., Howatt, K., Aderholdt, W., Ostlie, M., Sun, X., et al. *Weed-crop dataset in precision agriculture: Resource for AI-based robotic weed control systems.* Data in Brief, 2025. |
| **Modification** | Green crop markers, red weed markers and one red cloth distractor were **composited** onto the photograph. Contact shadows and ambient-luminance blending were applied so the cards sit in the scene. |

> **The figure must state that the markers are composited.** The photograph is
> real; the markers are not. What is being demonstrated is the *detector*
> deciding on real soil, real foliage and real outdoor illumination — not that
> the markers were physically present.

Suggested caption line:

> Base photograph: AgML Weed Crop Detection (Upadhyay et al., 2025), CC BY 4.0.
> Arena markers composited for illustration; detection output is unmodified.

### Tier 2 and Tier 3 panels — `data/demo/scene_field.jpg`

| | |
|---|---|
| **Photograph** | AgML *Weed Crop Detection*, image `train-00014-of-00016 #0050` |
| **Source / licence / citation** | as above — **CC BY 4.0** |
| **Modification** | Resized to 1500 px on the long side. Otherwise unaltered. |
| **Important** | This image is in the **validation split** — held out of training, so the Tier 2 result is on data the model has never seen. Its hand-drawn labels (2 crops, 2 weeds) are what both tiers are scored against in the captions. |

> The Tier 1 base image (#0048) is from the *training* split. That is fine and
> deliberate: Tier 1 is a deterministic HSV-and-geometry detector that is never
> trained on anything, so no split applies to it. Only Tiers 2 and 3 need a
> held-out frame, and they use #0050.

### Downloaded but not used

`data/demo/candidates/` holds three Wikimedia Commons images fetched while
choosing a base photograph. None appear in the final figures. If you ever do
use one, note that `Titan FT35 FarmWise3.jpg` is **CC BY-SA 4.0**, whose
share-alike clause would extend to the figure containing it — which is why it
was not used here.

---

## Models

| Tier | Model | Licence | Weights |
|---|---|---|---|
| 2 | YOLO11-nano, fine-tuned | AGPL-3.0 (Ultralytics) | `runs/detect/data/runs/weeds_demo/weights/best.pt` |
| 3 | YOLO-World v2-XL (`yolov8x-worldv2.pt`) | AGPL-3.0 (Ultralytics) | downloaded at runtime |

> **Ultralytics is AGPL-3.0.** For a student competition submission this is
> normally fine, but AGPL has obligations if the software is distributed or
> offered as a network service. The colour tier — the guaranteed point scorer —
> has **no** such dependency: it is OpenCV only. Worth knowing before any
> commercialisation conversation.

---

## Reproducing the figures

```bash
python tools/prepare_weed_dataset.py --delete-shards      # dataset → YOLO format
python tools/train_yolo.py --data data/demo/weeds/data.yaml --model yolo11n.pt \
       --epochs 110 --imgsz 640 --batch 16 --device 0
# Tier 1 composite, onto the bright cracked-soil frame
python tools/make_demo_scene.py --src data/demo/pool/0014-of-00016_0048.jpg

# All three panels. Passing the held-out frame by its dataset name is what
# lets the tool find its label file and score the captions against truth.
python tools/demo_three_tiers.py --field data/demo/pool/0014-of-00016_0050.jpg
```

Raw detections, with confidences and geometry metrics for every box drawn, are
written to `reports/three_tiers_detections.json`.

---

## Measured results

Hardware: RTX 4060 Laptop GPU (the Jetson Orin Nano will be slower; the
proposal's ~18 fps literature figure is the number to quote for on-robot Tier 2).

Ground truth for the Tier 2/3 frame (#0050): **2 crops, 2 weeds**, hand-labelled
in the source dataset. Detections are matched to labels at IoU > 0.3.

| Tier | What it needed | Result on these frames | Latency |
|---|---|---|---|
| 1 Colour + geometry | nothing | **4/4 markers** found exactly; red cloth rejected on `aspect 3.37 > 3.00` | 51 ms CPU |
| 2 YOLO11n fine-tuned | 287 labelled images | **3/4 correct**, 1 wrong class, 0 missed | 48 ms |
| 3 YOLO-World v2-XL | nothing but text | **3/4 correct**, 1 wrong class, 0 missed | 124 ms |

Both learned tiers score 3/4 on this frame but fail on *different* objects —
Tier 2 calls the small tray crop a weed, Tier 3 calls the large broadleaf crop a
weed. Neither miss is hidden: the captions state the score, and
`three_tiers_detections.json` carries every box.

> **On the two misclassifications.** Both are a crop called a weed, which is the
> direction that matters. On the robot this does not reach the nozzle: the
> crop veto fires on *any* crop evidence, and the colour tier sees the green
> marker independently. It is still the failure mode to watch, and it is the
> reason the fusion rule is asymmetric rather than a confidence comparison.

Tier 2 validation set (63 held-out images, 757 instances):

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| all | 0.782 | 0.745 | **0.802** | 0.452 |
| crop | 0.761 | 0.708 | 0.781 | 0.442 |
| weed | 0.803 | 0.781 | **0.822** | 0.462 |
