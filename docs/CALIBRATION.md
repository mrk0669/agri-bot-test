# Calibration

**Do this at the venue, on the actual arena surface, under the actual lighting.**
Arena lighting is never the lighting the thresholds were tuned under. Every
colour gate in the system reads from `config/robot.yaml`, so re-tuning and
pasting the result *is* the field-adaptation procedure — there is nothing else
to change.

Budget 30–40 minutes. Do them in this order; each depends on the one before.

---

## 0. Before you start

```bash
python -m agribot.app.preflight
```

Fix anything marked `[FAIL]`. Warnings are informational — the learned tier
being unavailable is expected and fine, since the colour tier is the guaranteed
scorer.

---

## 1. Camera mounting (do this once, then never touch it)

The mounting geometry is the most consequential mechanical decision in the
build, because one camera serves navigation, classification and targeting.

- The lower third of the frame must contain the guidance line immediately
  ahead of the wheels.
- The upper region must cover the marker working envelope, from the ground
  plane up to about 15 cm.
- **The mount must be rigid, and fixed to the chassis — not to the removable
  enclosure lid.** Any flex changes the pixel-to-angle calibration and
  therefore the spray accuracy. If you re-seat the camera, redo step 4.

Record the final `mount_pitch_deg` and `mount_height_m` in `config/robot.yaml`.

---

## 2. Guidance line HSV

```bash
python tools/calibrate_hsv.py --target line
```

Point the camera at the arena line. Drag the sliders until the line is solid
white in the mask and the soil is black. Press `s` to print the YAML, and paste
it under `navigation.line`.

Headless over SSH — put a patch of the line in the middle of the frame:

```bash
python tools/calibrate_hsv.py --target line --sample --frames 30
```

**What good looks like:** the line is one continuous blob spanning the full
height of the ROI, with no speckle. If soil is coming through, raise `V min`
first, then `S max`. If the line breaks up, lower `V min`.

Verify:

```bash
python -m agribot.app.simulate --seconds 10 --report   # sanity check the pipeline
```

---

## 3. Marker HSV — weed (red) and crop (green)

```bash
python tools/calibrate_hsv.py --target weed
python tools/calibrate_hsv.py --target crop
```

> **Red wraps the hue circle.** Red occupies both ends of the OpenCV H range
> (0–10 *and* 170–180). Keep **both** entries under `perception.color.weed.hsv_ranges`.
> A single range silently misses half the red markers depending on lighting —
> the tool prints the mirrored range for you.

**Get the crop gate right first, and make it generous.** The crop veto is what
protects the plants; a crop the detector misses is a crop that can be sprayed.
It is correct to have the green gate slightly too wide and the red gate
slightly too narrow — that bias costs you a missed weed, not a killed crop.

Check both classes together on a frame containing one of each:

```bash
python tools/bench_perception.py --source 0 --frames 60
```

---

## 4. Spray targeting (pixel → pan/tilt)

Fill the reservoir with plain water. Work over a tray.

```bash
python tools/calibrate_spray.py
```

For each of at least **six well-spread targets**:

1. Place the target where the camera sees it and the nozzle can reach it.
2. Click it in the video window.
3. Jog the servos with the arrow keys until the jet lands on it (`t` fires a
   test burst).
4. Press `SPACE` to record the pair.

Spread the targets across the frame — left, right, near, far. Six points
clustered in the middle produce a fit that is precise about nothing.

Press `f` to fit. Paste the printed block under `targeting`.

**Read the residual.** Under about one degree is good. Above that, the mount
has flexed or the camera has moved: the whole mapping assumes a fixed
camera-to-nozzle transform, so re-fitting will not rescue it. Make the mount
rigid and start step 4 again.

Include at least one target on the elevated (~15 cm) surface if the arena has
one, so the tilt range is exercised over its full span.

---

## 5. Flow sensor (turns the sustainability claim into a measurement)

Fire ten bursts into a measuring cylinder:

```bash
python - <<'EOF'
from agribot.config import load_config
from agribot.hal.mcu_link import McuLink
import time
cfg = load_config(); link = McuLink.from_config(cfg.mcu); link.open()
start = None
for i in range(10):
    link.send_heartbeat(); link.poll()
    if start is None: start = link.flow_ticks()
    link.send_pump(True);  time.sleep(cfg.spray.pump_spinup_ms / 1000)
    link.send_valve(True); time.sleep(cfg.spray.burst_ms / 1000)
    link.send_valve(False); link.send_pump(False); time.sleep(0.5)
    link.poll()
print("total ticks:", link.flow_ticks() - start)
link.stop(); link.close()
EOF
```

Read the collected volume, then set:

```
flow_ticks_per_ml = total_ticks / collected_ml
```

in `config/robot.yaml` under `spray`. Set `ml_per_burst_nominal` to
`collected_ml / 10` — it is the fallback used only if the sensor fails, and it
should be honest.

---

## 6. PID gains

The shipped gains are tuned against the closed-loop model and settle to about
0.2 mm RMS lateral error. Check them:

```bash
python tools/tune_pid.py --check
```

Only re-tune if the real robot oscillates or drifts, and if it does, tune
against the model first:

```bash
python tools/tune_pid.py --top 10
```

Then confirm on the robot with a `--dry-run` down the row. Symptoms:

| What you see | What to change |
|---|---|
| Weaves side to side, growing | `kp` too high, or `kd` too low |
| Slow to return to the line | `kp` too low |
| Twitchy, motors buzzing | `kd` too high — the derivative is amplifying centroid noise |
| Consistent offset to one side | Camera is not square to the chassis; fix it mechanically, do not trim with `ki` |

---

## 7. Final verification

```bash
python -m agribot.app.preflight
python -m agribot.app.main --dry-run --rows 1
```

Walk the robot down a practice row with the valve inhibited. Watch the log for
`crop_veto` events near the green markers — those are the crop protection
working, and they are the evidence for it.

Then, and only then, run for real.

---

## Where each value lives

| What you calibrated | Config key |
|---|---|
| Line HSV | `navigation.line.hsv_lower` / `hsv_upper` |
| Weed HSV (both ranges) | `perception.color.weed.hsv_ranges` |
| Crop HSV | `perception.color.crop.hsv_ranges` |
| Pixel → pan/tilt | `targeting.pan.*`, `targeting.tilt.*` |
| Flow sensor scale | `spray.flow_ticks_per_ml`, `spray.ml_per_burst_nominal` |
| PID gains | `navigation.pid.*` |
| Camera geometry | `camera.mount_pitch_deg`, `camera.mount_height_m` |

Keep venue-specific values in `config/robot.local.yaml` — it overrides
`robot.yaml` and is git-ignored, so the tuned-at-home defaults stay intact.
