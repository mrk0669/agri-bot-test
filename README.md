# AgriBot — Autonomous Precision Robot for Sustainable Smart Agriculture

Software stack for **Robofest Gujarat 6.0**, Application-Based Robot category,
Thematic Track (f) — *Robots for Sustainability*.
Team: Visvesvaraya National Institute of Technology, Nagpur.

The robot performs three coupled missions in a real field arena: it **navigates
crop rows** using computer vision, **discriminates crops from weeds** in real
time, and applies a **targeted non-toxic spray** (or a physical mark) only on
the weeds, leaving the crops untouched.

> **Scope of this repository.** Everything except the mechanical CAD. The
> chassis, the dimensioned drawings and the renders come from the design
> sub-team and slot into Sections 3, 7, 8 and 10 of the proposal.

---

## Quick start

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Run a complete mission in simulation — no hardware needed:

```bash
agribot-sim --seconds 45 --rows 1 --report
```

Reproduce the sensor-fusion study from Section 5.3 of the proposal:

```bash
python tools/kalman_sim.py --sweep 8
```

Run the test suite — 418 tests, 90 % coverage:

```bash
python -m pytest -q
```

Or run every check in the verification document at once:

```bash
bash verify.sh
```

---

## What is in here

| Path | What it is |
|---|---|
| `src/agribot/vision/` | Line following, the three detector tiers, crop-protective fusion, IR fail-safe |
| `src/agribot/control/` | PID, differential mixing, the Kalman heading and distance filters |
| `src/agribot/mission/` | The deterministic mission state machine (Section 5.4) |
| `src/agribot/targeting/` | Pixel→pan/tilt solving and metered spray sequencing |
| `src/agribot/hal/` | Jetson↔MCU wire protocol, serial link, simulated MCU |
| `src/agribot/telemetry/` | Run logging and the judging-criteria metrics |
| `src/agribot/sim/` | Synthetic arena, sensor models, software-in-the-loop harness |
| `src/agribot/app/` | Runtime, live entry point, simulator, preflight checks |
| `firmware/agribot_mcu/` | Arduino firmware: motors, encoders, IMU, servos, solenoid |
| `tools/` | Calibration, tuning, training, export and benchmarking |
| `config/robot.yaml` | **Every** tunable in the system, in one file |
| `docs/` | Architecture, calibration procedure, operator runbook, verification |

---

## Architecture in one paragraph

Computation is split across two tiers. An **NVIDIA Jetson Orin Nano** runs all
perception and decision-making — deep-learning inference, OpenCV navigation,
target solving and the mission state machine. A **microcontroller on a custom
PCB** handles hard real-time actuation: motor PWM, quadrature encoder counting,
IMU acquisition, servo positioning and solenoid switching. They talk over a
compact checksummed serial protocol. That separation stops real-time interrupts
competing with GPU inference, and it means the MCU can enforce safety
(heartbeat watchdog, valve timeout) even if the Jetson hangs.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

---

## The three design decisions worth knowing

**1. Perception is deliberately redundant, with a zero-data fallback.**
A deterministic HSV-plus-geometry stage recognises the arena's green crop and
red weed markers — no training, very high frame rate, and it is the guaranteed
point scorer. A YOLO11-nano detector on TensorRT generalises to real plant
appearance. An open-vocabulary zero-shot detector needs no training data at
all. Any tier can be absent and the system still runs.

**2. The fusion rule is asymmetric, and that is the point.**
An action fires when the colour tier sees a red marker *or* the learned
detector reports a weed with high confidence. But **any** crop evidence
suppresses actuation outright, whatever the weed detectors say. False positives
on a weed cost a little fluid; false positives on a crop damage the plant the
robot exists to protect. Encoding that asymmetry into the *rule* rather than
into threshold tuning makes crop protection structural — it cannot be undone by
someone nudging a confidence value in the field.

**3. The encoder innovation gate is a fixed physical bound, not an N-sigma one.**
On loose soil the wheels spin, the encoder over-reads, and pure odometry gains
distance that was never travelled. The distance filter rejects any encoder
sample disagreeing with its prediction by more than 0.05 m/s — a bound set by
the acceleration the rover can physically reach in one encoder interval. The
textbook adaptive gate widens as the covariance grows during the coast and
eventually admits the very sample it exists to reject. That is not asserted
here; `tests/test_kalman.py` measures both gates side by side.

---

## Verified behaviour

Every number below is produced by the test suite or by a tool in `tools/`, not
by hand. See [docs/VERIFICATION.md](docs/VERIFICATION.md) for how to reproduce
each one.

**Sensor fusion** (`python tools/kalman_sim.py`), against the values published
in Section 5.3:

| Quantity | Single sensor A | Single sensor B | Kalman fused |
|---|---|---|---|
| Heading RMSE | 18.25° (gyro integrated) | 2.75° (magnetometer) | **0.27°** |
| Distance RMSE | 16.44 m (accelerometer) | 0.299 m (encoder) | **0.004 m** |
| Final distance error | — | +0.573 m | **−0.002 m** |
| Gyro bias identified | true 0.780 °/s | — | estimated 0.785 °/s |

Wheel-spin samples rejected by the gate: **100 %**. Clean samples wrongly
rejected: **0 %**. Worst case across 8 independent seeds: fused distance RMSE
below 0.008 m.

**Navigation** (`python tools/tune_pid.py --check`): settles to **0.22 mm** RMS
lateral error from a 4 cm / 4° initial offset — the proposal's "centred to
within a few millimetres", measured.

**Perception** (`python tools/bench_perception.py`): the colour tier plus line
following runs the full pipeline at **~195 fps** on a laptop CPU — a 145×
margin over what the mission needs at 0.18 m/s.

**End-to-end** (`python -m agribot.app.simulate`): a complete single-row
mission treats every weed, sprays **zero** crops, and reports a measured
millilitres-per-weed figure with roughly a 3× reduction against blanket
spraying over the same ground.

---

## Hardware

| Subsystem | Part |
|---|---|
| Compute | NVIDIA Jetson Orin Nano (TensorRT, INT8) |
| Real-time | Arduino Uno / Mega on a custom PCB |
| Camera | Global-shutter RGB (Arducam) or Raspberry Pi HQ IMX477 |
| IMU | BNO055 9-DOF fused (or MPU-9250) |
| Drive | 4 × 12 V DC gear motors with quadrature encoders |
| Ranging | 2 × HC-SR04; optional RPLiDAR A1, RealSense D435i |
| Spray | MG996R pan/tilt, 12 V diaphragm pump, NC solenoid, in-line flow sensor |
| Fail-safe | Pololu QTR-8A IR array (silent backup only — vision is primary) |

Static envelope: **30 × 30 × 30 cm**, as the rules mandate. The nozzle working
envelope extends beyond it during spraying, which the rules permit.

---

## Deploying to the robot

```bash
sudo ./deploy/install_jetson.sh
```

Then follow [docs/RUNBOOK.md](docs/RUNBOOK.md). The short version: flash the
firmware, run `preflight`, calibrate HSV and the spray mapping on the actual
arena surface, do a `--dry-run`, then run for real.

**Always calibrate at the venue.** Arena lighting is never the lighting the
thresholds were tuned under, and every colour gate reads from `config/robot.yaml`,
so re-tuning is the whole field-adaptation procedure —
see [docs/CALIBRATION.md](docs/CALIBRATION.md).

---

## Datasets for the learned tier

The recommended sequence, from the proposal's Annexure:

1. **Sesame Crop & Weed** (Kaggle, Apache-2.0, ~1300 images, already YOLO
   format) — small, clean, enough to prove the pipeline.
2. **Your own marker photographs** taken on the practice field and labelled in
   Roboflow. This is the set that matters most; fifty of your own images beat
   ten thousand of someone else's.
3. **CropAndWeed** or **DeepWeeds** only if more robustness is needed.

```bash
python tools/train_yolo.py --data datasets/sesame/data.yaml --epochs 80
python tools/export_tensorrt.py --weights data/runs/weeds/weights/best.pt  # on the Jetson
```

Licences should be re-checked before submission.

---

## Licence

MIT for the code in this repository. The cited datasets remain the property of
their respective authors and are used under their stated licences.
