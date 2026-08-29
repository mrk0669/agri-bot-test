# Architecture

How the AgriBot software is put together, and why it is put together that way.
Section references are to the Robofest Gujarat 6.0 ideation proposal.

---

## 1. Two computational tiers

```
┌──────────────────────────── JETSON ORIN NANO ─────────────────────────────┐
│                                                                            │
│  Camera ──► LineFollower ─────────────► error ──► PID ──► DifferentialMixer│
│    │                                                              │        │
│    ├──► ColorDetector    (Tier 1) ─┐                              │        │
│    ├──► YoloDetector     (Tier 2) ─┼─► PerceptionFusion ──► TargetTracker  │
│    └──► ZeroShotDetector (Tier 3) ─┘        (crop veto)           │        │
│                                                    │              │        │
│                                                    ▼              ▼        │
│                                            PixelToAngleSolver  MissionFSM  │
│                                                    │              │        │
│                                                    └──► SprayController    │
│                                                                   │        │
│  MCU telemetry ──► FusionStack (Kalman heading + distance) ───────┘        │
│                                                                            │
│                          RunLogger  ·  MissionMetrics                      │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │  $payload*XX\n  @ 115200 baud
┌────────────────────────────────┴───────────────────────────────────────────┐
│                        MICROCONTROLLER (custom PCB)                        │
│  motor PWM · quadrature encoders · IMU · servos · solenoid · pump          │
│  ultrasonic · flow sensor · IR array · heartbeat watchdog · valve timer    │
└────────────────────────────────────────────────────────────────────────────┘
```

All heavy computation is consolidated on the Jetson; all time-critical
switching is delegated to the MCU. Sensors are routed to whichever tier
consumes them: imaging and ranging report to the Jetson, while inertial,
odometric and safety sensors report to the MCU. That keeps the control loop
latency low and independent of inference load.

The split is not just about throughput. Because the MCU owns the actuators, it
can enforce safety *without* the Jetson: if heartbeats stop arriving the
motors stop and the valve shuts, so a hung inference loop or a yanked USB lead
stops the robot instead of leaving it driving on the last command latched.

---

## 2. One tick of the runtime

`AgriBotRuntime.tick()` — the order is deliberate.

| # | Step | Why here |
|---|---|---|
| 1 | Drain the MCU, fuse the newest sample | Everything downstream reasons about the robot's *current* state |
| 2 | Grab a frame | |
| 3 | Extract the guidance line | |
| 4 | Run every enabled detector tier → fuse → track | Confirmation happens in the tracker, so the FSM only sees persistent targets |
| 5 | `MissionStateMachine.update()` | The **only** thing that produces a drive command |
| 6 | Send drive, run the spray sequence | |
| 7 | Heartbeat, then telemetry | Heartbeat last, so a tick that threw partway does not keep the watchdog fed |

Only stale frames are discarded, never queued: `McuLink.poll()` returns the
newest valid telemetry frame and drops older ones. Acting on a frame from
200 ms ago is worse than acting on nothing.

---

## 3. Navigation (Section 5.2)

The rules require navigation through a computer-vision technique, so the line
is tracked with the camera; the IR reflectance array is retained only as a
silent fail-safe.

1. Crop a region of interest near the bottom of the frame — react to the line
   immediately ahead of the wheels, not to distant curvature.
2. BGR → HSV, threshold on the line colour. HSV is far more robust to the
   uneven illumination of a real field than a fixed intensity threshold.
3. Morphological **opening then closing** — opening deletes soil speckle,
   closing then fills glare holes without re-growing what opening removed.
   The other order re-inflates the noise.
4. Select the line component (see below).
5. Centroid from image moments, `Cx = M10 / M00`.
6. Normalise the horizontal offset to `[-1, +1]`, so PID gains do not depend
   on capture resolution.

### Rejecting glare

Specular glare on wet soil saturates towards white, and adding a constant to
all three channels *lowers* saturation — so a glare patch lands squarely inside
a "bright, unsaturated" line threshold. Taking simply the largest blob steers
the robot towards a reflection of the sun.

Two shape gates separate them, both measured against the ROI so they survive a
resolution change:

* **Vertical span** — the guidance line runs through the ROI and spans
  essentially its full height. A glare patch is a disc and spans a fraction.
* **Broad *and* filled** — a line tilted far enough to be as wide as it is tall
  is a diagonal band, filling about half its bounding box. A disc that wide
  fills three quarters. Rejecting only on the *conjunction* keeps a steeply
  tilted line — exactly the frame a recovering robot needs — while dropping
  the disc.

Components are tried largest-first, so a real line beside a larger glare patch
is still found. When glare completely swallows the line the extractor reports
*lost* rather than guessing: the merged blob's centroid sits tens of pixels off
the true line, so the grace period and the IR fail-safe are the right response.

### The IR fail-safe

The rules require navigation by a computer-vision technique, so the camera is
the primary and only *navigating* sensor. The Pololu QTR-8A array is consulted
**only** after the vision pipeline has already reported the line lost, and the
observation it produces carries `source="ir"` so a run can always be audited
for what the robot was actually steering on.

It is deliberately conservative: the array sees a few centimetres of floor
directly under the chassis, so it can hold the robot across a glare patch or a
scuffed section but has no look-ahead and cannot anticipate a bend. It buys
seconds, not autonomy. Patterns that cannot be a single line — nothing lit, or
more than four of eight sensors lit — are rejected rather than averaged into a
meaningless centroid.

### Sign convention

One convention across the whole stack: **positive angular quantities are
counter-clockwise (left)**.

A line imaged to the *right* of centre gives a positive `LineObservation.error`.
The PID computes `setpoint − measurement` and so produces a *negative*
correction. `DifferentialMixer.mix()` speeds the left wheel and slows the
right, turning the robot right — towards the line, which is what a robot
sitting left of the row needs to do.

`mix()`, `turn()` and `wheel_speeds_to_body()` all agree on this. They did not
during development, and the closed loop diverged until they did.

---

## 4. Sensor fusion (Section 5.3)

Two independent Kalman filters, each fusing sources that fail in complementary
ways.

### Heading — 2 states, `[heading, gyro_bias]`

A gyroscope integrated over time is extremely smooth short-term but drifts
without bound, because its zero-rate bias is itself slowly varying. A
magnetometer is absolutely referenced and drift-free but noisy and vulnerable
to local magnetic disturbance.

Carrying the bias as an **explicit state** means the filter identifies and
subtracts the drift rather than merely smoothing its effect. Over the 40-second
run the fused estimate stays within a third of a degree while the integrated
gyroscope has walked 31° away.

The innovation is wrapped to `(-π, +π]` before use. Comparing raw angles across
the seam produces a ~2π innovation and a violent, wrong correction.

### Distance — 3 states, `[position, velocity, accel_bias]`

The accelerometer drives prediction; the wheel encoder corrects with a velocity
measurement.

The dominant field failure is not sensor noise but **wheel spin**: on loose
soil the wheels turn faster than the robot advances, the encoder over-reads,
and pure odometry gains distance that was never travelled.

The filter applies a **fixed physical innovation gate** — reject any encoder
sample disagreeing with the prediction by more than 0.05 m/s, a bound set by
the acceleration the rover can reach in one encoder interval.

Why fixed rather than adaptive N-sigma: while the filter coasts through a spin
its covariance grows, an N-sigma gate widens along with it, and the spin sample
is eventually admitted with high confidence — after which the state is
corrupted. Measured, on the shipped parameters: the fixed threshold is 0.0500
m/s after 60 s of coasting; a 3-sigma gate grows from 0.019 to 1.74 m/s over
the same coast and admits a 0.25 m/s spin after 14 s.

Two supporting behaviours:

* **First sample initialises.** The gate defends a *prediction*; at the first
  sample that "prediction" is the zero the filter was seeded with. Gating
  against it locks a filter started mid-motion out of ever acquiring.
* **Bounded coast.** After `max_consecutive_rejects` samples (2 s at 50 Hz) the
  filter re-acquires with inflated measurement noise, on the reasoning that two
  solid seconds of disagreement is more likely a drifted filter than a
  two-second spin. The cost — a longer spin *will* pull the estimate — is
  recorded in a test so it cannot change unnoticed.

---

## 5. Perception (Section 5.6)

| Tier | What it is | Role |
|---|---|---|
| 1 | HSV + geometry gating | The guaranteed point scorer. No training, very fast, robust in an unfamiliar arena |
| 2 | YOLO11-nano / YOLOv8-nano on TensorRT | Generalises to real plant appearance; redundancy under occlusion |
| 3 | YOLO-World open-vocabulary | An AI layer with **no dataset dependency** — text prompts, no training |

Tiers 2 and 3 load lazily and report `available == False` rather than raising,
so a missing `.engine` degrades to the colour tier instead of failing to start.

**Geometry gating is what separates a marker from a red object.** Area alone
admits a stray glove or a competitor's chassis. Extent (contour area over
bounding-rect area), aspect ratio and solidity (contour area over convex-hull
area) together reject shapes that are not compact marker-like blobs.

**Red wraps the hue circle.** Red occupies both ends of the OpenCV H range
(0–10 and 170–180) and needs two thresholds OR-ed together. A single range
silently misses half the red markers depending on lighting.

### The fusion rule

```
action  ⟸  (colour tier says weed)  OR  (learned tier says weed ≥ conf)
        AND NOT (any crop evidence ≥ veto_conf within IoU or radius)
```

The crop clause is a hard veto, not a weighted term. Detections of the same
physical weed from different tiers are merged first, so agreement between tiers
produces one action, not two — otherwise the robot sprays the same marker
twice.

Above the rule sits a small nearest-centroid tracker. A target must persist for
`confirm_frames` consecutive frames before the state machine will interrupt
travel for it; acting on a single frame is how a system sprays a glint of
sunlight. Sprayed tracks are retained twice as long as unsprayed ones, so a
marker briefly lost and re-acquired is not treated as fresh and hit again.

---

## 6. Mission sequencing (Section 5.4)

The route is dictated by the crop rows, so the planner is not searching a graph
for a shortest path — it manages a deterministic sequence of behaviours.

```
                ┌──────┐  line acquired
                │ INIT ├────────────────┐
                └──────┘                ▼
   ┌───────────────────────────► FOLLOW_LINE ◄───────────────┐
   │                            │    │    │                  │
   │  obstacle cleared          │    │    └── weed confirmed ─┼──► STOP_AND_AIM
   │                            │    │                        │        │
   ├── PAUSE ◄── obstacle ──────┘    │                        │        ▼
   │                                 │                        │      SPRAY
   │              blind & not moving │                        │        │
   ├── RECOVER ◄─────────────────────┤                        │        ▼
   │      │ probe exhausted          │ blind ≥ row_end_detect │    LOG_EVENT
   │      ▼                          ▼                        │        │
   │    ┌────────────────────────────────┐                    └────────┘
   └────┤ TURN  (rows left)              │
        │ MISSION_COMPLETE (last row)    │
        └────────────────────────────────┘
```

Safety conditions are evaluated **before** the per-state logic on every tick,
so no state can be written in a way that ignores them. An obstacle pre-empts
every state except an active burst — the nozzle must finish and shut rather
than be abandoned open.

Three details that are easy to get wrong:

* **Row end is decided on distance, not time.** The robot creeps blind until it
  has travelled `row_end_detect_m` without re-acquiring. The time limit is a
  *stall watchdog* — if that long passes without covering the distance, the
  robot is not making progress and something else is wrong.
* **RECOVER probes forward, bounded.** A purely stationary recovery deadlocks
  against a distance-based row-end test: a stopped robot can never travel the
  distance that would resolve it.
* **The turn closes on fused heading, and accumulates.** A timed turn is at the
  mercy of battery voltage and surface friction. And a 180° turn sits exactly
  on the ±180 wrap seam, so differencing against the start heading climbs to
  180 and then *falls* — the completion test is only true in a window the robot
  can step straight over. Accumulating small wrapped increments passes cleanly
  through and keeps counting.

---

## 7. Intervention (Section 5.7)

Pixel coordinates map linearly onto pan and tilt angles. The linear model is
what a rigid camera and fixed nozzle geometry actually produce over this small
working envelope, and its two coefficients per axis are directly measurable by
`tools/calibrate_spray.py` without full camera intrinsics.

**Elevated markers.** The rules allow signs on the floor or on a ~15 cm raised
surface. A pure image-to-angle map implicitly assumes every target is on the
ground plane, and that is what breaks on a raised marker: the same pixel row
means a different physical range. Where a depth camera supplies a true range,
the tilt is re-solved from geometry; otherwise the ground-plane solution is
used *and flagged*, so the log records which markers were engaged on an
assumption.

The burst sequence is a **non-blocking state machine** —
`AIMING → PRIMING → BURST → RECOVER`. A blocking `sleep(burst_ms)` would
suspend the ultrasonic safety layer and the heartbeat for the duration of every
burst. Ordering matters at both ends: the pump leads the valve so the line is
pressurised before the nozzle opens, and the valve shuts before the pump stops
so the line is never left pressurised behind a closed valve.

Volume is measured by the in-line flow sensor. Where no sensor is fitted, or
where no flow is registered, the nominal dose is used and the event is flagged
`measured=False` — an estimate is never silently presented as a measurement.

---

## 8. Configuration

`config/robot.yaml` is the single source of truth. Nothing in `src/` hard-codes
a threshold, gain or pin. Override layers, in increasing precedence:

1. `config/robot.yaml`
2. `config/robot.local.yaml` (per-machine, git-ignored)
3. `AGRIBOT_SECTION__KEY=value` environment variables
4. Programmatic overrides (CLI flags)

The environment layer exists so a systemd unit or a CI job can flip one value —
`AGRIBOT_SPRAY__ENABLED=false` — without editing files.

---

## 9. Testing strategy

| Layer | What it proves |
|---|---|
| Unit | Each component in isolation, including its failure modes |
| Study | `tests/test_kalman.py` locks the published Section 5.3 results and demonstrates the fixed-vs-adaptive gate difference |
| Software-in-the-loop | `tests/test_integration.py` runs the complete unmodified runtime against the mock MCU and synthetic arena |

The SIL harness is what makes the integration tests meaningful: the robot
steers on frames rendered from a pose its own drive commands produced, on a
virtual clock, deterministically. A test can therefore assert what the robot
*did* — converged onto the row, treated every weed, sprayed no crop — which no
amount of unit testing of the parts can establish.
