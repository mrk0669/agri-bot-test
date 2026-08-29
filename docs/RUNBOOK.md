# Operator runbook

What to do, in order, on competition day. And what to do when something goes
wrong.

---

## Pre-run checklist

Work down this list. Do not skip ahead.

- [ ] Battery above 11.5 V, BMS connected
- [ ] Reservoir filled, no air in the line (fire two priming bursts into a tray)
- [ ] Camera lens clean, mount screws tight
- [ ] Wheels clean and dry — mud in the tread is what causes the wheel spin the
      filter has to gate out
- [ ] MCU enumerated: `ls -l /dev/agribot-mcu`
- [ ] Firmware flashed and current
- [ ] Calibration done **at this venue** — see [CALIBRATION.md](CALIBRATION.md)
- [ ] `python -m agribot.app.preflight` → **READY**
- [ ] Dry run down a practice row completed
- [ ] Enough space behind the robot for a person to reach the kill switch

---

## Running

**Dry run** — navigation and perception, valve inhibited. Always do this first
on a new arena:

```bash
python -m agribot.app.main --dry-run --rows 1
```

**Marker mode** — servo felt stamp instead of wet spray. Use for an indoor
demonstration, or if the venue forbids liquids:

```bash
python -m agribot.app.main --mode mark
```

**The real run:**

```bash
sudo systemctl start agribot
journalctl -u agribot -f
```

or directly, which is easier to stop and gives the summary on stdout:

```bash
python -m agribot.app.main
```

**Stopping.** `Ctrl-C`, or `sudo systemctl stop agribot`. Both go through the
orderly shutdown path: motors off, valve shut, pump off. If the process is
killed outright, the MCU's own valve timer closes the solenoid within 1.5 s.

---

## Reading the console

```
14:22:31 INFO    agribot.mission.fsm    FOLLOW_LINE -> STOP_AND_AIM (weed confirmed (track 3))
14:22:31 INFO    agribot.targeting.spray aiming at track 3: pan=78.4 tilt=81.2
14:22:32 INFO    agribot.targeting.spray spray event 2 on track 3: 1.84 ml (measured), total 3.68 ml over 2 weeds
```

Every state transition carries its reason. Every burst reports its volume and
whether that volume was **measured** or **nominal** — if you see `nominal`, the
flow sensor is not reporting and the sustainability figure for that run is an
estimate, not a measurement.

At the end you get the mission summary: line lock rate, weeds treated, crops
seen and vetoed, millilitres per weed, and the saving against blanket spraying.

---

## Troubleshooting

### The robot will not start

| Symptom | Cause | Fix |
|---|---|---|
| `MCU link did not open` | Wrong port, or permissions | `ls -l /dev/agribot-mcu`; check the user is in `dialout` |
| `camera did not open` | Device busy or wrong index | `v4l2-ctl --list-devices`; close any other viewer |
| Preflight: resolution mismatch | Driver fell back to a smaller mode | Force the mode, or re-run the spray calibration at the actual size |
| Preflight: watchdog vs creep | Config edited inconsistently | Raise `navigation.line_lost_stop_s` |

### It drives but wanders

1. Check the line lock rate in the summary. Below ~70 % means the **HSV
   threshold** is wrong for this lighting — recalibrate, do not touch the PID.
   Check `line_source` in `timeseries.csv` too: a run with many `ir` rows was
   being carried by the fail-safe array, which has no look-ahead. That is a
   vision problem, not a control one.
2. If lock is high and it still weaves, the gains are wrong for the surface:
   `tools/tune_pid.py --check`, then the table in
   [CALIBRATION.md](CALIBRATION.md#6-pid-gains).
3. A consistent offset to one side is mechanical — the camera is not square to
   the chassis. Fix it mechanically; do not trim it out with `ki`.

### It stops and will not continue

Look at the last transition in the log.

- `-> PAUSE (obstacle)` — something is inside the 25 cm stop band. Clear it and
  the robot resumes on its own.
- `-> RECOVER (blind and not progressing)` — it lost the line and did not cover
  the row-end distance. It will probe forward up to `recover_probe_m` and then
  declare a row end. If it does this mid-row, the line contrast is marginal:
  recalibrate.
- `-> ESTOP (MCU link lost)` — serial dropped. Check the cable and connector.
  ESTOP is deliberately not self-clearing; restart the mission.
- `-> ESTOP (excessive tilt)` — the robot is on a slope beyond `max_tilt_deg`
  or has ridden up on something.

### It sprays nothing

| Check | Command / where |
|---|---|
| Is spray enabled? | `spray.enabled` in the config; also `--dry-run` inhibits it |
| Is the reservoir above one dose? | Summary: `blocked_by_reservoir` |
| Are weeds being *detected*? | Summary: `weeds_detected`; if 0, recalibrate the red HSV |
| Are they being **vetoed**? | `grep crop_veto data/logs/<run>/events.jsonl` |

A high `crop_veto` count near green markers is the crop protection working
correctly. A high count *away* from any crop means the green gate is far too
wide — tighten `perception.color.crop`.

### It sprays a crop

**Stop the run.** This is the one failure the whole design exists to prevent,
and the software treats it as a hard failure: `main.py` exits with code 2 and
`metrics.crops_sprayed` is non-zero.

1. Find the event: `grep '"crops_sprayed"' data/logs/<run>/summary.json`
2. Almost always the green HSV gate is too narrow, so the crop was never
   classified as a crop and could not veto anything. Recalibrate `crop` with a
   **generous** gate.
3. Re-run the SIL suite before going out again:
   `python -m pytest tests/test_integration.py -q`

### Odometry looks wrong

Check `encoder_reject_rate` in the summary.

- **Near 0 % on a clean run** — normal.
- **High during visible wheel spin** — the gate is working; that is what it is
  for.
- **High with no spin** — the filter and the encoder disagree persistently.
  Check `robot.wheel_radius_m` and `encoder_ticks_per_rev`; a wrong scale
  factor looks exactly like permanent slip.

---

## After the run

Logs land in `data/logs/run_<timestamp>/`:

| File | Contents |
|---|---|
| `summary.json` | Every judging figure, in one place — read this first |
| `events.jsonl` | Transitions, spray bursts, crop vetoes, faults |
| `timeseries.csv` | 20 Hz signals: line error, PID, fused heading and distance |
| `agribot.log` | Full text log |

Re-run the sensor-fusion analysis on the **real** logged data:

```bash
python tools/kalman_sim.py --csv data/logs/run_<timestamp>/timeseries.csv
```

Ground-truth columns are absent on hardware, so RMSE against truth is reported
as unavailable rather than silently computed against zeros. The filter output,
the gate statistics and the bias estimate are all still produced — that is the
evidence that the filter behaved on the robot the way it behaves in the study.

For the judges, `summary.json` carries the numbers that map onto the scoring
criteria: line lock rate (navigation), weeds treated versus detected and
`crops_sprayed` (discrimination), and `ml_per_weed` with `measured: true`
(efficient spraying).
