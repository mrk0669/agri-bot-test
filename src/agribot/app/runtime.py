"""The AgriBot runtime - one loop that owns every subsystem.

Order within a tick is deliberate and is the thing most worth understanding:

1. **Drain the MCU first.** Everything downstream reasons about the robot's
   current state, so telemetry is read before any decision is made, not after.
2. **Fuse before deciding.** The heading and distance filters are stepped with
   the fresh sample, so the state machine sees the same estimate the log records.
3. **Perceive, then fuse perception, then track.** Confirmation happens in the
   tracker, so the state machine only ever sees targets that have persisted.
4. **Decide.** The state machine is the only thing that produces a drive command.
5. **Actuate.** Drive, then aim/spray, then heartbeat - the heartbeat last so
   that a tick which threw partway through does not keep the watchdog fed.

The same class runs live and in simulation; only the camera and transport
differ. That is what makes the software-in-the-loop test meaningful.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..control.differential import DifferentialMixer
from ..control.kalman import FusionStack
from ..control.pid import PID
from ..mission.state_machine import MissionStateMachine
from ..mission.states import MissionInputs, MissionOutput, MissionState, TERMINAL_STATES
from ..targeting.pixel_to_angle import PixelToAngleSolver
from ..targeting.spray_controller import SprayController
from ..telemetry.logger import RunLogger
from ..telemetry.metrics import MissionMetrics
from ..types import DriveCommand, LineObservation, TargetClass
from ..utils.logging_setup import get_logger
from ..utils.rates import RateLimiter
from ..vision.color_detector import ColorDetector
from ..vision.fusion import PerceptionFusion, TargetTracker
from ..vision.ir_failsafe import IrLineSensor
from ..vision.line_follow import LineFollower
from ..vision.yolo_detector import YoloDetector
from ..vision.zeroshot_detector import ZeroShotDetector

__all__ = ["AgriBotRuntime"]

log = get_logger("app.runtime")


class AgriBotRuntime:
    """Owns and sequences every subsystem for one mission."""

    def __init__(
        self,
        cfg,
        camera,
        mcu_link,
        logger: Optional[RunLogger] = None,
        clock: Callable[[], float] = time.monotonic,
        dry_run: bool = False,
    ):
        self.cfg = cfg
        self.camera = camera
        self.mcu = mcu_link
        self.logger = logger
        self._clock = clock
        self.dry_run = dry_run

        # -- perception -----------------------------------------------------
        self.line_follower = LineFollower.from_config(cfg.navigation.line)
        self.ir_failsafe = IrLineSensor.from_config(cfg.navigation)
        self.color_detector = (
            ColorDetector.from_config(cfg.perception.color)
            if cfg.perception.color.get("enabled", True) else None
        )
        self.yolo = (
            YoloDetector.from_config(cfg.perception.yolo)
            if cfg.perception.yolo.get("enabled", False) else None
        )
        self.zeroshot = (
            ZeroShotDetector.from_config(cfg.perception.zeroshot)
            if cfg.perception.zeroshot.get("enabled", False) else None
        )
        self.perception_fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        self.tracker = TargetTracker.from_config(cfg.perception.fusion)
        # Crops are tracked too, purely so that one plant seen over fifty
        # frames counts as one crop encountered rather than fifty.
        self.crop_tracker = TargetTracker.from_config(cfg.perception.fusion)

        # -- control --------------------------------------------------------
        self.pid = PID.from_config(cfg.navigation.pid)
        self.mixer = DifferentialMixer.from_config(cfg.robot)
        self.state_fusion = FusionStack.from_config(cfg.fusion)
        self.fsm = MissionStateMachine.from_config(cfg, self.pid, self.mixer)

        # -- intervention ---------------------------------------------------
        self.solver = PixelToAngleSolver.from_config(cfg.targeting, cfg.camera)
        self.spray = SprayController.from_config(
            cfg.spray,
            cfg.targeting,
            set_aim=self._set_aim,
            set_pump=self._set_pump,
            set_valve=self._set_valve,
            read_flow_ticks=self._read_flow,
            set_marker=self._set_marker,
            clock=clock,
        )
        if dry_run:
            self.spray.enabled = False

        # -- bookkeeping ----------------------------------------------------
        self.metrics = MissionMetrics(
            swath_m=cfg.robot.bounding_box_cm.get("width", 30.0) / 100.0
        )
        self.metrics.spray = self.spray.stats
        self._heartbeat = RateLimiter(cfg.mcu.get("heartbeat_hz", 20), clock)
        self._sampler = RateLimiter(cfg.telemetry.get("timeseries_rate_hz", 20), clock)
        self._last_tick_t: Optional[float] = None
        self._start_t: Optional[float] = None
        self._line_error_sum = 0.0
        self.frames_on_ir_failsafe = 0
        self._seen_weed_tracks: set = set()
        self._seen_crop_tracks: set = set()
        self.last_frame: Optional[np.ndarray] = None
        self.last_output: Optional[MissionOutput] = None
        self.last_decision = None

    # -- MCU callbacks used by the spray controller -------------------------
    def _set_aim(self, pan_deg: float, tilt_deg: float) -> None:
        self.mcu.send_aim(
            pan_deg, tilt_deg,
            self.cfg.targeting.pan.get("servo_channel", 0),
            self.cfg.targeting.tilt.get("servo_channel", 1),
        )

    def _set_pump(self, on: bool) -> None:
        self.mcu.send_pump(on)

    def _set_valve(self, open_: bool) -> None:
        self.mcu.send_valve(open_)

    def _set_marker(self, degrees: float) -> None:
        self.mcu.send_marker(degrees)

    def _read_flow(self) -> int:
        return self.mcu.flow_ticks()

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> bool:
        """Open the hardware. Returns False if anything essential is missing."""
        if not self.mcu.open():
            log.error("MCU link did not open")
            return False
        if not self.camera.open():
            log.error("camera did not open")
            return False
        self.mcu.reset_encoders()
        self.mcu.send_heartbeat()
        self._set_aim(self.cfg.targeting.pan.centre_deg,
                      self.cfg.targeting.tilt.centre_deg)
        self._start_t = self._clock()
        self.fsm.reset(self._start_t)
        if self.logger:
            self.logger.event("startup", config=self.cfg.get("_meta", {}),
                              dry_run=self.dry_run)
        log.info("runtime ready (dry_run=%s)", self.dry_run)
        return True

    def shutdown(self, reason: str = "normal") -> MissionMetrics:
        """Stop everything and finalise the metrics. Safe to call twice."""
        log.info("shutting down: %s", reason)
        try:
            self.spray.abort()
            self.mcu.stop()
        finally:
            self.camera.release()
            self.mcu.close()

        self.metrics.duration_s = (
            self._clock() - self._start_t if self._start_t is not None else 0.0
        )
        self.metrics.distance_m = self.state_fusion.distance_m
        self.metrics.rows_completed = self.fsm.rows_done
        self.metrics.state_transitions = len(self.fsm.transitions)
        if self.metrics.frames_processed:
            self.metrics.mean_abs_line_error = (
                self._line_error_sum / self.metrics.frames_processed
            )
        self.metrics.encoder_samples = self.state_fusion.distance.n_updates + \
            self.state_fusion.distance.n_rejected
        self.metrics.encoder_rejected = self.state_fusion.distance.n_rejected

        if self.logger:
            self.logger.event("shutdown", reason=reason)
            self.logger.write_summary({
                "reason": reason,
                "metrics": self.metrics.to_dict(),
                "mission": self.fsm.summary(),
                "mcu": self.mcu.stats(),
                "detectors": {
                    "yolo": self.yolo.stats() if self.yolo else None,
                    "zeroshot": self.zeroshot.stats() if self.zeroshot else None,
                },
            })
            self.logger.close()
        return self.metrics

    # -- one iteration ------------------------------------------------------
    def tick(self) -> Optional[MissionOutput]:
        """Run one full control iteration. Returns None if no frame was available."""
        now = self._clock()
        dt = 0.0 if self._last_tick_t is None else now - self._last_tick_t
        self._last_tick_t = now

        # 1. Drain the MCU and fuse the newest sample.
        telemetry = self.mcu.poll()
        if telemetry is not None:
            self._fuse(telemetry)

        # 2. Frame.
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        self.last_frame = frame
        self.metrics.frames_processed += 1

        # 3. Navigation signal. Vision is primary and is what the rules
        #    require; the IR array is consulted only once vision has already
        #    given up, and the observation is tagged so the log shows it.
        line = self.line_follower.process(frame)
        if not line.found and telemetry is not None:
            fallback = self.ir_failsafe.observe(telemetry.ir_array)
            if fallback.found:
                line = fallback
                self.frames_on_ir_failsafe += 1
        if line.found:
            self.metrics.frames_line_found += 1
            self._line_error_sum += abs(line.error)
            self.metrics.max_abs_line_error = max(
                self.metrics.max_abs_line_error, abs(line.error))

        # 4. Perception -> fusion -> tracking.
        targets = self._perceive(frame)

        # 5. Decide.
        inputs = self._build_inputs(now, dt, line, targets, telemetry)
        output = self.fsm.update(inputs)
        self.last_output = output

        if output.transitioned and self.logger:
            last = self.fsm.transitions[-1] if self.fsm.transitions else None
            if last:
                self.logger.transition(last[1], last[2], last[3])

        # 6. Actuate.
        self._actuate(output, now)

        # 7. Telemetry.
        if self._sampler.due():
            self._sample(now, line, output, telemetry)

        return output

    def run(self, max_iterations: Optional[int] = None) -> MissionMetrics:
        """Run until the mission terminates, the camera ends, or the cap is hit."""
        if not self.setup():
            return self.shutdown("setup failed")

        reason = "mission complete"
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                output = self.tick()
                iterations += 1
                if output is None:
                    # No frame: the camera is exhausted (replay) or dropped one.
                    if not self.camera.is_open:
                        reason = "camera closed"
                        break
                    continue
                if output.state in TERMINAL_STATES:
                    reason = f"terminal state {output.state.value}"
                    break
            else:
                reason = "iteration limit"
        except KeyboardInterrupt:  # pragma: no cover - operator action
            reason = "interrupted"
        except Exception as exc:  # pragma: no cover - keep the robot safe
            log.exception("runtime error")
            self.metrics.faults.append(str(exc))
            reason = f"error: {exc}"
        return self.shutdown(reason)

    # -- internals ----------------------------------------------------------
    def _fuse(self, telemetry) -> None:
        """Step the heading and distance filters with one telemetry frame."""
        imu = telemetry.imu
        enc = telemetry.encoders
        if imu is None:
            return
        self.state_fusion.step(
            t=telemetry.t,
            gyro_z=imu.gyro_z,
            accel_x=imu.accel_x,
            mag_heading=imu.mag_heading_rad,
            encoder_vel=enc.linear_mps if enc is not None else None,
        )

    def _perceive(self, frame: np.ndarray) -> List:
        """Run every enabled tier, fuse, track, and return confirmed targets."""
        detection_sets = []
        if self.color_detector is not None:
            detection_sets.append(self.color_detector.detect(frame))
        if self.yolo is not None:
            detection_sets.append(self.yolo.detect(frame))
        if self.zeroshot is not None:
            detection_sets.append(self.zeroshot.detect(frame))

        decision = self.perception_fusion.fuse(*detection_sets)
        self.last_decision = decision

        # One plant seen across many frames is one crop encountered, so crops
        # are counted by track identity rather than per-frame detection.
        self.crop_tracker.update(decision.crops)
        for track in self.crop_tracker.confirmed_targets():
            if track.track_id not in self._seen_crop_tracks:
                self._seen_crop_tracks.add(track.track_id)
                self.metrics.crops_seen += 1

        for detection, reason in decision.vetoed:
            self.metrics.crop_vetoes += 1
            if self.logger:
                self.logger.veto(reason, detection.to_dict())

        self.tracker.update(decision.actionable)
        confirmed = self.tracker.confirmed_targets()
        for track in confirmed:
            if track.track_id not in self._seen_weed_tracks:
                self._seen_weed_tracks.add(track.track_id)
                self.metrics.weeds_detected += 1
        return confirmed

    def _build_inputs(self, now, dt, line, targets, telemetry) -> MissionInputs:
        ranges = telemetry.ranges if telemetry else None
        nearest = ranges.min_m if ranges else math.inf
        tilt_ok = True
        if telemetry and telemetry.imu:
            limit = self.cfg.safety.get("max_tilt_deg", 25.0)
            tilt_ok = (abs(telemetry.imu.roll_deg) <= limit
                       and abs(telemetry.imu.pitch_deg) <= limit)
        return MissionInputs(
            t=now,
            dt=dt,
            line=line,
            targets=targets,
            distance_m=self.state_fusion.distance_m,
            heading_deg=self.state_fusion.heading_deg,
            nearest_obstacle_m=nearest,
            spray_busy=self.spray.busy,
            spray_ready=self.spray.can_spray(now),
            mcu_ok=self.mcu.link_ok,
            tilt_ok=tilt_ok,
            battery_v=telemetry.battery_v if telemetry else 12.0,
        )

    def _actuate(self, output: MissionOutput, now: float) -> None:
        """Send the drive command, run the spray sequence, feed the watchdog."""
        self.mcu.send_drive(output.drive.left, output.drive.right)

        if output.engage_target is not None and not self.spray.busy:
            self._begin_spray(output.engage_target, now)

        event = self.spray.update(now)
        if event is not None:
            self.metrics.weeds_treated += 1
            self.tracker.mark_sprayed(event.track_id)
            if self.logger:
                self.logger.spray_event(event)

        if self._heartbeat.due():
            self.mcu.send_heartbeat()

    def _begin_spray(self, track, now: float) -> None:
        """Solve the aim for a confirmed target and start the burst."""
        # Structural guarantee, checked rather than trusted: a crop-classified
        # track must never reach actuation. If one ever does, that is a defect
        # and the run records it instead of quietly spraying the plant.
        if track.cls is not TargetClass.WEED:
            self.metrics.crops_sprayed += 1
            self.metrics.faults.append(
                f"non-weed track {track.track_id} reached actuation")
            log.error("refusing to actuate on a %s track", track.cls.value)
            return

        cx, cy = track.centroid
        aim = self.solver.solve(cx, cy)
        if not self.spray.begin(
            track.track_id,
            aim,
            distance_m=self.state_fusion.distance_m,
            detector_source=track.source.value,
            confidence=track.confidence,
        ):
            return
        self.fsm.notify_spray_started(now)

    def _sample(self, now, line, output, telemetry) -> None:
        if self.logger is None:
            return
        imu = telemetry.imu if telemetry else None
        enc = telemetry.encoders if telemetry else None
        # `self._start_t or now` would collapse to `now` whenever the run began
        # at t = 0.0 - which is exactly what a simulated clock does - stamping
        # every sample with zero and making the log useless for replay.
        start = self._start_t if self._start_t is not None else now
        self.logger.sample(
            t=round(now - start, 4),
            gyro_z=imu.gyro_z if imu else None,
            accel_x=imu.accel_x if imu else None,
            mag_heading_rad=imu.mag_heading_rad if imu else None,
            mag_valid=1 if (imu and imu.mag_heading_rad is not None) else 0,
            encoder_mps=enc.linear_mps if enc else None,
            enc_valid=1 if enc else 0,
            state=output.state.value,
            line_found=int(line.found),
            line_error=round(line.error, 4),
            line_source=line.source,
            pid_out=round(self.pid.state.output, 4),
            drive_left=round(output.drive.left, 4),
            drive_right=round(output.drive.right, 4),
            fused_heading_deg=round(self.state_fusion.heading_deg, 3),
            fused_distance_m=round(self.state_fusion.distance_m, 4),
            fused_velocity_mps=round(self.state_fusion.velocity_mps, 4),
            encoder_rejected=int(self.state_fusion.distance.last_rejected),
            obstacle_m=(telemetry.ranges.min_m if telemetry and telemetry.ranges else None),
            targets=len(self.tracker.confirmed_targets()),
            battery_v=telemetry.battery_v if telemetry else None,
        )
