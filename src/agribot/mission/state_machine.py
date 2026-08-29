"""The mission finite state machine (Section 5.4).

The machine begins in ``FOLLOW_LINE`` and continuously evaluates detection.
When a weed is confirmed it enters ``STOP_AND_AIM``, which halts the drive and
commands the pan and tilt head to the solved target angles, then ``SPRAY`` for
a metered burst, then ``LOG_EVENT``, before returning to ``FOLLOW_LINE``.

**A crop detection produces no intervention at all** - the robot simply
continues. That is the behaviour which protects the crop, and it is a property
of the fusion rule upstream rather than a case handled here: crop-vetoed
targets never reach the state machine.

An obstacle or the end of a row diverts the machine into ``PAUSE``,
``RECOVER`` or ``TURN``, and when every row is complete it terminates in
``MISSION_COMPLETE``. Every transition is deterministic and logged.

Safety conditions are evaluated **before** the per-state logic on every tick,
so no state can be written in a way that ignores them.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..control.differential import DifferentialMixer, safety_scale
from ..control.pid import PID
from ..types import DriveCommand, Track
from ..utils.logging_setup import get_logger
from .states import MissionInputs, MissionOutput, MissionState, TERMINAL_STATES

__all__ = ["MissionStateMachine"]

log = get_logger("mission.fsm")


class MissionStateMachine:
    """Deterministic mission sequencer.

    The machine owns the PID and the drive mixer because the drive command is
    a function of the state, not something layered on afterwards - stopping to
    aim, pausing for an obstacle and turning at a row end all produce different
    commands from the same line error.
    """

    def __init__(
        self,
        pid: PID,
        mixer: DifferentialMixer,
        *,
        cruise_speed_mps: float = 0.18,
        turn_speed_mps: float = 0.12,
        turn_angle_deg: float = 180.0,
        rows: int = 2,
        row_end_detect_m: float = 0.35,
        post_spray_advance_m: float = 0.05,
        line_lost_grace_s: float = 0.6,
        line_lost_stop_s: float = 2.0,
        ultrasonic_stop_m: float = 0.25,
        ultrasonic_slow_m: float = 0.45,
        max_mission_time_s: float = 600.0,
        blind_creep_scale: float = 0.6,
        recover_creep_scale: float = 0.4,
        recover_probe_m: Optional[float] = None,
    ):
        self.pid = pid
        self.mixer = mixer
        self.cruise_speed_mps = cruise_speed_mps
        self.turn_speed_mps = turn_speed_mps
        self.turn_angle_deg = turn_angle_deg
        self.rows = int(rows)
        self.row_end_detect_m = row_end_detect_m
        self.post_spray_advance_m = post_spray_advance_m
        self.line_lost_grace_s = line_lost_grace_s
        self.line_lost_stop_s = line_lost_stop_s
        self.ultrasonic_stop_m = ultrasonic_stop_m
        self.ultrasonic_slow_m = ultrasonic_slow_m
        self.max_mission_time_s = max_mission_time_s
        self.blind_creep_scale = blind_creep_scale
        self.recover_creep_scale = recover_creep_scale
        # Bounded forward probe from the point of line loss, so RECOVER always
        # terminates instead of deadlocking against a distance-based test.
        self.recover_probe_m = (recover_probe_m if recover_probe_m is not None
                                else row_end_detect_m * 2.0)

        self.state = MissionState.INIT
        self.rows_done = 0
        self.transitions: List[Tuple[float, str, str, str]] = []

        self._state_entered_t = 0.0
        self._mission_start_t: Optional[float] = None
        self._line_lost_since: Optional[float] = None
        self._last_line_error = 0.0
        self._distance_at_line_loss = 0.0
        self._distance_at_spray = 0.0
        self._heading_at_turn_start = 0.0
        self._turn_accumulated_deg = 0.0
        self._turn_prev_heading_deg: Optional[float] = None
        self._active_target: Optional[Track] = None
        self._paused_from = MissionState.FOLLOW_LINE

    @classmethod
    def from_config(cls, cfg, pid: PID, mixer: DifferentialMixer) -> "MissionStateMachine":
        """Build from the full config."""
        return cls(
            pid,
            mixer,
            cruise_speed_mps=cfg.robot.cruise_speed_mps,
            turn_speed_mps=cfg.mission.get("turn_speed_mps", 0.12),
            turn_angle_deg=cfg.mission.get("turn_angle_deg", 180.0),
            rows=cfg.mission.get("rows", 2),
            row_end_detect_m=cfg.mission.get("row_end_detect_m", 0.35),
            post_spray_advance_m=cfg.mission.get("post_spray_advance_m", 0.05),
            line_lost_grace_s=cfg.navigation.get("line_lost_grace_s", 0.6),
            line_lost_stop_s=cfg.navigation.get("line_lost_stop_s", 2.0),
            ultrasonic_stop_m=cfg.safety.get("ultrasonic_stop_m", 0.25),
            ultrasonic_slow_m=cfg.safety.get("ultrasonic_slow_m", 0.45),
            max_mission_time_s=cfg.mission.get("max_mission_time_s", 600.0),
            blind_creep_scale=cfg.navigation.get("blind_creep_scale", 0.6),
            recover_creep_scale=cfg.navigation.get("recover_creep_scale", 0.4),
            recover_probe_m=cfg.navigation.get("recover_probe_m", None),
        )

    # -- transitions --------------------------------------------------------
    def _transition(self, new_state: MissionState, t: float, reason: str) -> None:
        if new_state is self.state:
            return
        log.info("%s -> %s (%s)", self.state.value, new_state.value, reason)
        self.transitions.append((t, self.state.value, new_state.value, reason))
        self.state = new_state
        self._state_entered_t = t
        # A stale integrator from the previous state would kick the robot the
        # moment driving resumes, so the controller is cleared on every entry
        # into a driving state.
        if new_state in (MissionState.FOLLOW_LINE, MissionState.TURN, MissionState.RECOVER):
            self.pid.reset()

    def _time_in_state(self, t: float) -> float:
        return t - self._state_entered_t

    # -- main tick ----------------------------------------------------------
    def update(self, inputs: MissionInputs) -> MissionOutput:
        """Advance the machine one tick and return the drive decision."""
        before = self.state
        if self._mission_start_t is None:
            self._mission_start_t = inputs.t

        reason = ""

        # ---- global safety, evaluated before any per-state logic ----------
        fault = self._check_faults(inputs)
        if fault is not None:
            self._transition(MissionState.ESTOP, inputs.t, fault)
            return self._emit(MissionState.ESTOP, DriveCommand.stopped(), before,
                              reason=fault)

        if self.state in TERMINAL_STATES:
            return self._emit(self.state, DriveCommand.stopped(), before,
                              reason="terminal")

        if (inputs.t - self._mission_start_t) > self.max_mission_time_s:
            self._transition(MissionState.MISSION_COMPLETE, inputs.t, "time limit")
            return self._emit(MissionState.MISSION_COMPLETE, DriveCommand.stopped(),
                              before, reason="time limit")

        # Obstacle handling pre-empts everything except an active burst: the
        # nozzle must finish and close rather than be abandoned mid-open.
        obstructed = inputs.nearest_obstacle_m <= self.ultrasonic_stop_m
        if obstructed and self.state not in (MissionState.SPRAY, MissionState.STOP_AND_AIM,
                                             MissionState.PAUSE):
            self._paused_from = self.state
            self._transition(MissionState.PAUSE, inputs.t, "obstacle")

        # ---- line-loss bookkeeping ---------------------------------------
        self._track_line_loss(inputs)

        # ---- per-state behaviour ------------------------------------------
        handler = self._handlers()[self.state]
        drive, reason = handler(inputs)

        # ---- speed shaping from the independent safety layer --------------
        scale = safety_scale(
            inputs.nearest_obstacle_m, self.ultrasonic_stop_m, self.ultrasonic_slow_m
        )
        if scale < 1.0 and drive.linear > 0:
            drive = DriveCommand(drive.left * scale, drive.right * scale)

        engage = self._active_target if self.state is MissionState.STOP_AND_AIM else None
        return self._emit(self.state, drive, before, reason=reason, engage=engage)

    # -- helpers ------------------------------------------------------------
    def _handlers(self) -> Dict[MissionState, Callable[[MissionInputs], Tuple[DriveCommand, str]]]:
        return {
            MissionState.INIT: self._on_init,
            MissionState.FOLLOW_LINE: self._on_follow_line,
            MissionState.STOP_AND_AIM: self._on_stop_and_aim,
            MissionState.SPRAY: self._on_spray,
            MissionState.LOG_EVENT: self._on_log_event,
            MissionState.PAUSE: self._on_pause,
            MissionState.RECOVER: self._on_recover,
            MissionState.TURN: self._on_turn,
        }

    def _check_faults(self, inputs: MissionInputs) -> Optional[str]:
        if not inputs.mcu_ok:
            return "MCU link lost"
        if not inputs.tilt_ok:
            return "excessive tilt"
        return None

    def _track_line_loss(self, inputs: MissionInputs) -> None:
        if inputs.line.found:
            self._line_lost_since = None
            self._last_line_error = inputs.line.error
        elif self._line_lost_since is None:
            self._line_lost_since = inputs.t
            self._distance_at_line_loss = inputs.distance_m

    def _line_lost_for(self, t: float) -> float:
        if self._line_lost_since is None:
            return 0.0
        return t - self._line_lost_since

    def _reset_line_loss(self, inputs: MissionInputs) -> None:
        """Re-baseline blind-travel tracking at the start of a new row.

        Row-end detection measures distance travelled since the line was lost.
        Carrying the previous row's baseline across a turn means the *next*
        row is declared over the instant the turn finishes - the robot never
        gets the chance to travel looking for it, and a two-row mission ends
        after one. Every entry into a fresh row re-bases both the distance and
        the timer.
        """
        self._distance_at_line_loss = inputs.distance_m
        self._line_lost_since = None if inputs.line.found else inputs.t

    def _emit(
        self,
        state: MissionState,
        drive: DriveCommand,
        before: MissionState,
        reason: str = "",
        engage: Optional[Track] = None,
    ) -> MissionOutput:
        return MissionOutput(
            state=state,
            drive=drive,
            engage_target=engage,
            reason=reason,
            transitioned=(state is not before),
            rows_done=self.rows_done,
        )

    # -- state handlers -----------------------------------------------------
    def _on_init(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Hold still until the guidance line is actually visible."""
        if inputs.line.found:
            self._transition(MissionState.FOLLOW_LINE, inputs.t, "line acquired")
            return self._follow_drive(inputs), "line acquired"
        return DriveCommand.stopped(), "waiting for line"

    def _on_follow_line(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Vision-guided travel, interrupted by a confirmed weed or a row end."""
        # A confirmed, unsprayed target takes priority over continuing.
        if inputs.has_target and inputs.spray_ready and not inputs.spray_busy:
            self._active_target = inputs.targets[0]
            self._transition(MissionState.STOP_AND_AIM, inputs.t,
                             f"weed confirmed (track {self._active_target.track_id})")
            return DriveCommand.stopped(), "target confirmed"

        if not inputs.line.found:
            travelled_blind = inputs.distance_m - self._distance_at_line_loss
            lost_for = self._line_lost_for(inputs.t)

            # Row end is decided on DISTANCE, not on time: the robot has crept
            # the full row-end distance without re-acquiring, so the row really
            # has ended rather than the line having flickered.
            if travelled_blind >= self.row_end_detect_m:
                return self._begin_turn(inputs)

            # The time limit is a *stall* watchdog, not a row-end test. If this
            # much time has passed and the robot has not covered the row-end
            # distance, it is not making progress - blocked wheels, a stopped
            # drive - and creeping further is not the answer.
            if lost_for >= self.line_lost_stop_s:
                self._transition(MissionState.RECOVER, inputs.t,
                                 "blind and not progressing")
                return DriveCommand.stopped(), "line lost"

            if lost_for >= self.line_lost_grace_s:
                # Past the grace period: creep straight rather than steering on
                # a stale error, and let the distance test above resolve it.
                return (self.mixer.straight(self.cruise_speed_mps * self.blind_creep_scale),
                        "coasting")
            # Within the grace period, hold the last correction.
            return self.mixer.mix(self.cruise_speed_mps, self.pid.state.output), "grace"

        return self._follow_drive(inputs), "following"

    def _follow_drive(self, inputs: MissionInputs) -> DriveCommand:
        dt = inputs.dt if inputs.dt > 0 else 1.0 / 30.0
        correction = self.pid.update(inputs.line.error, dt)
        return self.mixer.mix(self.cruise_speed_mps, correction)

    def _on_stop_and_aim(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Halted while the pan/tilt head drives to the solved angles."""
        if inputs.spray_busy:
            self._transition(MissionState.SPRAY, inputs.t, "burst started")
            return DriveCommand.stopped(), "burst started"
        # The runtime refused to start (interval, empty reservoir, disabled).
        if self._time_in_state(inputs.t) > 2.0:
            self._transition(MissionState.FOLLOW_LINE, inputs.t,
                             "aim timed out, resuming")
            return DriveCommand.stopped(), "aim timeout"
        return DriveCommand.stopped(), "aiming"

    def _on_spray(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Stationary for the duration of the burst - never spray while moving."""
        if not inputs.spray_busy:
            self._distance_at_spray = inputs.distance_m
            self._transition(MissionState.LOG_EVENT, inputs.t, "burst complete")
            return DriveCommand.stopped(), "burst complete"
        return DriveCommand.stopped(), "spraying"

    def _on_log_event(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Nudge forward so the treated marker leaves the working envelope.

        Without this the robot re-acquires the same marker as a fresh target
        the moment it resumes. The tracker's ``sprayed`` flag is the primary
        guard; the nudge is the physical one.
        """
        advanced = inputs.distance_m - self._distance_at_spray
        if advanced >= self.post_spray_advance_m:
            self._transition(MissionState.FOLLOW_LINE, inputs.t, "clear of target")
            return self.mixer.straight(self.cruise_speed_mps), "clear of target"
        return self.mixer.straight(self.cruise_speed_mps * 0.6), "advancing clear"

    def _on_pause(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Hold while an obstacle is inside the stop band."""
        if inputs.nearest_obstacle_m > self.ultrasonic_stop_m:
            self._transition(self._paused_from, inputs.t, "obstacle cleared")
            return DriveCommand.stopped(), "obstacle cleared"
        return DriveCommand.stopped(), "obstacle"

    def _on_recover(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Blind and not progressing: probe forward a bounded distance.

        The robot does not wander looking for the line - on a real field that
        turns a recoverable stop into a lost robot. It creeps straight ahead,
        never turning, for at most twice the row-end distance measured from
        where the line was lost.

        A purely stationary recovery deadlocks: row-end detection is
        distance-based, and a robot that has stopped can never travel the
        distance that would resolve it. Bounding the probe keeps the "do not
        wander" property while guaranteeing the state is left.
        """
        if inputs.line.found:
            self._transition(MissionState.FOLLOW_LINE, inputs.t, "line re-acquired")
            return DriveCommand.stopped(), "line re-acquired"

        travelled_blind = inputs.distance_m - self._distance_at_line_loss
        if travelled_blind >= self.recover_probe_m:
            # Probe exhausted with no line: the row has ended (or the arena has).
            return self._begin_turn(inputs)
        return (self.mixer.straight(self.cruise_speed_mps * self.recover_creep_scale),
                "probing for line")

    def _begin_turn(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Enter the row-end manoeuvre, or finish the mission."""
        self.rows_done += 1
        if self.rows_done >= self.rows:
            self._transition(MissionState.MISSION_COMPLETE, inputs.t,
                             f"all {self.rows} rows complete")
            return DriveCommand.stopped(), "mission complete"
        self._heading_at_turn_start = inputs.heading_deg
        self._turn_accumulated_deg = 0.0
        self._turn_prev_heading_deg = inputs.heading_deg
        self._transition(MissionState.TURN, inputs.t, f"row {self.rows_done} end")
        return DriveCommand.stopped(), "row end"

    def _on_turn(self, inputs: MissionInputs) -> Tuple[DriveCommand, str]:
        """Turn in place until the fused heading has swept the row-end angle.

        Closing this on the *fused* heading rather than on a timer is the
        practical payoff of the sensor-fusion work: a timed turn is at the
        mercy of battery voltage and surface friction, both of which change
        during a run.
        """
        swept = abs(self._angle_swept(inputs.heading_deg))
        if swept >= abs(self.turn_angle_deg):
            self._reset_line_loss(inputs)
            self._transition(MissionState.FOLLOW_LINE, inputs.t, "turn complete")
            return DriveCommand.stopped(), "turn complete"
        # If the line reappears mid-turn we are already lined up on the next row.
        if inputs.line.found and swept > abs(self.turn_angle_deg) * 0.6:
            self._reset_line_loss(inputs)
            self._transition(MissionState.FOLLOW_LINE, inputs.t,
                             "line re-acquired mid-turn")
            return DriveCommand.stopped(), "line re-acquired"
        direction = 1.0 if self.turn_angle_deg >= 0 else -1.0
        return self.mixer.turn(direction, self.turn_speed_mps), "turning"

    def _angle_swept(self, heading_deg: float) -> float:
        """Total angle swept since the turn began, in degrees, unbounded.

        This **accumulates** per-tick increments rather than differencing
        against the start heading. Differencing cannot express a 180 degree
        turn: the wrapped difference climbs to 180 and then falls again on the
        far side of the seam, so ``>= 180`` is only true in a window the robot
        can step straight over - and a row-end turn that misses it spins for
        ever. Accumulating small wrapped increments passes cleanly through 180
        and keeps counting.

        Call once per tick while turning; it advances internal state.
        """
        previous = self._turn_prev_heading_deg
        if previous is None:
            self._turn_prev_heading_deg = heading_deg
            return self._turn_accumulated_deg

        delta = heading_deg - previous
        while delta > 180.0:
            delta -= 360.0
        while delta <= -180.0:
            delta += 360.0

        self._turn_accumulated_deg += delta
        self._turn_prev_heading_deg = heading_deg
        return self._turn_accumulated_deg

    # -- external events ----------------------------------------------------
    def notify_spray_started(self, t: float) -> None:
        """Called by the runtime when a burst actually begins."""
        if self.state is MissionState.STOP_AND_AIM:
            self._transition(MissionState.SPRAY, t, "burst started")

    def abort(self, t: float, reason: str = "operator abort") -> None:
        """Force the machine into ESTOP."""
        self._transition(MissionState.ESTOP, t, reason)

    def reset(self, t: float = 0.0) -> None:
        """Return to INIT for a fresh run."""
        self.state = MissionState.INIT
        self.rows_done = 0
        self.transitions.clear()
        self._state_entered_t = t
        self._mission_start_t = None
        self._line_lost_since = None
        self._turn_accumulated_deg = 0.0
        self._turn_prev_heading_deg = None
        self._active_target = None
        self.pid.reset()

    def summary(self) -> Dict[str, object]:
        return {
            "state": self.state.value,
            "rows_done": self.rows_done,
            "transitions": len(self.transitions),
            "history": [
                {"t": round(t, 3), "from": a, "to": b, "reason": r}
                for t, a, b, r in self.transitions
            ],
        }
