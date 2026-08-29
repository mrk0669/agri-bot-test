"""Metered spray / mark actuation and its telemetry (Section 5.7, Novelty 2/3).

The controller opens the solenoid valve for a calibrated burst while the
diaphragm pump maintains line pressure, then logs the event through the in-line
flow sensor. Metered bursts rather than continuous spraying are what make the
fluid consumed scale with the *number of weeds* rather than with the distance
travelled - the core sustainability argument of the whole track.

Because the flow sensor measures every event, spraying efficiency is reported
as a measured millilitres-per-weed figure rather than asserted (Novelty 3).
Where no flow sensor is fitted the nominal dose is used instead, and the event
is flagged ``measured=False`` so that an estimated number can never be silently
presented as a measurement.

The sequence is implemented as a **non-blocking state machine** driven by
``update()``. A blocking ``sleep(burst_ms)`` would suspend the ultrasonic
safety layer and the MCU heartbeat for the duration of every burst; stepping
the sequence instead keeps those alive while the nozzle is open.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from ..types import AimSolution, SprayEvent
from ..utils.logging_setup import get_logger

__all__ = ["SprayPhase", "SprayController", "SprayStats"]

log = get_logger("targeting.spray")


class SprayPhase(str, Enum):
    """Where the burst sequence has got to."""

    IDLE = "idle"
    AIMING = "aiming"        # servos commanded, waiting for settle
    PRIMING = "priming"      # pump running, building line pressure
    BURST = "burst"          # solenoid open
    RECOVER = "recover"      # solenoid shut, pump spinning down
    DONE = "done"


@dataclass
class SprayStats:
    """Running mission totals - the sustainability evidence."""

    events: int = 0
    total_ml: float = 0.0
    measured_events: int = 0
    reservoir_ml: float = 0.0
    blocked_by_interval: int = 0
    blocked_by_reservoir: int = 0

    @property
    def ml_per_weed(self) -> float:
        """The headline efficiency figure. NaN when nothing has been sprayed."""
        return self.total_ml / self.events if self.events else float("nan")

    @property
    def fully_measured(self) -> bool:
        """True only if every event was measured by the flow sensor."""
        return self.events > 0 and self.measured_events == self.events

    def to_dict(self) -> Dict[str, object]:
        return {
            "events": self.events,
            "total_ml": round(self.total_ml, 3),
            "ml_per_weed": (round(self.ml_per_weed, 3) if self.events else None),
            "measured_events": self.measured_events,
            "fully_measured": self.fully_measured,
            "reservoir_ml": round(self.reservoir_ml, 1),
            "blocked_by_interval": self.blocked_by_interval,
            "blocked_by_reservoir": self.blocked_by_reservoir,
        }


class SprayController:
    """Drives the pan/tilt head, pump, solenoid and marker servo.

    The controller does not talk to hardware directly; it calls the callbacks
    supplied at construction. That keeps it testable against a mock and means
    the same logic drives the real MCU link and the simulator.
    """

    def __init__(
        self,
        *,
        set_aim: Callable[[float, float], None],
        set_pump: Callable[[bool], None],
        set_valve: Callable[[bool], None],
        read_flow_ticks: Optional[Callable[[], int]] = None,
        set_marker: Optional[Callable[[float], None]] = None,
        mode: str = "spray",
        burst_ms: float = 220.0,
        pump_spinup_ms: float = 120.0,
        pump_off_delay_ms: float = 80.0,
        settle_time_s: float = 0.35,
        min_interval_s: float = 1.0,
        ml_per_burst_nominal: float = 1.8,
        flow_ticks_per_ml: float = 450.0,
        reservoir_ml: float = 500.0,
        reservoir_warn_ml: float = 60.0,
        marker_up_deg: float = 30.0,
        marker_down_deg: float = 95.0,
        marker_dwell_ms: float = 400.0,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._set_aim = set_aim
        self._set_pump = set_pump
        self._set_valve = set_valve
        self._read_flow_ticks = read_flow_ticks
        self._set_marker = set_marker

        self.mode = mode
        self.burst_s = burst_ms / 1000.0
        self.pump_spinup_s = pump_spinup_ms / 1000.0
        self.pump_off_delay_s = pump_off_delay_ms / 1000.0
        self.settle_time_s = settle_time_s
        self.min_interval_s = min_interval_s
        self.ml_per_burst_nominal = ml_per_burst_nominal
        self.flow_ticks_per_ml = flow_ticks_per_ml
        self.reservoir_warn_ml = reservoir_warn_ml
        self.marker_up_deg = marker_up_deg
        self.marker_down_deg = marker_down_deg
        self.marker_dwell_s = marker_dwell_ms / 1000.0
        self.enabled = enabled
        self._clock = clock

        self.phase = SprayPhase.IDLE
        self.stats = SprayStats(reservoir_ml=reservoir_ml)
        self.events: List[SprayEvent] = []

        self._phase_started = 0.0
        self._last_event_t = -float("inf")
        self._active_track_id: Optional[int] = None
        self._active_aim: Optional[AimSolution] = None
        self._flow_at_start = 0
        self._event_counter = 0
        self._pending_distance_m = 0.0

    @classmethod
    def from_config(cls, cfg, targeting_cfg, **callbacks) -> "SprayController":
        """Build from the ``spray`` and ``targeting`` config sections."""
        marker = cfg.get("marker", {}) or {}
        return cls(
            mode=cfg.get("mode", "spray"),
            burst_ms=cfg.get("burst_ms", 220.0),
            pump_spinup_ms=cfg.get("pump_spinup_ms", 120.0),
            pump_off_delay_ms=cfg.get("pump_off_delay_ms", 80.0),
            settle_time_s=targeting_cfg.get("settle_time_s", 0.35),
            min_interval_s=cfg.get("min_interval_s", 1.0),
            ml_per_burst_nominal=cfg.get("ml_per_burst_nominal", 1.8),
            flow_ticks_per_ml=cfg.get("flow_ticks_per_ml", 450.0),
            reservoir_ml=cfg.get("reservoir_ml", 500.0),
            reservoir_warn_ml=cfg.get("reservoir_warn_ml", 60.0),
            marker_up_deg=marker.get("up_deg", 30.0),
            marker_down_deg=marker.get("down_deg", 95.0),
            marker_dwell_ms=marker.get("dwell_ms", 400.0),
            enabled=cfg.get("enabled", True),
            **callbacks,
        )

    # -- lifecycle ----------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.phase not in (SprayPhase.IDLE, SprayPhase.DONE)

    def can_spray(self, now: Optional[float] = None) -> bool:
        """Is the controller allowed to start a burst right now?"""
        if not self.enabled or self.busy:
            return False
        now = self._clock() if now is None else now
        if (now - self._last_event_t) < self.min_interval_s:
            return False
        if self.stats.reservoir_ml < self.ml_per_burst_nominal:
            return False
        return True

    def begin(
        self,
        track_id: int,
        aim: AimSolution,
        distance_m: float = 0.0,
        detector_source: str = "fused",
        confidence: float = 0.0,
    ) -> bool:
        """Start a burst on a confirmed target. Returns False if refused."""
        now = self._clock()
        if not self.enabled or self.busy:
            return False
        if (now - self._last_event_t) < self.min_interval_s:
            self.stats.blocked_by_interval += 1
            log.debug("burst refused: min interval not elapsed (track %d)", track_id)
            return False
        if self.stats.reservoir_ml < self.ml_per_burst_nominal:
            self.stats.blocked_by_reservoir += 1
            log.warning("burst refused: reservoir %.1f ml below one dose",
                        self.stats.reservoir_ml)
            return False

        self._active_track_id = track_id
        self._active_aim = aim
        self._pending_distance_m = distance_m
        self._detector_source = detector_source
        self._confidence = confidence
        self._flow_at_start = self._read_flow_ticks() if self._read_flow_ticks else 0

        self._set_aim(aim.pan_deg, aim.tilt_deg)
        self._enter(SprayPhase.AIMING, now)
        log.info("aiming at track %d: pan=%.1f tilt=%.1f", track_id,
                 aim.pan_deg, aim.tilt_deg)
        return True

    def update(self, now: Optional[float] = None) -> Optional[SprayEvent]:
        """Advance the sequence. Returns the SprayEvent on the frame it finishes."""
        if self.phase in (SprayPhase.IDLE, SprayPhase.DONE):
            return None
        now = self._clock() if now is None else now
        elapsed = now - self._phase_started

        if self.phase is SprayPhase.AIMING:
            if elapsed >= self.settle_time_s:
                if self.mode == "mark":
                    self._begin_mark(now)
                else:
                    self._set_pump(True)
                    self._enter(SprayPhase.PRIMING, now)
            return None

        if self.phase is SprayPhase.PRIMING:
            if elapsed >= self.pump_spinup_s:
                self._set_valve(True)
                self._enter(SprayPhase.BURST, now)
            return None

        if self.phase is SprayPhase.BURST:
            if self.mode == "mark":
                if elapsed >= self.marker_dwell_s:
                    if self._set_marker:
                        self._set_marker(self.marker_up_deg)
                    self._enter(SprayPhase.RECOVER, now)
                return None
            if elapsed >= self.burst_s:
                self._set_valve(False)
                self._enter(SprayPhase.RECOVER, now)
            return None

        if self.phase is SprayPhase.RECOVER:
            settle = self.marker_dwell_s * 0.5 if self.mode == "mark" else self.pump_off_delay_s
            if elapsed >= settle:
                if self.mode != "mark":
                    self._set_pump(False)
                return self._finish(now)
            return None

        return None

    def abort(self) -> None:
        """Immediately shut everything and return to idle.

        Called by the safety layer. Ordering matters: the valve closes before
        the pump stops, so the line is never left pressurised behind a shut
        valve with the pump still driving into it.
        """
        if self.phase is SprayPhase.IDLE:
            return
        log.warning("spray aborted in phase %s", self.phase.value)
        try:
            self._set_valve(False)
            self._set_pump(False)
            if self._set_marker:
                self._set_marker(self.marker_up_deg)
        finally:
            self.phase = SprayPhase.IDLE
            self._active_track_id = None
            self._active_aim = None

    # -- internals ----------------------------------------------------------
    def _begin_mark(self, now: float) -> None:
        """Marker mode: drop the felt stamp instead of opening the valve."""
        if self._set_marker:
            self._set_marker(self.marker_down_deg)
        self._enter(SprayPhase.BURST, now)

    def _enter(self, phase: SprayPhase, now: float) -> None:
        self.phase = phase
        self._phase_started = now

    def _measure_volume(self) -> tuple:
        """Return ``(volume_ml, measured)`` for the burst just completed."""
        if self._read_flow_ticks is None or self.flow_ticks_per_ml <= 0:
            return self.ml_per_burst_nominal, False
        ticks = self._read_flow_ticks() - self._flow_at_start
        if ticks <= 0:
            # No flow registered. Either no sensor is fitted or the line is
            # dry/blocked - both are worth surfacing rather than papering over.
            log.warning("no flow ticks recorded for burst; using nominal dose")
            return self.ml_per_burst_nominal, False
        return ticks / self.flow_ticks_per_ml, True

    def _finish(self, now: float) -> SprayEvent:
        volume_ml, measured = (
            (0.0, False) if self.mode == "mark" else self._measure_volume()
        )

        self._event_counter += 1
        event = SprayEvent(
            event_id=self._event_counter,
            track_id=self._active_track_id if self._active_track_id is not None else -1,
            timestamp=now,
            aim=self._active_aim or AimSolution(0.0, 0.0, False),
            burst_ms=self.burst_s * 1000.0,
            volume_ml=volume_ml,
            measured=measured,
            mode=self.mode,
            distance_m=self._pending_distance_m,
            detector_source=getattr(self, "_detector_source", "fused"),
            confidence=getattr(self, "_confidence", 0.0),
        )

        self.events.append(event)
        self.stats.events += 1
        self.stats.total_ml += volume_ml
        if measured:
            self.stats.measured_events += 1
        self.stats.reservoir_ml = max(0.0, self.stats.reservoir_ml - volume_ml)
        self._last_event_t = now

        if 0 < self.stats.reservoir_ml <= self.reservoir_warn_ml:
            log.warning("reservoir low: %.1f ml remaining", self.stats.reservoir_ml)

        log.info(
            "%s event %d on track %d: %.2f ml (%s), total %.1f ml over %d weeds",
            self.mode, event.event_id, event.track_id, volume_ml,
            "measured" if measured else "nominal",
            self.stats.total_ml, self.stats.events,
        )

        self.phase = SprayPhase.IDLE
        self._active_track_id = None
        self._active_aim = None
        return event
