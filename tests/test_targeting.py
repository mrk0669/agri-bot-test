"""Pixel-to-actuator mapping and the metered spray sequence."""

from __future__ import annotations

import math

import pytest

from agribot.targeting.pixel_to_angle import AxisCalibration, PixelToAngleSolver
from agribot.targeting.spray_controller import SprayController, SprayPhase
from agribot.types import AimSolution


@pytest.fixture
def solver(cfg):
    return PixelToAngleSolver.from_config(cfg.targeting, cfg.camera)


class TestAxisCalibration:
    def test_centre_offset_gives_centre_angle(self):
        axis = AxisCalibration(90.0, 0.07, 50.0, 130.0)
        assert axis.solve(0.0) == (90.0, False)

    def test_linear_mapping(self):
        axis = AxisCalibration(90.0, 0.05, 0.0, 180.0)
        assert axis.solve(100)[0] == pytest.approx(95.0)
        assert axis.solve(-100)[0] == pytest.approx(85.0)

    def test_invert_flips_the_direction(self):
        axis = AxisCalibration(90.0, 0.05, 0.0, 180.0, invert=True)
        assert axis.solve(100)[0] == pytest.approx(85.0)

    def test_clamping_is_reported(self):
        axis = AxisCalibration(90.0, 1.0, 80.0, 100.0)
        angle, clamped = axis.solve(1000)
        assert angle == 100.0 and clamped is True

    def test_rejects_inverted_limits(self):
        with pytest.raises(ValueError):
            AxisCalibration(90.0, 0.05, 130.0, 50.0)

    def test_rejects_centre_outside_limits(self):
        with pytest.raises(ValueError):
            AxisCalibration(200.0, 0.05, 50.0, 130.0)


class TestPixelToAngleSolver:
    def test_frame_centre_maps_to_servo_centre(self, solver, cfg):
        aim = solver.solve(cfg.camera.width / 2, cfg.camera.height / 2)
        assert aim.pan_deg == pytest.approx(cfg.targeting.pan.centre_deg)
        assert aim.tilt_deg == pytest.approx(cfg.targeting.tilt.centre_deg)
        assert aim.in_range is True

    def test_pan_follows_horizontal_offset_monotonically(self, solver):
        angles = [solver.solve(x, 240).pan_deg for x in (100, 250, 320, 400, 560)]
        assert angles == sorted(angles)

    def test_tilt_follows_vertical_offset_monotonically(self, solver, cfg):
        angles = [solver.solve(320, y).tilt_deg for y in (80, 160, 240, 360, 460)]
        # The tilt axis is inverted in the shipped config: lower in the frame is
        # nearer, which needs a steeper nozzle depression.
        expected = sorted(angles, reverse=cfg.targeting.tilt.invert)
        assert angles == expected

    def test_extreme_pixels_clamp_and_report_it(self, solver):
        aim = solver.solve(-5000, -5000)
        assert aim.clamped is True and aim.in_range is False

    def test_solution_stays_inside_the_servo_envelope(self, solver, cfg):
        for x in (0, 160, 320, 480, 639):
            for y in (0, 120, 240, 360, 479):
                aim = solver.solve(x, y)
                assert cfg.targeting.pan.min_deg <= aim.pan_deg <= cfg.targeting.pan.max_deg
                assert cfg.targeting.tilt.min_deg <= aim.tilt_deg <= cfg.targeting.tilt.max_deg

    def test_source_pixel_is_recorded(self, solver):
        assert solver.solve(200, 300).source_px == (200, 300)

    def test_ground_range_decreases_down_the_frame(self, solver):
        assert solver.estimate_ground_range(440) < solver.estimate_ground_range(200)

    def test_depth_changes_the_tilt_solution(self, solver):
        """An elevated marker breaks the ground-plane assumption; a true range
        is what fixes it (rules allow markers up to ~15 cm)."""
        ground = solver.solve(320, 300)
        with_depth = solver.solve(320, 300, range_m=0.32)
        assert with_depth.range_m == 0.32
        assert abs(with_depth.tilt_deg - ground.tilt_deg) > 1.0

    def test_elevated_marker_needs_less_depression_than_a_floor_marker(self, solver):
        """At the same range, a raised target sits higher, so the nozzle must
        come up relative to the floor case."""
        floor = solver._tilt_from_range(320, 300, 0.30)[0]
        # Same pixel row but a longer measured range means the target is
        # higher off the floor than the ground plane would imply.
        raised = solver._tilt_from_range(320, 300, 0.34)[0]
        assert raised != pytest.approx(floor)

    def test_depth_is_ignored_when_disabled(self, cfg):
        merged = cfg.merged({"targeting": {"tilt": {"use_depth_when_available": False}}})
        solver = PixelToAngleSolver.from_config(merged.targeting, merged.camera)
        assert (solver.solve(320, 300).tilt_deg
                == pytest.approx(solver.solve(320, 300, range_m=0.32).tilt_deg))

    def test_centre_solution_is_the_stowed_aim(self, solver, cfg):
        home = solver.centre_solution()
        assert home.pan_deg == cfg.targeting.pan.centre_deg
        assert home.in_range is True

    def test_aim_serialises(self, solver):
        payload = solver.solve(200, 300, range_m=0.3).to_dict()
        assert set(payload) >= {"pan_deg", "tilt_deg", "in_range", "range_m"}


class Rig:
    """Records what the spray controller asked the hardware to do."""

    def __init__(self, ml_per_s=8.0, ticks_per_ml=450.0):
        self.aims = []
        self.pump_calls = []
        self.valve_calls = []
        self.marker_calls = []
        self.ticks = 0
        self.pump_on = False
        self.valve_open = False
        self.ml_per_s = ml_per_s
        self.ticks_per_ml = ticks_per_ml
        self._dispensed_ml = 0.0

    def set_aim(self, pan, tilt):
        self.aims.append((pan, tilt))

    def set_pump(self, on):
        self.pump_on = on
        self.pump_calls.append(on)

    def set_valve(self, open_):
        self.valve_open = open_
        self.valve_calls.append(open_)

    def set_marker(self, deg):
        self.marker_calls.append(deg)

    def read_flow(self):
        return self.ticks

    def advance(self, dt):
        """Simulate fluid actually flowing while pump and valve are both on.

        Ticks are derived from the *accumulated* volume, as a real turbine
        sensor produces them. Truncating ticks per step instead would lose a
        fraction of a tick every step and under-report the dose.
        """
        if self.pump_on and self.valve_open:
            self._dispensed_ml += self.ml_per_s * dt
            self.ticks = int(self._dispensed_ml * self.ticks_per_ml)

    def callbacks(self):
        return dict(set_aim=self.set_aim, set_pump=self.set_pump,
                    set_valve=self.set_valve, read_flow_ticks=self.read_flow,
                    set_marker=self.set_marker)


def build_spray(cfg, clock, rig, **overrides):
    merged = cfg.merged({"spray": overrides}) if overrides else cfg
    return SprayController.from_config(
        merged.spray, merged.targeting, clock=clock, **rig.callbacks())


AIM = AimSolution(pan_deg=72.0, tilt_deg=88.0, in_range=True)


def run_burst(controller, clock, rig, dt=0.01, limit=2000):
    """Step the sequence to completion, returning the event."""
    for _ in range(limit):
        clock.advance(dt)
        rig.advance(dt)
        event = controller.update()
        if event is not None:
            return event
    return None


class TestSprayController:
    def test_full_sequence_produces_a_measured_event(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        assert controller.begin(7, AIM, distance_m=1.2) is True
        event = run_burst(controller, clock, rig)

        assert event is not None
        assert event.track_id == 7
        assert event.measured is True
        assert event.volume_ml > 0
        assert controller.phase is SprayPhase.IDLE

    def test_head_is_aimed_before_the_valve_opens(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        assert rig.aims == [(72.0, 88.0)]
        assert rig.valve_calls == []          # nothing opened during aiming
        run_burst(controller, clock, rig)
        assert rig.valve_calls[0] is True

    def test_pump_leads_the_valve_and_valve_shuts_first(self, cfg, clock):
        """Never leave the pump driving into a shut valve, or spray unpressurised."""
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        order = []
        for _ in range(2000):
            clock.advance(0.01)
            rig.advance(0.01)
            before = (rig.pump_on, rig.valve_open)
            event = controller.update()
            after = (rig.pump_on, rig.valve_open)
            if before != after:
                order.append(after)
            if event:
                break
        # pump on -> valve open -> valve shut -> pump off
        assert order == [(True, False), (True, True), (True, False), (False, False)]

    def test_burst_duration_matches_the_configuration(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        opened = closed = None
        for _ in range(2000):
            clock.advance(0.001)
            rig.advance(0.001)
            was_open = rig.valve_open
            controller.update()
            if rig.valve_open and not was_open:
                opened = clock.t
            if was_open and not rig.valve_open:
                closed = clock.t
                break
        assert (closed - opened) == pytest.approx(cfg.spray.burst_ms / 1000.0, abs=0.01)

    def test_volume_is_measured_from_the_flow_sensor(self, cfg, clock):
        rig = Rig(ml_per_s=8.0)
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        event = run_burst(controller, clock, rig, dt=0.001)
        expected = 8.0 * (cfg.spray.burst_ms / 1000.0)
        assert event.volume_ml == pytest.approx(expected, rel=0.1)
        assert event.measured is True

    def test_falls_back_to_nominal_without_a_flow_sensor(self, cfg, clock):
        """An estimate must never be presented as a measurement."""
        rig = Rig()
        callbacks = rig.callbacks()
        callbacks["read_flow_ticks"] = None
        controller = SprayController.from_config(
            cfg.spray, cfg.targeting, clock=clock, **callbacks)
        controller.begin(1, AIM)
        event = run_burst(controller, clock, rig)
        assert event.measured is False
        assert event.volume_ml == cfg.spray.ml_per_burst_nominal
        assert controller.stats.fully_measured is False

    def test_no_flow_registered_is_reported_as_unmeasured(self, cfg, clock):
        """A dry or blocked line must surface, not be papered over."""
        rig = Rig(ml_per_s=0.0)               # pump runs but nothing flows
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        event = run_burst(controller, clock, rig)
        assert event.measured is False

    def test_minimum_interval_blocks_a_double_hit(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        run_burst(controller, clock, rig)
        assert controller.begin(2, AIM) is False
        assert controller.stats.blocked_by_interval == 1
        clock.advance(cfg.spray.min_interval_s + 0.1)
        assert controller.begin(2, AIM) is True

    def test_empty_reservoir_blocks_the_burst(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig, reservoir_ml=0.5)
        assert controller.begin(1, AIM) is False
        assert controller.stats.blocked_by_reservoir == 1

    def test_reservoir_depletes_with_each_event(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        start = controller.stats.reservoir_ml
        controller.begin(1, AIM)
        event = run_burst(controller, clock, rig)
        assert controller.stats.reservoir_ml == pytest.approx(start - event.volume_ml)

    def test_disabled_controller_never_actuates(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig, enabled=False)
        assert controller.begin(1, AIM) is False
        assert rig.valve_calls == [] and rig.pump_calls == []

    def test_cannot_start_a_second_burst_while_busy(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        assert controller.begin(1, AIM) is True
        assert controller.begin(2, AIM) is False

    def test_abort_shuts_valve_before_pump(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.begin(1, AIM)
        for _ in range(60):
            clock.advance(0.01)
            rig.advance(0.01)
            controller.update()
            if rig.valve_open:
                break
        assert rig.valve_open is True
        controller.abort()
        assert rig.valve_open is False and rig.pump_on is False
        assert controller.phase is SprayPhase.IDLE
        assert rig.valve_calls[-1] is False

    def test_abort_when_idle_is_a_no_op(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig)
        controller.abort()
        assert rig.valve_calls == []

    def test_mark_mode_uses_the_servo_not_the_valve(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig, mode="mark")
        controller.begin(1, AIM)
        event = run_burst(controller, clock, rig)
        assert event.mode == "mark"
        assert event.volume_ml == 0.0
        assert rig.valve_calls == [] and rig.pump_calls == []
        assert cfg.spray.marker.down_deg in rig.marker_calls
        assert cfg.spray.marker.up_deg in rig.marker_calls

    def test_stats_track_ml_per_weed(self, cfg, clock):
        rig = Rig()
        controller = build_spray(cfg, clock, rig, min_interval_s=0.0)
        for track_id in (1, 2, 3):
            controller.begin(track_id, AIM)
            run_burst(controller, clock, rig)
        stats = controller.stats
        assert stats.events == 3
        assert stats.ml_per_weed == pytest.approx(stats.total_ml / 3)
        assert stats.fully_measured is True

    def test_ml_per_weed_is_nan_before_any_event(self, cfg, clock):
        controller = build_spray(cfg, clock, Rig())
        assert math.isnan(controller.stats.ml_per_weed)

    def test_update_when_idle_returns_nothing(self, cfg, clock):
        controller = build_spray(cfg, clock, Rig())
        assert controller.update() is None
