"""Mission state machine: every transition in Section 5.4, and the safety pre-empts."""

from __future__ import annotations

import math

import pytest

from agribot.control.differential import DifferentialMixer
from agribot.control.pid import PID
from agribot.mission.state_machine import MissionStateMachine
from agribot.mission.states import MissionInputs, MissionState
from agribot.types import BBox, DetectionSource, LineObservation, TargetClass, Track


def line(found=True, error=0.0):
    return LineObservation(found=found, error=error, centroid_px=(320, 400),
                           mask_area_px=8000, confidence=0.2)


def weed_track(track_id=1, cx=320, cy=380):
    return Track(track_id=track_id, cls=TargetClass.WEED,
                 bbox=BBox(cx - 40, cy - 40, cx + 40, cy + 40),
                 confidence=0.95, source=DetectionSource.COLOR, hits=5)


@pytest.fixture
def fsm(cfg):
    return MissionStateMachine.from_config(
        cfg, PID.from_config(cfg.navigation.pid),
        DifferentialMixer.from_config(cfg.robot))


def step(fsm, t, **kwargs):
    kwargs.setdefault("line", line())
    kwargs.setdefault("dt", 1 / 30)
    return fsm.update(MissionInputs(t=t, **kwargs))


class TestStartup:
    def test_starts_in_init_and_holds_still(self, fsm):
        assert fsm.state is MissionState.INIT
        out = step(fsm, 0.0, line=line(found=False))
        assert out.state is MissionState.INIT
        assert out.drive.left == 0.0 and out.drive.right == 0.0

    def test_acquiring_the_line_starts_the_run(self, fsm):
        out = step(fsm, 0.0)
        assert out.state is MissionState.FOLLOW_LINE
        assert out.transitioned is True


class TestFollowLine:
    def test_drives_forward_when_centred(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1)
        assert out.state is MissionState.FOLLOW_LINE
        assert out.drive.linear > 0

    def test_steers_towards_a_line_seen_to_the_right(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1, line=line(error=0.4))
        # Line right of centre means the robot is left of the row: turn right,
        # i.e. speed the left wheel.
        assert out.drive.left > out.drive.right

    def test_steers_towards_a_line_seen_to_the_left(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1, line=line(error=-0.4))
        assert out.drive.right > out.drive.left


class TestWeedEngagement:
    def test_confirmed_weed_halts_the_drive(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1, targets=[weed_track()])
        assert out.state is MissionState.STOP_AND_AIM
        assert out.drive.left == 0.0 and out.drive.right == 0.0
        assert out.engage_target.track_id == 1

    def test_stays_stationary_for_the_whole_burst(self, fsm):
        """Spraying while moving smears the dose off the target."""
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        out = step(fsm, 0.2, targets=[weed_track()], spray_busy=True)
        assert out.state is MissionState.SPRAY
        for i in range(10):
            out = step(fsm, 0.3 + i * 0.1, targets=[weed_track()], spray_busy=True)
            assert out.drive.left == 0.0 and out.drive.right == 0.0

    def test_burst_completion_advances_to_log_event(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        step(fsm, 0.2, spray_busy=True)
        out = step(fsm, 0.5, spray_busy=False)
        assert out.state is MissionState.LOG_EVENT

    def test_nudges_clear_then_resumes(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        step(fsm, 0.2, spray_busy=True)
        step(fsm, 0.5, spray_busy=False, distance_m=1.0)
        out = step(fsm, 0.6, distance_m=1.0)
        assert out.state is MissionState.LOG_EVENT
        assert out.drive.linear > 0                    # advancing clear
        out = step(fsm, 0.7, distance_m=1.0 + cfg.mission.post_spray_advance_m)
        assert out.state is MissionState.FOLLOW_LINE

    def test_aim_times_out_if_the_burst_never_starts(self, fsm):
        """A refused burst (interval, empty tank) must not strand the mission."""
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        out = step(fsm, 3.0, targets=[weed_track()], spray_busy=False)
        assert out.state is MissionState.FOLLOW_LINE

    def test_no_target_means_no_interruption(self, fsm):
        """A crop produces no intervention: crop-vetoed targets never arrive."""
        step(fsm, 0.0)
        for i in range(20):
            out = step(fsm, 0.1 + i * 0.05, targets=[])
        assert out.state is MissionState.FOLLOW_LINE


class TestObstacle:
    def test_obstacle_inside_the_stop_band_pauses(self, fsm, cfg):
        step(fsm, 0.0)
        out = step(fsm, 0.1,
                   nearest_obstacle_m=cfg.safety.ultrasonic_stop_m - 0.05)
        assert out.state is MissionState.PAUSE
        assert out.drive.left == 0.0 and out.drive.right == 0.0

    def test_clearing_the_obstacle_resumes(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, nearest_obstacle_m=0.15)
        out = step(fsm, 0.2, nearest_obstacle_m=1.0)
        assert out.state is MissionState.FOLLOW_LINE

    def test_speed_ramps_down_in_the_slow_band(self, fsm, cfg):
        step(fsm, 0.0)
        fast = step(fsm, 0.1, nearest_obstacle_m=2.0).drive.linear
        slow = step(fsm, 0.2,
                    nearest_obstacle_m=(cfg.safety.ultrasonic_stop_m
                                        + cfg.safety.ultrasonic_slow_m) / 2).drive.linear
        assert 0 < slow < fast

    def test_an_active_burst_is_not_abandoned_mid_open(self, fsm):
        """The nozzle must finish and shut rather than be left open."""
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        step(fsm, 0.2, spray_busy=True)
        out = step(fsm, 0.3, spray_busy=True, nearest_obstacle_m=0.05)
        assert out.state is MissionState.SPRAY


class TestLineLossAndRowEnd:
    def test_grace_period_holds_the_last_correction(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(error=0.3))
        out = step(fsm, 0.2, line=line(found=False))
        assert out.state is MissionState.FOLLOW_LINE
        assert out.reason == "grace"

    def test_creeps_straight_past_the_grace_period(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False))            # loss starts here
        out = step(fsm, 0.1 + cfg.navigation.line_lost_grace_s + 0.2,
                   line=line(found=False))
        assert out.reason == "coasting"
        assert out.drive.left == pytest.approx(out.drive.right)

    def test_row_end_is_decided_on_distance(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        out = step(fsm, 1.0, line=line(found=False),
                   distance_m=cfg.mission.row_end_detect_m + 0.01)
        assert fsm.rows_done == 1
        assert out.state in (MissionState.TURN, MissionState.MISSION_COMPLETE)

    def test_regaining_the_line_cancels_the_row_end(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.2, line=line(found=False), distance_m=0.1)
        out = step(fsm, 0.4, line=line(), distance_m=0.2)
        assert out.state is MissionState.FOLLOW_LINE
        assert fsm.rows_done == 0

    def test_stall_watchdog_enters_recover_when_not_progressing(self, fsm, cfg):
        """Blind and stationary is a stalled robot, not a row end."""
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        out = step(fsm, cfg.navigation.line_lost_stop_s + 0.5,
                   line=line(found=False), distance_m=0.01)
        assert out.state is MissionState.RECOVER

    def test_recover_probes_forward_rather_than_deadlocking(self, fsm, cfg):
        """Row-end detection is distance-based; a stationary recovery could
        never travel the distance that resolves it."""
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        out = step(fsm, cfg.navigation.line_lost_stop_s + 0.5,
                   line=line(found=False), distance_m=0.01)
        assert out.state is MissionState.RECOVER
        out = step(fsm, cfg.navigation.line_lost_stop_s + 0.6,
                   line=line(found=False), distance_m=0.02)
        assert out.drive.linear > 0, "RECOVER must probe forward, not freeze"

    def test_recover_terminates_when_the_probe_is_exhausted(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        step(fsm, cfg.navigation.line_lost_stop_s + 0.5,
             line=line(found=False), distance_m=0.01)
        out = step(fsm, 20.0, line=line(found=False),
                   distance_m=cfg.navigation.recover_probe_m + 0.01)
        assert out.state in (MissionState.TURN, MissionState.MISSION_COMPLETE)

    def test_recover_exits_on_re_acquiring_the_line(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        step(fsm, cfg.navigation.line_lost_stop_s + 0.5,
             line=line(found=False), distance_m=0.01)
        out = step(fsm, 8.0, line=line(), distance_m=0.05)
        assert out.state is MissionState.FOLLOW_LINE


class TestTurnAndCompletion:
    @pytest.fixture
    def one_row_fsm(self, cfg):
        merged = cfg.merged({"mission": {"rows": 1}})
        return MissionStateMachine.from_config(
            merged, PID.from_config(merged.navigation.pid),
            DifferentialMixer.from_config(merged.robot))

    @pytest.fixture
    def two_row_fsm(self, cfg):
        merged = cfg.merged({"mission": {"rows": 2}})
        return MissionStateMachine.from_config(
            merged, PID.from_config(merged.navigation.pid),
            DifferentialMixer.from_config(merged.robot))

    def _reach_row_end(self, fsm, cfg):
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        return step(fsm, 1.0, line=line(found=False),
                    distance_m=cfg.mission.row_end_detect_m + 0.01)

    def test_first_row_end_starts_a_turn(self, two_row_fsm, cfg):
        out = self._reach_row_end(two_row_fsm, cfg)
        assert out.state is MissionState.TURN

    def test_turn_is_closed_on_fused_heading_not_a_timer(self, two_row_fsm, cfg):
        """A timed turn is at the mercy of battery voltage and friction."""
        self._reach_row_end(two_row_fsm, cfg)
        out = step(two_row_fsm, 2.0, line=line(found=False), heading_deg=30.0,
                   distance_m=0.4)
        assert out.state is MissionState.TURN
        assert out.drive.differential != 0

        # Sweep through the +/-180 seam in realistic increments; a turn that
        # steps over the boundary must still complete.
        for heading in (60.0, 90.0, 120.0, 150.0, 175.0, -175.0, -160.0):
            out = step(two_row_fsm, 6.0, line=line(found=False),
                       heading_deg=heading, distance_m=0.4)
            if out.state is MissionState.FOLLOW_LINE:
                break
        assert out.state is MissionState.FOLLOW_LINE

    def test_turn_ends_early_if_the_next_row_is_already_in_view(self, two_row_fsm, cfg):
        self._reach_row_end(two_row_fsm, cfg)
        out = step(two_row_fsm, 5.0, line=line(), heading_deg=140.0, distance_m=0.4)
        assert out.state is MissionState.FOLLOW_LINE

    def test_last_row_completes_the_mission(self, one_row_fsm, cfg):
        out = self._reach_row_end(one_row_fsm, cfg)
        assert out.state is MissionState.MISSION_COMPLETE
        assert one_row_fsm.rows_done == 1

    def test_terminal_state_is_sticky_and_stopped(self, one_row_fsm, cfg):
        self._reach_row_end(one_row_fsm, cfg)
        out = step(one_row_fsm, 30.0, line=line())
        assert out.state is MissionState.MISSION_COMPLETE
        assert out.drive.left == 0.0 and out.drive.right == 0.0


class TestFaultsAndSafety:
    def test_mcu_loss_forces_estop(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1, mcu_ok=False)
        assert out.state is MissionState.ESTOP
        assert out.drive.left == 0.0 and out.drive.right == 0.0

    def test_excessive_tilt_forces_estop(self, fsm):
        step(fsm, 0.0)
        out = step(fsm, 0.1, tilt_ok=False)
        assert out.state is MissionState.ESTOP

    def test_estop_is_not_recoverable_by_itself(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.1, mcu_ok=False)
        out = step(fsm, 0.2, mcu_ok=True)
        assert out.state is MissionState.ESTOP

    def test_safety_is_evaluated_before_state_logic(self, fsm):
        """No state can be written in a way that ignores a fault."""
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])       # STOP_AND_AIM
        out = step(fsm, 0.2, targets=[weed_track()], mcu_ok=False)
        assert out.state is MissionState.ESTOP

    def test_mission_time_limit_ends_the_run(self, fsm, cfg):
        step(fsm, 0.0)
        out = step(fsm, cfg.mission.max_mission_time_s + 1.0)
        assert out.state is MissionState.MISSION_COMPLETE

    def test_manual_abort(self, fsm):
        step(fsm, 0.0)
        fsm.abort(1.0, "operator")
        assert fsm.state is MissionState.ESTOP


class TestBookkeeping:
    def test_transitions_are_recorded_with_reasons(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        assert len(fsm.transitions) >= 2
        for _t, _from, _to, reason in fsm.transitions:
            assert reason

    def test_pid_is_reset_entering_a_driving_state(self, fsm):
        """A stale integrator would kick the robot when driving resumes."""
        step(fsm, 0.0)
        for i in range(20):
            step(fsm, 0.1 + i * 0.05, line=line(error=0.6))
        assert abs(fsm.pid.integral) > 0
        step(fsm, 2.0, targets=[weed_track()])       # -> STOP_AND_AIM
        step(fsm, 2.1, spray_busy=True)              # -> SPRAY
        step(fsm, 2.5, spray_busy=False)             # -> LOG_EVENT
        step(fsm, 2.6, distance_m=99.0)              # -> FOLLOW_LINE
        assert fsm.pid.integral == 0.0

    def test_reset_returns_to_init(self, fsm):
        step(fsm, 0.0)
        step(fsm, 0.1, targets=[weed_track()])
        fsm.reset(0.0)
        assert fsm.state is MissionState.INIT
        assert fsm.rows_done == 0 and fsm.transitions == []

    def test_summary_is_serialisable(self, fsm):
        step(fsm, 0.0)
        summary = fsm.summary()
        assert summary["state"] == "FOLLOW_LINE"
        assert isinstance(summary["history"], list)

    def test_angle_swept_accumulates_through_the_180_seam(self, fsm):
        """Differencing against the start heading tops out at 180 and then
        falls again; accumulating increments keeps counting past it."""
        fsm._turn_accumulated_deg = 0.0
        fsm._turn_prev_heading_deg = 0.0
        swept = 0.0
        for heading in (60.0, 120.0, 175.0, -175.0, -120.0, -60.0, 0.0):
            swept = fsm._angle_swept(heading)
        assert swept == pytest.approx(360.0, abs=1e-6)

    def test_angle_swept_reaches_180_without_overshoot_ambiguity(self, fsm):
        fsm._turn_accumulated_deg = 0.0
        fsm._turn_prev_heading_deg = 0.0
        swept = 0.0
        for heading in (60.0, 120.0, 175.0, -175.0):
            swept = fsm._angle_swept(heading)
        assert swept == pytest.approx(185.0, abs=1e-6)
        assert abs(swept) >= 180.0, "a 180 deg turn must register as complete"

    def test_angle_swept_counts_negative_for_a_clockwise_turn(self, fsm):
        fsm._turn_accumulated_deg = 0.0
        fsm._turn_prev_heading_deg = 0.0
        swept = 0.0
        for heading in (-60.0, -120.0, -175.0, 175.0):
            swept = fsm._angle_swept(heading)
        assert swept == pytest.approx(-185.0, abs=1e-6)


class TestMultiRowContinuity:
    """A turn must hand the next row a clean slate."""

    @pytest.fixture
    def three_row_fsm(self, cfg):
        merged = cfg.merged({"mission": {"rows": 3}})
        return MissionStateMachine.from_config(
            merged, PID.from_config(merged.navigation.pid),
            DifferentialMixer.from_config(merged.robot))

    def test_next_row_is_not_declared_over_the_instant_the_turn_ends(
            self, three_row_fsm, cfg):
        fsm = three_row_fsm
        step(fsm, 0.0)
        # Row 1 ends after the blind-travel distance.
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        out = step(fsm, 2.0, line=line(found=False),
                   distance_m=cfg.mission.row_end_detect_m + 0.02)
        assert out.state is MissionState.TURN

        # Sweep the turn to completion, still blind and still stationary.
        for heading in (60.0, 120.0, 175.0, -175.0):
            out = step(fsm, 3.0, line=line(found=False), heading_deg=heading,
                       distance_m=cfg.mission.row_end_detect_m + 0.02)
        assert out.state is MissionState.FOLLOW_LINE
        assert fsm.rows_done == 1

        # The very next tick must NOT end row 2: the blind counter was rebased.
        out = step(fsm, 3.1, line=line(found=False),
                   distance_m=cfg.mission.row_end_detect_m + 0.03)
        assert out.state is MissionState.FOLLOW_LINE
        assert fsm.rows_done == 1

    def test_row_two_ends_on_its_own_blind_distance(self, three_row_fsm, cfg):
        fsm = three_row_fsm
        base = cfg.mission.row_end_detect_m
        step(fsm, 0.0)
        step(fsm, 0.1, line=line(found=False), distance_m=0.0)
        step(fsm, 2.0, line=line(found=False), distance_m=base + 0.02)
        for heading in (60.0, 120.0, 175.0, -175.0):
            step(fsm, 3.0, line=line(found=False), heading_deg=heading,
                 distance_m=base + 0.02)
        assert fsm.rows_done == 1

        # Now travel a fresh row-end distance from the new baseline.
        out = step(fsm, 6.0, line=line(found=False), distance_m=base * 2 + 0.05)
        assert fsm.rows_done == 2
