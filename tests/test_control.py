"""PID controller, differential mixing and the shared geometry helpers."""

from __future__ import annotations

import math

import pytest

from agribot.control.differential import (
    DifferentialMixer,
    safety_scale,
    wheel_speeds_to_body,
)
from agribot.control.pid import PID
from agribot.types import DriveCommand
from agribot.utils.geometry import (
    angle_diff,
    clamp,
    deadband,
    low_pass_alpha,
    normalise_error,
    ground_range_from_pixel,
    wrap_pi,
)
from agribot.utils.rates import LoopTimer, RateLimiter


class TestGeometry:
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(11, 0, 10) == 10

    def test_clamp_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            clamp(1, 10, 0)

    def test_deadband_removes_the_step_at_the_edge(self):
        assert deadband(0.05, 0.1) == 0.0
        # Continuity: just outside the band the output is near zero, not 0.15.
        assert deadband(0.15, 0.1) == pytest.approx(0.05)
        assert deadband(-0.15, 0.1) == pytest.approx(-0.05)

    @pytest.mark.parametrize("angle,expected", [
        (0.0, 0.0),
        (math.pi, math.pi),
        (math.pi + 0.1, -math.pi + 0.1),
        (-math.pi - 0.1, math.pi - 0.1),
        (3 * math.pi, math.pi),
    ])
    def test_wrap_pi(self, angle, expected):
        assert wrap_pi(angle) == pytest.approx(expected, abs=1e-9)

    def test_angle_diff_takes_the_short_way_round_the_seam(self):
        """Comparing raw angles across +/-pi is how a robot spins the long way."""
        a, b = math.radians(179), math.radians(-179)
        assert angle_diff(a, b) == pytest.approx(math.radians(-2), abs=1e-9)

    def test_normalise_error_spans_minus_one_to_one(self):
        assert normalise_error(0, 640) == -1.0
        assert normalise_error(320, 640) == 0.0
        assert normalise_error(640, 640) == 1.0

    def test_normalise_error_is_resolution_independent(self):
        """PID gains must not need re-tuning when the capture size changes."""
        assert normalise_error(160, 640) == pytest.approx(normalise_error(320, 1280))

    def test_normalise_error_rejects_bad_width(self):
        with pytest.raises(ValueError):
            normalise_error(10, 0)

    def test_ground_range_grows_towards_the_horizon(self):
        near = ground_range_from_pixel(460, 480, 0.22, 35.0, 615.0, 240.0)
        far = ground_range_from_pixel(200, 480, 0.22, 35.0, 615.0, 240.0)
        assert 0 < near < far

    def test_ground_range_above_horizon_is_infinite(self):
        assert math.isinf(
            ground_range_from_pixel(-5000, 480, 0.22, 35.0, 615.0, 240.0))

    def test_low_pass_alpha_bounds(self):
        assert low_pass_alpha(0, 0.01) == 1.0        # disabled = pass through
        assert 0 < low_pass_alpha(12.0, 1 / 30) < 1


class TestRateLimiter:
    def test_fires_at_the_configured_rate(self, clock):
        limiter = RateLimiter(10.0, clock)
        assert limiter.due() is True          # fires immediately
        assert limiter.due() is False
        clock.advance(0.05)
        assert limiter.due() is False
        clock.advance(0.06)
        assert limiter.due() is True

    def test_a_long_stall_does_not_produce_a_burst(self, clock):
        """Re-basing rather than accumulating avoids catch-up storms."""
        limiter = RateLimiter(10.0, clock)
        limiter.due()
        clock.advance(5.0)
        assert limiter.due() is True
        assert limiter.due() is False         # not nine more queued firings

    def test_rejects_non_positive_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(0)


class TestLoopTimer:
    def test_counts_overruns(self, clock):
        slept = []
        timer = LoopTimer(10.0, clock, lambda s: slept.append(s))
        timer.sleep()
        clock.advance(0.5)                    # a 500 ms iteration at 10 Hz
        timer.sleep()
        assert timer.overruns == 1
        assert timer.stats()["target_hz"] == 10.0


class TestPID:
    def test_zero_error_gives_zero_output(self, cfg):
        pid = PID.from_config(cfg.navigation.pid)
        assert pid.update(0.0, 1 / 30) == pytest.approx(0.0, abs=1e-9)

    def test_sign_convention(self, cfg):
        """Line right of centre (positive error) must give a negative correction,
        which the mixer turns into a right turn."""
        pid = PID.from_config(cfg.navigation.pid)
        assert pid.update(0.3, 1 / 30) < 0
        pid.reset()
        assert pid.update(-0.3, 1 / 30) > 0

    def test_output_is_clamped(self):
        pid = PID(kp=100.0, ki=0.0, kd=0.0, output_limit=0.5)
        assert pid.update(1.0, 0.1) == pytest.approx(-0.5)
        assert pid.state.saturated is True

    def test_integral_is_clamped(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, output_limit=10.0, integral_limit=0.2)
        for _ in range(200):
            pid.update(1.0, 0.05)
        assert abs(pid.integral) <= 0.2 + 1e-9

    def test_saturation_unwinds_the_integral(self):
        """Without back-calculation the robot lurches when the line returns."""
        pid = PID(kp=1.0, ki=2.0, kd=0.0, output_limit=0.3, integral_limit=5.0)
        for _ in range(50):
            pid.update(1.0, 0.05)
        charged = abs(pid.integral)
        assert charged < 5.0, "integral kept charging against the output limit"

    def test_reset_clears_all_state(self, cfg):
        pid = PID.from_config(cfg.navigation.pid)
        for _ in range(10):
            pid.update(0.5, 1 / 30)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.state.output == 0.0
        assert pid.update(0.0, 1 / 30) == pytest.approx(0.0, abs=1e-9)

    def test_non_positive_dt_holds_the_previous_output(self, cfg):
        pid = PID.from_config(cfg.navigation.pid)
        first = pid.update(0.4, 1 / 30)
        assert pid.update(0.9, 0.0) == first

    def test_derivative_is_filtered(self):
        """A raw derivative of a quantised centroid is amplified noise."""
        unfiltered = PID(0.0, 0.0, 1.0, output_limit=1e6, derivative_filter_hz=0.0)
        filtered = PID(0.0, 0.0, 1.0, output_limit=1e6, derivative_filter_hz=5.0)
        unfiltered.update(0.0, 0.01); filtered.update(0.0, 0.01)
        raw_out = abs(unfiltered.update(1.0, 0.01))
        filt_out = abs(filtered.update(1.0, 0.01))
        assert filt_out < raw_out

    def test_gain_change_keeps_the_integral_contribution_continuous(self):
        pid = PID(0.0, 1.0, 0.0, output_limit=100.0, integral_limit=100.0)
        for _ in range(10):
            pid.update(0.5, 0.1)
        before = pid.ki * pid.integral
        pid.set_gains(ki=2.0)
        assert pid.ki * pid.integral == pytest.approx(before, rel=1e-9)

    def test_rejects_bad_limits(self):
        with pytest.raises(ValueError):
            PID(1, 0, 0, output_limit=0)
        with pytest.raises(ValueError):
            PID(1, 0, 0, integral_limit=-1)


class TestDriveCommand:
    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            DriveCommand(1.5, 0.0)

    def test_linear_and_differential(self):
        cmd = DriveCommand(0.2, 0.6)
        assert cmd.linear == pytest.approx(0.4)
        assert cmd.differential == pytest.approx(0.4)


class TestDifferentialMixer:
    def test_zero_correction_drives_straight(self, cfg):
        mixer = DifferentialMixer.from_config(cfg.robot)
        cmd = mixer.mix(0.18, 0.0)
        assert cmd.left == pytest.approx(cmd.right)

    def test_positive_correction_turns_left(self, cfg):
        """One sign convention across mix(), turn() and the yaw-rate helper."""
        mixer = DifferentialMixer.from_config(cfg.robot)
        cmd = mixer.mix(0.18, 0.3)
        assert cmd.right > cmd.left
        assert mixer.yaw_rate_for(cmd) > 0

    def test_turn_agrees_with_mix_on_sign(self, cfg):
        mixer = DifferentialMixer.from_config(cfg.robot)
        assert mixer.yaw_rate_for(mixer.turn(+1, 0.12)) > 0
        assert mixer.yaw_rate_for(mixer.turn(-1, 0.12)) < 0

    def test_saturation_preserves_the_turn(self, cfg):
        """Clamping each wheel independently straightens the robot mid-corner."""
        mixer = DifferentialMixer.from_config(cfg.robot)
        cmd = mixer.mix(0.35, 0.9)
        assert max(abs(cmd.left), abs(cmd.right)) <= 1.0 + 1e-9
        # The differential must survive, scaled but not destroyed.
        assert cmd.differential > 0.5

    def test_saturation_without_preservation_clips_the_differential(self, cfg):
        naive = DifferentialMixer(cfg.robot.max_speed_mps, cfg.robot.wheel_base_m,
                                  preserve_differential=False)
        smart = DifferentialMixer(cfg.robot.max_speed_mps, cfg.robot.wheel_base_m)
        assert smart.mix(0.35, 0.9).differential > naive.mix(0.35, 0.9).differential

    def test_forward_kinematics(self):
        linear, angular = wheel_speeds_to_body(0.1, 0.2, 0.24)
        assert linear == pytest.approx(0.15)
        assert angular == pytest.approx(0.1 / 0.24)

    def test_rejects_bad_geometry(self):
        with pytest.raises(ValueError):
            DifferentialMixer(0.0, 0.24)
        with pytest.raises(ValueError):
            DifferentialMixer(0.35, 0.0)


class TestSafetyScale:
    def test_full_stop_inside_the_stop_band(self):
        assert safety_scale(0.20, 0.25, 0.45) == 0.0

    def test_full_speed_beyond_the_slow_band(self):
        assert safety_scale(0.60, 0.25, 0.45) == 1.0

    def test_linear_ramp_between(self):
        assert safety_scale(0.35, 0.25, 0.45) == pytest.approx(0.5)

    def test_no_echo_is_treated_as_clear(self):
        assert safety_scale(math.inf, 0.25, 0.45) == 1.0

    def test_rejects_inverted_bands(self):
        with pytest.raises(ValueError):
            safety_scale(0.3, 0.45, 0.25)
