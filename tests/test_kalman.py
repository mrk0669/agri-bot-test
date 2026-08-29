"""Kalman heading and distance filters, and the Section 5.3 study.

The most important test in this file is
``TestFixedVsAdaptiveGate::test_adaptive_gate_admits_a_spin_sample`` - it
*demonstrates* the failure the proposal describes as the reason for choosing a
fixed physical innovation gate, rather than asserting it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agribot.control.kalman import (
    AdaptiveGateDistanceKalman,
    DistanceKalman,
    FusionStack,
    HeadingKalman,
    joseph_update,
)
from agribot.sim.fusion_study import analyse_run, run_seed_sweep
from agribot.sim.sensors import (
    RunProfile,
    SpinEvent,
    load_run_from_csv,
    save_run_to_csv,
    simulate_run,
)


class TestJosephForm:
    def test_keeps_the_covariance_symmetric(self):
        P = np.array([[2.0, 0.4], [0.4, 1.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.5]])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P_new = joseph_update(P, K, H, R)
        assert np.allclose(P_new, P_new.T)

    def test_stays_positive_semidefinite(self):
        P = np.diag([1e-8, 1e-8])
        H = np.array([[1.0, 0.0]])
        R = np.array([[1e-6]])
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        assert np.all(np.linalg.eigvalsh(joseph_update(P, K, H, R)) >= -1e-15)


class TestHeadingKalman:
    def test_identifies_a_constant_gyro_bias(self):
        """Carrying bias as a state means the drift is subtracted, not smoothed."""
        bias = math.radians(0.78)
        kf = HeadingKalman(mag_noise_std_rad=math.radians(1.5))
        for i in range(4000):
            kf.predict(bias, 0.01)          # true rate zero, gyro reads the bias
            if i % 5 == 0:
                kf.update(0.0)
        assert kf.bias_deg_s == pytest.approx(0.78, abs=0.05)
        assert abs(kf.heading_deg) < 0.5

    def test_tracks_a_real_turn(self):
        rate = math.radians(30.0)
        kf = HeadingKalman(mag_noise_std_rad=math.radians(1.5))
        truth = 0.0
        for i in range(300):
            kf.predict(rate, 0.01)
            truth += rate * 0.01
            if i % 5 == 0:
                kf.update(truth)
        assert math.degrees(kf.heading_rad) == pytest.approx(math.degrees(truth), abs=1.5)

    def test_innovation_is_wrapped_across_the_seam(self):
        """Raw angle subtraction near +/-pi produces a violent wrong correction."""
        kf = HeadingKalman(initial_heading=math.radians(179),
                           mag_noise_std_rad=math.radians(1.5),
                           innovation_gate_rad=None)
        kf.update(math.radians(-179))
        # The true correction is +2 deg, not -358 deg.
        assert kf.last_innovation_rad == pytest.approx(math.radians(2), abs=1e-6)
        assert abs(kf.heading_deg) > 170       # still near the seam, not near zero

    def test_gate_rejects_a_magnetic_disturbance(self):
        kf = HeadingKalman(mag_noise_std_rad=math.radians(1.5),
                           innovation_gate_rad=math.radians(8.0))
        for _ in range(200):
            kf.predict(0.0, 0.01)
            kf.update(0.0)
        settled = kf.heading_deg
        assert kf.update(math.radians(30.0)) is False
        assert kf.heading_deg == pytest.approx(settled, abs=0.2)
        assert kf.n_rejected == 1

    def test_gate_accepts_normal_noise(self):
        kf = HeadingKalman(mag_noise_std_rad=math.radians(1.5),
                           innovation_gate_rad=math.radians(8.0))
        rng = np.random.default_rng(3)
        accepted = 0
        for _ in range(500):
            kf.predict(0.0, 0.01)
            accepted += kf.update(math.radians(rng.normal(0, 1.5)))
        assert accepted > 490

    def test_variance_shrinks_with_updates(self):
        kf = HeadingKalman(mag_noise_std_rad=math.radians(1.5))
        initial = kf.heading_var
        for _ in range(100):
            kf.predict(0.0, 0.01)
            kf.update(0.0)
        assert kf.heading_var < initial

    def test_zero_dt_predict_is_a_no_op(self):
        kf = HeadingKalman()
        before = kf.x.copy()
        kf.predict(1.0, 0.0)
        assert np.allclose(kf.x, before)

    def test_from_config(self, cfg):
        kf = HeadingKalman.from_config(cfg.fusion.heading)
        assert kf.innovation_gate_rad == pytest.approx(
            math.radians(cfg.fusion.heading.mag_innovation_gate_deg))


class TestDistanceKalman:
    def test_tracks_constant_velocity(self):
        kf = DistanceKalman()
        for i in range(2000):
            kf.predict(0.0, 0.01)
            if i % 2 == 0:
                kf.update(0.18)
        assert kf.velocity_mps == pytest.approx(0.18, abs=0.01)
        assert kf.position_m == pytest.approx(0.18 * 20.0, rel=0.02)

    def test_estimates_the_accelerometer_bias(self):
        bias = 0.05
        kf = DistanceKalman()
        for i in range(4000):
            kf.predict(bias, 0.01)          # stationary, accel reads only bias
            if i % 2 == 0:
                kf.update(0.0)
        assert kf.accel_bias == pytest.approx(bias, abs=0.02)
        assert abs(kf.position_m) < 0.05

    def test_fixed_gate_rejects_wheel_spin(self):
        kf = DistanceKalman(innovation_gate_mps=0.05)
        for i in range(500):
            kf.predict(0.0, 0.01)
            if i % 2 == 0:
                kf.update(0.18)
        assert kf.update(0.60) is False       # a spin: encoder reads 0.6 m/s
        assert kf.last_rejected is True

    def test_gate_accepts_a_physically_reachable_change(self):
        kf = DistanceKalman(innovation_gate_mps=0.05)
        for i in range(200):
            kf.predict(0.0, 0.01)
            if i % 2 == 0:
                kf.update(0.18)
        assert kf.update(0.20) is True

    def test_forced_reacquire_after_a_long_reject_run(self):
        """Two solid seconds of disagreement means the filter drifted, not the wheels."""
        kf = DistanceKalman(innovation_gate_mps=0.05, max_consecutive_rejects=10)
        kf.update(0.18)                       # first sample initialises the filter
        for _ in range(9):
            assert kf.update(5.0) is False
        assert kf.update(5.0) is True         # re-acquire on the tenth
        assert kf.consecutive_rejects == 0

    def test_reset_position_leaves_velocity_alone(self):
        kf = DistanceKalman()
        for i in range(200):
            kf.predict(0.0, 0.01)
            kf.update(0.18)
        velocity = kf.velocity_mps
        kf.reset_position()
        assert kf.position_m == 0.0
        assert kf.velocity_mps == pytest.approx(velocity)

    def test_from_config(self, cfg):
        kf = DistanceKalman.from_config(cfg.fusion.distance)
        assert kf.innovation_gate_mps == cfg.fusion.distance.encoder_innovation_gate_mps


class TestFixedVsAdaptiveGate:
    """Why the design uses a fixed physical bound - demonstrated, not asserted.

    The proposal argues that an adaptive N-sigma gate widens as the covariance
    grows during a coast and eventually admits the very sample it exists to
    reject. These tests measure that happening, and measure the fixed gate not
    doing it, using the same filter class with only the gate swapped.
    """

    CRUISE = 0.18
    SPIN_EXCESS = 0.25          # encoder over-reads by this much while spinning
    ENCODER_EVERY = 2           # 50 Hz encoder against a 100 Hz predict

    @classmethod
    def _settle(cls, kf, ramp_s=1.0, hold_s=5.0, dt=0.01):
        """Accelerate to cruise, then hold, with physically consistent signals.

        The accelerometer must report the robot actually accelerating during
        the ramp. Feeding a velocity ramp with a zero accelerometer is not a
        cold start, it is an inconsistent one, and it makes the filter look
        broken when it is being lied to.
        """
        accel = cls.CRUISE / ramp_s
        velocity = 0.0
        for i in range(int(ramp_s / dt)):
            kf.predict(accel, dt)
            velocity += accel * dt
            if i % cls.ENCODER_EVERY == 0:
                kf.update(velocity)
        for i in range(int(hold_s / dt)):
            kf.predict(0.0, dt)
            if i % cls.ENCODER_EVERY == 0:
                kf.update(cls.CRUISE)
        return kf

    @classmethod
    def _spin(cls, kf, seconds):
        """Apply a continuous wheel spin. Returns how many samples were admitted."""
        admitted = 0
        for i in range(int(seconds / 0.01)):
            kf.predict(0.0, 0.01)
            if i % cls.ENCODER_EVERY == 0:
                if kf.update(cls.CRUISE + cls.SPIN_EXCESS):
                    admitted += 1
        return admitted

    def test_fixed_gate_threshold_never_moves(self):
        """The bound is physical - the rover's reachable acceleration - so no
        amount of coasting can widen it."""
        kf = self._settle(DistanceKalman(innovation_gate_mps=0.05,
                                         max_consecutive_rejects=10 ** 9))
        assert kf._gate_threshold() == pytest.approx(0.05)
        for _ in range(6000):                     # 60 s of coasting
            kf.predict(0.0, 0.01)
        assert kf._gate_threshold() == pytest.approx(0.05)

    def test_adaptive_gate_threshold_grows_without_bound(self):
        """The same coast widens an N-sigma gate by more than an order of magnitude."""
        kf = self._settle(AdaptiveGateDistanceKalman(n_sigma=3.0,
                                                     max_consecutive_rejects=10 ** 9))
        settled = kf._gate_threshold()
        assert settled < 0.05, "N-sigma starts tighter than the fixed bound"
        for _ in range(3000):                     # 30 s of coasting
            kf.predict(0.0, 0.01)
        assert kf._gate_threshold() > 10 * settled
        # And it has now grown past the very spin magnitude it must reject.
        assert kf._gate_threshold() > self.SPIN_EXCESS

    def test_adaptive_gate_admits_a_spin_the_fixed_gate_never_does(self):
        """The failure observed during development, side by side.

        Forced re-acquire is disabled on both filters so that what is measured
        is the gate itself, not the separate bounded-coast policy.
        """
        adaptive = self._settle(
            AdaptiveGateDistanceKalman(n_sigma=3.0, max_consecutive_rejects=10 ** 9))
        fixed = self._settle(
            DistanceKalman(innovation_gate_mps=0.05, max_consecutive_rejects=10 ** 9))

        assert self._spin(adaptive, 20.0) > 0, "adaptive gate should let the spin in"
        assert self._spin(fixed, 20.0) == 0, "fixed gate must never let it in"

        # Once admitted the adaptive filter tracks the spin; the fixed one does
        # not, and its velocity stays at the speed the robot is really doing.
        assert adaptive.velocity_mps > self.CRUISE + 0.05
        assert fixed.velocity_mps == pytest.approx(self.CRUISE, abs=0.02)

    def test_fixed_gate_rejects_a_realistic_spin_entirely(self):
        """A field wheel-spin lasts on the order of a second, not twenty."""
        kf = self._settle(DistanceKalman(innovation_gate_mps=0.05))
        before = kf.position_m
        assert self._spin(kf, 1.2) == 0
        # Coasting on the bias-corrected accelerometer, so distance still
        # advances at roughly the true speed rather than the spun-up one.
        gained = kf.position_m - before
        assert gained == pytest.approx(self.CRUISE * 1.2, abs=0.05)

    def test_forced_reacquire_bounds_how_long_the_filter_coasts(self):
        """The bounded-coast policy is deliberate, and it has a cost.

        After ``max_consecutive_rejects`` samples the filter re-acquires even
        against a still-spinning wheel, on the reasoning that two solid seconds
        of disagreement is more likely a drifted filter than a two-second spin.
        The consequence is that a spin longer than that WILL pull the estimate.
        That trade is recorded here so it cannot change unnoticed.
        """
        rejects = 100
        kf = self._settle(DistanceKalman(innovation_gate_mps=0.05,
                                         max_consecutive_rejects=rejects))
        # 100 rejects at a 50 Hz encoder = 2 s of coasting before re-acquire.
        assert self._spin(kf, 1.9) == 0
        assert self._spin(kf, 1.0) > 0

    def test_first_sample_initialises_rather_than_being_gated(self):
        """A filter started while the robot is already moving must acquire.

        The gate defends a prediction; at the first sample the "prediction" is
        the zero the filter was seeded with. Gating against it locks the filter
        out permanently, because every subsequent sample looks equally wrong.
        """
        kf = DistanceKalman(innovation_gate_mps=0.05,
                            max_consecutive_rejects=10 ** 9)
        assert kf.update(0.18) is True          # far outside the gate, yet accepted
        assert kf.velocity_mps == pytest.approx(0.18)
        # And the gate is live from the second sample onwards.
        assert kf.update(0.90) is False


class TestFusionStack:
    def test_steps_both_filters(self, cfg):
        stack = FusionStack.from_config(cfg.fusion)
        for i in range(500):
            stack.step(t=i * 0.01, gyro_z=0.0, accel_x=0.0,
                       mag_heading=0.0 if i % 5 == 0 else None,
                       encoder_vel=0.18 if i % 2 == 0 else None)
        assert stack.distance_m > 0
        assert abs(stack.heading_deg) < 1.0

    def test_snapshot_reports_both_states(self, cfg):
        stack = FusionStack.from_config(cfg.fusion)
        stack.step(0.0, 0.0, 0.0)
        stack.step(0.01, 0.0, 0.0, encoder_vel=0.18)
        snapshot = stack.snapshot(0.01)
        assert snapshot.t == 0.01
        assert "distance_m" in snapshot.to_dict()

    def test_absent_measurements_are_skipped_not_faked(self, cfg):
        stack = FusionStack.from_config(cfg.fusion)
        mag_ok, enc_ok = stack.step(0.01, 0.0, 0.0)
        assert mag_ok is False and enc_ok is False


class TestSection53Study:
    """Regression lock on the published Section 5.3 results."""

    @pytest.fixture(scope="class")
    def result(self):
        return analyse_run(simulate_run(RunProfile()))

    def test_single_sensor_baselines_match_the_proposal(self, result):
        m = result.metrics
        assert m["heading_rmse_gyro_deg"] == pytest.approx(18.27, abs=0.5)
        assert m["heading_rmse_mag_deg"] == pytest.approx(2.77, abs=0.15)
        assert m["distance_rmse_accel_m"] == pytest.approx(16.44, abs=0.5)
        assert m["distance_rmse_encoder_m"] == pytest.approx(0.300, abs=0.02)
        assert m["distance_final_encoder_m"] == pytest.approx(0.567, abs=0.02)

    def test_fused_estimates_meet_or_beat_the_proposal(self, result):
        m = result.metrics
        assert m["heading_rmse_fused_deg"] <= 0.60
        assert m["distance_rmse_fused_m"] <= 0.006
        assert abs(m["distance_final_fused_m"]) <= 0.010

    def test_gyro_bias_is_recovered(self, result):
        m = result.metrics
        assert m["gyro_bias_est_deg_s"] == pytest.approx(
            m["gyro_bias_true_deg_s"], abs=0.02)

    def test_fusion_beats_both_single_sensors_by_a_wide_margin(self, result):
        m = result.metrics
        assert m["heading_improvement_vs_gyro"] > 30.0
        assert m["distance_improvement_vs_encoder"] > 30.0

    def test_gate_separates_spin_from_clean_samples_perfectly(self, result):
        m = result.metrics
        assert m["spin_reject_recall"] == 1.0
        assert m["clean_reject_rate"] == 0.0

    def test_magnetometer_disturbance_is_gated_out(self, result):
        assert result.metrics["mag_rejected"] > 0

    def test_summary_table_has_the_published_shape(self, result):
        rows = result.summary_table()
        assert rows[0][0] == "Quantity"
        assert len(rows) == 5 and all(len(r) == 4 for r in rows)

    @pytest.mark.slow
    def test_holds_across_independent_seeds(self):
        """A filter that works on one noise realisation has not been verified."""
        sweep = run_seed_sweep([12, 19, 26, 33, 40, 47, 54, 61])
        assert sweep["distance_rmse_fused_m_worst"] < 0.014
        assert sweep["heading_rmse_fused_deg_worst"] < 1.0
        assert sweep["spin_reject_recall_min"] == 1.0
        assert sweep["clean_reject_rate_max"] < 0.01


class TestRunProfileAndCsv:
    def test_spin_events_inject_the_stated_distance(self):
        spin = SpinEvent(1.0, 2.0, 0.25)
        assert spin.distance_gained_m == pytest.approx(0.25)

    def test_truth_distance_matches_the_speed_profile(self):
        run = simulate_run(RunProfile())
        assert run.true_distance_m[-1] > 4.0
        assert np.all(np.diff(run.true_distance_m) >= -1e-12)   # monotonic

    def test_csv_round_trip_preserves_the_analysis(self, tmp_path):
        """The proposal claims the same script runs on logged data; check it."""
        run = simulate_run(RunProfile())
        path = save_run_to_csv(run, tmp_path / "run.csv")
        reloaded = load_run_from_csv(path)
        assert len(reloaded) == len(run)
        original = analyse_run(run).metrics
        replayed = analyse_run(reloaded).metrics
        assert replayed["distance_rmse_fused_m"] == pytest.approx(
            original["distance_rmse_fused_m"], rel=1e-6)
        assert replayed["heading_rmse_fused_deg"] == pytest.approx(
            original["heading_rmse_fused_deg"], rel=1e-6)

    def test_csv_without_ground_truth_reports_unavailable(self, tmp_path):
        """Hardware logs have no external reference; RMSE must not be faked."""
        import csv as _csv
        path = tmp_path / "hw.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.writer(fh)
            writer.writerow(["t", "gyro_z", "accel_x", "mag_heading_rad",
                             "mag_valid", "encoder_mps", "enc_valid"])
            for i in range(300):
                writer.writerow([i * 0.01, 0.0136, 0.04, 0.0, int(i % 5 == 0),
                                 0.18, int(i % 2 == 0)])
        result = analyse_run(load_run_from_csv(path))
        assert result.metrics["have_ground_truth"] is False
        assert math.isnan(result.metrics["heading_rmse_fused_deg"])
        # The filter still produces an estimate even without truth to score it.
        assert result.distance_fused[-1] > 0
