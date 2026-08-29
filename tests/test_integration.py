"""End-to-end software-in-the-loop missions.

These run the *complete, unmodified runtime* - fusion, perception, state
machine, spray sequencing, telemetry - against the mock MCU and the synthetic
arena, on a virtual clock. What they assert is what the robot *did*, which no
amount of unit testing of the parts can establish.

The single most important assertion in the suite is that no run ever actuates
on a crop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agribot.app.simulate import SimulationHarness
from agribot.mission.states import MissionState
from agribot.sim.arena import ArenaLayout, ArenaMarker
from agribot.telemetry.logger import RunLogger
from agribot.types import TargetClass


@pytest.fixture(scope="module")
def one_row_cfg():
    """A single-row mission, so a run terminates inside the test budget."""
    from agribot.config import load_config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "robot.yaml", use_local=False, use_env=False)
    return cfg.merged({"mission": {"rows": 1}})


@pytest.fixture(scope="module")
def demo_run(one_row_cfg):
    """One full mission over the demonstration arena, shared by several tests."""
    harness = SimulationHarness(
        one_row_cfg, start_lateral_m=0.04, start_heading_deg=-4.0, seed=7)
    metrics = harness.run(seconds=45.0)
    return harness, metrics


class TestFullMission:
    def test_mission_reaches_completion(self, demo_run):
        harness, metrics = demo_run
        assert harness.runtime.fsm.state is MissionState.MISSION_COMPLETE
        assert metrics.rows_completed >= 1

    def test_robot_actually_travelled_the_row(self, demo_run, one_row_cfg):
        harness, metrics = demo_run
        assert metrics.distance_m > 2.5
        # Fused odometry must agree with the simulator's ground truth.
        assert metrics.distance_m == pytest.approx(harness.mock.distance_m, rel=0.05)

    def test_every_weed_is_treated(self, demo_run, one_row_cfg):
        _harness, metrics = demo_run
        expected = sum(1 for m in ArenaLayout.default_demo().markers
                       if m.cls is TargetClass.WEED)
        assert metrics.weeds_treated == expected

    def test_no_crop_is_ever_sprayed(self, demo_run):
        """The one assertion that makes the whole design worth having."""
        _harness, metrics = demo_run
        assert metrics.crops_sprayed == 0
        assert metrics.crop_protection_rate == 1.0

    def test_crops_are_seen_and_counted_once_each(self, demo_run):
        _harness, metrics = demo_run
        expected = sum(1 for m in ArenaLayout.default_demo().markers
                       if m.cls is TargetClass.CROP)
        assert metrics.crops_seen == expected

    def test_navigation_converges_onto_the_row(self, demo_run):
        """Section 5.2 claims centring to within a few millimetres."""
        harness, metrics = demo_run
        assert harness.lateral_rms_after(3.0) < 0.005      # 5 mm RMS
        assert abs(harness.final_lateral_error_m) < 0.02

    def test_line_is_held_for_most_of_the_run(self, demo_run):
        _harness, metrics = demo_run
        # The tail of the run is deliberately blind - that is how the row end
        # is detected - so a perfect rate is neither expected nor desirable.
        assert metrics.line_lock_rate > 0.75

    def test_fluid_use_scales_with_weeds_not_distance(self, demo_run):
        _harness, metrics = demo_run
        assert metrics.spray.events == metrics.weeds_treated
        assert metrics.ml_per_weed == pytest.approx(
            metrics.total_ml / metrics.weeds_treated)

    def test_every_dose_is_flow_measured(self, demo_run):
        """Reported ml/weed must be a measurement, never a nominal estimate."""
        _harness, metrics = demo_run
        assert metrics.spray.fully_measured is True

    def test_saving_against_blanket_spraying_is_reported(self, demo_run):
        _harness, metrics = demo_run
        assert metrics.saving_percent > 0
        assert metrics.saving_ratio > 1.0

    def test_no_faults_recorded(self, demo_run):
        _harness, metrics = demo_run
        assert metrics.faults == []

    def test_encoder_gate_stays_quiet_on_a_clean_run(self, demo_run):
        """With no wheel spin the gate must not be rejecting good samples."""
        _harness, metrics = demo_run
        assert metrics.encoder_reject_rate < 0.02

    def test_report_renders(self, demo_run):
        _harness, metrics = demo_run
        assert "MISSION SUMMARY" in metrics.report()


class TestCropProtectionUnderPressure:
    def test_weed_touching_a_crop_is_not_sprayed(self, one_row_cfg):
        """The asymmetric veto must hold when the markers are adjacent."""
        layout = ArenaLayout(
            row_length_m=1.6,
            markers=[
                ArenaMarker(TargetClass.CROP, along_m=0.80, lateral_m=0.00),
                ArenaMarker(TargetClass.WEED, along_m=0.82, lateral_m=0.045),
            ],
        )
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=5)
        metrics = harness.run(seconds=35.0)
        assert metrics.crops_sprayed == 0
        assert metrics.crop_vetoes > 0, "the adjacent weed should have been vetoed"

    def test_a_row_of_crops_alone_produces_no_actuation(self, one_row_cfg):
        layout = ArenaLayout(
            row_length_m=2.0,
            markers=[ArenaMarker(TargetClass.CROP, along_m=a, lateral_m=lat)
                     for a, lat in ((0.6, -0.05), (1.0, 0.05), (1.4, -0.04))],
        )
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=5)
        metrics = harness.run(seconds=35.0)
        assert metrics.spray.events == 0
        assert metrics.crops_sprayed == 0
        assert harness.mock.dispensed_ml == 0.0

    def test_elevated_weed_is_still_engaged(self, one_row_cfg):
        """Rules allow markers on a ~15 cm raised surface."""
        layout = ArenaLayout(
            row_length_m=1.6,
            markers=[ArenaMarker(TargetClass.WEED, along_m=0.9, lateral_m=0.0,
                                 height_m=0.15)],
        )
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=5)
        metrics = harness.run(seconds=35.0)
        assert metrics.weeds_treated == 1


class TestSafetyBehaviours:
    def test_obstacle_halts_the_robot(self, one_row_cfg):
        layout = ArenaLayout(row_length_m=8.0, markers=[])
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=3)
        runtime = harness.runtime
        assert runtime.setup()

        # Drive clear for a moment, then place an obstacle in front.
        for _ in range(90):
            harness.mock.step(0.01)
            runtime.tick()
        travelled_before = harness.mock.x
        harness.mock.set_obstacle(front_m=0.12)
        for _ in range(150):
            harness.mock.step(0.01)
            runtime.tick()

        assert runtime.fsm.state is MissionState.PAUSE
        assert harness.mock.x - travelled_before < 0.15
        runtime.shutdown("test")

    def test_obstacle_clearing_resumes_the_run(self, one_row_cfg):
        layout = ArenaLayout(row_length_m=8.0, markers=[])
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=3)
        runtime = harness.runtime
        runtime.setup()
        for _ in range(60):
            harness.mock.step(0.01)
            runtime.tick()
        harness.mock.set_obstacle(front_m=0.12)
        for _ in range(90):
            harness.mock.step(0.01)
            runtime.tick()
        assert runtime.fsm.state is MissionState.PAUSE

        harness.mock.set_obstacle(front_m=math.inf)
        for _ in range(90):
            harness.mock.step(0.01)
            runtime.tick()
        assert runtime.fsm.state is MissionState.FOLLOW_LINE
        runtime.shutdown("test")

    def test_wheel_spin_is_gated_out_of_the_odometry(self, one_row_cfg):
        """The Section 5.3 claim, exercised through the whole stack."""
        layout = ArenaLayout(row_length_m=10.0, markers=[])
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=3)
        runtime = harness.runtime
        runtime.setup()

        for _ in range(300):
            harness.mock.step(0.01)
            runtime.tick()

        harness.mock.set_slip(True)
        for _ in range(120):                     # 1.2 s of wheel spin
            harness.mock.step(0.01)
            runtime.tick()
        harness.mock.set_slip(False)
        for _ in range(200):
            harness.mock.step(0.01)
            runtime.tick()

        fused = runtime.state_fusion.distance_m
        truth = harness.mock.distance_m
        assert runtime.state_fusion.distance.n_rejected > 0, "gate never fired"
        # Raw odometry would have gained ~0.3 m of phantom distance.
        assert abs(fused - truth) < 0.10
        runtime.shutdown("test")

    def test_mcu_silence_escalates_to_estop(self, one_row_cfg):
        layout = ArenaLayout(row_length_m=8.0, markers=[])
        harness = SimulationHarness(one_row_cfg, layout=layout, seed=3)
        runtime = harness.runtime
        runtime.setup()
        for _ in range(60):
            harness.mock.step(0.01)
            runtime.tick()
        assert runtime.fsm.state is MissionState.FOLLOW_LINE

        # Simulate the link going quiet: advance time without stepping the MCU.
        harness.mock._open = False
        for _ in range(60):
            harness.mock.t += 0.01
            runtime.tick()
        assert runtime.fsm.state is MissionState.ESTOP
        runtime.shutdown("test")

    def test_dry_run_never_opens_the_valve(self, one_row_cfg):
        harness = SimulationHarness(one_row_cfg, seed=7, dry_run=True)
        metrics = harness.run(seconds=35.0)
        assert metrics.spray.events == 0
        assert harness.mock.dispensed_ml == 0.0
        assert harness.mock.valve_open is False

    def test_shutdown_leaves_the_hardware_safe(self, one_row_cfg):
        harness = SimulationHarness(one_row_cfg, seed=7)
        harness.run(seconds=20.0)
        assert harness.mock.valve_open is False
        assert harness.mock.pump_on is False
        assert harness.mock.estopped is True


class TestRunLogging:
    def test_a_run_writes_a_complete_log_set(self, one_row_cfg, tmp_path):
        logger = RunLogger(tmp_path, run_name="sil")
        harness = SimulationHarness(one_row_cfg, seed=7, logger=logger)
        metrics = harness.run(seconds=45.0)

        run_dir = tmp_path / "sil"
        assert (run_dir / "events.jsonl").is_file()
        assert (run_dir / "timeseries.csv").is_file()
        assert (run_dir / "summary.json").is_file()

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["metrics"]["perception"]["crops_sprayed"] == 0
        assert summary["metrics"]["sustainability"]["ml_per_weed"] is not None

        kinds = [json.loads(line)["kind"] for line in
                 (run_dir / "events.jsonl").read_text(
                     encoding="utf-8").strip().splitlines()]
        assert "startup" in kinds and "shutdown" in kinds
        assert kinds.count("spray") == metrics.weeds_treated
        assert "transition" in kinds

    def test_logged_timeseries_feeds_the_fusion_analysis(self, one_row_cfg, tmp_path):
        """The proposal's claim that the same script runs on logged data."""
        from agribot.sim.fusion_study import analyse_run
        from agribot.sim.sensors import load_run_from_csv

        logger = RunLogger(tmp_path, run_name="sil")
        harness = SimulationHarness(one_row_cfg, seed=7, logger=logger)
        harness.run(seconds=25.0)

        run = load_run_from_csv(tmp_path / "sil" / "timeseries.csv")
        assert len(run) > 50
        result = analyse_run(run)
        # No ground truth on a real log, so RMSE is unavailable by design.
        assert result.metrics["have_ground_truth"] is False
        assert result.distance_fused[-1] > 0


class TestDeterminismAndRobustness:
    def test_the_same_seed_gives_the_same_run(self, one_row_cfg):
        """A non-deterministic simulation cannot be a regression test."""
        first = SimulationHarness(one_row_cfg, seed=11).run(seconds=30.0)
        second = SimulationHarness(one_row_cfg, seed=11).run(seconds=30.0)
        assert first.weeds_treated == second.weeds_treated
        assert first.total_ml == pytest.approx(second.total_ml)
        assert first.distance_m == pytest.approx(second.distance_m, rel=1e-9)

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [3, 7, 11, 19])
    def test_crop_is_never_sprayed_across_seeds(self, one_row_cfg, seed):
        metrics = SimulationHarness(one_row_cfg, seed=seed).run(seconds=45.0)
        assert metrics.crops_sprayed == 0

    @pytest.mark.slow
    @pytest.mark.parametrize("lateral,heading", [
        (0.0, 0.0), (0.05, -6.0), (-0.05, 6.0), (0.07, 0.0), (0.0, -10.0),
    ])
    def test_converges_from_a_range_of_bad_starts(self, one_row_cfg, lateral, heading):
        layout = ArenaLayout(row_length_m=6.0, markers=[])
        harness = SimulationHarness(
            one_row_cfg, layout=layout, start_lateral_m=lateral,
            start_heading_deg=heading, seed=5)
        harness.run(seconds=20.0)
        assert harness.lateral_rms_after(6.0) < 0.01
