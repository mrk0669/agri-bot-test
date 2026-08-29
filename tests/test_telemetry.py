"""Run logging and the mission metrics that answer the judging criteria."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from agribot.targeting.spray_controller import SprayStats
from agribot.telemetry.logger import TIMESERIES_COLUMNS, RunLogger, json_safe
from agribot.telemetry.metrics import (
    BLANKET_ML_PER_M2,
    MissionMetrics,
    blanket_equivalent_ml,
)
from agribot.types import AimSolution, SprayEvent


class TestJsonSafe:
    def test_converts_enums_and_dataclasses(self):
        aim = AimSolution(pan_deg=72.0, tilt_deg=88.0, in_range=True)
        event = SprayEvent(event_id=1, track_id=2, timestamp=1.0, aim=aim,
                           burst_ms=220.0, volume_ml=1.8, measured=True)
        payload = json_safe(event)
        assert json.dumps(payload)              # must be serialisable

    def test_converts_numpy_scalars_and_arrays(self):
        assert json_safe(np.float64(1.5)) == 1.5
        assert json_safe(np.array([1, 2, 3])) == [1, 2, 3]

    def test_non_finite_floats_become_null(self):
        """JSON has no NaN or Infinity; emitting them makes the log unparseable."""
        assert json_safe(float("nan")) is None
        assert json_safe(float("inf")) is None
        json.dumps(json_safe({"range": float("inf")}))   # must not raise

    def test_nested_structures(self):
        payload = json_safe({"a": [np.int64(1), {"b": float("nan")}]})
        assert payload == {"a": [1, {"b": None}]}


class TestRunLogger:
    def test_creates_a_run_directory(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        assert (tmp_path / "demo").is_dir()
        logger.close()

    def test_events_are_one_json_object_per_line(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        logger.event("startup", dry_run=False)
        logger.transition("INIT", "FOLLOW_LINE", "line acquired")
        logger.close()

        lines = (tmp_path / "demo" / "events.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert records[0]["kind"] == "startup"
        assert records[1]["from"] == "INIT" and records[1]["to"] == "FOLLOW_LINE"

    def test_events_are_flushed_immediately(self, tmp_path):
        """The run that ends because the battery died is the one that matters."""
        logger = RunLogger(tmp_path, run_name="demo")
        logger.event("startup")
        content = (tmp_path / "demo" / "events.jsonl").read_text(encoding="utf-8")
        assert "startup" in content              # before close()
        logger.close()

    def test_spray_events_and_vetoes_are_recorded(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        aim = AimSolution(pan_deg=72.0, tilt_deg=88.0, in_range=True)
        logger.spray_event(SprayEvent(1, 2, 1.0, aim, 220.0, 1.8, True))
        logger.veto("crop_proximity(d=40px)", {"cls": "weed"})
        logger.close()

        kinds = [json.loads(line)["kind"] for line in
                 (tmp_path / "demo" / "events.jsonl").read_text(
                     encoding="utf-8").strip().splitlines()]
        assert kinds == ["spray", "crop_veto"]

    def test_timeseries_header_matches_the_fusion_schema(self, tmp_path):
        """tools/kalman_sim.py --csv reads these columns back."""
        logger = RunLogger(tmp_path, run_name="demo")
        logger.close()
        header = (tmp_path / "demo" / "timeseries.csv").read_text(
            encoding="utf-8").splitlines()[0]
        assert header.split(",")[:7] == TIMESERIES_COLUMNS[:7]
        assert header.split(",")[:3] == ["t", "gyro_z", "accel_x"]

    def test_unknown_sample_keys_are_ignored(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        logger.sample(t=0.1, line_error=0.2, something_unknown=5)
        logger.close()
        rows = (tmp_path / "demo" / "timeseries.csv").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(rows) == 2

    def test_summary_is_written(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        path = logger.write_summary({"reason": "done", "value": np.float64(2)})
        logger.close()
        assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2

    def test_close_is_idempotent(self, tmp_path):
        logger = RunLogger(tmp_path, run_name="demo")
        logger.close()
        logger.close()

    def test_context_manager(self, tmp_path):
        with RunLogger(tmp_path, run_name="demo") as logger:
            logger.event("x")
        assert (tmp_path / "demo" / "events.jsonl").is_file()


class TestBlanketEquivalent:
    def test_area_times_rate(self):
        assert blanket_equivalent_ml(10.0, 0.3) == pytest.approx(
            10.0 * 0.3 * BLANKET_ML_PER_M2)

    def test_rejects_bad_geometry(self):
        with pytest.raises(ValueError):
            blanket_equivalent_ml(-1.0, 0.3)
        with pytest.raises(ValueError):
            blanket_equivalent_ml(1.0, 0.0)


class TestMissionMetrics:
    def _metrics(self, **kwargs):
        metrics = MissionMetrics(swath_m=0.30, **kwargs)
        return metrics

    def test_line_lock_rate(self):
        metrics = self._metrics(frames_processed=100, frames_line_found=87)
        assert metrics.line_lock_rate == pytest.approx(0.87)

    def test_line_lock_rate_with_no_frames(self):
        assert self._metrics().line_lock_rate == 0.0

    def test_ml_per_weed_comes_from_the_spray_stats(self):
        metrics = self._metrics()
        metrics.spray = SprayStats(events=3, total_ml=5.4, measured_events=3)
        assert metrics.ml_per_weed == pytest.approx(1.8)

    def test_saving_against_blanket_spraying(self):
        """The sustainability claim, computed rather than asserted."""
        metrics = self._metrics(distance_m=6.0)
        metrics.spray = SprayStats(events=4, total_ml=7.2, measured_events=4)
        assert metrics.blanket_ml == pytest.approx(6.0 * 0.30 * BLANKET_ML_PER_M2)
        assert metrics.saving_ratio == pytest.approx(metrics.blanket_ml / 7.2)
        assert 0 < metrics.saving_percent < 100

    def test_saving_is_infinite_when_nothing_was_sprayed(self):
        metrics = self._metrics(distance_m=6.0)
        metrics.spray = SprayStats()
        assert math.isinf(metrics.saving_ratio)

    def test_crop_protection_is_perfect_by_construction(self):
        metrics = self._metrics(crops_seen=5, crops_sprayed=0)
        assert metrics.crop_protection_rate == 1.0

    def test_crop_protection_falls_if_a_crop_is_sprayed(self):
        """Below 1.0 is a defect signal, not a statistic to be tuned."""
        metrics = self._metrics(crops_seen=4, crops_sprayed=1)
        assert metrics.crop_protection_rate == pytest.approx(0.75)

    def test_crop_protection_with_no_crops_seen(self):
        assert self._metrics().crop_protection_rate == 1.0

    def test_encoder_reject_rate(self):
        metrics = self._metrics(encoder_samples=1000, encoder_rejected=60)
        assert metrics.encoder_reject_rate == pytest.approx(0.06)

    def test_measured_flag_propagates(self):
        metrics = self._metrics()
        metrics.spray = SprayStats(events=2, total_ml=3.6, measured_events=1)
        assert metrics.to_dict()["sustainability"]["measured"] is False

    def test_to_dict_is_json_serialisable(self):
        metrics = self._metrics(distance_m=3.0, frames_processed=100,
                                frames_line_found=90, crops_seen=2)
        metrics.spray = SprayStats(events=3, total_ml=5.4, measured_events=3)
        json.dumps(metrics.to_dict())

    def test_report_mentions_the_headline_figures(self):
        metrics = self._metrics(distance_m=3.0, frames_processed=100,
                                frames_line_found=90, weeds_detected=3,
                                weeds_treated=3, crops_seen=2, crop_vetoes=1)
        metrics.spray = SprayStats(events=3, total_ml=5.4, measured_events=3)
        report = metrics.report()
        assert "ml" in report and "saving" in report
        assert "protection" in report

    def test_report_without_spray_events(self):
        assert "no spray events" in self._metrics().report()

    def test_faults_are_listed(self):
        metrics = self._metrics()
        metrics.faults.append("MCU link lost")
        assert "MCU link lost" in metrics.report()
