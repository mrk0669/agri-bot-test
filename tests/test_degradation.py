"""Graceful degradation when optional components are absent.

The layered perception design is only worth having if a missing tier degrades
instead of crashing: the colour baseline is the guaranteed scorer, the learned
tiers are additions. These tests run on a machine with no trained weights -
which is also the state of a freshly imaged Jetson - and assert that the system
still starts, still runs, and still reports honestly that a tier is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from agribot.app.preflight import FAIL, PASS, WARN, run_checks
from agribot.sim.field import Marker, render_frame
from agribot.types import TargetClass
from agribot.utils.logging_setup import get_logger, setup_logging
from agribot.vision.camera import FrameListCamera, VideoFileCamera
from agribot.vision.yolo_detector import YoloDetector
from agribot.vision.zeroshot_detector import ZeroShotDetector


@pytest.fixture
def frame():
    return render_frame(markers=[Marker(TargetClass.WEED, 300, 200, 80)])


class TestYoloDegradation:
    def test_missing_weights_reports_unavailable_rather_than_raising(self):
        detector = YoloDetector(weights="models/definitely-not-here.engine")
        assert detector.available is False
        assert detector.load_error

    def test_detect_returns_empty_when_unavailable(self, frame):
        detector = YoloDetector(weights="models/definitely-not-here.engine")
        assert detector.detect(frame) == []

    def test_load_is_attempted_only_once(self):
        """A per-frame retry of a failing load would stall the control loop."""
        detector = YoloDetector(weights="models/definitely-not-here.engine")
        assert detector.available is False
        first_error = detector.load_error
        for _ in range(5):
            assert detector.available is False
        assert detector.load_error is first_error

    def test_engine_falls_back_to_the_pytorch_checkpoint(self, tmp_path):
        """One config must work on both the Jetson (.engine) and a laptop (.pt)."""
        checkpoint = tmp_path / "weeds.pt"
        checkpoint.write_bytes(b"not really a model")
        detector = YoloDetector(
            weights=str(tmp_path / "weeds.engine"),
            fallback_weights=str(checkpoint),
        )
        assert detector._resolve_weights() == str(checkpoint)

    def test_missing_engine_and_missing_fallback_resolves_to_none(self, tmp_path):
        detector = YoloDetector(
            weights=str(tmp_path / "weeds.engine"),
            fallback_weights=str(tmp_path / "weeds.pt"),
        )
        assert detector._resolve_weights() is None

    def test_class_map_translates_indices(self):
        detector = YoloDetector(weights="x.pt", class_map={0: "crop", 1: "weed"})
        assert detector._map_class(0, "0") is TargetClass.CROP
        assert detector._map_class(1, "1") is TargetClass.WEED

    def test_unmapped_classes_are_dropped(self):
        """A 58-species dataset can be used by mapping only what matters."""
        detector = YoloDetector(weights="x.pt", class_map={0: "crop", 1: "weed"})
        assert detector._map_class(7, "soil") is None

    def test_label_names_are_matched_when_no_index_mapping_exists(self):
        detector = YoloDetector(weights="x.pt", class_map={})
        assert detector._map_class(3, "Weed") is TargetClass.WEED
        assert detector._map_class(4, "seedling") is TargetClass.CROP

    def test_stats_are_reportable_while_unavailable(self):
        detector = YoloDetector(weights="nope.engine")
        detector.available
        stats = detector.stats()
        assert stats["available"] is False and stats["load_error"]

    def test_empty_frame_is_handled(self):
        detector = YoloDetector(weights="nope.engine")
        assert detector.detect(None) == []
        assert detector.detect(np.zeros((0, 0, 3), np.uint8)) == []

    def test_from_config(self, cfg):
        detector = YoloDetector.from_config(cfg.perception.yolo)
        assert detector.conf == cfg.perception.yolo.conf
        assert detector.class_map == {0: "crop", 1: "weed"}


class TestZeroShotDegradation:
    def test_missing_model_reports_unavailable(self):
        detector = ZeroShotDetector(model="models/not-a-world-model.pt")
        assert detector.available is False
        assert detector.detect(render_frame()) == []

    def test_vocabulary_is_built_from_the_prompts(self, cfg):
        detector = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        vocabulary = detector.vocabulary
        assert "red marker" in vocabulary
        assert "green plant" in vocabulary

    def test_prompts_map_back_to_the_two_mission_classes(self, cfg):
        detector = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        assert detector._map_class(0, "red marker") is TargetClass.WEED
        assert detector._map_class(0, "green plant") is TargetClass.CROP

    def test_positional_fallback_when_the_label_is_an_index(self, cfg):
        detector = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        first = detector.vocabulary[0]
        assert detector._map_class(0, "0") is detector._label_to_class[first.lower()]

    def test_unknown_label_maps_to_nothing(self, cfg):
        detector = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        assert detector._map_class(99, "tractor") is None

    def test_prompts_can_be_changed_at_runtime(self, cfg):
        """If the markers turn out orange, change the prompt, not the model."""
        detector = ZeroShotDetector.from_config(cfg.perception.zeroshot)
        detector.set_prompts(weed=["orange marker"], crop=["green leaf"])
        assert detector.vocabulary == ["orange marker", "green leaf"]
        assert detector._map_class(0, "orange marker") is TargetClass.WEED

    def test_empty_prompt_set_is_refused(self):
        with pytest.raises(ValueError):
            ZeroShotDetector(prompts={"weed": [], "crop": []})


class TestRuntimeWithoutLearnedTiers:
    def test_runtime_builds_and_runs_with_the_colour_tier_alone(self, cfg):
        """The shipped config has both learned tiers disabled by design."""
        from agribot.app.simulate import SimulationHarness

        harness = SimulationHarness(cfg.merged({"mission": {"rows": 1}}), seed=7)
        assert harness.runtime.yolo is None
        assert harness.runtime.zeroshot is None
        assert harness.runtime.color_detector is not None
        metrics = harness.run(seconds=25.0)
        assert metrics.weeds_treated > 0

    def test_runtime_survives_learned_tiers_enabled_but_absent(self, cfg):
        """Flipping the flag before the weights exist must not stop the run."""
        from agribot.app.simulate import SimulationHarness

        merged = cfg.merged({
            "mission": {"rows": 1},
            "perception": {"yolo": {"enabled": True,
                                    "weights": "models/absent.engine",
                                    "fallback_weights": "models/absent.pt"}},
        })
        harness = SimulationHarness(merged, seed=7)
        assert harness.runtime.yolo is not None
        assert harness.runtime.yolo.available is False
        metrics = harness.run(seconds=25.0)
        assert metrics.weeds_treated > 0        # colour tier carried the run
        assert metrics.crops_sprayed == 0


class TestPreflight:
    def test_shipped_config_passes_every_required_check(self, cfg):
        checks = run_checks(cfg, skip_hardware=True)
        failures = [c for c in checks if c.status == FAIL and c.required]
        assert failures == [], [f"{c.name}: {c.detail}" for c in failures]

    def test_oversized_robot_is_rejected(self, cfg):
        merged = cfg.merged({"robot": {"bounding_box_cm": {"length": 35.0}}})
        checks = run_checks(merged, skip_hardware=True)
        assert any(c.status == FAIL and "bounding box" in c.name for c in checks)

    def test_inconsistent_watchdog_is_rejected(self, cfg):
        """The stall watchdog firing before a row end can be declared."""
        merged = cfg.merged({"navigation": {"line_lost_stop_s": 0.5}})
        checks = run_checks(merged, skip_hardware=True)
        assert any(c.status == FAIL and "watchdog" in c.name for c in checks)

    def test_inverted_ultrasonic_bands_are_rejected(self, cfg):
        merged = cfg.merged({"safety": {"ultrasonic_slow_m": 0.1}})
        checks = run_checks(merged, skip_hardware=True)
        assert any(c.status == FAIL and "ultrasonic" in c.name for c in checks)

    def test_missing_learned_weights_warns_but_does_not_fail(self, cfg):
        """Degrading to the colour tier is by design, not an error."""
        merged = cfg.merged({"perception": {"yolo": {
            "enabled": True, "weights": "models/absent.engine",
            "fallback_weights": "models/absent.pt"}}})
        checks = run_checks(merged, skip_hardware=True)
        learned = [c for c in checks if "learned detector" in c.name]
        assert learned and learned[0].status == WARN
        assert all(c.status != FAIL for c in checks if c.required)

    def test_a_raising_check_is_reported_not_propagated(self, cfg):
        """One broken check must not prevent the others from being reported."""
        broken = cfg.merged({"targeting": {"pan": {"min_deg": 200.0}}})
        checks = run_checks(broken, skip_hardware=True)
        assert any(c.status == FAIL for c in checks)


class TestCameraBackends:
    def test_video_file_camera_reports_a_missing_file(self, tmp_path):
        camera = VideoFileCamera(tmp_path / "nope.mp4")
        assert camera.open() is False

    def test_frame_list_camera_context_manager(self):
        with FrameListCamera([render_frame()]) as camera:
            ok, frame = camera.read()
            assert ok and frame is not None

    def test_context_manager_raises_on_a_camera_that_will_not_open(self, tmp_path):
        with pytest.raises(RuntimeError):
            with VideoFileCamera(tmp_path / "nope.mp4"):
                pass

    def test_frames_iterator_respects_the_limit(self):
        camera = FrameListCamera([render_frame()] * 10, loop=True)
        camera.open()
        assert len(list(camera.frames(limit=4))) == 4


class TestLoggingSetup:
    def test_creates_a_log_file(self, tmp_path):
        setup_logging(log_dir=tmp_path, force=True)
        get_logger("test.module").warning("hello")
        for handler in logging.getLogger("agribot").handlers:
            handler.flush()
        assert (tmp_path / "agribot.log").is_file()
        assert "hello" in (tmp_path / "agribot.log").read_text(encoding="utf-8")

    def test_child_logger_naming(self):
        assert get_logger("vision.line").name == "agribot.vision.line"
        assert get_logger("agribot.already").name == "agribot.already"

    def test_unknown_level_falls_back_to_info(self, tmp_path):
        setup_logging(log_dir=tmp_path, console_level="NOT_A_LEVEL", force=True)
        assert logging.getLogger("agribot").handlers
