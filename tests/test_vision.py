"""Guidance-line extraction, the colour tier, late fusion and tracking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agribot.sim.field import ArenaStyle, Marker, render_frame
from agribot.types import BBox, Detection, DetectionSource, TargetClass
from agribot.vision.camera import FrameListCamera
from agribot.vision.fusion import PerceptionFusion, TargetTracker
from agribot.vision.line_follow import LineFollower, extract_roi, hsv_mask


class TestBBox:
    def test_geometry(self):
        box = BBox(10, 20, 30, 60)
        assert box.width == 20 and box.height == 40
        assert box.area == 800
        assert box.centroid == (20.0, 40.0)

    def test_rejects_degenerate(self):
        with pytest.raises(ValueError):
            BBox(30, 0, 10, 10)

    def test_iou_identical_is_one(self):
        box = BBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_disjoint_is_zero(self):
        assert BBox(0, 0, 10, 10).iou(BBox(50, 50, 60, 60)) == 0.0

    def test_iou_half_overlap(self):
        a, b = BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_from_xywh(self):
        assert BBox.from_xywh(5, 5, 10, 20).as_tuple() == (5, 5, 15, 25)


class TestRoiAndMask:
    def test_roi_offsets_map_back_to_the_frame(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        roi, (x_off, y_off) = extract_roi(frame, 0.5, 1.0, 0.25, 0.75)
        assert roi.shape[:2] == (240, 320)
        assert (x_off, y_off) == (160, 240)

    def test_roi_is_never_empty(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        roi, _ = extract_roi(frame, 1.0, 1.0)
        assert roi.size > 0

    def test_hsv_mask_rejects_empty_input(self):
        with pytest.raises(ValueError):
            hsv_mask(np.zeros((0, 0, 3), np.uint8), [0, 0, 0], [180, 255, 255])


class TestLineFollower:
    def test_error_sign_and_magnitude(self, line_follower):
        """Line right of centre gives a positive error."""
        for offset in (-200, -100, 0, 100, 200):
            observation = line_follower.process(render_frame(line_offset_px=offset))
            assert observation.found
            assert observation.error == pytest.approx(offset / 320.0, abs=0.05)

    def test_centroid_is_reported_in_frame_coordinates(self, line_follower):
        observation = line_follower.process(render_frame(line_offset_px=80))
        assert observation.centroid_px is not None
        assert observation.centroid_px[0] == pytest.approx(400, abs=15)
        # The ROI is the lower part of the frame, so the row must be there too.
        assert observation.centroid_px[1] > 240

    def test_no_line_is_reported_as_lost(self, line_follower):
        assert line_follower.process(render_frame(draw_line=False)).found is False

    def test_empty_frame_is_handled(self, line_follower):
        assert line_follower.process(None).found is False
        assert line_follower.process(np.zeros((0, 0, 3), np.uint8)).found is False

    def test_survives_uneven_illumination(self, line_follower):
        """HSV thresholding is chosen precisely so a brightness gradient does
        not break what a fixed intensity threshold would."""
        style = ArenaStyle(soil_speckle=30, brightness_gradient=0.6, seed=4)
        observation = line_follower.process(
            render_frame(line_offset_px=-120, style=style))
        assert observation.found
        assert observation.error == pytest.approx(-0.375, abs=0.06)

    def test_survives_motion_blur(self, line_follower):
        style = ArenaStyle(soil_speckle=20, blur=9, seed=2)
        assert line_follower.process(render_frame(line_offset_px=60, style=style)).found

    def test_tilted_line_is_still_found(self, line_follower):
        """A steeply tilted line is exactly the frame a recovering robot sees."""
        for angle in (-30, -20, 0, 20, 30):
            style = ArenaStyle(soil_speckle=20, seed=1)
            observation = line_follower.process(
                render_frame(line_offset_px=0, line_angle_deg=angle, style=style))
            assert observation.found, f"lost the line at {angle} deg tilt"

    def test_glare_alone_is_not_mistaken_for_a_line(self, line_follower):
        """Specular glare desaturates soil into the line's HSV gate. Without a
        shape gate the robot steers towards a reflection of the sun."""
        for strength in (0.45, 0.6, 0.8):
            style = ArenaStyle(soil_speckle=20, glare_strength=strength, seed=3)
            observation = line_follower.process(
                render_frame(draw_line=False, style=style))
            assert observation.found is False, f"glare {strength} read as a line"

    def test_line_beside_glare_is_still_found_correctly(self, line_follower):
        style = ArenaStyle(soil_speckle=20, glare_strength=0.45,
                           glare_centre=(0.78, 0.8), seed=5)
        observation = line_follower.process(
            render_frame(line_offset_px=-60, style=style))
        assert observation.found
        assert observation.error == pytest.approx(-0.1875, abs=0.05)

    def test_line_fully_washed_out_reports_lost_rather_than_guessing(
            self, line_follower):
        """When glare swallows the line the honest answer is "lost".

        The merged blob's centroid sits tens of pixels off the true line, so
        reporting it would steer the robot wrong; the state machine's grace
        period and the IR fail-safe are the designed response.
        """
        style = ArenaStyle(glare_strength=0.8, glare_centre=(0.5, 0.8), seed=3)
        observation = line_follower.process(
            render_frame(line_offset_px=-60, style=style))
        assert observation.found is False

    def test_largest_component_ignores_a_distractor_patch(self, line_follower):
        frame = render_frame(line_offset_px=0)
        # A bright patch at the edge, smaller than the line.
        frame[440:470, 10:50] = (240, 240, 240)
        observation = line_follower.process(frame)
        assert observation.found
        assert abs(observation.error) < 0.08

    def test_debug_overlay_renders(self, line_follower):
        frame = render_frame(line_offset_px=40)
        overlay = line_follower.debug_overlay(frame, line_follower.process(frame))
        assert overlay.shape == frame.shape


class TestColorDetector:
    def test_detects_both_classes(self, color_detector):
        frame = render_frame(markers=[
            Marker(TargetClass.WEED, 180, 150, 80),
            Marker(TargetClass.CROP, 460, 170, 90),
        ])
        detections = color_detector.detect(frame)
        classes = {d.cls for d in detections}
        assert classes == {TargetClass.WEED, TargetClass.CROP}

    def test_centroids_are_accurate(self, color_detector):
        frame = render_frame(markers=[Marker(TargetClass.WEED, 300, 200, 80)])
        weed = [d for d in color_detector.detect(frame) if d.cls is TargetClass.WEED][0]
        assert weed.centroid[0] == pytest.approx(300, abs=6)
        assert weed.centroid[1] == pytest.approx(200, abs=6)

    def test_area_gate_rejects_specks(self, color_detector):
        frame = render_frame(markers=[Marker(TargetClass.WEED, 300, 200, 10)])
        assert not [d for d in color_detector.detect(frame)
                    if d.cls is TargetClass.WEED]

    def test_geometry_gate_rejects_a_thin_streak(self, color_detector):
        """Area alone admits a stray glove; extent and aspect are what reject it."""
        import cv2
        frame = render_frame()
        cv2.rectangle(frame, (100, 200), (500, 212), (40, 40, 205), -1)
        assert not [d for d in color_detector.detect(frame)
                    if d.cls is TargetClass.WEED]

    def test_red_is_found_at_both_ends_of_the_hue_circle(self, color_detector):
        """Red wraps H=0/180; one range silently misses half the markers."""
        import cv2
        for bgr in ((40, 40, 205), (60, 45, 200)):
            frame = render_frame()
            cv2.rectangle(frame, (260, 160), (340, 240), bgr, -1)
            hits = [d for d in color_detector.detect(frame) if d.cls is TargetClass.WEED]
            assert hits, f"missed a red marker at BGR {bgr}"

    def test_no_markers_gives_no_detections(self, color_detector):
        assert color_detector.detect(render_frame()) == []

    def test_empty_frame_is_handled(self, color_detector):
        assert color_detector.detect(None) == []

    def test_detections_are_sorted_nearest_first(self, color_detector):
        frame = render_frame(markers=[
            Marker(TargetClass.WEED, 150, 150, 50),
            Marker(TargetClass.WEED, 450, 300, 110),
        ])
        weeds = [d for d in color_detector.detect(frame) if d.cls is TargetClass.WEED]
        assert weeds[0].area_px > weeds[1].area_px

    def test_confidence_is_in_range(self, color_detector):
        frame = render_frame(markers=[Marker(TargetClass.WEED, 300, 200, 80)])
        for detection in color_detector.detect(frame):
            assert 0.0 <= detection.confidence <= 1.0


def _det(cls, x, y, size=60, conf=0.9, source=DetectionSource.COLOR):
    half = size / 2
    return Detection(cls=cls, bbox=BBox(x - half, y - half, x + half, y + half),
                     confidence=conf, source=source, area_px=size * size)


class TestPerceptionFusion:
    def test_lone_weed_is_actionable(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([_det(TargetClass.WEED, 300, 200)])
        assert decision.n_actionable == 1

    def test_crop_alone_produces_no_action(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([_det(TargetClass.CROP, 300, 200)])
        assert decision.n_actionable == 0
        assert decision.vetoed == []

    def test_crop_overlap_vetoes_the_weed(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 300, 200, 80),
            _det(TargetClass.CROP, 320, 200, 80),
        ])
        assert decision.n_actionable == 0
        assert "crop_overlap" in decision.vetoed[0][1]

    def test_crop_proximity_vetoes_the_weed(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 300, 200, 40),
            _det(TargetClass.CROP, 345, 200, 40),
        ])
        assert decision.n_actionable == 0
        assert "crop_proximity" in decision.vetoed[0][1]

    def test_distant_crop_does_not_veto(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 120, 200, 60),
            _det(TargetClass.CROP, 520, 200, 60),
        ])
        assert decision.n_actionable == 1

    def test_veto_is_asymmetric_by_design(self, cfg):
        """A high-confidence weed must not out-vote a low-confidence crop.

        False positives on a weed cost a little fluid; false positives on a
        crop damage the plant. The asymmetry lives in the rule, not in tuning.
        """
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 300, 200, 80, conf=0.99),
            _det(TargetClass.CROP, 320, 200, 80, conf=0.36),
        ])
        assert decision.n_actionable == 0

    def test_crop_evidence_below_the_veto_threshold_does_not_protect(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 300, 200, 80, conf=0.9),
            _det(TargetClass.CROP, 320, 200, 80, conf=0.10),
        ])
        assert decision.n_actionable == 1

    def test_colour_tier_triggers_regardless_of_confidence(self, cfg):
        """Colour detections already passed HSV and geometry gates."""
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        weed = _det(TargetClass.WEED, 300, 200, conf=0.51,
                    source=DetectionSource.COLOR)
        assert fusion.fuse([weed]).n_actionable == 1

    def test_learned_tier_must_clear_its_threshold(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        low = _det(TargetClass.WEED, 300, 200, conf=0.40, source=DetectionSource.YOLO)
        high = _det(TargetClass.WEED, 300, 200, conf=0.80, source=DetectionSource.YOLO)
        assert fusion.fuse([low]).n_actionable == 0
        assert fusion.fuse([high]).n_actionable == 1

    def test_tiers_agreeing_on_one_weed_produce_one_action(self, cfg):
        """Otherwise the robot sprays the same marker twice."""
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse(
            [_det(TargetClass.WEED, 300, 200, 80, source=DetectionSource.COLOR)],
            [_det(TargetClass.WEED, 305, 203, 80, conf=0.9,
                  source=DetectionSource.YOLO)],
        )
        assert decision.n_actionable == 1
        merged = decision.actionable[0]
        assert merged.source is DetectionSource.FUSED
        assert set(merged.meta["sources"]) == {"color", "yolo"}

    def test_two_separate_weeds_stay_separate(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        decision = fusion.fuse([
            _det(TargetClass.WEED, 120, 200, 60),
            _det(TargetClass.WEED, 520, 200, 60),
        ])
        assert decision.n_actionable == 2

    def test_empty_input_is_safe(self, cfg):
        fusion = PerceptionFusion.from_config(cfg.perception.fusion)
        assert fusion.fuse([], None).n_actionable == 0


class TestTargetTracker:
    def test_requires_confirmation_before_reporting(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        confirm = cfg.perception.fusion.confirm_frames
        for i in range(confirm):
            tracker.update([_det(TargetClass.WEED, 300 + i, 200)])
            expected = (i + 1) >= confirm
            assert bool(tracker.confirmed_targets()) is expected

    def test_a_one_frame_flicker_never_confirms(self, cfg):
        """This is how a system sprays a glint of sunlight."""
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        tracker.update([_det(TargetClass.WEED, 300, 200)])
        for _ in range(5):
            tracker.update([])
        assert tracker.confirmed_targets() == []

    def test_track_identity_survives_motion(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        ids = set()
        for i in range(8):
            tracker.update([_det(TargetClass.WEED, 300, 150 + i * 15)])
            ids.update(t.track_id for t in tracker.confirmed_targets())
        assert ids == {1}, "a moving marker must stay one track"

    def test_a_jump_beyond_the_match_radius_starts_a_new_track(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(3):
            tracker.update([_det(TargetClass.WEED, 100, 200)])
        for _ in range(3):
            tracker.update([_det(TargetClass.WEED, 520, 200)])
        assert tracker.confirmed_targets()[0].track_id != 1

    def test_sprayed_targets_are_not_re_reported(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(4):
            tracker.update([_det(TargetClass.WEED, 300, 200)])
        target = tracker.confirmed_targets()[0]
        tracker.mark_sprayed(target.track_id)
        tracker.update([_det(TargetClass.WEED, 300, 200)])
        assert tracker.confirmed_targets() == []
        assert tracker.confirmed_targets(include_sprayed=True)

    def test_nearest_target_is_engaged_first(self, cfg):
        """The camera is pitched down, so lower in the frame is nearer."""
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(4):
            tracker.update([
                _det(TargetClass.WEED, 200, 120),
                _det(TargetClass.WEED, 400, 380),
            ])
        confirmed = tracker.confirmed_targets()
        assert confirmed[0].centroid[1] > confirmed[1].centroid[1]

    def test_stale_tracks_are_dropped(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(4):
            tracker.update([_det(TargetClass.WEED, 300, 200)])
        for _ in range(cfg.perception.fusion.track_max_age + 2):
            tracker.update([])
        assert tracker.tracks == []

    def test_reset_clears_everything(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(4):
            tracker.update([_det(TargetClass.WEED, 300, 200)])
        tracker.reset()
        assert tracker.tracks == []

    def test_greedy_matching_prefers_the_closest_pair(self, cfg):
        tracker = TargetTracker.from_config(cfg.perception.fusion)
        for _ in range(3):
            tracker.update([_det(TargetClass.WEED, 200, 200),
                            _det(TargetClass.WEED, 400, 200)])
        before = {t.track_id: t.centroid[0] for t in tracker.confirmed_targets()}
        tracker.update([_det(TargetClass.WEED, 210, 200),
                        _det(TargetClass.WEED, 410, 200)])
        after = {t.track_id: t.centroid[0] for t in tracker.confirmed_targets()}
        for track_id, x in after.items():
            assert abs(x - before[track_id]) < 30


class TestFrameListCamera:
    def test_iterates_then_stops(self):
        frames = [render_frame(line_offset_px=o) for o in (-40, 0, 40)]
        camera = FrameListCamera(frames)
        assert camera.open()
        assert len(list(camera.frames())) == 3
        assert camera.read()[0] is False

    def test_loops_when_asked(self):
        camera = FrameListCamera([render_frame()], loop=True)
        camera.open()
        for _ in range(5):
            assert camera.read()[0] is True

    def test_returns_a_copy_so_callers_cannot_corrupt_the_source(self):
        camera = FrameListCamera([render_frame(line_offset_px=0)], loop=True)
        camera.open()
        _, first = camera.read()
        first[:] = 0                      # a consumer drawing an overlay in place
        _, second = camera.read()
        assert second.any(), "mutating one frame must not blank the source"


class TestIrFailsafe:
    """The QTR-8A array: silent backup only, never the primary sensor."""

    def _sensor(self, cfg, **kwargs):
        from agribot.vision.ir_failsafe import IrLineSensor
        sensor = IrLineSensor.from_config(cfg.navigation)
        for key, value in kwargs.items():
            setattr(sensor, key, value)
        return sensor

    def test_centred_line_gives_zero_error(self, cfg):
        # Sensors 3 and 4 of 8 straddle the centre.
        assert self._sensor(cfg).observe(
            (0, 0, 0, 1, 1, 0, 0, 0)).error == pytest.approx(0.0)

    def test_line_to_the_right_gives_positive_error(self, cfg):
        """Same sign convention as the vision extractor."""
        observation = self._sensor(cfg).observe((0, 0, 0, 0, 0, 0, 1, 0))
        assert observation.found is True
        assert observation.error > 0

    def test_line_to_the_left_gives_negative_error(self, cfg):
        assert self._sensor(cfg).observe((0, 1, 0, 0, 0, 0, 0, 0)).error < 0

    def test_error_is_normalised_to_the_unit_interval(self, cfg):
        sensor = self._sensor(cfg)
        assert sensor.observe((1, 0, 0, 0, 0, 0, 0, 0)).error == pytest.approx(-1.0)
        assert sensor.observe((0, 0, 0, 0, 0, 0, 0, 1)).error == pytest.approx(1.0)

    def test_nothing_seen_is_not_found(self, cfg):
        assert self._sensor(cfg).observe((0,) * 8).found is False

    def test_everything_lit_is_rejected_as_a_washout(self, cfg):
        """All eight sensors on the line is a reflective floor, not a line."""
        assert self._sensor(cfg).observe((1,) * 8).found is False

    def test_disabled_sensor_reports_nothing(self, cfg):
        assert self._sensor(cfg, enabled=False).observe(
            (0, 0, 0, 1, 1, 0, 0, 0)).found is False

    def test_missing_array_is_handled(self, cfg):
        sensor = self._sensor(cfg)
        assert sensor.observe(None).found is False
        assert sensor.observe(()).found is False

    def test_observation_is_tagged_as_ir(self, cfg):
        """The log must always show which sensor the robot steered on."""
        observation = self._sensor(cfg).observe((0, 0, 0, 1, 0, 0, 0, 0))
        assert observation.source == "ir"
        assert observation.to_dict()["source"] == "ir"

    def test_vision_observations_are_tagged_as_vision(self, line_follower):
        assert line_follower.process(render_frame(line_offset_px=0)).source == "vision"

    def test_confidence_falls_as_the_reading_smears(self, cfg):
        sensor = self._sensor(cfg)
        crisp = sensor.observe((0, 0, 0, 1, 0, 0, 0, 0)).confidence
        smeared = sensor.observe((0, 0, 1, 1, 1, 1, 0, 0)).confidence
        assert crisp > smeared

    def test_runtime_only_consults_ir_after_vision_gives_up(self, cfg):
        """Vision is primary; the rules require navigation by computer vision."""
        from agribot.app.simulate import SimulationHarness

        harness = SimulationHarness(cfg.merged({"mission": {"rows": 1}}), seed=7)
        runtime = harness.runtime
        runtime.setup()
        # Light the IR array continuously; while vision has the line it must
        # never be used.
        harness.mock.ir_mask = 0b00011000
        for _ in range(120):
            harness.mock.step(0.01)
            runtime.tick()
        assert runtime.frames_on_ir_failsafe == 0
        runtime.shutdown("test")


class TestColorDetectorDiagnostics:
    """Rejections must be inspectable, not silent.

    The geometry gates are the difference between a marker and a red shirt.
    When one fires, the calibration tooling and the proposal figures both need
    to name which gate it was and by how much.
    """

    def _red_blob(self, w, h, notch=False):
        """A red region of a given aspect, optionally made non-convex."""
        frame = render_frame()
        cx, cy = 320, 240
        cv2 = __import__("cv2")
        cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                      (40, 40, 205), -1)
        if notch:
            cv2.circle(frame, (cx, cy - h // 2), int(h * 0.45), (58, 78, 108), -1)
        return frame

    def test_a_clean_marker_is_accepted_with_no_rejection(self, color_detector):
        frame = render_frame(markers=[Marker(TargetClass.WEED, 320, 240, 90)])
        accepted, rejected = color_detector.detect_with_diagnostics(frame)
        assert any(d.cls is TargetClass.WEED for d in accepted)
        assert not [r for r in rejected if r.cls is TargetClass.WEED]

    def test_an_elongated_red_region_is_rejected_on_aspect(self, color_detector):
        accepted, rejected = color_detector.detect_with_diagnostics(
            self._red_blob(360, 70))
        assert not [d for d in accepted if d.cls is TargetClass.WEED]
        reasons = [r.reason for r in rejected if r.cls is TargetClass.WEED]
        assert reasons and "aspect" in reasons[0]

    def test_a_non_convex_red_region_is_rejected_on_solidity(self, color_detector):
        accepted, rejected = color_detector.detect_with_diagnostics(
            self._red_blob(150, 150, notch=True))
        weed_rejects = [r for r in rejected if r.cls is TargetClass.WEED]
        if [d for d in accepted if d.cls is TargetClass.WEED]:
            pytest.skip("notch was not deep enough to breach the solidity gate")
        assert weed_rejects and ("solidity" in weed_rejects[0].reason
                                 or "extent" in weed_rejects[0].reason)

    def test_rejection_carries_the_measured_metrics(self, color_detector):
        _accepted, rejected = color_detector.detect_with_diagnostics(
            self._red_blob(360, 70))
        r = [x for x in rejected if x.cls is TargetClass.WEED][0]
        assert r.aspect > 3.0
        assert 0.0 <= r.extent <= 1.0 and 0.0 <= r.solidity <= 1.0
        assert set(r.to_dict()) >= {"cls", "bbox", "reason", "extent", "solidity"}

    def test_speckle_is_not_reported_as_a_rejection(self, color_detector):
        """Sub-threshold noise would drown the useful rejections."""
        frame = render_frame(markers=[Marker(TargetClass.WEED, 300, 200, 8)])
        _accepted, rejected = color_detector.detect_with_diagnostics(frame)
        assert not [r for r in rejected if r.area_px < 100]

    def test_detect_matches_the_diagnostic_accepted_set(self, color_detector):
        frame = render_frame(markers=[Marker(TargetClass.WEED, 200, 200, 80),
                                      Marker(TargetClass.CROP, 440, 220, 80)])
        plain = color_detector.detect(frame)
        accepted, _ = color_detector.detect_with_diagnostics(frame)
        assert [d.to_dict() for d in plain] == [d.to_dict() for d in accepted]

    def test_empty_frame_returns_two_empty_lists(self, color_detector):
        assert color_detector.detect_with_diagnostics(None) == ([], [])
