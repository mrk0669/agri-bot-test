"""Configuration loading, override layering and self-consistency."""

from __future__ import annotations

import pytest
import yaml

from agribot.config import Config, load_config


class TestConfigAccess:
    def test_dotted_attribute_access(self, cfg):
        assert cfg.robot.name == "AgriBot-1"
        assert isinstance(cfg.navigation.pid.kp, float)

    def test_mapping_interface(self, cfg):
        assert "robot" in cfg
        assert cfg["robot"]["name"] == "AgriBot-1"
        assert len(cfg) > 5

    def test_dotted_path_lookup(self, cfg):
        assert cfg.dotted("navigation.pid.kp") == cfg.navigation.pid.kp
        assert cfg.dotted("nope.not.here", "fallback") == "fallback"

    def test_missing_key_names_the_alternatives(self, cfg):
        with pytest.raises(AttributeError, match="available"):
            _ = cfg.does_not_exist

    def test_config_is_read_only(self, cfg):
        with pytest.raises(TypeError):
            cfg.robot = {}

    def test_to_dict_is_a_deep_copy(self, cfg):
        snapshot = cfg.to_dict()
        snapshot["robot"]["name"] = "mutated"
        assert cfg.robot.name == "AgriBot-1"

    def test_merged_does_not_mutate_the_original(self, cfg):
        merged = cfg.merged({"robot": {"cruise_speed_mps": 0.05}})
        assert merged.robot.cruise_speed_mps == 0.05
        assert cfg.robot.cruise_speed_mps != 0.05
        # Sibling keys in the merged section must survive the merge.
        assert merged.robot.name == cfg.robot.name

    def test_lists_of_mappings_are_wrapped(self, cfg):
        first = cfg.perception.color.weed.hsv_ranges[0]
        assert first["lower"][0] == 0


class TestOverrideLayers:
    def test_explicit_overrides_win(self, tmp_path):
        path = tmp_path / "robot.yaml"
        path.write_text(yaml.safe_dump({"robot": {"a": 1, "b": 2}}), encoding="utf-8")
        cfg = load_config(path, use_local=False, use_env=False,
                          overrides={"robot": {"b": 99}})
        assert cfg.robot.a == 1 and cfg.robot.b == 99

    def test_local_file_is_merged(self, tmp_path):
        (tmp_path / "robot.yaml").write_text(
            yaml.safe_dump({"robot": {"a": 1, "b": 2}}), encoding="utf-8")
        (tmp_path / "robot.local.yaml").write_text(
            yaml.safe_dump({"robot": {"b": 42}}), encoding="utf-8")
        cfg = load_config(tmp_path / "robot.yaml", use_env=False)
        assert cfg.robot.a == 1 and cfg.robot.b == 42

    def test_environment_overrides_are_typed(self, tmp_path):
        (tmp_path / "robot.yaml").write_text(
            yaml.safe_dump({"spray": {"enabled": True, "burst_ms": 220}}),
            encoding="utf-8")
        cfg = load_config(
            tmp_path / "robot.yaml", use_local=False,
            environ={"AGRIBOT_SPRAY__ENABLED": "false",
                     "AGRIBOT_SPRAY__BURST_MS": "150"},
        )
        assert cfg.spray.enabled is False        # parsed as a bool, not "false"
        assert cfg.spray.burst_ms == 150         # parsed as an int

    def test_precedence_env_below_explicit(self, tmp_path):
        (tmp_path / "robot.yaml").write_text(
            yaml.safe_dump({"a": {"b": 1}}), encoding="utf-8")
        cfg = load_config(tmp_path / "robot.yaml", use_local=False,
                          environ={"AGRIBOT_A__B": "2"},
                          overrides={"a": {"b": 3}})
        assert cfg.a.b == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path):
        path = tmp_path / "robot.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(path, use_local=False, use_env=False)


class TestShippedConfigIsSane:
    """The committed config is what runs in the arena; check it holds together."""

    def test_bounding_box_within_rules(self, cfg):
        box = cfg.robot.bounding_box_cm
        assert max(box.length, box.width, box.height) <= 30.0

    def test_cruise_below_max_speed(self, cfg):
        assert 0 < cfg.robot.cruise_speed_mps <= cfg.robot.max_speed_mps

    def test_ultrasonic_bands_ordered(self, cfg):
        assert cfg.safety.ultrasonic_slow_m > cfg.safety.ultrasonic_stop_m > 0

    def test_servo_limits_bracket_centre(self, cfg):
        for axis in (cfg.targeting.pan, cfg.targeting.tilt):
            assert axis.min_deg <= axis.centre_deg <= axis.max_deg

    def test_red_weed_range_covers_both_ends_of_the_hue_circle(self, cfg):
        """Red wraps H=0/180; a single range silently misses half the markers."""
        ranges = cfg.perception.color.weed.hsv_ranges
        assert len(ranges) >= 2
        lows = sorted(r["lower"][0] for r in ranges)
        assert lows[0] <= 10 and lows[-1] >= 160

    def test_encoder_gate_is_a_fixed_physical_bound(self, cfg):
        gate = cfg.fusion.distance.encoder_innovation_gate_mps
        assert 0 < gate < 0.5

    def test_pid_limit_does_not_reverse_a_wheel_while_following(self, cfg):
        normalised_cruise = cfg.robot.cruise_speed_mps / cfg.robot.max_speed_mps
        assert cfg.navigation.pid.output_limit <= normalised_cruise + 1e-9

    def test_line_loss_watchdog_allows_the_row_end_creep(self, cfg):
        """The stall watchdog must not fire before a row end can be declared."""
        creep = cfg.robot.cruise_speed_mps * cfg.navigation.blind_creep_scale
        needed = cfg.mission.row_end_detect_m / creep
        assert cfg.navigation.line_lost_stop_s > needed
