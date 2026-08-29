"""Command-line entry points.

These are thin, but they are the layer the operator actually touches under
time pressure, and a typo in an override key would only surface at the venue.
The tests exercise parsing and override construction without ever opening
hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agribot.app import main as main_mod
from agribot.app import preflight as preflight_mod
from agribot.app import simulate as simulate_mod
from agribot.config import load_config


class TestMainParser:
    def test_defaults(self):
        args = main_mod.build_parser().parse_args([])
        assert args.dry_run is False
        assert args.mode is None
        assert args.preflight is False

    def test_dry_run_and_mode(self):
        args = main_mod.build_parser().parse_args(["--dry-run", "--mode", "mark"])
        assert args.dry_run is True and args.mode == "mark"

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(SystemExit):
            main_mod.build_parser().parse_args(["--mode", "flamethrower"])

    def test_overrides_are_built_under_the_right_keys(self):
        args = main_mod.build_parser().parse_args(
            ["--mode", "mark", "--rows", "3", "--port", "COM7"])
        overrides = main_mod._overrides(args)
        assert overrides["spray"]["mode"] == "mark"
        assert overrides["mission"]["rows"] == 3
        assert overrides["mcu"]["port"] == "COM7"

    def test_overrides_actually_apply_to_the_config(self):
        args = main_mod.build_parser().parse_args(["--rows", "4"])
        cfg = load_config(use_local=False, use_env=False,
                          overrides=main_mod._overrides(args))
        assert cfg.mission.rows == 4

    def test_numeric_camera_source_becomes_an_int(self):
        """`--camera 0` must select /dev/video0, not a file called "0"."""
        args = main_mod.build_parser().parse_args(["--camera", "0"])
        assert main_mod._overrides(args)["camera"]["source"] == 0

    def test_path_camera_source_stays_a_string(self):
        args = main_mod.build_parser().parse_args(["--camera", "clip.mp4"])
        assert main_mod._overrides(args)["camera"]["source"] == "clip.mp4"

    def test_no_flags_produces_no_overrides(self):
        args = main_mod.build_parser().parse_args([])
        assert main_mod._overrides(args) == {}


class TestSimulateParser:
    def test_defaults(self):
        args = simulate_mod.build_parser().parse_args([])
        assert args.seconds > 0
        assert args.dry_run is False

    def test_rows_and_seed(self):
        args = simulate_mod.build_parser().parse_args(
            ["--rows", "2", "--seed", "42", "--seconds", "10"])
        assert (args.rows, args.seed, args.seconds) == (2, 42, 10.0)

    def test_simulate_main_runs_and_returns_zero(self, tmp_path):
        """A run that sprayed a crop returns non-zero; a clean one returns 0."""
        code = simulate_mod.main(
            ["--seconds", "20", "--rows", "1", "--quiet", "--seed", "7"])
        assert code == 0


class TestPreflightCli:
    def test_returns_zero_on_the_shipped_config(self):
        assert preflight_mod.main(["--skip-hardware"]) == 0

    def test_returns_non_zero_when_a_required_check_fails(self, tmp_path):
        import yaml
        base = load_config(use_local=False, use_env=False).to_dict()
        base["robot"]["bounding_box_cm"]["length"] = 45.0
        path = tmp_path / "robot.yaml"
        path.write_text(yaml.safe_dump(base), encoding="utf-8")
        assert preflight_mod.main(["--config", str(path), "--skip-hardware"]) == 1

    def test_check_dataclass_ok_property(self):
        assert preflight_mod.Check("x", preflight_mod.PASS).ok is True
        assert preflight_mod.Check("x", preflight_mod.WARN).ok is True
        assert preflight_mod.Check("x", preflight_mod.FAIL).ok is False


class TestConsoleScripts:
    """The entry points declared in pyproject must actually resolve."""

    @pytest.mark.parametrize("module,attr", [
        ("agribot.app.main", "main"),
        ("agribot.app.simulate", "main"),
        ("agribot.app.preflight", "main"),
    ])
    def test_entry_point_is_importable_and_callable(self, module, attr):
        import importlib
        assert callable(getattr(importlib.import_module(module), attr))


class TestToolsAreImportable:
    """Every tool must at least import - a syntax error in a calibration
    script is only discovered when you need it, in the arena."""

    @pytest.mark.parametrize("name", [
        "kalman_sim", "tune_pid", "bench_perception",
        "calibrate_hsv", "calibrate_spray", "train_yolo", "export_tensorrt",
    ])
    def test_tool_imports(self, name):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
        assert path.is_file(), f"missing tool: {path}"
        spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main")

    def test_train_yolo_rejects_a_transposed_class_map(self, tmp_path):
        """A silently transposed map sprays every crop and spares every weed."""
        import importlib.util
        import yaml

        path = Path(__file__).resolve().parents[1] / "tools" / "train_yolo.py"
        spec = importlib.util.spec_from_file_location("_tool_train", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        data = tmp_path / "data.yaml"
        data.write_text(yaml.safe_dump({
            "train": "train", "val": "val", "names": ["weed", "crop"],
        }), encoding="utf-8")

        problems = module.check_dataset(data)
        assert any("class order" in p for p in problems)

    def test_train_yolo_accepts_the_correct_class_order(self, tmp_path):
        import importlib.util
        import yaml

        path = Path(__file__).resolve().parents[1] / "tools" / "train_yolo.py"
        spec = importlib.util.spec_from_file_location("_tool_train2", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        data = tmp_path / "data.yaml"
        data.write_text(yaml.safe_dump({
            "train": "train", "val": "val", "names": ["crop", "weed"],
        }), encoding="utf-8")

        assert module.check_dataset(data) == []

    def test_spray_calibration_fit_recovers_a_known_linear_model(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "tools" / "calibrate_spray.py"
        spec = importlib.util.spec_from_file_location("_tool_spray", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Synthesise samples from a known model and check the fit recovers it.
        centre_pan, slope_pan = 90.0, 0.0725
        centre_tilt, slope_tilt = 75.0, -0.0610
        pairs = []
        for px in (80, 200, 320, 440, 560):
            for py in (120, 240, 360):
                pan = centre_pan + slope_pan * (px - 320)
                tilt = centre_tilt + slope_tilt * (py - 240)
                pairs.append((px, py, pan, tilt))

        fit = module.fit_pairs(pairs, 640, 480)
        assert fit["pan"]["centre_deg"] == pytest.approx(centre_pan, abs=1e-6)
        assert fit["pan"]["deg_per_px"] == pytest.approx(slope_pan, abs=1e-9)
        assert fit["tilt"]["invert"] is True
        assert fit["pan"]["rms_residual_deg"] == pytest.approx(0.0, abs=1e-9)

    def test_spray_calibration_refuses_degenerate_input(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "tools" / "calibrate_spray.py"
        spec = importlib.util.spec_from_file_location("_tool_spray2", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(ValueError, match="at least 3"):
            module.fit_pairs([(1, 1, 90, 75)], 640, 480)
        # All samples at one pixel offset cannot determine a slope.
        with pytest.raises(ValueError, match="spread"):
            module.fit_pairs([(320, y, 90.0, 75.0) for y in (100, 200, 300)],
                             640, 480)
