"""Shared pytest fixtures for the AgriBot suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agribot.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """The checked-in configuration, with local and env layers disabled.

    Tests must assert against the committed defaults, not against whatever a
    developer happens to have in ``robot.local.yaml`` or their environment.
    """
    return load_config(
        PROJECT_ROOT / "config" / "robot.yaml",
        use_local=False,
        use_env=False,
    )


@pytest.fixture
def demo_layout():
    from agribot.sim.arena import ArenaLayout
    return ArenaLayout.default_demo()


@pytest.fixture
def arena(cfg, demo_layout):
    from agribot.sim.arena import SimulatedArena
    return SimulatedArena.from_config(cfg, demo_layout)


@pytest.fixture
def empty_arena(cfg):
    """A long, marker-free row - for navigation tests."""
    from agribot.sim.arena import ArenaLayout, SimulatedArena
    return SimulatedArena.from_config(cfg, ArenaLayout(row_length_m=30.0, markers=[]))


@pytest.fixture
def line_follower(cfg):
    from agribot.vision.line_follow import LineFollower
    return LineFollower.from_config(cfg.navigation.line)


@pytest.fixture
def color_detector(cfg):
    from agribot.vision.color_detector import ColorDetector
    return ColorDetector.from_config(cfg.perception.color)


class FakeClock:
    """Manually advanced monotonic clock, so timing is tested without sleeping."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@pytest.fixture
def clock():
    return FakeClock()
