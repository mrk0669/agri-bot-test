"""Mission states and the data crossing the state-machine boundary (Section 5.4).

Path planning for an in-row agricultural task is fundamentally different from
planning in a corridor network. The route is dictated by the crop rows, so the
planner is not searching a graph for a shortest path; it manages a
deterministic sequence of behaviours along a known route, deciding when to
interrupt travel to intervene on a weed, when to pause for an obstacle, and
when a row has ended.

Expressing that as a finite state machine gives the judges a clear and
repeatable demonstration and gives the team clean telemetry for tuning. Every
transition is deterministic and logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..types import DriveCommand, LineObservation, SprayEvent, Track

__all__ = ["MissionState", "MissionInputs", "MissionOutput", "TERMINAL_STATES"]


class MissionState(str, Enum):
    """The complete mission state set."""

    INIT = "INIT"                       # power-on checks, waiting for the line
    FOLLOW_LINE = "FOLLOW_LINE"         # vision-guided travel along the row
    STOP_AND_AIM = "STOP_AND_AIM"       # halted, pan/tilt head driving to target
    SPRAY = "SPRAY"                     # metered burst (or mark) in progress
    LOG_EVENT = "LOG_EVENT"             # event recorded, nudge clear of the target
    PAUSE = "PAUSE"                     # obstacle within the ultrasonic stop band
    RECOVER = "RECOVER"                 # line lost beyond the grace period
    TURN = "TURN"                       # row-end manoeuvre
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ESTOP = "ESTOP"                     # unrecoverable fault - drive inhibited


#: States from which the machine will not resume driving on its own.
TERMINAL_STATES = frozenset({MissionState.MISSION_COMPLETE, MissionState.ESTOP})


@dataclass
class MissionInputs:
    """Everything the state machine needs to decide, for one tick."""

    t: float
    line: LineObservation
    targets: Sequence[Track] = field(default_factory=list)
    distance_m: float = 0.0
    heading_deg: float = 0.0
    nearest_obstacle_m: float = float("inf")
    dt: float = 0.0
    spray_busy: bool = False
    spray_ready: bool = True
    mcu_ok: bool = True
    tilt_ok: bool = True
    battery_v: float = 12.0

    @property
    def has_target(self) -> bool:
        return len(self.targets) > 0


@dataclass
class MissionOutput:
    """What the state machine decided this tick."""

    state: MissionState
    drive: DriveCommand
    # Set on the tick a burst should be started, consumed by the runtime.
    engage_target: Optional[Track] = None
    # Set on the tick an event completed, for the telemetry writer.
    completed_event: Optional[SprayEvent] = None
    reason: str = ""
    transitioned: bool = False
    rows_done: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "drive": self.drive.to_dict(),
            "reason": self.reason,
            "transitioned": self.transitioned,
            "rows_done": self.rows_done,
            "engage_track": self.engage_target.track_id if self.engage_target else None,
        }
