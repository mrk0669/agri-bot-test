"""Run logging: JSONL events and a CSV time series.

Two streams, because they answer different questions.

* **Events (JSONL).** Discrete, irregular things: state transitions, spray
  bursts, crop vetoes, faults. One self-describing JSON object per line, so a
  run can be grepped and replayed without a schema.
* **Time series (CSV).** Regular samples of the continuous signals: line error,
  PID terms, fused heading and distance, wheel commands. This is the file the
  Kalman analysis script re-reads, which is what makes the same analysis run on
  logged hardware data as on the simulator.

Both are flushed on write. A run that ends because the battery died is exactly
the run whose log matters most, and a buffered writer loses it.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

import numpy as np

from ..types import SprayEvent
from ..utils.logging_setup import get_logger

__all__ = ["RunLogger", "json_safe"]

log = get_logger("telemetry.logger")

#: Columns of the time-series CSV. The first ten are the schema the fusion
#: analysis reads back, so their names and order must not drift.
TIMESERIES_COLUMNS = [
    "t", "gyro_z", "accel_x", "mag_heading_rad", "mag_valid",
    "encoder_mps", "enc_valid", "true_heading_rad", "true_velocity_mps",
    "true_distance_m",
    # Runtime-only columns, ignored by the fusion loader.
    "state", "line_found", "line_error", "line_source", "pid_out",
    "drive_left", "drive_right",
    "fused_heading_deg", "fused_distance_m", "fused_velocity_mps",
    "encoder_rejected", "obstacle_m", "targets", "battery_v",
]


def json_safe(value: Any) -> Any:
    """Convert dataclasses, enums and numpy scalars into JSON-serialisable data."""
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return json_safe(value.to_dict())
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    # Mapping, not dict: Config is a Mapping and is not a dict subclass, so a
    # dict-only check silently falls through and json.dumps then raises on it.
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        # JSON has no NaN/Infinity; emit null rather than invalid JSON that a
        # strict parser will reject when the log is analysed later.
        return None
    return value


class RunLogger:
    """Writes one run's event and time-series logs into a timestamped directory."""

    def __init__(
        self,
        log_dir: Path,
        run_name: Optional[str] = None,
        jsonl_events: bool = True,
        csv_timeseries: bool = True,
    ):
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = Path(log_dir) / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self._events_fh: Optional[IO[str]] = None
        self._csv_fh: Optional[IO[str]] = None
        self._csv_writer = None
        self._t0 = time.monotonic()
        self.event_count = 0
        self.sample_count = 0

        if jsonl_events:
            self._events_fh = (self.dir / "events.jsonl").open("w", encoding="utf-8")
        if csv_timeseries:
            self._csv_fh = (self.dir / "timeseries.csv").open(
                "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_fh, fieldnames=TIMESERIES_COLUMNS, extrasaction="ignore")
            self._csv_writer.writeheader()

        log.info("logging run to %s", self.dir)

    @classmethod
    def from_config(cls, cfg, run_name: Optional[str] = None) -> "RunLogger":
        """Build from the ``telemetry`` config section."""
        return cls(
            log_dir=Path(cfg.get("log_dir", "data/logs")),
            run_name=run_name,
            jsonl_events=cfg.get("jsonl_events", True),
            csv_timeseries=cfg.get("csv_timeseries", True),
        )

    # -- events -------------------------------------------------------------
    def event(self, kind: str, **fields: Any) -> None:
        """Append one event record."""
        if self._events_fh is None:
            return
        record = {
            "t": round(time.monotonic() - self._t0, 4),
            "wall": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
        }
        record.update({k: json_safe(v) for k, v in fields.items()})
        self._events_fh.write(json.dumps(record) + "\n")
        self._events_fh.flush()
        self.event_count += 1

    def spray_event(self, event: SprayEvent) -> None:
        self.event("spray", **event.to_dict())

    def transition(self, from_state: str, to_state: str, reason: str) -> None:
        self.event("transition", **{"from": from_state, "to": to_state, "reason": reason})

    def veto(self, reason: str, detection: Dict[str, Any]) -> None:
        """Record a crop-protected suppression - the evidence for Novelty 4."""
        self.event("crop_veto", reason=reason, detection=detection)

    def fault(self, message: str, **fields: Any) -> None:
        self.event("fault", message=message, **fields)

    # -- time series --------------------------------------------------------
    def sample(self, **fields: Any) -> None:
        """Append one time-series row. Unknown keys are ignored."""
        if self._csv_writer is None:
            return
        row = {k: json_safe(v) for k, v in fields.items()}
        row.setdefault("t", round(time.monotonic() - self._t0, 4))
        self._csv_writer.writerow(row)
        self._csv_fh.flush()
        self.sample_count += 1

    # -- lifecycle ----------------------------------------------------------
    def write_summary(self, summary: Dict[str, Any]) -> Path:
        """Write ``summary.json`` - the file a judge or a teammate reads first."""
        path = self.dir / "summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(json_safe(summary), fh, indent=2)
        log.info("wrote %s", path)
        return path

    def close(self) -> None:
        for handle in (self._events_fh, self._csv_fh):
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:  # pragma: no cover
                    pass
        self._events_fh = None
        self._csv_fh = None
        self._csv_writer = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
