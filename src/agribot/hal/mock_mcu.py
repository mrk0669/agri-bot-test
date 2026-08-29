"""Simulated microcontroller - a transport that behaves like the real PCB.

Drop-in replacement for :class:`SerialTransport`. It decodes the same command
frames the firmware decodes, integrates a differential-drive model, and emits
the same telemetry frames the firmware emits, complete with sensor bias and
noise.

That makes software-in-the-loop testing possible: the entire runtime - fusion,
state machine, spray sequencing, telemetry - runs unmodified against this, and
a test can assert on what the robot *did*, not merely on what each unit
returned in isolation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.geometry import clamp, wrap_pi
from . import protocol

__all__ = ["MockMcu", "MockMcuConfig"]


@dataclass
class MockMcuConfig:
    """Physical and sensor parameters of the simulated robot."""

    wheel_base_m: float = 0.24
    max_speed_mps: float = 0.35
    # First-order motor response: how quickly actual speed reaches commanded.
    motor_tau_s: float = 0.12

    gyro_bias_deg_s: float = 0.78
    gyro_noise_std: float = 0.0035
    accel_bias_mps2: float = 0.043
    accel_noise_std: float = 0.06
    mag_noise_std_deg: float = 1.5
    encoder_noise_std: float = 0.006

    battery_v: float = 12.4
    # Flow sensor: ticks per millilitre, and pump delivery rate.
    flow_ticks_per_ml: float = 450.0
    pump_ml_per_s: float = 8.0

    # Wheel slip: fraction of commanded speed the encoders over-read while
    # ``slip_active`` is set, used to exercise the innovation gate end to end.
    slip_excess_mps: float = 0.25

    seed: int = 7


class MockMcu:
    """Transport-compatible simulated MCU.

    Call :meth:`step` to advance simulated time; it produces telemetry frames
    at the configured rate which :meth:`read_lines` then returns.
    """

    def __init__(
        self,
        config: Optional[MockMcuConfig] = None,
        telemetry_hz: float = 100.0,
    ):
        self.cfg = config or MockMcuConfig()
        self.telemetry_period = 1.0 / float(telemetry_hz)
        self._rng = random.Random(self.cfg.seed)

        # -- true state -----------------------------------------------------
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading_rad = 0.0
        self.distance_m = 0.0
        self.left_mps = 0.0
        self.right_mps = 0.0
        self._prev_linear_mps = 0.0

        # -- commanded ------------------------------------------------------
        self.cmd_left = 0.0
        self.cmd_right = 0.0
        self.servos: Dict[int, float] = {0: 90.0, 1: 75.0, 2: 30.0}
        self.valve_open = False
        self.pump_on = False
        self.estopped = False

        # -- simulated sensors ---------------------------------------------
        self.flow_ticks = 0
        self.dispensed_ml = 0.0
        self.obstacle_front_m = math.inf
        self.obstacle_left_m = math.inf
        self.slip_active = False
        self.ir_mask = 0

        # -- link -----------------------------------------------------------
        self._outgoing: List[str] = []
        self._open = False
        self._seq = 0
        self._next_telemetry_t = 0.0
        self._last_heartbeat_t = -math.inf
        self.heartbeat_timeout_s = 0.5

        self.commands_received: List[str] = []
        self.valve_open_time_s = 0.0

    # -- transport interface ------------------------------------------------
    def open(self) -> bool:
        self._open = True
        return True

    @property
    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False

    def write(self, data: str) -> bool:
        """Decode and apply a command frame, exactly as the firmware would."""
        if not self._open:
            return False
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = protocol.parse_frame(line)
            except protocol.ProtocolError:
                # A corrupt frame is dropped silently, as the firmware does.
                continue
            self.commands_received.append(payload)
            self._apply(payload)
        return True

    def read_lines(self) -> List[str]:
        out, self._outgoing = self._outgoing, []
        return out

    # -- command handling ---------------------------------------------------
    def _apply(self, payload: str) -> None:
        parts = payload.split(",")
        kind = parts[0]

        if kind == "M" and len(parts) >= 3:
            if self.estopped:
                return
            self.cmd_left = clamp(int(parts[1]) / 1000.0, -1.0, 1.0)
            self.cmd_right = clamp(int(parts[2]) / 1000.0, -1.0, 1.0)
        elif kind == "S" and len(parts) >= 3:
            self.servos[int(parts[1])] = int(parts[2]) / 10.0
        elif kind == "K" and len(parts) >= 2:
            self.servos[2] = int(parts[1]) / 10.0
        elif kind == "V" and len(parts) >= 2:
            self.valve_open = parts[1] == "1"
        elif kind == "P" and len(parts) >= 2:
            self.pump_on = parts[1] == "1"
        elif kind == "H":
            self._last_heartbeat_t = self.t
            self.estopped = False
        elif kind == "Z":
            self.distance_m = 0.0
        elif kind == "X":
            self._estop()

    def _estop(self) -> None:
        self.estopped = True
        self.cmd_left = self.cmd_right = 0.0
        self.left_mps = self.right_mps = 0.0
        self.valve_open = False
        self.pump_on = False

    # -- simulation ---------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the simulated robot and emit telemetry when due."""
        if dt <= 0:
            return
        self.t += dt

        # The firmware stops the drive if heartbeats stop arriving. Reproducing
        # that here is what lets the watchdog be tested rather than assumed.
        if (self._last_heartbeat_t > -math.inf
                and (self.t - self._last_heartbeat_t) > self.heartbeat_timeout_s):
            self.cmd_left = self.cmd_right = 0.0

        target_left = self.cmd_left * self.cfg.max_speed_mps
        target_right = self.cmd_right * self.cfg.max_speed_mps
        alpha = clamp(dt / max(self.cfg.motor_tau_s, 1e-6), 0.0, 1.0)
        self.left_mps += alpha * (target_left - self.left_mps)
        self.right_mps += alpha * (target_right - self.right_mps)

        linear = 0.5 * (self.left_mps + self.right_mps)
        angular = (self.right_mps - self.left_mps) / self.cfg.wheel_base_m
        # True longitudinal acceleration. The accelerometer must report the
        # robot actually accelerating, not just its bias - otherwise the
        # distance filter cannot predict through a speed change and gates out
        # every encoder sample during acceleration.
        accel_true = (linear - self._prev_linear_mps) / dt
        self._prev_linear_mps = linear

        self.heading_rad = wrap_pi(self.heading_rad + angular * dt)
        self.x += linear * math.cos(self.heading_rad) * dt
        self.y += linear * math.sin(self.heading_rad) * dt
        self.distance_m += linear * dt

        if self.pump_on and self.valve_open:
            self.valve_open_time_s += dt
            delivered = self.cfg.pump_ml_per_s * dt
            self.dispensed_ml += delivered
            self.flow_ticks += int(delivered * self.cfg.flow_ticks_per_ml)

        while self.t >= self._next_telemetry_t:
            self._emit_telemetry(linear, angular, accel_true)
            self._next_telemetry_t += self.telemetry_period

    def _emit_telemetry(self, linear: float, angular: float,
                        accel_true: float = 0.0) -> None:
        g = self._rng.gauss
        self._seq = (self._seq + 1) & 0xFFFF

        gyro = (angular
                + math.radians(self.cfg.gyro_bias_deg_s)
                + g(0.0, self.cfg.gyro_noise_std))
        accel = accel_true + self.cfg.accel_bias_mps2 + g(0.0, self.cfg.accel_noise_std)
        mag = self.heading_rad + math.radians(g(0.0, self.cfg.mag_noise_std_deg))

        slip = self.cfg.slip_excess_mps if self.slip_active else 0.0
        enc_l = self.left_mps + slip + g(0.0, self.cfg.encoder_noise_std)
        enc_r = self.right_mps + slip + g(0.0, self.cfg.encoder_noise_std)

        self._outgoing.append(protocol.encode_telemetry(
            seq=self._seq,
            ms=int(self.t * 1000.0),
            gyro_z_rad_s=gyro,
            accel_x_mps2=accel,
            mag_heading_rad=mag,
            enc_l_mps=enc_l,
            enc_r_mps=enc_r,
            us_front_m=self.obstacle_front_m,
            us_left_m=self.obstacle_left_m,
            flow_ticks=self.flow_ticks,
            battery_v=self.cfg.battery_v,
            roll_deg=0.0,
            pitch_deg=0.0,
            ir_mask=self.ir_mask,
        ))

    # -- scenario helpers ---------------------------------------------------
    def set_obstacle(self, front_m: float = math.inf, left_m: float = math.inf) -> None:
        self.obstacle_front_m = front_m
        self.obstacle_left_m = left_m

    def set_slip(self, active: bool) -> None:
        """Enable/disable wheel spin, so the encoder over-reads."""
        self.slip_active = active

    @property
    def pan_deg(self) -> float:
        return self.servos.get(0, 90.0)

    @property
    def tilt_deg(self) -> float:
        return self.servos.get(1, 75.0)

    @property
    def marker_deg(self) -> float:
        return self.servos.get(2, 30.0)

    def state(self) -> dict:
        return {
            "t": round(self.t, 3),
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "heading_deg": round(math.degrees(self.heading_rad), 2),
            "distance_m": round(self.distance_m, 4),
            "left_mps": round(self.left_mps, 4),
            "right_mps": round(self.right_mps, 4),
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "valve": self.valve_open,
            "pump": self.pump_on,
            "dispensed_ml": round(self.dispensed_ml, 3),
            "estopped": self.estopped,
        }
