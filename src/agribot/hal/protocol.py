"""Serial wire protocol between the Jetson and the microcontroller.

Computation is deliberately split across two tiers (Section 2): the Jetson does
perception and decision-making, the MCU on the custom PCB does hard real-time
actuation. This module defines the compact frame format they exchange.

Frame format, NMEA-style::

    $<payload>*<XX>\\n

``XX`` is the XOR of every payload byte, as two uppercase hex digits. The
checksum is not ceremony: motor wiring runs beside the USB lead, and a single
corrupted digit in a motor command is the difference between a correction and a
lurch. Frames that fail the checksum are dropped, and a dropped command is
harmless because commands are re-sent at 50 Hz.

Fields are integers wherever possible - Arduino string-to-float parsing is slow
and lossy, so angles are sent in tenths of a degree and velocities in
millimetres per second, which keeps everything in ``long`` range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..types import EncoderSample, ImuSample, McuTelemetry, RangeSample

__all__ = [
    "checksum",
    "frame",
    "parse_frame",
    "encode_motor",
    "encode_servo",
    "encode_valve",
    "encode_pump",
    "encode_marker",
    "encode_heartbeat",
    "encode_reset_encoders",
    "encode_estop",
    "decode_telemetry",
    "encode_telemetry",
    "ProtocolError",
    "TELEMETRY_FIELDS",
]

START = "$"
CHECKSUM_SEP = "*"
TERMINATOR = "\n"

#: Telemetry payload layout after the leading "T".
TELEMETRY_FIELDS = (
    "seq",          # rolling frame counter
    "ms",           # MCU millis()
    "gyro_mdps",    # yaw rate, milli-degrees per second
    "accel_mmss",   # longitudinal accel, mm/s^2
    "mag_cdeg",     # magnetometer heading, centi-degrees (-1 = no fix)
    "enc_l_mmps",   # left wheel velocity, mm/s
    "enc_r_mmps",   # right wheel velocity, mm/s
    "us_front_mm",  # front ultrasonic, mm (-1 = no echo)
    "us_left_mm",   # front-left ultrasonic, mm (-1 = no echo)
    "flow",         # cumulative flow-sensor ticks
    "batt_mv",      # battery millivolts
    "roll_cdeg",
    "pitch_cdeg",
    "ir",           # IR reflectance array as a bitmask
)


class ProtocolError(ValueError):
    """Raised when a frame is malformed or fails its checksum."""


def checksum(payload: str) -> int:
    """XOR checksum over the payload bytes."""
    value = 0
    for char in payload.encode("ascii", errors="strict"):
        value ^= char
    return value


def frame(payload: str) -> str:
    """Wrap a payload into a complete, checksummed frame."""
    if START in payload or CHECKSUM_SEP in payload or "\n" in payload:
        raise ProtocolError(f"payload contains a delimiter: {payload!r}")
    return f"{START}{payload}{CHECKSUM_SEP}{checksum(payload):02X}{TERMINATOR}"


def parse_frame(line: str) -> str:
    """Validate a received frame and return its payload.

    Raises:
        ProtocolError: on a missing delimiter or a checksum mismatch.
    """
    text = line.strip()
    if not text.startswith(START):
        raise ProtocolError(f"frame does not start with {START!r}: {text!r}")
    if CHECKSUM_SEP not in text:
        raise ProtocolError(f"frame has no checksum separator: {text!r}")

    payload, _, received = text[1:].rpartition(CHECKSUM_SEP)
    if not payload:
        raise ProtocolError(f"empty payload: {text!r}")
    try:
        expected = int(received, 16)
    except ValueError:
        raise ProtocolError(f"checksum not hex: {received!r}") from None
    actual = checksum(payload)
    if actual != expected:
        raise ProtocolError(
            f"checksum mismatch: got {expected:02X}, computed {actual:02X}"
        )
    return payload


# ---------------------------------------------------------------------------
# Host -> MCU commands
# ---------------------------------------------------------------------------


def _clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def encode_motor(left: float, right: float) -> str:
    """Motor command. ``left``/``right`` are normalised to ``[-1, +1]``."""
    l = _clamp_int(left * 1000.0, -1000, 1000)
    r = _clamp_int(right * 1000.0, -1000, 1000)
    return frame(f"M,{l},{r}")


def encode_servo(channel: int, degrees: float) -> str:
    """Servo position in tenths of a degree."""
    return frame(f"S,{int(channel)},{_clamp_int(degrees * 10.0, 0, 1800)}")


def encode_valve(open_: bool) -> str:
    """Solenoid valve state."""
    return frame(f"V,{1 if open_ else 0}")


def encode_pump(on: bool) -> str:
    """Diaphragm pump state."""
    return frame(f"P,{1 if on else 0}")


def encode_marker(degrees: float) -> str:
    """Marker/stamp servo position, tenths of a degree."""
    return frame(f"K,{_clamp_int(degrees * 10.0, 0, 1800)}")


def encode_heartbeat(seq: int = 0) -> str:
    """Liveness ping. The MCU stops the drive if these stop arriving."""
    return frame(f"H,{int(seq) & 0xFFFF}")


def encode_reset_encoders() -> str:
    """Zero the encoder accumulators (used at a row start)."""
    return frame("Z")


def encode_estop() -> str:
    """Immediate stop: motors, pump and valve all off."""
    return frame("X")


# ---------------------------------------------------------------------------
# MCU -> host telemetry
# ---------------------------------------------------------------------------


def encode_telemetry(
    seq: int,
    ms: int,
    gyro_z_rad_s: float,
    accel_x_mps2: float,
    mag_heading_rad: Optional[float],
    enc_l_mps: float,
    enc_r_mps: float,
    us_front_m: float,
    us_left_m: float,
    flow_ticks: int,
    battery_v: float,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    ir_mask: int = 0,
) -> str:
    """Build a telemetry frame. Used by the mock MCU and the firmware tests."""
    mag = -1 if mag_heading_rad is None else _clamp_int(
        math.degrees(mag_heading_rad) * 100.0, -36000, 36000)
    us_f = -1 if not math.isfinite(us_front_m) else _clamp_int(us_front_m * 1000.0, 0, 65535)
    us_l = -1 if not math.isfinite(us_left_m) else _clamp_int(us_left_m * 1000.0, 0, 65535)
    fields = [
        int(seq) & 0xFFFF,
        int(ms),
        _clamp_int(math.degrees(gyro_z_rad_s) * 1000.0, -2000000, 2000000),
        _clamp_int(accel_x_mps2 * 1000.0, -200000, 200000),
        mag,
        _clamp_int(enc_l_mps * 1000.0, -10000, 10000),
        _clamp_int(enc_r_mps * 1000.0, -10000, 10000),
        us_f,
        us_l,
        int(flow_ticks),
        _clamp_int(battery_v * 1000.0, 0, 65535),
        _clamp_int(roll_deg * 100.0, -18000, 18000),
        _clamp_int(pitch_deg * 100.0, -18000, 18000),
        int(ir_mask) & 0xFF,
    ]
    return frame("T," + ",".join(str(f) for f in fields))


def decode_telemetry(payload: str, t: float) -> McuTelemetry:
    """Parse a validated telemetry payload into a :class:`McuTelemetry`.

    Raises:
        ProtocolError: if the payload is not a telemetry frame or is truncated.
    """
    parts = payload.split(",")
    if not parts or parts[0] != "T":
        raise ProtocolError(f"not a telemetry payload: {payload!r}")
    values = parts[1:]
    if len(values) < len(TELEMETRY_FIELDS):
        raise ProtocolError(
            f"telemetry has {len(values)} fields, expected {len(TELEMETRY_FIELDS)}"
        )

    try:
        nums = [int(v) for v in values[: len(TELEMETRY_FIELDS)]]
    except ValueError as exc:
        raise ProtocolError(f"non-integer telemetry field: {exc}") from None

    (seq, ms, gyro_mdps, accel_mmss, mag_cdeg, enc_l, enc_r,
     us_f, us_l, flow, batt_mv, roll_cd, pitch_cd, ir_mask) = nums

    mag_heading = None if mag_cdeg == -1 else math.radians(mag_cdeg / 100.0)
    front = math.inf if us_f < 0 else us_f / 1000.0
    left = math.inf if us_l < 0 else us_l / 1000.0

    return McuTelemetry(
        t=t,
        seq=seq,
        imu=ImuSample(
            t=t,
            gyro_z=math.radians(gyro_mdps / 1000.0),
            accel_x=accel_mmss / 1000.0,
            mag_heading_rad=mag_heading,
            roll_deg=roll_cd / 100.0,
            pitch_deg=pitch_cd / 100.0,
        ),
        encoders=EncoderSample(t=t, left_mps=enc_l / 1000.0, right_mps=enc_r / 1000.0),
        ranges=RangeSample(t=t, front_m=front, front_left_m=left),
        flow_ticks=flow,
        battery_v=batt_mv / 1000.0,
        ir_array=tuple((ir_mask >> i) & 1 for i in range(8)),
        ok=True,
    )
