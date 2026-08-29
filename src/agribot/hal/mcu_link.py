"""Serial transport to the microcontroller.

Owns the port, the receive buffer and the liveness watchdog. Two properties
matter more than throughput:

* **A stale telemetry frame must never look fresh.** ``telemetry`` returns the
  last decoded frame together with its age, and ``link_ok`` goes false once no
  valid frame has arrived within the configured timeout, which the mission
  state machine escalates to ESTOP.
* **Failure to write must not raise into the control loop.** A USB re-enumeration
  mid-run would otherwise abort the mission with a traceback rather than
  stopping the robot cleanly, so writes are caught and reported through
  ``link_ok``.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional, Tuple

from ..types import McuTelemetry
from ..utils.logging_setup import get_logger
from . import protocol

__all__ = ["McuLink", "SerialTransport", "LoopbackTransport"]

log = get_logger("hal.mcu")


class SerialTransport:
    """Thin wrapper over ``pyserial`` with a line-oriented read."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.05):
        self.port = port
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self._serial = None
        self._buffer = b""

    def open(self) -> bool:
        try:
            import serial  # noqa: WPS433 - optional dependency at import time
        except Exception as exc:  # pragma: no cover
            log.error("pyserial not available: %s", exc)
            return False
        try:
            self._serial = serial.Serial(
                self.port, self.baud, timeout=self.timeout_s, write_timeout=0.1
            )
            # Opening the port resets an Arduino; give the bootloader time to
            # finish before the first command, or it is silently swallowed.
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            log.info("MCU link open on %s @ %d", self.port, self.baud)
            return True
        except Exception as exc:
            log.error("cannot open %s: %s", self.port, exc)
            self._serial = None
            return False

    @property
    def is_open(self) -> bool:
        return self._serial is not None and getattr(self._serial, "is_open", False)

    def write(self, data: str) -> bool:
        if self._serial is None:
            return False
        try:
            self._serial.write(data.encode("ascii"))
            return True
        except Exception as exc:
            log.error("MCU write failed: %s", exc)
            return False

    def read_lines(self) -> List[str]:
        """Return every complete line currently buffered."""
        if self._serial is None:
            return []
        try:
            waiting = self._serial.in_waiting
            if waiting:
                self._buffer += self._serial.read(waiting)
        except Exception as exc:
            log.error("MCU read failed: %s", exc)
            return []

        lines: List[str] = []
        while b"\n" in self._buffer:
            raw, _, self._buffer = self._buffer.partition(b"\n")
            try:
                lines.append(raw.decode("ascii", errors="strict").strip())
            except UnicodeDecodeError:
                # Line noise on the wire; the checksum would reject it anyway.
                continue
        # Bound the buffer so a wedged MCU emitting no newline cannot grow it
        # without limit over a long run.
        if len(self._buffer) > 4096:
            self._buffer = self._buffer[-1024:]
        return lines

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # pragma: no cover
                pass
            self._serial = None


class LoopbackTransport:
    """In-memory transport used by the tests and the simulator.

    Commands written by the host are captured in ``sent``; frames pushed with
    :meth:`push` are returned by :meth:`read_lines`.
    """

    def __init__(self):
        self.sent: List[str] = []
        self._incoming: List[str] = []
        self._open = False
        self.fail_writes = False

    def open(self) -> bool:
        self._open = True
        return True

    @property
    def is_open(self) -> bool:
        return self._open

    def write(self, data: str) -> bool:
        if self.fail_writes:
            return False
        self.sent.append(data)
        return True

    def push(self, line: str) -> None:
        self._incoming.append(line)

    def read_lines(self) -> List[str]:
        out, self._incoming = self._incoming, []
        return out

    def close(self) -> None:
        self._open = False

    # -- test helpers -------------------------------------------------------
    def sent_payloads(self) -> List[str]:
        """Validated payloads of everything the host wrote."""
        return [protocol.parse_frame(s) for s in self.sent]

    def last_payload(self, kind: Optional[str] = None) -> Optional[str]:
        for raw in reversed(self.sent):
            payload = protocol.parse_frame(raw)
            if kind is None or payload.startswith(kind):
                return payload
        return None


class McuLink:
    """High-level MCU interface: send commands, receive telemetry, watch liveness."""

    def __init__(
        self,
        transport,
        timeout_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.transport = transport
        self.timeout_s = float(timeout_s)
        self._clock = clock

        self._telemetry: Optional[McuTelemetry] = None
        self._last_rx_t: Optional[float] = None
        self._heartbeat_seq = 0
        self._write_failed = False

        self.frames_received = 0
        self.frames_dropped = 0
        self.commands_sent = 0

    @classmethod
    def from_config(cls, cfg) -> "McuLink":
        """Build a serial link from the ``mcu`` config section."""
        transport = SerialTransport(
            cfg.get("port", "/dev/ttyACM0"),
            cfg.get("baud", 115200),
            cfg.get("timeout_s", 0.05),
        )
        return cls(transport, timeout_s=cfg.get("timeout_s", 0.5) or 0.5)

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> bool:
        ok = self.transport.open()
        if ok:
            self._write_failed = False
        return ok

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.transport.close()

    # -- receive ------------------------------------------------------------
    def poll(self) -> Optional[McuTelemetry]:
        """Drain the port and decode the newest valid telemetry frame.

        Older frames in the buffer are discarded rather than processed: acting
        on a queued frame from 200 ms ago is worse than acting on nothing.
        """
        newest: Optional[McuTelemetry] = None
        for line in self.transport.read_lines():
            if not line:
                continue
            try:
                payload = protocol.parse_frame(line)
                telemetry = protocol.decode_telemetry(payload, self._clock())
            except protocol.ProtocolError as exc:
                self.frames_dropped += 1
                log.debug("dropped frame: %s", exc)
                continue
            newest = telemetry
            self.frames_received += 1

        if newest is not None:
            self._telemetry = newest
            self._last_rx_t = newest.t
        return newest

    @property
    def telemetry(self) -> Optional[McuTelemetry]:
        return self._telemetry

    @property
    def age_s(self) -> float:
        """Seconds since the last valid frame; ``inf`` if none has arrived."""
        if self._last_rx_t is None:
            return float("inf")
        return self._clock() - self._last_rx_t

    @property
    def link_ok(self) -> bool:
        """False when the link has gone quiet or a write has failed."""
        return (not self._write_failed) and self.age_s <= self.timeout_s

    # -- transmit -----------------------------------------------------------
    def _send(self, data: str) -> bool:
        ok = self.transport.write(data)
        if ok:
            self.commands_sent += 1
        else:
            self._write_failed = True
        return ok

    def send_drive(self, left: float, right: float) -> bool:
        return self._send(protocol.encode_motor(left, right))

    def send_aim(self, pan_deg: float, tilt_deg: float,
                 pan_channel: int = 0, tilt_channel: int = 1) -> bool:
        ok = self._send(protocol.encode_servo(pan_channel, pan_deg))
        return self._send(protocol.encode_servo(tilt_channel, tilt_deg)) and ok

    def send_valve(self, open_: bool) -> bool:
        return self._send(protocol.encode_valve(open_))

    def send_pump(self, on: bool) -> bool:
        return self._send(protocol.encode_pump(on))

    def send_marker(self, degrees: float) -> bool:
        return self._send(protocol.encode_marker(degrees))

    def send_heartbeat(self) -> bool:
        self._heartbeat_seq = (self._heartbeat_seq + 1) & 0xFFFF
        return self._send(protocol.encode_heartbeat(self._heartbeat_seq))

    def reset_encoders(self) -> bool:
        return self._send(protocol.encode_reset_encoders())

    def stop(self) -> bool:
        """Emergency stop: motors, pump and valve off in one frame."""
        return self._send(protocol.encode_estop())

    def flow_ticks(self) -> int:
        return self._telemetry.flow_ticks if self._telemetry else 0

    def stats(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "commands_sent": self.commands_sent,
            "link_ok": self.link_ok,
            "age_ms": round(self.age_s * 1e3, 1) if self._last_rx_t else None,
        }
