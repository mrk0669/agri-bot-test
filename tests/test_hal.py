"""Wire protocol, MCU link and the simulated microcontroller.

The protocol tests matter more than they look: the Python side and the Arduino
firmware implement the same format independently, so these lock the contract
that ``firmware/agribot_mcu/agribot_mcu.ino`` must satisfy.
"""

from __future__ import annotations

import math

import pytest

from agribot.hal import protocol
from agribot.hal.mcu_link import LoopbackTransport, McuLink
from agribot.hal.mock_mcu import MockMcu, MockMcuConfig


class TestChecksumAndFraming:
    def test_frame_round_trip(self):
        assert protocol.parse_frame(protocol.frame("M,500,-250")) == "M,500,-250"

    def test_frame_shape(self):
        text = protocol.frame("H,1")
        assert text.startswith("$") and text.endswith("\n") and "*" in text

    def test_checksum_is_xor_of_payload_bytes(self):
        expected = 0
        for char in b"M,500,-250":
            expected ^= char
        assert protocol.checksum("M,500,-250") == expected

    def test_corrupted_payload_is_rejected(self):
        """Motor wiring runs beside the USB lead; a flipped digit must not
        become a lurch."""
        good = protocol.frame("M,500,-250")
        bad = good.replace("M,500", "M,900")
        with pytest.raises(protocol.ProtocolError, match="checksum"):
            protocol.parse_frame(bad)

    def test_corrupted_checksum_is_rejected(self):
        good = protocol.frame("M,500,-250")
        payload, _, tail = good.rstrip("\n").rpartition("*")
        flipped = f"{payload}*{'00' if tail != '00' else '01'}\n"
        with pytest.raises(protocol.ProtocolError):
            protocol.parse_frame(flipped)

    @pytest.mark.parametrize("line", [
        "M,500,-250*62\n",      # no leading $
        "$M,500,-250\n",        # no checksum separator
        "$*62\n",               # empty payload
        "$M,1*ZZ\n",            # non-hex checksum
        "",                     # nothing at all
    ])
    def test_malformed_frames_are_rejected(self, line):
        with pytest.raises(protocol.ProtocolError):
            protocol.parse_frame(line)

    def test_payload_containing_a_delimiter_is_refused(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.frame("M,1*2")

    def test_whitespace_is_tolerated(self):
        assert protocol.parse_frame("  " + protocol.frame("Z").strip() + " \r\n") == "Z"


class TestCommandEncoding:
    def test_motor_scaling_and_clamping(self):
        assert protocol.parse_frame(protocol.encode_motor(0.5, -0.25)) == "M,500,-250"
        assert protocol.parse_frame(protocol.encode_motor(9.0, -9.0)) == "M,1000,-1000"

    def test_servo_in_tenths_of_a_degree(self):
        assert protocol.parse_frame(protocol.encode_servo(1, 88.4)) == "S,1,884"

    def test_servo_is_clamped_to_the_physical_range(self):
        assert protocol.parse_frame(protocol.encode_servo(0, 999.0)) == "S,0,1800"
        assert protocol.parse_frame(protocol.encode_servo(0, -50.0)) == "S,0,0"

    def test_valve_pump_marker_heartbeat_estop(self):
        assert protocol.parse_frame(protocol.encode_valve(True)) == "V,1"
        assert protocol.parse_frame(protocol.encode_pump(False)) == "P,0"
        assert protocol.parse_frame(protocol.encode_marker(95.0)) == "K,950"
        assert protocol.parse_frame(protocol.encode_heartbeat(7)) == "H,7"
        assert protocol.parse_frame(protocol.encode_estop()) == "X"
        assert protocol.parse_frame(protocol.encode_reset_encoders()) == "Z"

    def test_heartbeat_sequence_wraps_at_16_bits(self):
        assert protocol.parse_frame(protocol.encode_heartbeat(70000)) == "H,4464"


class TestTelemetryCodec:
    def test_round_trip_preserves_every_field(self):
        frame = protocol.encode_telemetry(
            seq=7, ms=1234, gyro_z_rad_s=0.0136, accel_x_mps2=0.05,
            mag_heading_rad=0.1, enc_l_mps=0.18, enc_r_mps=0.17,
            us_front_m=0.9, us_left_m=math.inf, flow_ticks=900,
            battery_v=12.4, roll_deg=1.0, pitch_deg=-2.0, ir_mask=0b1010,
        )
        telemetry = protocol.decode_telemetry(protocol.parse_frame(frame), t=1.0)

        assert telemetry.seq == 7
        assert telemetry.imu.gyro_z == pytest.approx(0.0136, abs=2e-5)
        assert telemetry.imu.accel_x == pytest.approx(0.05, abs=1e-3)
        assert telemetry.imu.mag_heading_rad == pytest.approx(0.1, abs=2e-4)
        assert telemetry.imu.roll_deg == pytest.approx(1.0, abs=0.01)
        assert telemetry.encoders.left_mps == pytest.approx(0.18, abs=1e-3)
        assert telemetry.encoders.linear_mps == pytest.approx(0.175, abs=1e-3)
        assert telemetry.ranges.front_m == pytest.approx(0.9, abs=1e-3)
        assert telemetry.flow_ticks == 900
        assert telemetry.battery_v == pytest.approx(12.4, abs=1e-3)
        assert telemetry.ir_array == (0, 1, 0, 1, 0, 0, 0, 0)

    def test_no_echo_decodes_to_infinity(self):
        """A missing echo means "clear", not "an obstacle at zero metres"."""
        frame = protocol.encode_telemetry(
            1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf, math.inf, 0, 12.0)
        telemetry = protocol.decode_telemetry(protocol.parse_frame(frame), t=0.0)
        assert math.isinf(telemetry.ranges.front_m)
        assert math.isinf(telemetry.ranges.min_m)

    def test_absent_magnetometer_fix_decodes_to_none(self):
        """Better to skip the correction step than to fuse a heading the
        firmware could not actually measure."""
        frame = protocol.encode_telemetry(
            1, 0, 0.0, 0.0, None, 0.0, 0.0, 1.0, 1.0, 0, 12.0)
        telemetry = protocol.decode_telemetry(protocol.parse_frame(frame), t=0.0)
        assert telemetry.imu.mag_heading_rad is None

    def test_truncated_telemetry_is_rejected(self):
        with pytest.raises(protocol.ProtocolError, match="fields"):
            protocol.decode_telemetry("T,1,2,3", t=0.0)

    def test_non_telemetry_payload_is_rejected(self):
        with pytest.raises(protocol.ProtocolError, match="not a telemetry"):
            protocol.decode_telemetry("M,0,0", t=0.0)

    def test_non_integer_field_is_rejected(self):
        payload = "T," + ",".join(["1"] * 13 + ["oops"])
        with pytest.raises(protocol.ProtocolError):
            protocol.decode_telemetry(payload, t=0.0)


class TestMcuLink:
    def test_commands_are_framed_on_the_wire(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        link.send_drive(0.4, -0.2)
        assert transport.last_payload("M") == "M,400,-200"

    def test_aim_sends_both_servos(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        link.send_aim(72.0, 88.0)
        payloads = transport.sent_payloads()
        assert "S,0,720" in payloads and "S,1,880" in payloads

    def test_telemetry_is_decoded(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        transport.push(protocol.encode_telemetry(
            1, 0, 0.0, 0.0, 0.0, 0.18, 0.18, 1.0, 1.0, 0, 12.4).strip())
        telemetry = link.poll()
        assert telemetry is not None
        assert link.frames_received == 1

    def test_only_the_newest_frame_is_acted_on(self, clock):
        """Acting on a queued frame from 200 ms ago is worse than acting on none."""
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        for seq in (1, 2, 3):
            transport.push(protocol.encode_telemetry(
                seq, 0, 0.0, 0.0, 0.0, 0.18, 0.18, 1.0, 1.0, 0, 12.4).strip())
        assert link.poll().seq == 3
        assert link.frames_received == 3

    def test_corrupt_frames_are_counted_and_dropped(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        transport.push("$T,garbage*00")
        assert link.poll() is None
        assert link.frames_dropped == 1

    def test_link_is_not_ok_before_any_telemetry(self, clock):
        link = McuLink(LoopbackTransport(), timeout_s=0.5, clock=clock)
        link.open()
        assert link.link_ok is False
        assert math.isinf(link.age_s)

    def test_link_goes_stale_after_the_timeout(self, clock):
        """A stale frame must never look fresh - the FSM escalates this to ESTOP."""
        transport = LoopbackTransport()
        link = McuLink(transport, timeout_s=0.5, clock=clock)
        link.open()
        transport.push(protocol.encode_telemetry(
            1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 12.4).strip())
        link.poll()
        assert link.link_ok is True
        clock.advance(0.6)
        assert link.link_ok is False

    def test_a_failed_write_marks_the_link_down(self, clock):
        """A USB re-enumeration must stop the robot, not raise into the loop."""
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        transport.push(protocol.encode_telemetry(
            1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 12.4).strip())
        link.poll()
        assert link.link_ok is True
        transport.fail_writes = True
        assert link.send_drive(0.1, 0.1) is False
        assert link.link_ok is False

    def test_heartbeat_sequence_increments(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        link.send_heartbeat()
        link.send_heartbeat()
        payloads = [p for p in transport.sent_payloads() if p.startswith("H")]
        assert payloads == ["H,1", "H,2"]

    def test_close_stops_the_robot_first(self, clock):
        transport = LoopbackTransport()
        link = McuLink(transport, clock=clock)
        link.open()
        link.close()
        assert transport.sent_payloads()[-1] == "X"


class TestMockMcu:
    def test_drives_forward_on_a_symmetric_command(self):
        mcu = MockMcu(MockMcuConfig())
        mcu.open()
        mcu.write(protocol.encode_motor(0.6, 0.6))
        for _ in range(200):
            mcu.step(0.01)
        assert mcu.x > 0.3
        assert abs(mcu.y) < 1e-6
        assert abs(mcu.heading_rad) < 1e-6

    def test_turns_on_a_differential_command(self):
        mcu = MockMcu(MockMcuConfig())
        mcu.open()
        mcu.write(protocol.encode_motor(-0.4, 0.4))
        for _ in range(200):
            mcu.step(0.01)
        assert mcu.heading_rad > 0.5           # positive differential turns left

    def test_emits_decodable_telemetry(self):
        mcu = MockMcu(MockMcuConfig(), telemetry_hz=100)
        mcu.open()
        mcu.step(0.05)
        lines = mcu.read_lines()
        assert lines
        telemetry = protocol.decode_telemetry(protocol.parse_frame(lines[0]), 0.0)
        assert telemetry.battery_v == pytest.approx(12.4, abs=0.01)

    def test_accelerometer_reports_real_acceleration(self):
        """Reporting only bias would make the distance filter gate out every
        encoder sample during a speed change."""
        mcu = MockMcu(MockMcuConfig(accel_noise_std=0.0, accel_bias_mps2=0.0))
        mcu.open()
        mcu.write(protocol.encode_motor(0.6, 0.6))
        mcu.step(0.01)
        mcu.read_lines()
        mcu.step(0.01)
        telemetry = protocol.decode_telemetry(
            protocol.parse_frame(mcu.read_lines()[0]), 0.0)
        assert telemetry.imu.accel_x > 0.1

    def test_servo_commands_are_applied(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_servo(0, 72.0))
        mcu.write(protocol.encode_servo(1, 88.0))
        mcu.write(protocol.encode_marker(95.0))
        assert (mcu.pan_deg, mcu.tilt_deg, mcu.marker_deg) == (72.0, 88.0, 95.0)

    def test_flow_accumulates_only_with_pump_and_valve(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_valve(True))
        mcu.step(0.2)
        assert mcu.flow_ticks == 0            # valve open but pump off
        mcu.write(protocol.encode_pump(True))
        mcu.step(0.2)
        assert mcu.flow_ticks > 0

    def test_estop_stops_everything(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_motor(0.8, 0.8))
        mcu.write(protocol.encode_pump(True))
        mcu.write(protocol.encode_valve(True))
        mcu.step(0.1)
        mcu.write(protocol.encode_estop())
        assert mcu.estopped and not mcu.pump_on and not mcu.valve_open
        mcu.write(protocol.encode_motor(0.8, 0.8))   # ignored while estopped
        assert mcu.cmd_left == 0.0

    def test_heartbeat_rearms_after_estop(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_estop())
        mcu.write(protocol.encode_heartbeat(1))
        assert mcu.estopped is False

    def test_watchdog_stops_the_drive_when_heartbeats_stop(self):
        """A crashed Jetson must stop the robot, not leave it driving on the
        last command."""
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_heartbeat(1))
        mcu.write(protocol.encode_motor(0.6, 0.6))
        mcu.step(0.2)
        assert mcu.cmd_left > 0
        for _ in range(80):                   # 0.8 s with no heartbeat
            mcu.step(0.01)
        assert mcu.cmd_left == 0.0

    def test_corrupt_command_is_ignored(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write("$M,900,900*00\n")
        assert mcu.cmd_left == 0.0

    def test_slip_makes_the_encoder_over_read(self):
        mcu = MockMcu(MockMcuConfig(encoder_noise_std=0.0, gyro_noise_std=0.0))
        mcu.open()
        mcu.write(protocol.encode_motor(0.5, 0.5))
        for _ in range(200):
            mcu.step(0.01)
        mcu.read_lines()
        mcu.set_slip(True)
        mcu.step(0.01)
        telemetry = protocol.decode_telemetry(
            protocol.parse_frame(mcu.read_lines()[0]), 0.0)
        assert telemetry.encoders.linear_mps > mcu.left_mps + 0.1

    def test_reset_encoders_zeroes_distance(self):
        mcu = MockMcu()
        mcu.open()
        mcu.write(protocol.encode_motor(0.6, 0.6))
        mcu.step(1.0)
        assert mcu.distance_m > 0
        mcu.write(protocol.encode_reset_encoders())
        assert mcu.distance_m == 0.0

    def test_state_snapshot_is_serialisable(self):
        mcu = MockMcu()
        mcu.open()
        mcu.step(0.1)
        assert set(mcu.state()) >= {"t", "x", "y", "heading_deg", "distance_m"}
