#!/usr/bin/env python3
"""
rpi_pico_lsl.py

Raspberry Pi LSL bridge for the unified Raspberry Pi Pico acquisition firmware:

    BNO085   -> Pico I2C0
    MAX30102 -> Pico I2C1
                 |
                 +---- Protocol V2, 42-byte binary frames over native USB CDC
                 |
                 v
            Raspberry Pi
                 |
                 +---- daq_accel
                 +---- daq_gyro
                 +---- daq_quat
                 +---- daq_ppg

Protocol V2 frame (42 bytes)
----------------------------
    uint8   magic0              0xA5
    uint8   magic1              0x5A
    uint8   version             2
    uint8   packet_type
                                  1 = IMU
                                  2 = IMU stats
                                  3 = PPG
                                  4 = PPG stats
    uint32  packet_sequence     global Pico packet sequence
    uint64  sensor_timestamp_us Pico time_us_64() sample timestamp
    uint32  host_timestamp_us   low 32 bits of the same Pico timestamp
    uint8   sensor_sequence
                                  IMU: BNO085 report sequence
                                  PPG: low 8 bits of PPG sample_index
    uint8   sensor_type
                                  IMU: 1=ACC, 2=GYRO, 3=QUAT
                                  PPG: 1=Red+IR
    uint8   sensor_status       BNO085 status for IMU, 0 for PPG
    uint8   n_values
    uint8   payload[16]
    uint16  crc16_ccitt

IMU payload
-----------
    4 x float32
    ACC/GYRO use first 3 values
    QUAT uses all 4 values

PPG payload
-----------
    uint32 red
    uint32 ir
    uint32 sample_index
    uint32 reserved

IMU stats payload
-----------------
    uint32 imu_ring_overflow
    uint32 imu_max_queue_depth
    uint32 imu_current_queue_depth
    uint32 bno_reset_count

PPG stats payload
-----------------
    uint32 ppg_ring_overflow
    uint32 ppg_max_queue_depth
    uint32 ppg_current_queue_depth
    uint32 max30102_fifo_overflow

Timing
------
Both IMU and PPG samples are timestamped on the Pico using the SAME 64-bit
hardware clock. The Raspberry Pi estimates one Pico-clock -> LSL-clock offset
during a short calibration interval, then applies the same mapping to every
sensor stream.

The 32-bit timestamp field is retained only for Protocol V2 compatibility.
It is not used for LSL timing.

Dependencies
------------
    pip install pyserial pylsl

Examples
--------
    python3 rpi_pico_lsl.py
    python3 rpi_pico_lsl.py --verbose
    python3 rpi_pico_lsl.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import serial
from pylsl import StreamInfo, StreamOutlet, local_clock


# =============================================================================
# Protocol V2
# =============================================================================

MAGIC = b"\xA5\x5A"
MAGIC0 = 0xA5
MAGIC1 = 0x5A
PROTOCOL_VERSION = 2

PKT_IMU = 1
PKT_STATS = 2
PKT_PPG = 3
PKT_PPG_STATS = 4

SENSOR_ACCEL = 1
SENSOR_GYRO = 2
SENSOR_QUAT = 3

PPG_RED_IR = 1

HEADER_FMT = "<BBBBIQIBBBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 24
PAYLOAD_SIZE = 16
CRC_SIZE = 2
PACKET_SIZE = HEADER_SIZE + PAYLOAD_SIZE + CRC_SIZE  # 42

if PACKET_SIZE != 42:
    raise RuntimeError(f"Protocol V2 packet size is {PACKET_SIZE}, expected 42")


# =============================================================================
# Defaults
# =============================================================================

SERIAL_PORT = "/dev/ttyACM0"

# Native USB CDC on the Pico ignores the physical UART baud rate, but pyserial
# still requires one. Keep 500000 for compatibility with the previous setup.
SERIAL_BAUD = 500_000

STREAM_PREFIX = "daq"

ACCEL_RATE_HZ = 200.0
GYRO_RATE_HZ = 200.0
QUAT_RATE_HZ = 100.0
PPG_RATE_HZ = 200.0

DIAG_PERIOD_S = 1.0
CLOCK_CALIBRATION_S = 1.0


# =============================================================================
# CRC16-CCITT
# =============================================================================

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# =============================================================================
# Protocol parser
# =============================================================================

@dataclass
class ParsedFrame:
    packet_type: int
    packet_sequence: int
    sensor_timestamp_us: int
    host_timestamp_us_raw: int
    sensor_sequence: int
    sensor_type: int
    sensor_status: int
    n_values: int
    payload: bytes


class ProtocolV2Parser:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.crc_errors = 0
        self.framing_bytes_discarded = 0
        self.version_errors = 0

    def feed(self, chunk: bytes) -> List[ParsedFrame]:
        if chunk:
            self.buffer.extend(chunk)

        frames: List[ParsedFrame] = []

        while len(self.buffer) >= 2:
            pos = self.buffer.find(MAGIC)

            if pos < 0:
                # Keep a trailing 0xA5 in case it is the first byte of the next
                # frame's magic marker.
                if self.buffer and self.buffer[-1] == MAGIC0:
                    discarded = len(self.buffer) - 1
                    self.framing_bytes_discarded += discarded
                    self.buffer[:] = self.buffer[-1:]
                else:
                    self.framing_bytes_discarded += len(self.buffer)
                    self.buffer.clear()
                break

            if pos > 0:
                self.framing_bytes_discarded += pos
                del self.buffer[:pos]

            if len(self.buffer) < PACKET_SIZE:
                break

            raw = bytes(self.buffer[:PACKET_SIZE])

            received_crc = struct.unpack_from("<H", raw, PACKET_SIZE - 2)[0]
            calculated_crc = crc16_ccitt(raw[:-2])

            if received_crc != calculated_crc:
                self.crc_errors += 1
                # Drop one byte, then search for the next valid magic marker.
                del self.buffer[0]
                continue

            (
                magic0,
                magic1,
                version,
                packet_type,
                packet_sequence,
                sensor_timestamp_us,
                host_timestamp_us_raw,
                sensor_sequence,
                sensor_type,
                sensor_status,
                n_values,
            ) = struct.unpack_from(HEADER_FMT, raw, 0)

            if magic0 != MAGIC0 or magic1 != MAGIC1:
                self.framing_bytes_discarded += 1
                del self.buffer[0]
                continue

            if version != PROTOCOL_VERSION:
                self.version_errors += 1
                del self.buffer[:PACKET_SIZE]
                continue

            payload = raw[HEADER_SIZE:HEADER_SIZE + PAYLOAD_SIZE]

            frames.append(
                ParsedFrame(
                    packet_type=packet_type,
                    packet_sequence=packet_sequence,
                    sensor_timestamp_us=sensor_timestamp_us,
                    host_timestamp_us_raw=host_timestamp_us_raw,
                    sensor_sequence=sensor_sequence,
                    sensor_type=sensor_type,
                    sensor_status=sensor_status,
                    n_values=n_values,
                    payload=payload,
                )
            )

            del self.buffer[:PACKET_SIZE]

        return frames


# =============================================================================
# Continuous USB serial reader
# =============================================================================

class SerialReader(threading.Thread):
    """
    Continuously drain the Pico USB CDC stream so LSL publishing cannot block
    serial reception.
    """

    def __init__(
        self,
        ser: serial.Serial,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="pico-serial-reader")
        self.ser = ser
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.parser = ProtocolV2Parser()

        self.queue_drops = 0
        self.read_errors = 0
        self.bytes_received = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                waiting = self.ser.in_waiting
                if waiting:
                    chunk = self.ser.read(min(waiting, 8192))
                else:
                    chunk = self.ser.read(1)
            except (serial.SerialException, OSError) as exc:
                self.read_errors += 1
                print(f"[SERIAL] read error: {exc}", file=sys.stderr, flush=True)
                self.stop_event.set()
                return

            if not chunk:
                continue

            self.bytes_received += len(chunk)

            # Timestamp the USB receive event once for this chunk. Clock
            # calibration uses the minimum observed transport delay.
            receive_lsl_s = local_clock()
            frames = self.parser.feed(chunk)

            for frame in frames:
                try:
                    self.output_queue.put_nowait((frame, receive_lsl_s))
                except queue.Full:
                    self.queue_drops += 1


# =============================================================================
# Pico -> LSL clock mapping
# =============================================================================

class PicoLslClockMapper:
    """
    Estimate the offset between the Pico's time_us_64() clock and
    pylsl.local_clock().

    During calibration we retain the minimum:
        receive_lsl - pico_timestamp

    USB/queueing delays are non-negative, so the minimum observed offset is the
    best simple estimate of the clock relationship available from one-way
    packets. After the calibration window the mapping is frozen, matching the
    previous acquisition approach while giving every sensor one common clock.
    """

    def __init__(self, calibration_s: float) -> None:
        self.calibration_s = float(calibration_s)
        self.started_mono: Optional[float] = None
        self.min_offset_s: Optional[float] = None
        self.offset_s: Optional[float] = None
        self.frozen = False
        self.observations = 0

    def observe(self, pico_us: int, receive_lsl_s: float) -> None:
        if self.frozen:
            return

        now_mono = time.monotonic()
        if self.started_mono is None:
            self.started_mono = now_mono

        candidate = float(receive_lsl_s) - int(pico_us) * 1e-6

        if self.min_offset_s is None or candidate < self.min_offset_s:
            self.min_offset_s = candidate

        self.observations += 1

    def ready_to_freeze(self) -> bool:
        if self.frozen or self.started_mono is None:
            return False

        return (
            self.observations >= 20
            and (time.monotonic() - self.started_mono) >= self.calibration_s
        )

    def freeze(self) -> None:
        if self.frozen:
            return

        if self.min_offset_s is None:
            raise RuntimeError("No Pico clock observations available")

        self.offset_s = self.min_offset_s
        self.frozen = True

    def pico_to_lsl(self, pico_us: int) -> float:
        if not self.frozen or self.offset_s is None:
            raise RuntimeError("Pico/LSL clock mapping not frozen")

        return int(pico_us) * 1e-6 + self.offset_s


# =============================================================================
# Sequence tracking
# =============================================================================

class SequenceTracker:
    """
    Track a wrapping unsigned sequence counter.

    A very large apparent backward jump is treated as a device restart/resync
    rather than billions of dropped samples.
    """

    def __init__(self, modulus: int) -> None:
        self.modulus = int(modulus)
        self.previous: Optional[int] = None
        self.dropped = 0
        self.restarts = 0

    def update(self, sequence: int) -> None:
        sequence %= self.modulus

        if self.previous is None:
            self.previous = sequence
            return

        expected = (self.previous + 1) % self.modulus
        gap = (sequence - expected) % self.modulus

        if gap > self.modulus // 2:
            self.restarts += 1
            gap = 0

        self.dropped += gap
        self.previous = sequence


# =============================================================================
# LSL metadata / outlets
# =============================================================================

def add_channel_metadata(
    info: StreamInfo,
    channels: Sequence[Tuple[str, str, str]],
) -> None:
    root = info.desc().append_child("channels")

    for label, unit, channel_type in channels:
        ch = root.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", channel_type)


def make_outlet(
    *,
    name: str,
    stream_type: str,
    channels: Sequence[Tuple[str, str, str]],
    nominal_rate_hz: float,
    source_id: str,
    manufacturer: str,
    timestamp_source: str,
) -> StreamOutlet:
    info = StreamInfo(
        name=name,
        type=stream_type,
        channel_count=len(channels),
        nominal_srate=float(nominal_rate_hz),
        channel_format="float32",
        source_id=source_id,
    )

    add_channel_metadata(info, channels)

    desc = info.desc()
    desc.append_child_value("manufacturer", manufacturer)
    desc.append_child_value("transport", "Raspberry Pi Pico native USB CDC")
    desc.append_child_value("protocol", "Pico Protocol V2, 42-byte CRC16 frames")
    desc.append_child_value("timestamp_source", timestamp_source)

    return StreamOutlet(info, chunk_size=0, max_buffered=60)


def make_imu_outlets(
    prefix: str,
    serial_port: str,
    accel_rate: float,
    gyro_rate: float,
    quat_rate: float,
) -> Dict[int, StreamOutlet]:
    host = socket.gethostname()
    port_id = serial_port.replace("/", "_").replace("\\", "_").replace(":", "_")
    source_base = f"{prefix}_{host}_{port_id}"

    common_timestamp_source = (
        "Raspberry Pi Pico time_us_64() sample timestamp mapped to "
        "pylsl.local_clock(); shared with MAX30102 PPG"
    )

    return {
        SENSOR_ACCEL: make_outlet(
            name=f"{prefix}_accel",
            stream_type="Accelerometer",
            channels=[
                ("ax", "m/s^2", "Accelerometer"),
                ("ay", "m/s^2", "Accelerometer"),
                ("az", "m/s^2", "Accelerometer"),
            ],
            nominal_rate_hz=accel_rate,
            source_id=f"{source_base}_accel",
            manufacturer="BNO085 via Raspberry Pi Pico",
            timestamp_source=common_timestamp_source,
        ),
        SENSOR_GYRO: make_outlet(
            name=f"{prefix}_gyro",
            stream_type="Gyroscope",
            channels=[
                ("gx", "rad/s", "Gyroscope"),
                ("gy", "rad/s", "Gyroscope"),
                ("gz", "rad/s", "Gyroscope"),
            ],
            nominal_rate_hz=gyro_rate,
            source_id=f"{source_base}_gyro",
            manufacturer="BNO085 via Raspberry Pi Pico",
            timestamp_source=common_timestamp_source,
        ),
        SENSOR_QUAT: make_outlet(
            name=f"{prefix}_quat",
            stream_type="Quaternion",
            channels=[
                ("qw", "unitless", "Quaternion"),
                ("qx", "unitless", "Quaternion"),
                ("qy", "unitless", "Quaternion"),
                ("qz", "unitless", "Quaternion"),
            ],
            nominal_rate_hz=quat_rate,
            source_id=f"{source_base}_quat",
            manufacturer="BNO085 via Raspberry Pi Pico",
            timestamp_source=common_timestamp_source,
        ),
    }


def make_ppg_outlet(
    prefix: str,
    serial_port: str,
    ppg_rate: float,
) -> StreamOutlet:
    host = socket.gethostname()
    port_id = serial_port.replace("/", "_").replace("\\", "_").replace(":", "_")

    return make_outlet(
        name=f"{prefix}_ppg",
        stream_type="PPG",
        channels=[
            ("red", "counts", "PPG"),
            ("ir", "counts", "PPG"),
        ],
        nominal_rate_hz=ppg_rate,
        source_id=f"{prefix}_{host}_{port_id}_ppg",
        manufacturer="MAX30102 via Raspberry Pi Pico",
        timestamp_source=(
            "Raspberry Pi Pico time_us_64() sample timestamp mapped to "
            "pylsl.local_clock(); shared with BNO085 IMU"
        ),
    )


# =============================================================================
# Pending sample used during clock calibration
# =============================================================================

@dataclass
class PendingSample:
    kind: str              # "imu" or "ppg"
    pico_us: int
    values: List[float]
    sensor_type: int = 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Publish Raspberry Pi Pico BNO085 + MAX30102 Protocol V2 data to LSL."
        )
    )

    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--prefix", default=STREAM_PREFIX)

    ap.add_argument("--accel-rate", type=float, default=ACCEL_RATE_HZ)
    ap.add_argument("--gyro-rate", type=float, default=GYRO_RATE_HZ)
    ap.add_argument("--quat-rate", type=float, default=QUAT_RATE_HZ)
    ap.add_argument("--ppg-rate", type=float, default=PPG_RATE_HZ)

    ap.add_argument("--diag-period", type=float, default=DIAG_PERIOD_S)
    ap.add_argument("--clock-calibration", type=float, default=CLOCK_CALIBRATION_S)

    ap.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after opening Pico USB CDC before clearing stale input.",
    )
    ap.add_argument("--serial-queue", type=int, default=20000)

    ap.add_argument(
        "--no-ppg",
        action="store_true",
        help="Do not publish the PPG LSL stream (PPG packets are still parsed).",
    )
    ap.add_argument(
        "--no-imu",
        action="store_true",
        help="Do not publish IMU LSL streams (IMU packets are still parsed).",
    )
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    stop_event = threading.Event()

    print(
        f"[DAQ] Opening Pico {args.port} at logical baud {args.baud:,} "
        f"(Protocol V2, {PACKET_SIZE}-byte frames)"
    )

    try:
        ser = serial.Serial(
            args.port,
            baudrate=args.baud,
            timeout=0.02,
        )
    except serial.SerialException as exc:
        print(f"[DAQ] Could not open serial port: {exc}", file=sys.stderr)
        return 2

    # Pico native USB CDC does not require UART baud matching. Give a newly
    # connected/rebooted board a short settling period, then discard any partial
    # frame already sitting in the Linux tty buffer.
    time.sleep(max(0.0, args.startup_delay))
    ser.reset_input_buffer()

    imu_outlets: Dict[int, StreamOutlet] = {}
    if not args.no_imu:
        imu_outlets = make_imu_outlets(
            prefix=args.prefix,
            serial_port=args.port,
            accel_rate=args.accel_rate,
            gyro_rate=args.gyro_rate,
            quat_rate=args.quat_rate,
        )

    ppg_outlet: Optional[StreamOutlet] = None
    if not args.no_ppg:
        ppg_outlet = make_ppg_outlet(
            prefix=args.prefix,
            serial_port=args.port,
            ppg_rate=args.ppg_rate,
        )

    print("[LSL] Published streams:")
    if imu_outlets:
        print(
            f"      {args.prefix}_accel : 3 ch @ nominal "
            f"{args.accel_rate:g} Hz (ax ay az)"
        )
        print(
            f"      {args.prefix}_gyro  : 3 ch @ nominal "
            f"{args.gyro_rate:g} Hz (gx gy gz)"
        )
        print(
            f"      {args.prefix}_quat  : 4 ch @ nominal "
            f"{args.quat_rate:g} Hz (qw qx qy qz)"
        )
    if ppg_outlet is not None:
        print(
            f"      {args.prefix}_ppg   : 2 ch @ nominal "
            f"{args.ppg_rate:g} Hz (red ir)"
        )

    frame_queue: queue.Queue = queue.Queue(
        maxsize=max(1000, int(args.serial_queue))
    )

    serial_reader = SerialReader(
        ser=ser,
        output_queue=frame_queue,
        stop_event=stop_event,
    )
    serial_reader.start()

    print("[DAQ] Serial reader started; waiting for Pico Protocol V2 packets...")

    clock_mapper = PicoLslClockMapper(args.clock_calibration)

    # One global sequence spans IMU, PPG and stats packets.
    usb_sequence = SequenceTracker(1 << 32)

    # Independent PPG sample-index sequence catches lost PPG samples even when
    # other packet types are interleaved.
    ppg_sample_sequence = SequenceTracker(1 << 32)

    pending: List[PendingSample] = []

    imu_counts: Dict[int, int] = {
        SENSOR_ACCEL: 0,
        SENSOR_GYRO: 0,
        SENSOR_QUAT: 0,
    }
    ppg_count = 0

    latest_bno_sequence: Dict[int, Optional[int]] = {
        SENSOR_ACCEL: None,
        SENSOR_GYRO: None,
        SENSOR_QUAT: None,
    }

    latest_ppg_sample_index: Optional[int] = None

    # Pico IMU-side stats
    imu_ring_overflow = 0
    imu_max_queue_depth = 0
    imu_current_queue_depth = 0
    bno_reset_count = 0

    # Pico PPG-side stats
    ppg_ring_overflow = 0
    ppg_max_queue_depth = 0
    ppg_current_queue_depth = 0
    ppg_fifo_overflow = 0

    unknown_packets = 0
    unknown_sensors = 0
    invalid_ppg_packets = 0

    last_diag = time.monotonic()
    last_valid_frame = time.monotonic()

    def publish_pending_sample(sample: PendingSample) -> None:
        ts_lsl = clock_mapper.pico_to_lsl(sample.pico_us)

        if sample.kind == "imu":
            outlet = imu_outlets.get(sample.sensor_type)
            if outlet is not None:
                outlet.push_sample(sample.values, timestamp=ts_lsl)

        elif sample.kind == "ppg":
            if ppg_outlet is not None:
                ppg_outlet.push_sample(sample.values, timestamp=ts_lsl)

    try:
        while not stop_event.is_set():
            try:
                frame, receive_lsl_s = frame_queue.get(timeout=0.05)
            except queue.Empty:
                frame = None
                receive_lsl_s = None

            if frame is not None:
                last_valid_frame = time.monotonic()

                # Every transmitted Pico frame consumes the same global sequence.
                usb_sequence.update(frame.packet_sequence)

                # Protocol V2 Pico build:
                # sensor_timestamp_us == common time_us_64() clock.
                pico_us = int(frame.sensor_timestamp_us)

                if frame.packet_type == PKT_IMU:
                    if frame.sensor_type not in (
                        SENSOR_ACCEL,
                        SENSOR_GYRO,
                        SENSOR_QUAT,
                    ):
                        unknown_sensors += 1

                    else:
                        expected_n = (
                            4 if frame.sensor_type == SENSOR_QUAT else 3
                        )

                        if frame.n_values != expected_n:
                            unknown_sensors += 1

                        else:
                            values = struct.unpack_from("<4f", frame.payload, 0)
                            sample_values = list(values[:frame.n_values])

                            clock_mapper.observe(
                                pico_us,
                                float(receive_lsl_s),
                            )

                            latest_bno_sequence[frame.sensor_type] = (
                                frame.sensor_sequence
                            )
                            imu_counts[frame.sensor_type] += 1

                            pending_sample = PendingSample(
                                kind="imu",
                                sensor_type=frame.sensor_type,
                                pico_us=pico_us,
                                values=sample_values,
                            )

                            if clock_mapper.frozen:
                                publish_pending_sample(pending_sample)
                            else:
                                pending.append(pending_sample)

                elif frame.packet_type == PKT_PPG:
                    if (
                        frame.sensor_type != PPG_RED_IR
                        or frame.n_values != 2
                    ):
                        invalid_ppg_packets += 1

                    else:
                        red, ir, sample_index, _reserved = struct.unpack_from(
                            "<4I",
                            frame.payload,
                            0,
                        )

                        clock_mapper.observe(
                            pico_us,
                            float(receive_lsl_s),
                        )

                        ppg_sample_sequence.update(sample_index)
                        latest_ppg_sample_index = sample_index
                        ppg_count += 1

                        pending_sample = PendingSample(
                            kind="ppg",
                            pico_us=pico_us,
                            values=[float(red), float(ir)],
                        )

                        if clock_mapper.frozen:
                            publish_pending_sample(pending_sample)
                        else:
                            pending.append(pending_sample)

                elif frame.packet_type == PKT_STATS:
                    (
                        imu_ring_overflow,
                        imu_max_queue_depth,
                        imu_current_queue_depth,
                        bno_reset_count,
                    ) = struct.unpack_from("<4I", frame.payload, 0)

                elif frame.packet_type == PKT_PPG_STATS:
                    (
                        ppg_ring_overflow,
                        ppg_max_queue_depth,
                        ppg_current_queue_depth,
                        ppg_fifo_overflow,
                    ) = struct.unpack_from("<4I", frame.payload, 0)

                else:
                    unknown_packets += 1

            # Once enough one-way USB observations have been collected, freeze
            # the common Pico -> LSL mapping and release the short pending buffer.
            if not clock_mapper.frozen and clock_mapper.ready_to_freeze():
                clock_mapper.freeze()

                print(
                    "[CLOCK] Pico time_us_64() -> LSL mapping frozen: "
                    f"offset={clock_mapper.offset_s:.6f} s; "
                    f"calibration_samples={clock_mapper.observations}",
                    flush=True,
                )

                for sample in pending:
                    publish_pending_sample(sample)

                pending.clear()

            now = time.monotonic()

            if now - last_diag >= max(0.2, args.diag_period):
                elapsed = now - last_diag
                last_diag = now

                acc_n = imu_counts[SENSOR_ACCEL]
                gyro_n = imu_counts[SENSOR_GYRO]
                quat_n = imu_counts[SENSOR_QUAT]

                imu_counts = {
                    SENSOR_ACCEL: 0,
                    SENSOR_GYRO: 0,
                    SENSOR_QUAT: 0,
                }

                ppg_n = ppg_count
                ppg_count = 0

                print(
                    f"[LSL] "
                    f"ACC={acc_n:3d}({acc_n / elapsed:6.1f}Hz) "
                    f"GYRO={gyro_n:3d}({gyro_n / elapsed:6.1f}Hz) "
                    f"QUAT={quat_n:3d}({quat_n / elapsed:6.1f}Hz) "
                    f"PPG={ppg_n:3d}({ppg_n / elapsed:6.1f}Hz) | "
                    f"IMUring={imu_ring_overflow} "
                    f"depth={imu_current_queue_depth}/{imu_max_queue_depth} "
                    f"resets={bno_reset_count} | "
                    f"PPGring={ppg_ring_overflow} "
                    f"depth={ppg_current_queue_depth}/{ppg_max_queue_depth} "
                    f"FIFOovf={ppg_fifo_overflow} | "
                    f"CRC={serial_reader.parser.crc_errors} "
                    f"USBdrops={usb_sequence.dropped} "
                    f"PPGdrops={ppg_sample_sequence.dropped} "
                    f"framing={serial_reader.parser.framing_bytes_discarded} "
                    f"rxQ={frame_queue.qsize()} "
                    f"qdrop={serial_reader.queue_drops}",
                    flush=True,
                )

                if args.verbose:
                    print(
                        "[PICO] "
                        f"BNOseq A/G/Q="
                        f"{latest_bno_sequence[SENSOR_ACCEL]}/"
                        f"{latest_bno_sequence[SENSOR_GYRO]}/"
                        f"{latest_bno_sequence[SENSOR_QUAT]} "
                        f"PPGindex={latest_ppg_sample_index} "
                        f"USBrestarts={usb_sequence.restarts} "
                        f"PPGrestarts={ppg_sample_sequence.restarts} "
                        f"unknown_packets={unknown_packets} "
                        f"unknown_sensors={unknown_sensors} "
                        f"invalid_ppg={invalid_ppg_packets}",
                        flush=True,
                    )

                if now - last_valid_frame > 2.0:
                    print(
                        "[DAQ] WARNING: no valid Pico Protocol V2 frames for >2 s",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\n[DAQ] Stopping.", flush=True)

    finally:
        stop_event.set()
        serial_reader.join(timeout=1.0)

        try:
            ser.close()
        except Exception:
            pass

    print("[DAQ] Pico serial port closed.")

    print(
        "[DAQ] Final diagnostics: "
        f"CRC={serial_reader.parser.crc_errors}, "
        f"USBdrops={usb_sequence.dropped}, "
        f"USBrestarts={usb_sequence.restarts}, "
        f"PPGdrops={ppg_sample_sequence.dropped}, "
        f"PPGrestarts={ppg_sample_sequence.restarts}, "
        f"framing={serial_reader.parser.framing_bytes_discarded}, "
        f"version_errors={serial_reader.parser.version_errors}, "
        f"serial_queue_drops={serial_reader.queue_drops}, "
        f"serial_read_errors={serial_reader.read_errors}, "
        f"unknown_packets={unknown_packets}, "
        f"unknown_sensors={unknown_sensors}, "
        f"invalid_ppg_packets={invalid_ppg_packets}, "
        f"IMU_ring_overflow={imu_ring_overflow}, "
        f"PPG_ring_overflow={ppg_ring_overflow}, "
        f"MAX30102_fifo_overflow={ppg_fifo_overflow}, "
        f"BNO_resets={bno_reset_count}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
