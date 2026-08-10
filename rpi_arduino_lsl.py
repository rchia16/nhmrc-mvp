#!/usr/bin/env python3
"""
rpi_arduino_lsl_v2_ppg_syncfix.py

Raspberry Pi LSL bridge for:
  1) Arduino Mega + BNO085 Protocol V2 over USB serial
  2) MAX30102 PPG connected directly to Raspberry Pi I2C

Arduino Protocol V2 frame (42 bytes)
------------------------------------
    uint8   magic0              0xA5
    uint8   magic1              0x5A
    uint8   version             2
    uint8   packet_type         1=IMU, 2=STATS
    uint32  packet_sequence
    uint64  sensor_timestamp_us BNO085 SH2 timestamp (retained for diagnostics)
    uint32  host_timestamp_us   Arduino micros() when event was retrieved
    uint8   sensor_sequence     BNO085 report sequence (diagnostic only)
    uint8   sensor_type         1=ACC, 2=GYRO, 3=QUAT
    uint8   sensor_status       BNO085 accuracy/status
    uint8   n_values
    uint8   payload[16]         4x float32 for IMU; 4x uint32 for STATS
    uint16  crc16_ccitt

LSL streams
-----------
    daq_ACCEL : ax, ay, az           [m/s^2]
    daq_GYRO  : gx, gy, gz           [rad/s]
    daq_QUAT  : qw, qx, qy, qz       [unitless]
    daq_ppg   : red, ir               [ADC counts]

Timing
------
BNO085:
    The authoritative LSL timestamp is derived from the Arduino Mega's
    host_timestamp_us (micros()) and mapped into pylsl.local_clock().

    sensor_timestamp_us is retained in Protocol V2 but is NOT used for LSL
    timing because live SH2 values were observed to have discontinuities / an
    incompatible epoch for direct host-clock mapping.

PPG:
    MAX30102 FIFO samples are drained in batches, but timestamps are maintained
    on one persistent sample clock across FIFO batches. A small phase-locked
    correction follows pylsl.local_clock() without allowing timestamps to move
    backward, while a large real gap re-anchors the clock.

Architecture
------------
A dedicated serial-reader thread continuously drains /dev/ttyACM0 so LSL
publishing cannot overflow the Linux tty buffer. This is important at ~400+
42-byte Protocol V2 frames/s.

Dependencies
------------
    pip install pyserial pylsl smbus2 RPi.GPIO

The repository-local max30102.py must be importable (normally this script lives
beside it in nhmrc-mvp).

Examples
--------
    python3 rpi_arduino_lsl_v2_ppg_syncfix.py
    python3 rpi_arduino_lsl_v2_ppg_syncfix.py --verbose
    python3 rpi_arduino_lsl_v2_ppg_syncfix.py --port /dev/ttyACM0 --baud 500000
    python3 rpi_arduino_lsl_v2_ppg_syncfix.py --no-ppg
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

try:
    import max30102
except ImportError:
    max30102 = None


# =============================================================================
# Protocol V2
# =============================================================================

MAGIC = b"\xA5\x5A"
MAGIC0 = 0xA5
MAGIC1 = 0x5A
PROTOCOL_VERSION = 2

PKT_IMU = 1
PKT_STATS = 2

SENSOR_ACCEL = 1
SENSOR_GYRO = 2
SENSOR_QUAT = 3

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
SERIAL_BAUD = 500_000
STREAM_PREFIX = "daq"

ACCEL_RATE_HZ = 200.0
GYRO_RATE_HZ = 200.0
QUAT_RATE_HZ = 100.0
PPG_RATE_HZ = 200.0

DIAG_PERIOD_S = 1.0
CLOCK_CALIBRATION_S = 1.0
PPG_POLL_MS = 5.0
PPG_REANCHOR_GAP_S = 0.25
PPG_PHASE_ALPHA = 0.05
PPG_MAX_PHASE_CORRECTION_FRACTION = 0.10


# =============================================================================
# CRC
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

            frame = bytes(self.buffer[:PACKET_SIZE])
            received_crc = struct.unpack_from("<H", frame, PACKET_SIZE - 2)[0]
            calculated_crc = crc16_ccitt(frame[:-2])

            if received_crc != calculated_crc:
                self.crc_errors += 1
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
            ) = struct.unpack_from(HEADER_FMT, frame, 0)

            if magic0 != MAGIC0 or magic1 != MAGIC1:
                self.framing_bytes_discarded += 1
                del self.buffer[0]
                continue

            if version != PROTOCOL_VERSION:
                self.version_errors += 1
                del self.buffer[:PACKET_SIZE]
                continue

            payload = frame[HEADER_SIZE:HEADER_SIZE + PAYLOAD_SIZE]

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
# Continuous serial reader
# =============================================================================

class SerialReader(threading.Thread):
    """Continuously drain the tty so LSL work cannot overflow the serial RX path."""

    def __init__(
        self,
        ser: serial.Serial,
        output_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="arduino-serial-reader")
        self.ser = ser
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.parser = ProtocolV2Parser()
        self.queue_drops = 0
        self.read_errors = 0

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

            receive_lsl_s = local_clock()
            frames = self.parser.feed(chunk)

            for frame in frames:
                try:
                    self.output_queue.put_nowait((frame, receive_lsl_s))
                except queue.Full:
                    self.queue_drops += 1


# =============================================================================
# Clock mapping
# =============================================================================

class HostMicrosExtender:
    MODULUS = 1 << 32
    HALF = 1 << 31

    def __init__(self) -> None:
        self.previous: Optional[int] = None
        self.upper = 0

    def extend(self, raw_us: int) -> int:
        raw_us &= 0xFFFFFFFF
        if self.previous is not None:
            if raw_us < self.previous and (self.previous - raw_us) > self.HALF:
                self.upper += self.MODULUS
        self.previous = raw_us
        return self.upper + raw_us


class HostLslClockMapper:
    """Map extended Arduino micros() into pylsl.local_clock()."""

    def __init__(self, calibration_s: float) -> None:
        self.calibration_s = float(calibration_s)
        self.started_mono: Optional[float] = None
        self.min_offset_s: Optional[float] = None
        self.offset_s: Optional[float] = None
        self.frozen = False
        self.observations = 0

    def observe(self, host_us: int, receive_lsl_s: float) -> None:
        if self.frozen:
            return

        now_mono = time.monotonic()
        if self.started_mono is None:
            self.started_mono = now_mono

        candidate = receive_lsl_s - host_us * 1e-6
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
            raise RuntimeError("No Arduino clock observations available")
        self.offset_s = self.min_offset_s
        self.frozen = True

    def host_to_lsl(self, host_us: int) -> float:
        if not self.frozen or self.offset_s is None:
            raise RuntimeError("Arduino/LSL clock mapping not frozen")
        return host_us * 1e-6 + self.offset_s


# =============================================================================
# Sequence tracking
# =============================================================================

class SequenceTracker:
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

        # Large apparent gaps are treated as restart / resync rather than loss.
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
            manufacturer="BNO085 via Arduino Mega",
            timestamp_source="Arduino Mega micros() mapped to pylsl.local_clock()",
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
            manufacturer="BNO085 via Arduino Mega",
            timestamp_source="Arduino Mega micros() mapped to pylsl.local_clock()",
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
            manufacturer="BNO085 via Arduino Mega",
            timestamp_source="Arduino Mega micros() mapped to pylsl.local_clock()",
        ),
    }


def make_ppg_outlet(prefix: str, ppg_rate: float) -> StreamOutlet:
    host = socket.gethostname()
    return make_outlet(
        name=f"{prefix}_ppg",
        stream_type="PPG",
        channels=[
            ("red", "counts", "PPG"),
            ("ir", "counts", "PPG"),
        ],
        nominal_rate_hz=ppg_rate,
        source_id=f"{prefix}_{host}_max30102_ppg",
        manufacturer="MAX30102 via Raspberry Pi I2C",
        timestamp_source=(
            "Persistent MAX30102 sample clock at configured interval; "
            "initially anchored and re-anchored after large gaps to "
            "pylsl.local_clock()"
        ),
    )


# =============================================================================
# MAX30102 PPG worker
# =============================================================================

class PPGPublisher(threading.Thread):
    def __init__(
        self,
        outlet: StreamOutlet,
        stop_event: threading.Event,
        sample_rate_hz: float,
        poll_ms: float,
        reanchor_gap_s: float,
        verbose: bool = False,
    ) -> None:
        super().__init__(daemon=True, name="max30102-ppg")
        self.outlet = outlet
        self.stop_event = stop_event
        self.sample_rate_hz = float(sample_rate_hz)
        self.poll_ms = float(poll_ms)
        self.reanchor_gap_s = float(reanchor_gap_s)
        self.verbose = bool(verbose)

        self.sensor = None
        self._lock = threading.Lock()
        self._count_since_report = 0
        self.total_samples = 0
        self.read_errors = 0
        self.init_error: Optional[str] = None

        # Persistent PPG sample clock state.
        self.last_sample_lsl: Optional[float] = None
        self.clock_reanchors = 0
        self.clock_clamps = 0
        self.clock_phase_corrections = 0
        self.last_clock_error_ms = 0.0

    def _add_samples(self, n: int) -> None:
        with self._lock:
            self._count_since_report += n
            self.total_samples += n

    def snapshot_count(self, reset: bool = True) -> int:
        with self._lock:
            n = self._count_since_report
            if reset:
                self._count_since_report = 0
            return n

    def run(self) -> None:
        if max30102 is None:
            self.init_error = "could not import repository-local max30102.py"
            print(f"[PPG] ERROR: {self.init_error}", file=sys.stderr, flush=True)
            return

        try:
            # gpio_pin=None: polling FIFO over I2C; no RPi.GPIO interrupt required.
            self.sensor = max30102.MAX30102(gpio_pin=None)
            self.sensor.setup(
                led_mode=0x03,
                sample_rate=int(self.sample_rate_hz),
                pulse_width=118,
                adc_range=4096,
                fifo_average=1,
                fifo_rollover=False,
                fifo_a_full=15,
            )
            print(
                f"[PPG] MAX30102 ready on Pi I2C: red/IR @ {self.sample_rate_hz:g} Hz",
                flush=True,
            )
        except Exception as exc:
            self.init_error = str(exc)
            print(f"[PPG] init failed: {exc}", file=sys.stderr, flush=True)
            return

        sample_period_s = 1.0 / self.sample_rate_hz

        try:
            while not self.stop_event.is_set():
                try:
                    batch = self.sensor.i2c_thread_func(
                        max_batch=32,
                        require_ppg_rdy=False,
                    )
                except Exception as exc:
                    self.read_errors += 1
                    if self.verbose:
                        print(f"[PPG] read error: {exc}", file=sys.stderr, flush=True)
                    time.sleep(max(0.001, self.poll_ms / 1000.0))
                    continue

                if batch:
                    n = len(batch)
                    now_lsl = local_clock()

                    # Candidate timing if this FIFO batch were independently
                    # anchored to its drain time. We use it only for initial
                    # anchoring and detecting a genuine acquisition gap.
                    candidate_first = now_lsl - (n - 1) * sample_period_s

                    if self.last_sample_lsl is None:
                        first_ts = candidate_first
                    else:
                        expected_first = self.last_sample_lsl + sample_period_s
                        gap_s = candidate_first - expected_first
                        self.last_clock_error_ms = gap_s * 1000.0

                        if gap_s > self.reanchor_gap_s:
                            # A real pause / sensor interruption is more likely
                            # than normal FIFO polling jitter. Preserve the gap.
                            first_ts = candidate_first
                            self.clock_reanchors += 1
                            if self.verbose:
                                print(
                                    f"[PPG] clock re-anchor after gap={gap_s*1000.0:.1f} ms",
                                    flush=True,
                                )
                        else:
                            # Phase-lock the persistent sample clock gently to
                            # local_clock(). This removes batch-boundary backward
                            # jumps without allowing long-term drift if the real
                            # MAX30102 oscillator is slightly above/below 200 Hz.
                            max_correction = (
                                PPG_MAX_PHASE_CORRECTION_FRACTION * sample_period_s
                            )
                            correction = PPG_PHASE_ALPHA * gap_s
                            correction = max(
                                -max_correction,
                                min(max_correction, correction),
                            )
                            first_ts = expected_first + correction
                            if abs(correction) > 0.0:
                                self.clock_phase_corrections += 1

                    # Absolute monotonicity guard. Even under a very large negative
                    # phase error, never allow the next XDF timestamp to move back.
                    if self.last_sample_lsl is not None:
                        min_step = max(1e-6, 0.10 * sample_period_s)
                        min_first = self.last_sample_lsl + min_step
                        if first_ts < min_first:
                            first_ts = min_first
                            self.clock_clamps += 1

                    for i, sample in enumerate(batch):
                        # max30102.py returns (time.time(), red, ir); that value is
                        # batch-level, so FIFO order defines within-batch timing.
                        _, red, ir = sample
                        ts_lsl = first_ts + i * sample_period_s
                        self.outlet.push_sample(
                            [float(red), float(ir)],
                            timestamp=ts_lsl,
                            pushthrough=(i == n - 1),
                        )

                    self.last_sample_lsl = first_ts + (n - 1) * sample_period_s
                    self._add_samples(n)

                time.sleep(max(0.0, self.poll_ms / 1000.0))

        finally:
            if self.sensor is not None:
                try:
                    self.sensor.shutdown()
                except Exception:
                    pass


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Publish Arduino BNO085 Protocol V2 plus Raspberry Pi MAX30102 PPG to LSL."
        )
    )
    ap.add_argument("--port", default=SERIAL_PORT)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--prefix", default=STREAM_PREFIX)
    ap.add_argument("--accel-rate", type=float, default=ACCEL_RATE_HZ)
    ap.add_argument("--gyro-rate", type=float, default=GYRO_RATE_HZ)
    ap.add_argument("--quat-rate", type=float, default=QUAT_RATE_HZ)
    ap.add_argument("--ppg-rate", type=float, default=PPG_RATE_HZ)
    ap.add_argument("--ppg-poll-ms", type=float, default=PPG_POLL_MS)
    ap.add_argument(
        "--ppg-reanchor-gap",
        type=float,
        default=PPG_REANCHOR_GAP_S,
        help=(
            "Re-anchor persistent PPG sample clock when reconstructed FIFO "
            "timing is this many seconds late (default: 0.25)."
        ),
    )
    ap.add_argument("--diag-period", type=float, default=DIAG_PERIOD_S)
    ap.add_argument("--clock-calibration", type=float, default=CLOCK_CALIBRATION_S)
    ap.add_argument("--startup-delay", type=float, default=2.0)
    ap.add_argument("--serial-queue", type=int, default=20000)
    ap.add_argument("--no-ppg", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    stop_event = threading.Event()

    print(
        f"[DAQ] Opening {args.port} at {args.baud:,} baud "
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

    # Mega normally resets when the serial connection opens.
    time.sleep(max(0.0, args.startup_delay))
    ser.reset_input_buffer()

    imu_outlets = make_imu_outlets(
        prefix=args.prefix,
        serial_port=args.port,
        accel_rate=args.accel_rate,
        gyro_rate=args.gyro_rate,
        quat_rate=args.quat_rate,
    )

    ppg_outlet: Optional[StreamOutlet] = None
    ppg_worker: Optional[PPGPublisher] = None

    if not args.no_ppg:
        ppg_outlet = make_ppg_outlet(args.prefix, args.ppg_rate)
        ppg_worker = PPGPublisher(
            outlet=ppg_outlet,
            stop_event=stop_event,
            sample_rate_hz=args.ppg_rate,
            poll_ms=args.ppg_poll_ms,
            reanchor_gap_s=args.ppg_reanchor_gap,
            verbose=args.verbose,
        )

    print("[LSL] Published streams:")
    print(f"      {args.prefix}_accel : 3 ch @ nominal {args.accel_rate:g} Hz (ax ay az)")
    print(f"      {args.prefix}_gyro  : 3 ch @ nominal {args.gyro_rate:g} Hz (gx gy gz)")
    print(f"      {args.prefix}_quat  : 4 ch @ nominal {args.quat_rate:g} Hz (qw qx qy qz)")
    if ppg_worker is not None:
        print(f"      {args.prefix}_ppg   : 2 ch @ nominal {args.ppg_rate:g} Hz (red ir)")

    frame_queue: queue.Queue = queue.Queue(maxsize=max(1000, args.serial_queue))
    serial_reader = SerialReader(ser, frame_queue, stop_event)

    serial_reader.start()
    if ppg_worker is not None:
        ppg_worker.start()

    print("[DAQ] Serial reader thread started; waiting for Protocol V2 packets...")

    host_clock = HostMicrosExtender()
    clock_mapper = HostLslClockMapper(args.clock_calibration)
    uart_sequence = SequenceTracker(1 << 32)

    # Hold only the short clock-calibration interval before publishing BNO samples.
    pending: List[Tuple[int, List[float], int]] = []

    imu_counts: Dict[int, int] = {
        SENSOR_ACCEL: 0,
        SENSOR_GYRO: 0,
        SENSOR_QUAT: 0,
    }

    last_diag = time.monotonic()
    last_valid_frame = time.monotonic()

    ring_overflow = 0
    max_queue_depth = 0
    current_queue_depth = 0
    bno_reset_count = 0

    unknown_packets = 0
    unknown_sensors = 0
    latest_bno_sequence: Dict[int, Optional[int]] = {
        SENSOR_ACCEL: None,
        SENSOR_GYRO: None,
        SENSOR_QUAT: None,
    }

    sensor_names = {
        SENSOR_ACCEL: "acc",
        SENSOR_GYRO: "gyro",
        SENSOR_QUAT: "quat",
    }

    try:
        while not stop_event.is_set():
            try:
                frame, receive_lsl_s = frame_queue.get(timeout=0.05)
            except queue.Empty:
                frame = None
                receive_lsl_s = None

            if frame is not None:
                last_valid_frame = time.monotonic()
                uart_sequence.update(frame.packet_sequence)
                host_us64 = host_clock.extend(frame.host_timestamp_us_raw)

                if frame.packet_type == PKT_IMU:
                    if frame.sensor_type not in imu_outlets:
                        unknown_sensors += 1
                    else:
                        expected_n = 4 if frame.sensor_type == SENSOR_QUAT else 3
                        if frame.n_values != expected_n:
                            unknown_sensors += 1
                        else:
                            values = struct.unpack_from("<4f", frame.payload, 0)
                            sample = list(values[:frame.n_values])

                            clock_mapper.observe(host_us64, float(receive_lsl_s))
                            latest_bno_sequence[frame.sensor_type] = frame.sensor_sequence
                            imu_counts[frame.sensor_type] += 1

                            if clock_mapper.frozen:
                                imu_outlets[frame.sensor_type].push_sample(
                                    sample,
                                    timestamp=clock_mapper.host_to_lsl(host_us64),
                                )
                            else:
                                pending.append((frame.sensor_type, sample, host_us64))

                elif frame.packet_type == PKT_STATS:
                    (
                        ring_overflow,
                        max_queue_depth,
                        current_queue_depth,
                        bno_reset_count,
                    ) = struct.unpack_from("<4I", frame.payload, 0)

                else:
                    unknown_packets += 1

            if not clock_mapper.frozen and clock_mapper.ready_to_freeze():
                clock_mapper.freeze()
                print(
                    "[CLOCK] Arduino micros() -> LSL mapping frozen: "
                    f"offset={clock_mapper.offset_s:.6f} s; "
                    f"calibration_samples={clock_mapper.observations}",
                    flush=True,
                )

                for sensor_type, sample, host_us64 in pending:
                    imu_outlets[sensor_type].push_sample(
                        sample,
                        timestamp=clock_mapper.host_to_lsl(host_us64),
                    )
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

                ppg_n = 0
                ppg_errors = 0
                ppg_reanchors = 0
                ppg_clamps = 0
                ppg_phase_corrections = 0
                ppg_clock_error_ms = 0.0
                if ppg_worker is not None:
                    ppg_n = ppg_worker.snapshot_count(reset=True)
                    ppg_errors = ppg_worker.read_errors
                    ppg_reanchors = ppg_worker.clock_reanchors
                    ppg_clamps = ppg_worker.clock_clamps
                    ppg_phase_corrections = ppg_worker.clock_phase_corrections
                    ppg_clock_error_ms = ppg_worker.last_clock_error_ms

                print(
                    f"[LSL] "
                    f"ACC={acc_n:3d}({acc_n / elapsed:6.1f}Hz) "
                    f"GYRO={gyro_n:3d}({gyro_n / elapsed:6.1f}Hz) "
                    f"QUAT={quat_n:3d}({quat_n / elapsed:6.1f}Hz) "
                    + (
                        f"PPG={ppg_n:3d}({ppg_n / elapsed:6.1f}Hz) | "
                        if ppg_worker is not None
                        else "| "
                    )
                    + f"ring={ring_overflow} depth={current_queue_depth}/{max_queue_depth} "
                    f"resets={bno_reset_count} | "
                    f"CRC={serial_reader.parser.crc_errors} "
                    f"UARTdrops={uart_sequence.dropped} "
                    f"framing={serial_reader.parser.framing_bytes_discarded} "
                    f"rxQ={frame_queue.qsize()} qdrop={serial_reader.queue_drops} "
                    f"PPGerr={ppg_errors} "
                    + (
                        f"PPGreanchor={ppg_reanchors} "
                        f"PPGclamp={ppg_clamps} "
                        f"PPGphase={ppg_phase_corrections} "
                        f"PPGclkerr={ppg_clock_error_ms:+.2f}ms"
                        if ppg_worker is not None
                        else ""
                    ),
                    flush=True,
                )

                if args.verbose:
                    print(
                        "[BNO] latest sequence "
                        f"A/G/Q={latest_bno_sequence[SENSOR_ACCEL]}/"
                        f"{latest_bno_sequence[SENSOR_GYRO]}/"
                        f"{latest_bno_sequence[SENSOR_QUAT]} "
                        "(diagnostic only; not interpreted as per-sensor drop counters)",
                        flush=True,
                    )

                if now - last_valid_frame > 2.0:
                    print(
                        "[DAQ] WARNING: no valid Arduino Protocol V2 frames for >2 s",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\n[DAQ] Stopping.", flush=True)

    finally:
        stop_event.set()

        serial_reader.join(timeout=1.0)
        if ppg_worker is not None:
            ppg_worker.join(timeout=2.0)

        try:
            ser.close()
        except Exception:
            pass

    print("[DAQ] Serial port closed.")
    print(
        "[DAQ] Final diagnostics: "
        f"CRC={serial_reader.parser.crc_errors}, "
        f"UARTdrops={uart_sequence.dropped}, "
        f"framing={serial_reader.parser.framing_bytes_discarded}, "
        f"version_errors={serial_reader.parser.version_errors}, "
        f"serial_queue_drops={serial_reader.queue_drops}, "
        f"serial_read_errors={serial_reader.read_errors}, "
        f"unknown_packets={unknown_packets}, "
        f"unknown_sensors={unknown_sensors}, "
        f"ring_overflow={ring_overflow}, "
        f"BNO_resets={bno_reset_count}"
        + (
            f", PPG_read_errors={ppg_worker.read_errors}, "
            f"PPG_reanchors={ppg_worker.clock_reanchors}, "
            f"PPG_clamps={ppg_worker.clock_clamps}, "
            f"PPG_phase_corrections={ppg_worker.clock_phase_corrections}"
            if ppg_worker is not None
            else ""
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

