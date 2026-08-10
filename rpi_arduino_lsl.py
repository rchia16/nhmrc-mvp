#!/usr/bin/env python3
"""
rpi_arduino_ppg_lsl.py

Raspberry Pi acquisition bridge for the NHMRC MVP hardware.

Inputs
------
1. BNO085 IMU from Arduino Mega over USB serial.

   Expected Arduino record:
       I,t_us,accel_seq,gyro_seq,ax,ay,az,gx,gy,gz

   This matches bno085_usb_stream.ino:
       accel = 200 Hz
       gyro  = 200 Hz
       serial = 1,000,000 baud

2. MAX30102 PPG connected directly to Raspberry Pi I2C bus 1.

   Wiring from the nhmrc-mvp README:
       PPG SDA -> Pi SDA1 / BCM2, physical pin 3
       PPG SCL -> Pi SCL1 / BCM3, physical pin 5
       PPG INT -> Pi BCM25 / D25, physical pin 22

   The current reader polls the FIFO, so the PPG INT pin is not required
   by this script even if it remains connected.

Outputs
-------
Exactly three Lab Streaming Layer streams:

    name="ppg"   type="PPG"   channels=[red, ir]   nominal rate=200 Hz
    name="acc"   type="ACC"   channels=[x, y, z]   nominal rate=200 Hz
    name="gyr"   type="GYR"   channels=[x, y, z]   nominal rate=200 Hz

Timing
------
- PPG timestamps are assigned in the Raspberry Pi pylsl.local_clock() domain.
  FIFO batches are reconstructed at the configured 200 Hz sample interval.
- Arduino timestamps are device microseconds. A low-latency serial clock
  mapper continuously maps that device time into the Raspberry Pi LSL clock
  domain. Acceleration and gyroscope from the same Arduino record therefore
  receive the same LSL timestamp.

Dependencies
------------
The repository's rpi environment already provides pylsl, smbus2 and
rpi-lgpio. This script additionally needs pyserial:

    python -m pip install pyserial

The repository's max30102.py must be importable, normally by running this
script from the nhmrc-mvp repository root.

Examples
--------
    python rpi_arduino_ppg_lsl.py

    python rpi_arduino_ppg_lsl.py \
        --serial-port /dev/ttyACM0 \
        --rate-print

Check PPG first:
    i2cdetect -y 1

Expected address:
    0x57
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial
from pylsl import StreamInfo, StreamOutlet, local_clock

import max30102


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_SERIAL_BAUD = 1_000_000

DEFAULT_IMU_RATE_HZ = 200.0
DEFAULT_PPG_RATE_HZ = 200.0

DEFAULT_PPG_I2C_BUS = 1
DEFAULT_PPG_ADDRESS = 0x57
DEFAULT_PPG_POLL_MS = 5.0

ACC_STREAM_NAME = "acc"
GYR_STREAM_NAME = "gyr"
PPG_STREAM_NAME = "ppg"


# =============================================================================
# Utility classes
# =============================================================================

@dataclass
class RateCounter:
    label: str
    enabled: bool = False
    interval_s: float = 2.0

    def __post_init__(self) -> None:
        self._count = 0
        self._t0 = time.monotonic()

    def add(self, n: int = 1) -> None:
        self._count += int(n)

        if not self.enabled:
            return

        now = time.monotonic()
        dt = now - self._t0

        if dt >= self.interval_s:
            hz = self._count / dt if dt > 0 else 0.0
            print(
                f"[RATE][{self.label}] {hz:.1f} Hz "
                f"({self._count} samples / {dt:.2f} s)",
                flush=True,
            )
            self._count = 0
            self._t0 = now


class SequenceTracker:
    """Track 8-bit BNO report sequence gaps."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.previous: Optional[int] = None
        self.dropped = 0

    def update(self, sequence: int) -> int:
        sequence &= 0xFF

        if self.previous is None:
            self.previous = sequence
            return 0

        expected = (self.previous + 1) & 0xFF
        gap = (sequence - expected) & 0xFF

        # Ignore large backwards/restart discontinuities.
        if gap >= 128:
            gap = 0

        self.dropped += gap
        self.previous = sequence
        return gap


class ArduinoClockMapper:
    """
    Map Arduino microsecond timestamps to pylsl.local_clock().

    Serial transport latency is positive and variable. Within each one-second
    Arduino-time block, the minimum observed arrival offset is used as the best
    available estimate of the clock offset. The estimate is then smoothed
    between blocks so slow Arduino/Pi clock drift is followed without injecting
    normal USB scheduling jitter into the sample timestamps.
    """

    def __init__(self, alpha: float = 0.20) -> None:
        self.alpha = float(alpha)

        self.offset_s: Optional[float] = None

        self._block_id: Optional[int] = None
        self._block_min_offset: Optional[float] = None

        self._last_device_us: Optional[int] = None
        self._last_output_ts: Optional[float] = None

        self.reset_count = 0

    def reset(self) -> None:
        self.offset_s = None
        self._block_id = None
        self._block_min_offset = None
        self._last_device_us = None
        self._last_output_ts = None
        self.reset_count += 1

    def map(self, device_us: int, arrival_lsl_s: float) -> float:
        # Detect an Arduino reboot/device-clock reset.
        if (
            self._last_device_us is not None
            and device_us + 1_000_000 < self._last_device_us
        ):
            print(
                "[CLOCK] Arduino timestamp moved backwards; "
                "resetting serial clock mapper",
                flush=True,
            )
            self.reset()

        self._last_device_us = device_us

        device_s = float(device_us) * 1e-6
        observed_offset = float(arrival_lsl_s) - device_s
        block_id = int(device_s)

        if self.offset_s is None:
            self.offset_s = observed_offset
            self._block_id = block_id
            self._block_min_offset = observed_offset

        elif block_id == self._block_id:
            if (
                self._block_min_offset is None
                or observed_offset < self._block_min_offset
            ):
                self._block_min_offset = observed_offset

            # A newly observed lower-latency path is useful immediately.
            if observed_offset < self.offset_s:
                self.offset_s = observed_offset

        else:
            # Finalize the previous block's minimum-latency observation.
            if self._block_min_offset is not None:
                target = self._block_min_offset
                self.offset_s = (
                    (1.0 - self.alpha) * self.offset_s
                    + self.alpha * target
                )

            self._block_id = block_id
            self._block_min_offset = observed_offset

        mapped = device_s + self.offset_s

        # Keep timestamps strictly monotonic even if the offset estimate makes
        # a very small correction.
        if self._last_output_ts is not None and mapped <= self._last_output_ts:
            mapped = self._last_output_ts + 1e-6

        self._last_output_ts = mapped
        return mapped


class PPGTimestampReconstructor:
    """
    Reconstruct MAX30102 timestamps on a continuous sample clock.

    First batch:
        Anchor to the Raspberry Pi LSL clock.

    Subsequent batches:
        Continue exactly at 1 / sample_rate_hz.

    This guarantees:
        - strictly increasing timestamps
        - exactly 5 ms spacing at 200 Hz
        - no backwards re-anchoring
    """

    def __init__(self, sample_rate_hz: float) -> None:
        self.rate_hz = float(sample_rate_hz)
        self.period_s = 1.0 / self.rate_hz
        self.last_timestamp: Optional[float] = None

        # Diagnostics only.
        self.max_anchor_error_s = 0.0
        self.anchor_warning_count = 0

    def timestamps(
        self,
        n: int,
        batch_anchor_lsl_s: float,
    ) -> list[float]:

        if n <= 0:
            return []

        # Estimate where the oldest sample in this FIFO batch would lie
        # according to the Pi clock.
        candidate_first = (
            float(batch_anchor_lsl_s)
            - (n - 1) * self.period_s
        )

        if self.last_timestamp is None:
            # Only the first batch establishes the absolute time origin.
            first = candidate_first

        else:
            # Thereafter follow the MAX30102 sample clock exactly.
            expected_first = (
                self.last_timestamp
                + self.period_s
            )

            # Compare against the Pi/FIFO timing, but DO NOT use that noisy
            # estimate to move timestamps backwards.
            anchor_error = (
                candidate_first
                - expected_first
            )

            self.max_anchor_error_s = max(
                self.max_anchor_error_s,
                abs(anchor_error),
            )

            # Diagnostic only. Does not alter timestamps.
            if abs(anchor_error) > 0.050:
                self.anchor_warning_count += 1

                print(
                    "[PPG][CLOCK] FIFO anchor differs from "
                    f"sample clock by {anchor_error * 1000:.1f} ms; "
                    "keeping continuous 200 Hz timestamps",
                    flush=True,
                )

            first = expected_first

        timestamps = [
            first + i * self.period_s
            for i in range(n)
        ]

        self.last_timestamp = timestamps[-1]

        return timestamps


# =============================================================================
# LSL stream creation
# =============================================================================

def append_channel_metadata(
    info: StreamInfo,
    channels: list[tuple[str, str]],
) -> None:
    root = info.desc().append_child("channels")

    for label, unit in channels:
        ch = root.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)


def create_lsl_outlets(
    imu_rate_hz: float,
    ppg_rate_hz: float,
) -> tuple[StreamOutlet, StreamOutlet, StreamOutlet]:
    host = socket.gethostname()

    acc_info = StreamInfo(
        ACC_STREAM_NAME,
        "ACC",
        3,
        imu_rate_hz,
        "float32",
        f"nhmrc-{host}-acc",
    )
    append_channel_metadata(
        acc_info,
        [
            ("x", "m/s^2"),
            ("y", "m/s^2"),
            ("z", "m/s^2"),
        ],
    )
    acc_info.desc().append_child_value("sensor", "BNO085")
    acc_info.desc().append_child_value("transport", "Arduino Mega USB serial")
    acc_info.desc().append_child_value(
        "timestamp_source",
        "Arduino micros64 mapped to pylsl.local_clock",
    )

    gyr_info = StreamInfo(
        GYR_STREAM_NAME,
        "GYR",
        3,
        imu_rate_hz,
        "float32",
        f"nhmrc-{host}-gyr",
    )
    append_channel_metadata(
        gyr_info,
        [
            ("x", "rad/s"),
            ("y", "rad/s"),
            ("z", "rad/s"),
        ],
    )
    gyr_info.desc().append_child_value("sensor", "BNO085")
    gyr_info.desc().append_child_value("transport", "Arduino Mega USB serial")
    gyr_info.desc().append_child_value(
        "timestamp_source",
        "Arduino micros64 mapped to pylsl.local_clock",
    )

    ppg_info = StreamInfo(
        PPG_STREAM_NAME,
        "PPG",
        2,
        ppg_rate_hz,
        "int32",
        f"nhmrc-{host}-ppg",
    )
    append_channel_metadata(
        ppg_info,
        [
            ("red", "ADC counts"),
            ("ir", "ADC counts"),
        ],
    )
    ppg_info.desc().append_child_value("sensor", "MAX30102")
    ppg_info.desc().append_child_value("transport", "Raspberry Pi I2C bus 1")
    ppg_info.desc().append_child_value("i2c_address", "0x57")
    ppg_info.desc().append_child_value(
        "timestamp_source",
        "Raspberry Pi pylsl.local_clock with FIFO interval reconstruction",
    )

    # chunk_size=0 lets liblsl choose buffering while we provide explicit
    # timestamps for every sample.
    return (
        StreamOutlet(acc_info, chunk_size=0, max_buffered=60),
        StreamOutlet(gyr_info, chunk_size=0, max_buffered=60),
        StreamOutlet(ppg_info, chunk_size=0, max_buffered=60),
    )


# =============================================================================
# Arduino IMU worker
# =============================================================================

class ArduinoIMUWorker:
    def __init__(
        self,
        port: str,
        baud: int,
        startup_delay_s: float,
        acc_outlet: StreamOutlet,
        gyr_outlet: StreamOutlet,
        rate_print: bool,
        debug: bool,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.startup_delay_s = float(startup_delay_s)

        self.acc_outlet = acc_outlet
        self.gyr_outlet = gyr_outlet

        self.rate_acc = RateCounter("acc", enabled=rate_print)
        self.rate_gyr = RateCounter("gyr", enabled=rate_print)

        self.acc_seq = SequenceTracker("acc")
        self.gyr_seq = SequenceTracker("gyr")

        self.mapper = ArduinoClockMapper()
        self.debug = bool(debug)

        self.bad_lines = 0

    def run(self, stop_event: threading.Event) -> None:
        print(
            f"[ARDUINO] opening {self.port} at {self.baud:,} baud",
            flush=True,
        )

        try:
            ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.10,
            )
        except Exception as exc:
            print(f"[ARDUINO] failed to open serial port: {exc}", flush=True)
            stop_event.set()
            return

        try:
            # Mega commonly resets when USB serial is opened.
            time.sleep(max(0.0, self.startup_delay_s))
            ser.reset_input_buffer()

            print("[ARDUINO] reading BNO085 IMU stream", flush=True)

            while not stop_event.is_set():
                raw = ser.readline()
                arrival = local_clock()

                if not raw:
                    continue

                try:
                    line = raw.decode("ascii", errors="strict").strip()
                except UnicodeDecodeError:
                    self.bad_lines += 1
                    continue

                if not line:
                    continue

                if line.startswith("#"):
                    if self.debug:
                        print(f"[ARDUINO] {line}", flush=True)
                    continue

                fields = line.split(",")

                if fields[0] != "I" or len(fields) != 10:
                    self.bad_lines += 1
                    if self.debug:
                        print(f"[ARDUINO] ignored: {line}", flush=True)
                    continue

                try:
                    device_us = int(fields[1])
                    accel_sequence = int(fields[2])
                    gyro_sequence = int(fields[3])

                    ax = float(fields[4])
                    ay = float(fields[5])
                    az = float(fields[6])

                    gx = float(fields[7])
                    gy = float(fields[8])
                    gz = float(fields[9])

                except ValueError:
                    self.bad_lines += 1
                    continue

                timestamp = self.mapper.map(device_us, arrival)

                self.acc_seq.update(accel_sequence)
                self.gyr_seq.update(gyro_sequence)

                self.acc_outlet.push_sample(
                    [ax, ay, az],
                    timestamp=timestamp,
                )

                self.gyr_outlet.push_sample(
                    [gx, gy, gz],
                    timestamp=timestamp,
                )

                self.rate_acc.add()
                self.rate_gyr.add()

        except Exception as exc:
            print(f"[ARDUINO] stream error: {exc}", flush=True)
            stop_event.set()

        finally:
            try:
                ser.close()
            except Exception:
                pass

            print(
                "[ARDUINO] stopped "
                f"drop_acc={self.acc_seq.dropped} "
                f"drop_gyr={self.gyr_seq.dropped} "
                f"bad_lines={self.bad_lines}",
                flush=True,
            )


# =============================================================================
# Raspberry Pi MAX30102 worker
# =============================================================================

class MAX30102PPGWorker:
    def __init__(
        self,
        bus: int,
        address: int,
        sample_rate_hz: float,
        poll_ms: float,
        outlet: StreamOutlet,
        rate_print: bool,
        debug: bool,
    ) -> None:
        self.bus = int(bus)
        self.address = int(address)
        self.sample_rate_hz = float(sample_rate_hz)
        self.poll_ms = float(poll_ms)
        self.outlet = outlet
        self.rate = RateCounter("ppg", enabled=rate_print)
        self.debug = bool(debug)

        self.sensor: Optional[max30102.MAX30102] = None
        self.timestamps = PPGTimestampReconstructor(self.sample_rate_hz)

        self.read_errors = 0
        self.samples_total = 0

    def start(self) -> None:
        print(
            f"[PPG] opening MAX30102 on I2C bus {self.bus}, "
            f"address 0x{self.address:02X}",
            flush=True,
        )

        # README/current repo streamer polls FIFO and does not use PPG INT.
        self.sensor = max30102.MAX30102(
            channel=self.bus,
            address=self.address,
            gpio_pin=None,
            led_mode=0x03,
        )

        # Match the current rpi_lsl_imu_ppg.py acquisition settings:
        # red + IR, 200 Hz, no FIFO averaging.
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
            f"[PPG] MAX30102 configured at {self.sample_rate_hz:g} Hz "
            "(red + IR, FIFO polling)",
            flush=True,
        )

    def run(self, stop_event: threading.Event) -> None:
        try:
            self.start()
        except Exception as exc:
            print(f"[PPG] initialization failed: {exc}", flush=True)
            stop_event.set()
            return

        try:
            while not stop_event.is_set():
                batch = None

                # Anchor immediately before the FIFO status/pointer snapshot.
                batch_anchor = local_clock()

                try:
                    if self.sensor is not None:
                        batch = self.sensor.i2c_thread_func(
                            max_batch=32,
                            require_ppg_rdy=False,
                        )
                except Exception as exc:
                    self.read_errors += 1
                    if self.debug or self.read_errors <= 5:
                        print(f"[PPG] FIFO read error: {exc}", flush=True)

                if batch:
                    n = len(batch)
                    timestamps = self.timestamps.timestamps(
                        n,
                        batch_anchor,
                    )

                    for (_, red, ir), ts in zip(batch, timestamps):
                        self.outlet.push_sample(
                            [int(red), int(ir)],
                            timestamp=ts,
                        )

                    self.samples_total += n
                    self.rate.add(n)

                time.sleep(self.poll_ms / 1000.0)

        finally:
            if self.sensor is not None:
                try:
                    self.sensor.shutdown()
                except Exception:
                    pass

            print(
                f"[PPG] stopped samples={self.samples_total} "
                f"read_errors={self.read_errors}",
                flush=True,
            )


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Publish Arduino BNO085 acceleration/gyroscope and "
            "Raspberry Pi MAX30102 PPG as three LSL streams: "
            "acc, gyr and ppg."
        )
    )

    ap.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help=f"Arduino USB serial device (default: {DEFAULT_SERIAL_PORT})",
    )
    ap.add_argument(
        "--serial-baud",
        type=int,
        default=DEFAULT_SERIAL_BAUD,
        help=f"Arduino serial baud (default: {DEFAULT_SERIAL_BAUD})",
    )
    ap.add_argument(
        "--arduino-startup-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after opening USB serial (default: 2.0)",
    )

    ap.add_argument(
        "--imu-rate",
        type=float,
        default=DEFAULT_IMU_RATE_HZ,
        help=f"Nominal acc/gyr LSL rate (default: {DEFAULT_IMU_RATE_HZ:g})",
    )

    ap.add_argument(
        "--ppg-bus",
        type=int,
        default=DEFAULT_PPG_I2C_BUS,
        help=f"Raspberry Pi I2C bus (default: {DEFAULT_PPG_I2C_BUS})",
    )
    ap.add_argument(
        "--ppg-address",
        type=lambda x: int(x, 0),
        default=DEFAULT_PPG_ADDRESS,
        help="MAX30102 I2C address (default: 0x57)",
    )
    ap.add_argument(
        "--ppg-rate",
        type=float,
        default=DEFAULT_PPG_RATE_HZ,
        help=f"MAX30102/LSL PPG rate (default: {DEFAULT_PPG_RATE_HZ:g})",
    )
    ap.add_argument(
        "--ppg-poll-ms",
        type=float,
        default=DEFAULT_PPG_POLL_MS,
        help=f"PPG FIFO polling interval ms (default: {DEFAULT_PPG_POLL_MS:g})",
    )

    ap.add_argument(
        "--rate-print",
        action="store_true",
        help="Print measured acc/gyr/ppg publication rates",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print Arduino diagnostics and PPG read errors",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.imu_rate <= 0:
        raise ValueError("--imu-rate must be > 0")
    if args.ppg_rate <= 0:
        raise ValueError("--ppg-rate must be > 0")
    if args.ppg_poll_ms <= 0:
        raise ValueError("--ppg-poll-ms must be > 0")

    print("[LSL] creating streams:", flush=True)
    print(
        f"      acc : 3 channels @ {args.imu_rate:g} Hz "
        "(x,y,z m/s^2)",
        flush=True,
    )
    print(
        f"      gyr : 3 channels @ {args.imu_rate:g} Hz "
        "(x,y,z rad/s)",
        flush=True,
    )
    print(
        f"      ppg : 2 channels @ {args.ppg_rate:g} Hz "
        "(red,ir ADC counts)",
        flush=True,
    )

    acc_outlet, gyr_outlet, ppg_outlet = create_lsl_outlets(
        imu_rate_hz=args.imu_rate,
        ppg_rate_hz=args.ppg_rate,
    )

    stop_event = threading.Event()

    def request_stop(signum=None, frame=None) -> None:
        if not stop_event.is_set():
            print("\n[MAIN] stopping...", flush=True)
            stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    imu_worker = ArduinoIMUWorker(
        port=args.serial_port,
        baud=args.serial_baud,
        startup_delay_s=args.arduino_startup_delay,
        acc_outlet=acc_outlet,
        gyr_outlet=gyr_outlet,
        rate_print=args.rate_print,
        debug=args.debug,
    )

    ppg_worker = MAX30102PPGWorker(
        bus=args.ppg_bus,
        address=args.ppg_address,
        sample_rate_hz=args.ppg_rate,
        poll_ms=args.ppg_poll_ms,
        outlet=ppg_outlet,
        rate_print=args.rate_print,
        debug=args.debug,
    )

    imu_thread = threading.Thread(
        target=imu_worker.run,
        args=(stop_event,),
        name="arduino-bno085",
        daemon=True,
    )

    ppg_thread = threading.Thread(
        target=ppg_worker.run,
        args=(stop_event,),
        name="max30102-ppg",
        daemon=True,
    )

    imu_thread.start()
    ppg_thread.start()

    print(
        "[LSL] outlets active: ppg, acc, gyr",
        flush=True,
    )

    try:
        while not stop_event.is_set():
            # If either acquisition thread dies unexpectedly, stop the other.
            if not imu_thread.is_alive():
                print("[MAIN] Arduino IMU thread stopped", flush=True)
                stop_event.set()
                break

            if not ppg_thread.is_alive():
                print("[MAIN] PPG thread stopped", flush=True)
                stop_event.set()
                break

            time.sleep(0.25)

    finally:
        stop_event.set()
        imu_thread.join(timeout=2.0)
        ppg_thread.join(timeout=2.0)

    print(
        "[MAIN] final diagnostics: "
        f"drop_acc={imu_worker.acc_seq.dropped}, "
        f"drop_gyr={imu_worker.gyr_seq.dropped}, "
        f"serial_bad_lines={imu_worker.bad_lines}, "
        f"ppg_read_errors={ppg_worker.read_errors}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
