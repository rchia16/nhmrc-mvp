#!/usr/bin/env python3
"""
bno085_lsl_streamer.py

Runs on a Raspberry Pi connected to a BNO085/BNO08x over I2C.

Publishes the IMU's calibrated, fused, raw, and classifier outputs to Lab
Streaming Layer as one numeric stream. Channels whose report has not produced
data yet are sent as NaN so the stream layout stays stable.

Run:
  python bno085_lsl_streamer.py --reports accelerometer gyroscope magnetometer
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


STABILITY_CODES = {
    "Unknown": 0.0,
    "On Table": 1.0,
    "Stationary": 2.0,
    "Stable": 3.0,
    "In motion": 4.0,
}

ACTIVITY_NAMES = [
    "Unknown",
    "In-Vehicle",
    "On-Bicycle",
    "On-Foot",
    "Still",
    "Tilting",
    "Walking",
    "Running",
    "OnStairs",
]

ACTIVITY_CODES = {name: float(i) for i, name in enumerate(ACTIVITY_NAMES)}


@dataclass(frozen=True)
class ReportSpec:
    name: str
    feature_const: str
    property_name: str
    channel_names: tuple[str, ...]
    default_interval_us: int
    converter: Callable[[Any], list[float]]


MAX_REPORT_RATES_HZ = {
    "accelerometer": 500.0,
    "gyroscope": 400.0,
    "magnetometer": 100.0,
    "linear_acceleration": 400.0,
    "gravity": 400.0,
    "rotation_vector": 400.0,
    "game_rotation_vector": 400.0,
    "geomagnetic_rotation_vector": 90.0,
    "raw_accelerometer": 500.0,
    "raw_gyroscope": 400.0,
    "raw_magnetometer": 100.0,
}


def interval_us_for_hz(rate_hz: float) -> int:
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    return max(1, int(round(1_000_000.0 / rate_hz)))


def report_rate_hz(report: ReportSpec) -> float:
    return 1_000_000.0 / float(report.default_interval_us)


def max_enabled_report_rate_hz(reports: Iterable[ReportSpec]) -> float:
    return max((report_rate_hz(report) for report in reports), default=100.0)


def float_tuple(value: Any, count: int) -> list[float]:
    if value is None:
        return [math.nan] * count
    values = list(value)
    if len(values) != count:
        return [math.nan] * count
    return [float(x) for x in values]


def scalar(value: Any) -> list[float]:
    if value is None:
        return [math.nan]
    return [float(value)]


def boolean(value: Any) -> list[float]:
    if value is None:
        return [math.nan]
    return [1.0 if bool(value) else 0.0]


def stability(value: Any) -> list[float]:
    if value is None:
        return [math.nan]
    return [STABILITY_CODES.get(str(value), math.nan)]


def activity(value: Any) -> list[float]:
    if not isinstance(value, dict):
        return [math.nan] * (1 + len(ACTIVITY_NAMES))

    most_likely = str(value.get("most_likely", "Unknown"))
    channels = [ACTIVITY_CODES.get(most_likely, math.nan)]
    channels.extend(float(value.get(name, math.nan)) for name in ACTIVITY_NAMES)
    return channels


REPORTS = [
    ReportSpec(
        "accelerometer",
        "BNO_REPORT_ACCELEROMETER",
        "acceleration",
        ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["accelerometer"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "gyroscope",
        "BNO_REPORT_GYROSCOPE",
        "gyro",
        ("gyro_x_rps", "gyro_y_rps", "gyro_z_rps"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["gyroscope"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "magnetometer",
        "BNO_REPORT_MAGNETOMETER",
        "magnetic",
        ("mag_x_uT", "mag_y_uT", "mag_z_uT"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["magnetometer"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "linear_acceleration",
        "BNO_REPORT_LINEAR_ACCELERATION",
        "linear_acceleration",
        ("linear_accel_x_mps2", "linear_accel_y_mps2", "linear_accel_z_mps2"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["linear_acceleration"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "gravity",
        "BNO_REPORT_GRAVITY",
        "gravity",
        ("gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["gravity"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "rotation_vector",
        "BNO_REPORT_ROTATION_VECTOR",
        "quaternion",
        ("quat_i", "quat_j", "quat_k", "quat_real"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["rotation_vector"]),
        lambda value: float_tuple(value, 4),
    ),
    ReportSpec(
        "game_rotation_vector",
        "BNO_REPORT_GAME_ROTATION_VECTOR",
        "game_quaternion",
        ("game_quat_i", "game_quat_j", "game_quat_k", "game_quat_real"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["game_rotation_vector"]),
        lambda value: float_tuple(value, 4),
    ),
    ReportSpec(
        "geomagnetic_rotation_vector",
        "BNO_REPORT_GEOMAGNETIC_ROTATION_VECTOR",
        "geomagnetic_quaternion",
        ("geomag_quat_i", "geomag_quat_j", "geomag_quat_k", "geomag_quat_real"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["geomagnetic_rotation_vector"]),
        lambda value: float_tuple(value, 4),
    ),
    ReportSpec(
        "raw_accelerometer",
        "BNO_REPORT_RAW_ACCELEROMETER",
        "raw_acceleration",
        ("raw_accel_x", "raw_accel_y", "raw_accel_z"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["raw_accelerometer"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "raw_gyroscope",
        "BNO_REPORT_RAW_GYROSCOPE",
        "raw_gyro",
        ("raw_gyro_x", "raw_gyro_y", "raw_gyro_z"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["raw_gyroscope"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "raw_magnetometer",
        "BNO_REPORT_RAW_MAGNETOMETER",
        "raw_magnetic",
        ("raw_mag_x", "raw_mag_y", "raw_mag_z"),
        interval_us_for_hz(MAX_REPORT_RATES_HZ["raw_magnetometer"]),
        lambda value: float_tuple(value, 3),
    ),
    ReportSpec(
        "step_counter",
        "BNO_REPORT_STEP_COUNTER",
        "steps",
        ("steps",),
        100_000,
        scalar,
    ),
    ReportSpec(
        "shake_detector",
        "BNO_REPORT_SHAKE_DETECTOR",
        "shake",
        ("shake",),
        20_000,
        boolean,
    ),
    ReportSpec(
        "stability_classifier",
        "BNO_REPORT_STABILITY_CLASSIFIER",
        "stability_classification",
        ("stability_code",),
        100_000,
        stability,
    ),
    ReportSpec(
        "activity_classifier",
        "BNO_REPORT_ACTIVITY_CLASSIFIER",
        "activity_classification",
        (
            "activity_most_likely_code",
            "activity_unknown_conf",
            "activity_in_vehicle_conf",
            "activity_on_bicycle_conf",
            "activity_on_foot_conf",
            "activity_still_conf",
            "activity_tilting_conf",
            "activity_walking_conf",
            "activity_running_conf",
            "activity_on_stairs_conf",
        ),
        100_000,
        activity,
    ),
]


def selected_reports(names: Iterable[str]) -> list[ReportSpec]:
    wanted = set(names)
    if "all" in wanted:
        return REPORTS

    reports = [report for report in REPORTS if report.name in wanted]
    unknown = sorted(wanted - {report.name for report in REPORTS})
    if unknown:
        raise ValueError(f"Unknown report name(s): {', '.join(unknown)}")
    return reports




def make_reset_pin(board: Any, digitalio: Any, pin_name: Optional[str]) -> Any:
    if not pin_name:
        return None

    normalized = str(pin_name).strip().upper().replace("BOARD.", "").replace("PIN", "")
    physical_pin_map = {
        "11": "D17",
        "GPIO0": "D17",  # WiringPi GPIO0 is physical pin 11 / BCM17.
    }
    attr_name = physical_pin_map.get(normalized, normalized)
    if attr_name.isdigit():
        attr_name = f"D{attr_name}"
    if not hasattr(board, attr_name):
        raise ValueError(f"Unknown reset pin {pin_name!r}; use a Blinka pin name like D17")
    return digitalio.DigitalInOut(getattr(board, attr_name))



def pulse_reset_pin(reset_pin: Any, hold_s: float = 0.05, boot_s: float = 0.75) -> None:
    if reset_pin is None:
        return
    reset_pin.switch_to_output(value=True)
    time.sleep(0.01)
    reset_pin.value = False
    time.sleep(hold_s)
    reset_pin.value = True
    time.sleep(boot_s)

def require_modules():
    try:
        import board
        import busio
        import digitalio
        from adafruit_bno08x.i2c import BNO08X_I2C
        import adafruit_bno08x as bno08x
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. On the Raspberry Pi install:\n"
            "  pip install adafruit-blinka adafruit-circuitpython-bno08x pylsl\n"
            f"Import error: {exc}"
        ) from exc

    return board, busio, digitalio, BNO08X_I2C, bno08x, StreamInfo, StreamOutlet, local_clock




def create_bno085_i2c(
    board: Any,
    busio: Any,
    digitalio: Any,
    BNO08X_I2C: Any,
    address: int,
    i2c_frequency: int,
    reset_pin_name: Optional[str],
    debug: bool,
    attempts: int = 5,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_frequency)
        reset_pin = make_reset_pin(board, digitalio, reset_pin_name)
        try:
            pulse_reset_pin(reset_pin)
            return BNO08X_I2C(i2c, address=address, reset=reset_pin, debug=debug)
        except Exception as exc:
            last_error = exc
            try:
                i2c.deinit()
            except Exception:
                pass
            if attempt >= attempts:
                raise
            print(f"BNO085 init failed on attempt {attempt}/{attempts}: {exc}; retrying", file=sys.stderr)
            time.sleep(0.75)

    if last_error is not None:
        raise last_error
    raise RuntimeError("BNO085 init failed")

def add_lsl_metadata(info: Any, channel_names: list[str], enabled_names: list[str]) -> None:
    info.desc().append_child_value("manufacturer", "Bosch/Sense BNO085 via Adafruit BNO08x")
    info.desc().append_child_value("source", "Raspberry Pi I2C")
    reports = info.desc().append_child("reports")
    for name in enabled_names:
        reports.append_child_value("report", name)

    channels = info.desc().append_child("channels")
    for name in channel_names:
        channel = channels.append_child("channel")
        channel.append_child_value("label", name)
        channel.append_child_value("unit", infer_unit(name))
        channel.append_child_value("type", "IMU")


def infer_unit(name: str) -> str:
    if name.endswith("_mps2"):
        return "m/s^2"
    if name.endswith("_rps"):
        return "rad/s"
    if name.endswith("_uT"):
        return "uT"
    if name.startswith("quat_") or "_quat_" in name:
        return "unitless"
    if name.endswith("_conf"):
        return "percent"
    return "unitless"


def open_csv(path: Optional[str], channel_names: list[str]):
    if not path:
        return None, None

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(["lsl_timestamp", "unix_time_ns", *channel_names])
    handle.flush()
    return handle, writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream BNO085/BNO08x I2C IMU channels to Lab Streaming Layer."
    )
    parser.add_argument("--stream-name", default="BNO085")
    parser.add_argument("--stream-type", default="IMU")
    parser.add_argument("--source-id", default="bno085_rpi_i2c")
    parser.add_argument("--rate-hz", type=float, default=None, help="LSL output rate. Defaults to the fastest enabled report rate.")
    parser.add_argument("--i2c-frequency", type=int, default=100000)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x4A)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reset-pin", default=None, help="Optional BNO085 reset pin, e.g. D17 for physical pin 11.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=["accelerometer", "gyroscope", "magnetometer", "rotation_vector"],
        help="Report groups to stream, or 'all'.",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV mirror path.")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = selected_reports(args.reports)
    if args.rate_hz is None:
        args.rate_hz = max_enabled_report_rate_hz(reports)
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be positive")
    channel_names = [name for report in reports for name in report.channel_names]

    board, busio, digitalio, BNO08X_I2C, bno08x, StreamInfo, StreamOutlet, local_clock = require_modules()

    bno = create_bno085_i2c(
        board,
        busio,
        digitalio,
        BNO08X_I2C,
        address=args.address,
        i2c_frequency=args.i2c_frequency,
        reset_pin_name=args.reset_pin,
        debug=args.debug,
    )

    enabled_reports: list[ReportSpec] = []
    for report in reports:
        feature_id = getattr(bno08x, report.feature_const, None)
        if feature_id is None:
            print(f"Skipping {report.name}: missing {report.feature_const}", file=sys.stderr)
            continue
        try:
            bno.enable_feature(feature_id, report.default_interval_us)
        except Exception as exc:
            print(f"Skipping {report.name}: could not enable report: {exc}", file=sys.stderr)
            continue
        enabled_reports.append(report)

    if not enabled_reports:
        raise SystemExit("No BNO085 reports were enabled.")

    channel_names = [name for report in enabled_reports for name in report.channel_names]
    info = StreamInfo(
        args.stream_name,
        args.stream_type,
        len(channel_names),
        args.rate_hz,
        "float32",
        args.source_id,
    )
    add_lsl_metadata(info, channel_names, [report.name for report in enabled_reports])
    outlet = StreamOutlet(info)
    csv_handle, csv_writer = open_csv(args.csv, channel_names)

    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print(f"\nReceived signal {signum}, stopping...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    period_s = 1.0 / args.rate_hz
    next_sample_time = time.monotonic()
    n = 0

    print(
        f"Streaming {len(channel_names)} channels from {len(enabled_reports)} BNO085 reports "
        f"to LSL stream '{args.stream_name}' at {args.rate_hz:g} Hz."
    )

    try:
        while running:
            sample: list[float] = []
            for report in enabled_reports:
                try:
                    value = getattr(bno, report.property_name)
                    sample.extend(report.converter(value))
                except Exception as exc:
                    print(f"Read failed for {report.name}: {exc}", file=sys.stderr)
                    sample.extend([math.nan] * len(report.channel_names))

            timestamp = local_clock()
            outlet.push_sample(sample, timestamp)

            if csv_writer is not None:
                csv_writer.writerow([timestamp, time.time_ns(), *sample])
                if n % max(1, args.print_every) == 0:
                    csv_handle.flush()

            n += 1
            if args.print_every > 0 and n % args.print_every == 0:
                finite_count = sum(1 for x in sample if math.isfinite(x))
                print(f"samples={n} finite_channels={finite_count}/{len(sample)}")

            next_sample_time += period_s
            sleep_s = next_sample_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_sample_time = time.monotonic()

    finally:
        if csv_handle is not None:
            csv_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
