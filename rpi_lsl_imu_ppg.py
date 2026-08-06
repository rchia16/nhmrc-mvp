#!/usr/bin/env python3
"""
Publish only IMU and PPG data to Lab Streaming Layer.

Streams:
  NHMRC_IMU: utc_unix_s, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
  NHMRC_PPG: utc_unix_s, red, ir

The LSL timestamp is set from pylsl.local_clock(); each sample also carries an
explicit UTC Unix timestamp in seconds.
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import max30102
import pyrealsense2 as rs
from pylsl import StreamInfo, StreamOutlet, local_clock
from RPi import GPIO

from config import deep_get, load_config
from imu_reader import IMUReader


@dataclass
class RateCounter:
    label: str
    interval_s: float = 2.0
    enabled: bool = False

    def __post_init__(self):
        self._count = 0
        self._t0 = time.time()

    def add(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self._count += int(n)
        now = time.time()
        dt = now - self._t0
        if dt >= self.interval_s:
            rate = self._count / dt if dt > 0 else 0.0
            print(f"[LSL][{self.label}] {rate:.1f} Hz ({self._count} samples / {dt:.2f} s)")
            self._count = 0
            self._t0 = now


def _append_channels(info: StreamInfo, labels: Iterable[str]) -> None:
    channels = info.desc().append_child("channels")
    for label in labels:
        channels.append_child("channel").append_child_value("label", str(label))


class LSLIMUPublisher:
    CHANNELS = (
        "utc_unix_s",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )

    def __init__(self, name: str, source_id: str, accel_hz: int, gyro_hz: int, poll_hz: float, rate_print: bool):
        info = StreamInfo(name, "IMU", len(self.CHANNELS), 0.0, "float64", source_id)
        _append_channels(info, self.CHANNELS)
        info.desc().append_child_value("clock", "sample channel 0 is UTC Unix seconds")

        self.outlet = StreamOutlet(info)
        self.accel_hz = int(accel_hz)
        self.gyro_hz = int(gyro_hz)
        self.poll_hz = float(poll_hz)
        self.rate = RateCounter("IMU", enabled=rate_print)
        self.reader: Optional[IMUReader] = None
        self._last_accel_seq = -1
        self._last_gyro_seq = -1

    def start(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) < 1:
            raise RuntimeError("No RealSense device found for IMU streaming.")
        self.reader = IMUReader(devices[0], accel_hz=self.accel_hz, gyro_hz=self.gyro_hz)
        print(f"[LSL][IMU] Outlet ready: accel@{self.accel_hz}Hz gyro@{self.gyro_hz}Hz")

    def stop(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader = None

    def run(self, stop_evt: threading.Event) -> None:
        self.start()
        sleep_s = 1.0 / max(1.0, self.poll_hz)
        try:
            while not stop_evt.is_set():
                sample = self.reader.get_latest_timestamped() if self.reader is not None else None
                if sample is None:
                    time.sleep(sleep_s)
                    continue

                accel_seq = int(sample["accel_seq"])
                gyro_seq = int(sample["gyro_seq"])
                if accel_seq == self._last_accel_seq and gyro_seq == self._last_gyro_seq:
                    time.sleep(sleep_s)
                    continue

                accel = sample["accel"]
                gyro = sample["gyro"]
                utc = max(
                    float(sample["accel_utc"] or 0.0),
                    float(sample["gyro_utc"] or 0.0),
                ) or time.time()

                row = [
                    utc,
                    *(accel if accel is not None else (math.nan, math.nan, math.nan)),
                    *(gyro if gyro is not None else (math.nan, math.nan, math.nan)),
                ]
                self.outlet.push_sample(row, timestamp=local_clock())
                self._last_accel_seq = accel_seq
                self._last_gyro_seq = gyro_seq
                self.rate.add()
                time.sleep(sleep_s)
        finally:
            self.stop()


class LSLPPGPublisher:
    CHANNELS = ("utc_unix_s", "red", "ir")

    def __init__(self, name: str, source_id: str, poll_sleep_ms: float, rate_print: bool):
        info = StreamInfo(name, "PPG", len(self.CHANNELS), 200.0, "float64", source_id)
        _append_channels(info, self.CHANNELS)
        info.desc().append_child_value("clock", "sample channel 0 is UTC Unix seconds")

        self.outlet = StreamOutlet(info)
        self.poll_sleep_ms = float(poll_sleep_ms)
        self.rate = RateCounter("PPG", enabled=rate_print)
        self.sensor: Optional[max30102.MAX30102] = None

    def start(self) -> None:
        self.sensor = max30102.MAX30102()
        self.sensor.setup(
            led_mode=0x03,
            sample_rate=200,
            pulse_width=118,
            adc_range=4096,
            fifo_average=1,
            fifo_rollover=False,
            fifo_a_full=15,
        )
        print("[LSL][PPG] Outlet ready: MAX30102 red/IR")

    def stop(self) -> None:
        try:
            GPIO.cleanup()
        except Exception:
            pass
        self.sensor = None

    def run(self, stop_evt: threading.Event) -> None:
        self.start()
        try:
            while not stop_evt.is_set():
                batch = None
                try:
                    batch = self.sensor.i2c_thread_func(max_batch=32, require_ppg_rdy=False) if self.sensor else None
                except Exception as e:
                    print(f"[LSL][PPG] read error: {e}")

                if batch:
                    for utc, red, ir in batch:
                        self.outlet.push_sample([float(utc), float(red), float(ir)], timestamp=local_clock())
                    self.rate.add(len(batch))

                time.sleep(self.poll_sleep_ms / 1000.0)
        finally:
            self.stop()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Publish only IMU and PPG to LSL.")
    ap.add_argument("--config", default="./streaming_config.yaml")
    ap.add_argument("--imu-stream-name", default=None)
    ap.add_argument("--ppg-stream-name", default=None)
    ap.add_argument("--source-id-prefix", default=None)
    ap.add_argument("--imu-poll-hz", type=float, default=None)
    ap.add_argument("--accel-hz", type=int, default=None)
    ap.add_argument("--gyro-hz", type=int, default=None)
    ap.add_argument("--ppg-poll-sleep-ms", type=float, default=None)
    ap.add_argument("--rate-print", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    imu_stream_name = args.imu_stream_name or deep_get(cfg, "lsl.imu_stream_name", "NHMRC_IMU")
    ppg_stream_name = args.ppg_stream_name or deep_get(cfg, "lsl.ppg_stream_name", "NHMRC_PPG")
    source_id_prefix = args.source_id_prefix or deep_get(cfg, "lsl.source_id_prefix", "nhmrc")
    imu_poll_hz = args.imu_poll_hz or float(deep_get(cfg, "lsl.imu_poll_hz", 500.0))
    accel_hz = args.accel_hz or int(deep_get(cfg, "lsl.accel_hz", 250))
    gyro_hz = args.gyro_hz or int(deep_get(cfg, "lsl.gyro_hz", 400))
    ppg_poll_sleep_ms = args.ppg_poll_sleep_ms or float(deep_get(cfg, "ppg.poll_sleep_ms", 5.0))
    rate_print = bool(args.rate_print or deep_get(cfg, "lsl.rate_print", False))

    stop_evt = threading.Event()

    def _sig_handler(_sig, _frame):
        stop_evt.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    imu_pub = LSLIMUPublisher(
        name=imu_stream_name,
        source_id=f"{source_id_prefix}_imu",
        accel_hz=accel_hz,
        gyro_hz=gyro_hz,
        poll_hz=imu_poll_hz,
        rate_print=rate_print,
    )
    ppg_pub = LSLPPGPublisher(
        name=ppg_stream_name,
        source_id=f"{source_id_prefix}_ppg",
        poll_sleep_ms=ppg_poll_sleep_ms,
        rate_print=rate_print,
    )

    def _run_publisher(label, publisher):
        try:
            publisher.run(stop_evt)
        except Exception as e:
            print(f"[LSL][{label}] fatal error: {e}")
            stop_evt.set()

    threads = [
        threading.Thread(target=_run_publisher, args=("IMU", imu_pub), daemon=True),
        threading.Thread(target=_run_publisher, args=("PPG", ppg_pub), daemon=True),
    ]
    for thread in threads:
        thread.start()

    print("[LSL] Streaming only IMU and PPG. Press Ctrl+C to stop.")
    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
    finally:
        stop_evt.set()
        for thread in threads:
            thread.join(timeout=2.0)
        print("[LSL] Stopped.")


if __name__ == "__main__":
    main()
