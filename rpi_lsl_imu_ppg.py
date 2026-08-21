#!/usr/bin/env python3
"""
Publish BNO085 IMU and PPG data to Lab Streaming Layer.

Streams:
  NHMRC_IMU: utc_unix_s plus configurable BNO085 9-DoF channels
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
import traceback
from dataclasses import dataclass, field
from typing import Iterable, Optional

from bno085_lsl_streamer import ReportSpec, create_bno085_i2c, create_bno085_spi, max_enabled_report_rate_hz, require_modules, selected_reports
import max30102
from pylsl import StreamInfo, StreamOutlet, local_clock

from config import deep_get, load_config


I2C_LOCK = threading.Lock()


@dataclass
class DiagnosticLogger:
    label: str
    enabled: bool = False
    interval_s: float = 2.0
    _last_by_key: dict[str, float] = field(default_factory=dict, init=False)

    def log(self, message: str, key: str = "default", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        last = self._last_by_key.get(key, 0.0)
        if force or now - last >= self.interval_s:
            print(f"[LSL][DIAG][{self.label}] {message}", flush=True)
            self._last_by_key[key] = now


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
            print(f"[LSL][{self.label}] {rate:.1f} Hz ({self._count} samples / {dt:.2f} s)", flush=True)
            self._count = 0
            self._t0 = now


def _append_channels(info: StreamInfo, labels: Iterable[str]) -> None:
    channels = info.desc().append_child("channels")
    for label in labels:
        channels.append_child("channel").append_child_value("label", str(label))


class BNO085LSLIMUPublisher:
    DEFAULT_REPORTS = (
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "rotation_vector",
    )

    def __init__(
        self,
        name: str,
        source_id: str,
        poll_hz: float,
        reports: Iterable[str],
        address: int,
        i2c_frequency: int,
        transport: str,
        spi_cs_pin: str,
        spi_int_pin: str,
        spi_baudrate: int,
        debug: bool,
        reset_pin: Optional[str],
        rate_print: bool,
        diagnostics: bool,
    ):
        self.name = name
        self.source_id = source_id
        self.poll_hz = float(poll_hz)
        self.report_names = tuple(reports)
        self.address = int(address)
        self.i2c_frequency = int(i2c_frequency)
        self.transport = str(transport).strip().lower()
        self.spi_cs_pin = spi_cs_pin
        self.spi_int_pin = spi_int_pin
        self.spi_baudrate = int(spi_baudrate)
        self.debug = bool(debug)
        self.io_lock = I2C_LOCK if self.transport == "i2c" else threading.Lock()
        self.reset_pin_name = reset_pin
        self.reports: list[ReportSpec] = []
        self.channel_names: tuple[str, ...] = ()
        self.outlet: Optional[StreamOutlet] = None
        self.rate = RateCounter("IMU", enabled=rate_print)
        self.diag = DiagnosticLogger("IMU", enabled=diagnostics)
        self.ready = threading.Event()
        self.bno = None

    def start(self) -> None:
        self.diag.log("loading hardware modules", force=True)
        board, busio, digitalio, BNO08X_I2C, BNO08X_SPI, bno08x, _, _, _ = require_modules()

        enabled_reports: list[ReportSpec] = []
        self.diag.log(f"waiting for {self.transport.upper()} lock during BNO085 init", force=True)
        with self.io_lock:
            self.diag.log(f"acquired {self.transport.upper()} lock during BNO085 init", force=True)
            self.diag.log(f"creating BNO085 {self.transport.upper()} object", force=True)
            if self.transport == "spi":
                self.bno = create_bno085_spi(
                    board,
                    busio,
                    digitalio,
                    BNO08X_SPI,
                    cs_pin_name=self.spi_cs_pin,
                    int_pin_name=self.spi_int_pin,
                    reset_pin_name=self.reset_pin_name,
                    spi_baudrate=self.spi_baudrate,
                    debug=self.debug,
                )
            elif self.transport == "i2c":
                self.bno = create_bno085_i2c(
                    board,
                    busio,
                    digitalio,
                    BNO08X_I2C,
                    address=self.address,
                    i2c_frequency=self.i2c_frequency,
                    reset_pin_name=self.reset_pin_name,
                    debug=self.debug,
                )
            else:
                raise ValueError(f"Unsupported BNO085 transport {self.transport!r}; use 'spi' or 'i2c'")

            self.diag.log("BNO085 object created; enabling reports", force=True)
            for report in selected_reports(self.report_names):
                feature_id = getattr(bno08x, report.feature_const, None)
                if feature_id is None:
                    print(f"[LSL][IMU] skipping {report.name}: missing {report.feature_const}", flush=True)
                    continue
                try:
                    requested_interval_us = max(report.default_interval_us, int(1_000_000 / max(1.0, self.poll_hz)))
                    self.diag.log(
                        f"enabling {report.name} interval_us={requested_interval_us}",
                        key=f"enable_{report.name}",
                        force=True,
                    )
                    self.bno.enable_feature(feature_id, requested_interval_us)
                except Exception as e:
                    print(f"[LSL][IMU] skipping {report.name}: could not enable report: {e}", flush=True)
                    continue
                enabled_reports.append(report)
                configured_hz = 1_000_000.0 / float(requested_interval_us)
                print(
                    f"[LSL][IMU] Set {report.name} rate: {configured_hz:g} Hz "
                    f"(interval_us={requested_interval_us})",
                    flush=True,
                )
                self.diag.log(f"enabled {report.name}", key=f"enabled_{report.name}", force=True)

        if not enabled_reports:
            raise RuntimeError("No BNO085 reports were enabled for IMU streaming.")

        self.reports = enabled_reports
        self.channel_names = tuple(
            ["utc_unix_s", *[name for report in enabled_reports for name in report.channel_names]]
        )

        info = StreamInfo(self.name, "IMU", len(self.channel_names), self.poll_hz, "double64", self.source_id)
        _append_channels(info, self.channel_names)
        info.desc().append_child_value("clock", "sample channel 0 is UTC Unix seconds")
        info.desc().append_child_value("sensor", "BNO085/BNO08x over Raspberry Pi I2C")
        reports_meta = info.desc().append_child("reports")
        for report in enabled_reports:
            reports_meta.append_child_value("report", report.name)

        self.diag.log("creating IMU LSL outlet", force=True)
        self.outlet = StreamOutlet(info)
        self.ready.set()
        print(
            f"[LSL][IMU] Outlet ready: BNO085 reports={','.join(report.name for report in enabled_reports)} "
            f"rate={self.poll_hz:g}Hz",
            flush=True,
        )

    def stop(self) -> None:
        self.bno = None

    def run(self, stop_evt: threading.Event) -> None:
        self.start()
        self.diag.log("entering IMU sample loop", force=True)
        period_s = 1.0 / max(1.0, self.poll_hz)
        next_sample_time = time.monotonic()
        try:
            while not stop_evt.is_set():
                row = [time.time()]
                for report in self.reports:
                    try:
                        self.diag.log(f"waiting for {self.transport.upper()} lock to read {report.name}", key=f"wait_read_{report.name}")
                        with self.io_lock:
                            self.diag.log(f"acquired {self.transport.upper()} lock; reading {report.name}", key=f"read_{report.name}")
                            value = getattr(self.bno, report.property_name) if self.bno is not None else None
                        self.diag.log(f"read {report.name} value={value}", key=f"read_ok_{report.name}")
                        row.extend(report.converter(value))
                    except Exception as e:
                        print(f"[LSL][IMU] read failed for {report.name}: {e}", flush=True)
                        row.extend([math.nan] * len(report.channel_names))

                if self.outlet is not None:
                    self.outlet.push_sample(row, timestamp=local_clock())
                    self.diag.log(f"pushed IMU sample len={len(row)}", key="push")
                self.rate.add()

                next_sample_time += period_s
                sleep_s = next_sample_time - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_sample_time = time.monotonic()
        finally:
            self.stop()


class LSLPPGPublisher:
    CHANNELS = ("utc_unix_s", "red", "ir")

    def __init__(self, name: str, source_id: str, poll_sleep_ms: float, sample_rate_hz: float, rate_print: bool, diagnostics: bool):
        self.sample_rate_hz = float(sample_rate_hz)
        info = StreamInfo(name, "PPG", len(self.CHANNELS), self.sample_rate_hz, "double64", source_id)
        _append_channels(info, self.CHANNELS)
        info.desc().append_child_value("clock", "sample channel 0 is UTC Unix seconds")

        self.outlet = StreamOutlet(info)
        self.poll_sleep_ms = float(poll_sleep_ms)
        self.rate = RateCounter("PPG", enabled=rate_print)
        self.diag = DiagnosticLogger("PPG", enabled=diagnostics)
        self.sensor: Optional[max30102.MAX30102] = None

    def start(self) -> None:
        self.diag.log("waiting for I2C lock during MAX30102 init", force=True)
        with I2C_LOCK:
            self.diag.log("acquired I2C lock during MAX30102 init", force=True)
            self.sensor = max30102.MAX30102(gpio_pin=None)
            self.diag.log("MAX30102 object created; running setup", force=True)
            ppg_sensor_rate_hz = 200
            print(f"[LSL][PPG] Set MAX30102 sample rate: {ppg_sensor_rate_hz:g} Hz", flush=True)
            self.sensor.setup(
                led_mode=0x03,
                sample_rate=ppg_sensor_rate_hz,
                pulse_width=118,
                adc_range=4096,
                fifo_average=1,
                fifo_rollover=False,
                fifo_a_full=15,
            )
        self.diag.log("MAX30102 setup complete", force=True)
        print("[LSL][PPG] Outlet ready: MAX30102 red/IR", flush=True)

    def stop(self) -> None:
        self.sensor = None

    def run(self, stop_evt: threading.Event) -> None:
        self.start()
        self.diag.log("entering PPG sample loop", force=True)
        try:
            while not stop_evt.is_set():
                batch = None
                try:
                    self.diag.log("waiting for I2C lock to read FIFO", key="wait_read")
                    with I2C_LOCK:
                        self.diag.log("acquired I2C lock; reading FIFO", key="read")
                        batch = self.sensor.i2c_thread_func(max_batch=32, require_ppg_rdy=False) if self.sensor else None
                    self.diag.log(f"FIFO read returned {len(batch) if batch else 0} samples", key="read_ok")
                except Exception as e:
                    print(f"[LSL][PPG] read error: {e}", flush=True)

                if batch:
                    n = len(batch)
                    for i, (utc, red, ir) in enumerate(batch):
                        sample_utc = float(utc) - max(0, n - 1 - i) / max(1.0, self.sample_rate_hz)
                        self.outlet.push_sample([sample_utc, float(red), float(ir)], timestamp=local_clock())
                    self.diag.log(f"pushed PPG batch n={n}", key="push")
                    self.rate.add(n)

                time.sleep(self.poll_sleep_ms / 1000.0)
        finally:
            self.stop()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Publish BNO085 IMU and PPG to LSL.")
    ap.add_argument("--config", default="./streaming_config.yaml")
    ap.add_argument("--imu-stream-name", default=None)
    ap.add_argument("--ppg-stream-name", default=None)
    ap.add_argument("--source-id-prefix", default=None)
    ap.add_argument("--imu-poll-hz", type=float, default=None)
    ap.add_argument("--bno-address", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--bno-i2c-frequency", type=int, default=None)
    ap.add_argument("--bno-transport", choices=("i2c", "spi"), default=None)
    ap.add_argument("--bno-spi-cs-pin", default=None)
    ap.add_argument("--bno-spi-int-pin", default=None)
    ap.add_argument("--bno-spi-baudrate", type=int, default=None)
    ap.add_argument("--bno-debug", action="store_true")
    ap.add_argument("--bno-reset-pin", default=None)
    ap.add_argument("--bno-reports", nargs="+", default=None)
    ap.add_argument("--ppg-poll-sleep-ms", type=float, default=None)
    ap.add_argument("--ppg-sample-rate-hz", type=float, default=None)
    ap.add_argument("--imu-ready-timeout-s", type=float, default=None)
    ap.add_argument("--rate-print", action="store_true")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--no-ppg", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    imu_stream_name = args.imu_stream_name or deep_get(cfg, "lsl.imu_stream_name", "NHMRC_IMU")
    ppg_stream_name = args.ppg_stream_name or deep_get(cfg, "lsl.ppg_stream_name", "NHMRC_PPG")
    source_id_prefix = args.source_id_prefix or deep_get(cfg, "lsl.source_id_prefix", "nhmrc")
    imu_poll_hz = args.imu_poll_hz
    if imu_poll_hz is None:
        configured_imu_poll_hz = deep_get(cfg, "lsl.imu_poll_hz", None)
        if configured_imu_poll_hz is not None:
            imu_poll_hz = float(configured_imu_poll_hz)
    bno_address = args.bno_address if args.bno_address is not None else int(deep_get(cfg, "bno085.address", 0x4A))
    bno_i2c_frequency = args.bno_i2c_frequency or int(deep_get(cfg, "bno085.i2c_frequency", 100000))
    bno_transport = args.bno_transport or str(deep_get(cfg, "bno085.transport", "i2c")).lower()
    bno_spi_cs_pin = args.bno_spi_cs_pin or deep_get(cfg, "bno085.spi.cs_pin", "CE0")
    bno_spi_int_pin = args.bno_spi_int_pin or deep_get(cfg, "bno085.spi.int_pin", None)
    bno_spi_baudrate = args.bno_spi_baudrate or int(deep_get(cfg, "bno085.spi.baudrate", 1000000))
    bno_debug = bool(args.bno_debug or deep_get(cfg, "bno085.debug", False))
    bno_reset_pin = args.bno_reset_pin or deep_get(cfg, "bno085.reset_pin", None)
    bno_reports = args.bno_reports or deep_get(cfg, "bno085.reports", BNO085LSLIMUPublisher.DEFAULT_REPORTS)
    if imu_poll_hz is None:
        imu_poll_hz = max_enabled_report_rate_hz(selected_reports(bno_reports))
    ppg_poll_sleep_ms = args.ppg_poll_sleep_ms or float(deep_get(cfg, "ppg.poll_sleep_ms", 5.0))
    ppg_sample_rate_hz = args.ppg_sample_rate_hz or float(deep_get(cfg, "lsl.ppg_sample_rate_hz", 200.0))
    imu_ready_timeout_s = args.imu_ready_timeout_s
    if imu_ready_timeout_s is None:
        imu_ready_timeout_s = float(deep_get(cfg, "lsl.imu_ready_timeout_s", 30.0))
    rate_print = bool(args.rate_print or deep_get(cfg, "lsl.rate_print", False))
    diagnostics = bool(args.diagnostics or deep_get(cfg, "lsl.diagnostics", False))
    imu_enabled = bool(deep_get(cfg, "streams.imu.enabled", True)) and not args.no_imu
    ppg_enabled = bool(deep_get(cfg, "streams.ppg.enabled", True)) and not args.no_ppg
    if not imu_enabled and not ppg_enabled:
        raise SystemExit("No streams enabled. Remove --no-imu/--no-ppg or enable a stream in config.")

    print(
        f"[LSL] Config: imu_stream={imu_stream_name} ppg_stream={ppg_stream_name} "
        f"imu_poll_hz={imu_poll_hz:g} reports={','.join(str(r) for r in bno_reports)} "
        f"bno_transport={bno_transport} bno_addr=0x{bno_address:02x} i2c_frequency={bno_i2c_frequency} "
        f"spi_cs={bno_spi_cs_pin or 'none'} spi_int={bno_spi_int_pin or 'none'} spi_baudrate={bno_spi_baudrate} reset_pin={bno_reset_pin or 'none'} "
        f"imu_enabled={imu_enabled} ppg_enabled={ppg_enabled} rate_print={rate_print} diagnostics={diagnostics}",
        flush=True,
    )

    stop_evt = threading.Event()

    def _sig_handler(_sig, _frame):
        print("[LSL] Stop requested; waiting for publisher threads to exit...", flush=True)
        stop_evt.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    imu_pub = BNO085LSLIMUPublisher(
        name=imu_stream_name,
        source_id=f"{source_id_prefix}_imu",
        poll_hz=imu_poll_hz,
        reports=bno_reports,
        address=bno_address,
        i2c_frequency=bno_i2c_frequency,
        transport=bno_transport,
        spi_cs_pin=bno_spi_cs_pin,
        spi_int_pin=bno_spi_int_pin,
        spi_baudrate=bno_spi_baudrate,
        debug=bno_debug,
        reset_pin=bno_reset_pin,
        rate_print=rate_print,
        diagnostics=diagnostics,
    )
    ppg_pub = LSLPPGPublisher(
        name=ppg_stream_name,
        source_id=f"{source_id_prefix}_ppg",
        poll_sleep_ms=ppg_poll_sleep_ms,
        sample_rate_hz=ppg_sample_rate_hz,
        rate_print=rate_print,
        diagnostics=diagnostics,
    )

    def _run_publisher(label, publisher):
        try:
            if diagnostics:
                print(f"[LSL][DIAG][{label}] publisher thread starting", flush=True)
            publisher.run(stop_evt)
            if diagnostics:
                print(f"[LSL][DIAG][{label}] publisher thread exited", flush=True)
        except Exception as e:
            print(f"[LSL][{label}] fatal error: {e}", flush=True)
            traceback.print_exc()
            stop_evt.set()

    threads = []
    imu_thread = threading.Thread(target=_run_publisher, args=("IMU", imu_pub), daemon=True, name="LSL-IMU")
    ppg_thread = threading.Thread(target=_run_publisher, args=("PPG", ppg_pub), daemon=True, name="LSL-PPG")

    if imu_enabled:
        threads.append(imu_thread)
        imu_thread.start()
        if ppg_enabled:
            if diagnostics:
                print(f"[LSL][DIAG][MAIN] waiting up to {imu_ready_timeout_s:g}s for IMU readiness", flush=True)
            if imu_pub.ready.wait(timeout=max(0.0, imu_ready_timeout_s)) and not stop_evt.is_set():
                if diagnostics:
                    print("[LSL][DIAG][MAIN] IMU ready; starting PPG thread", flush=True)
                threads.append(ppg_thread)
                ppg_thread.start()
            elif not stop_evt.is_set():
                print(
                    f"[LSL] IMU did not become ready within {imu_ready_timeout_s:g}s; "
                    "not starting PPG on the shared I2C bus.",
                    flush=True,
                )
                stop_evt.set()
    elif ppg_enabled:
        threads.append(ppg_thread)
        ppg_thread.start()

    active = ", ".join(label for label, enabled in (("BNO085 IMU", imu_enabled), ("PPG", ppg_enabled)) if enabled)
    print(f"[LSL] Streaming {active}. Press Ctrl+C to stop.", flush=True)
    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
    finally:
        stop_evt.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"[LSL] Warning: {thread.name} did not exit within 2s.", flush=True)
        print("[LSL] Stopped.", flush=True)


if __name__ == "__main__":
    main()