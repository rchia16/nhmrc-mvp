#!/usr/bin/env python3
"""
Raspberry Pi "all-in-one" sender for main_sonify_live_with_depth.py

Opens TWO TCP connections to the PC:
  1) RealSense stream (color JPEG + depth PNG + IMU) -> rs_port
     Packet format: length-prefixed pickled dict with keys:
        t, jpeg, depth_png, depth_scale, depth_wh, imu{accel,gyro}

  2) PPG stream (MAX30102) -> ppg_port
     Packet format: raw TCP stream of fixed-size records:
        struct '!dii' => (timestamp float64, red int32, ir int32)

This matches the expectations of main_sonify_live_with_depth.py:
  - RealSenseYOLOFromTCP listens on rs_port
  - PPGTCPReceiver listens on ppg_port
"""

import argparse
import socket
import struct
import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2
import pyrealsense2 as rs

import max30102  # expects max30102.py in the same project folder


# -----------------------------
# TCP helpers
# -----------------------------
class TCPClient:
    def __init__(self, host: str, port: int, reconnect_sec: float = 2.0, timeout_sec: float = 5.0):
        self.host = host
        self.port = int(port)
        self.reconnect_sec = float(reconnect_sec)
        self.timeout_sec = float(timeout_sec)
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self):
        with self._lock:
            if self._sock is not None:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout_sec)
            s.connect((self.host, self.port))
            s.settimeout(None)
            self._sock = s
            print(f"[TCP] Connected -> {self.host}:{self.port}")

    def close(self):
        with self._lock:
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None

    def sendall(self, data: bytes):
        """Send bytes reliably; reconnect on failure."""
        while True:
            try:
                if self._sock is None:
                    self.connect()
                with self._lock:
                    if self._sock is None:
                        continue
                    self._sock.sendall(data)
                return
            except Exception as e:
                print(f"[TCP] Send failed: {e}. Reconnecting in {self.reconnect_sec}s")
                self.close()
                time.sleep(self.reconnect_sec)


# -----------------------------
# RealSense (color+depth) + IMU
# -----------------------------
class IMUReader:
    """Reads accel+gyro in a background thread and stores the latest values."""
    def __init__(self, accel_hz: int = 63, gyro_hz: int = 200):
        self.accel_hz = int(accel_hz)
        self.gyro_hz = int(gyro_hz)
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.accel_hz)
        self._config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.gyro_hz)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._accel: Optional[Tuple[float, float, float]] = None
        self._gyro: Optional[Tuple[float, float, float]] = None

    def start(self):
        self._pipeline.start(self._config)
        print("[RS] IMU started (accel + gyro).")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def latest(self):
        with self._lock:
            return self._accel, self._gyro

    def _loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                continue
            for f in frames:
                try:
                    if f.is_motion_frame():
                        motion = f.as_motion_frame().get_motion_data()
                        v = (float(motion.x), float(motion.y), float(motion.z))
                        st = f.get_profile().stream_type()
                        with self._lock:
                            if st == rs.stream.accel:
                                self._accel = v
                            elif st == rs.stream.gyro:
                                self._gyro = v
                except Exception:
                    continue


class RealSenseColorDepthSender:
    def __init__(
        self,
        host: str,
        port: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 80,
        png_compression: int = 3,
        reconnect_sec: float = 2.0,
        accel_hz: int = 63,
        gyro_hz: int = 200,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.png_compression = int(png_compression)

        self.client = TCPClient(host, port, reconnect_sec=reconnect_sec)
        self.imu = IMUReader(accel_hz=accel_hz, gyro_hz=gyro_hz)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Enable BOTH color and depth (same size/fps) to avoid "Couldn't resolve requests"
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = None
        self.align = None
        self.depth_scale = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        print("[RS] RealSense started (color + depth).")
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"[RS] Depth scale: {self.depth_scale} m/unit")

        self.imu.start()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.imu.stop()
        self.client.close()

    def _loop(self):
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        png_params = [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]

        sent = 0
        last_print = time.time()

        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print(f"[RS] wait_for_frames error: {e}")
                continue

            try:
                frames = self.align.process(frames)
            except Exception:
                pass

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # BGR8
            depth = np.asanyarray(depth_frame.get_data())  # uint16

            ok_c, enc_c = cv2.imencode(".jpg", color, jpeg_params)
            if not ok_c:
                continue
            jpeg_bytes = enc_c.tobytes()

            ok_d, enc_d = cv2.imencode(".png", depth, png_params)  # lossless uint16 PNG
            if not ok_d:
                continue
            depth_png = enc_d.tobytes()

            accel, gyro = self.imu.latest()

            packet = {
                "t": time.time(),
                "jpeg": jpeg_bytes,
                "depth_png": depth_png,
                "depth_scale": float(self.depth_scale),
                "depth_wh": (self.width, self.height),
                "imu": {"accel": accel, "gyro": gyro},
            }

            payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
            header = struct.pack("!I", len(payload))
            self.client.sendall(header + payload)

            sent += 1
            now = time.time()
            if now - last_print >= 2.0:
                mb = (len(payload) / (1024 * 1024))
                print(f"[RS SEND] {sent} packets, last_payload={mb:.2f} MiB")
                last_print = now


# -----------------------------
# PPG (MAX30102) -> TCP
# -----------------------------
class PPGTCPSender:
    PACK_FMT = "!dii"

    def __init__(
        self,
        host: str,
        port: int,
        reconnect_sec: float = 2.0,
        sample_rate: int = 200,
        pulse_width: int = 118,
        adc_range: int = 4096,
        fifo_average: int = 1,
        poll_sleep_ms: float = 5.0,
        max_batch: int = 32,
        rate_print: bool = True,
    ):
        self.client = TCPClient(host, port, reconnect_sec=reconnect_sec)
        self.poll_sleep = float(poll_sleep_ms) / 1000.0
        self.max_batch = int(max_batch)
        self.rate_print = bool(rate_print)

        self.m = max30102.MAX30102()
        self.m.setup(
            led_mode=0x03,
            sample_rate=int(sample_rate),
            pulse_width=int(pulse_width),
            adc_range=int(adc_range),
            fifo_average=int(fifo_average),
            fifo_rollover=False,
            fifo_a_full=15,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._c = 0
        self._t0 = time.time()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[PPG] Started (poll) -> TCP {self.client.host}:{self.client.port}")

    def stop(self):
        self._running = False
        self.client.close()

    def _loop(self):
        while self._running:
            try:
                batch = self.m.i2c_thread_func(max_batch=self.max_batch, require_ppg_rdy=False)
            except TypeError:
                batch = self.m.i2c_thread_func()
            except Exception as e:
                print(f"[PPG] Read error: {e}")
                batch = None

            if batch:
                out = bytearray()
                for ts, red, ir in batch:
                    out.extend(struct.pack(self.PACK_FMT, float(ts), int(red), int(ir)))

                self.client.sendall(bytes(out))

                if self.rate_print:
                    self._c += len(batch)
                    now = time.time()
                    dt = now - self._t0
                    if dt >= 2.0:
                        print(f"[PPG SEND] {self._c/dt:.1f} Hz ({self._c} samples / {dt:.2f}s)")
                        self._c = 0
                        self._t0 = now

            time.sleep(self.poll_sleep)


# -----------------------------
# App
# -----------------------------
class PiAllInOneSender:
    def __init__(self, args):
        self.rs = RealSenseColorDepthSender(
            host=args.host,
            port=args.rs_port,
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            png_compression=args.png_compression,
            reconnect_sec=args.reconnect_sec,
            accel_hz=args.accel_hz,
            gyro_hz=args.gyro_hz,
        )
        self.ppg = PPGTCPSender(
            host=args.host,
            port=args.ppg_port,
            reconnect_sec=args.reconnect_sec,
            sample_rate=args.ppg_sr,
            pulse_width=args.ppg_pw,
            adc_range=args.ppg_adc,
            fifo_average=args.ppg_avg,
            poll_sleep_ms=args.ppg_poll_ms,
            max_batch=args.ppg_max_batch,
            rate_print=not args.ppg_no_rate_print,
        )

    def start(self):
        self.rs.start()
        self.ppg.start()

    def stop(self):
        self.rs.stop()
        self.ppg.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="PC IP running main_sonify_live_with_depth.py")
    ap.add_argument("--rs-port", type=int, default=50000, help="TCP port for RealSense stream (default 50000)")
    ap.add_argument("--ppg-port", type=int, default=9999, help="TCP port for PPG stream (default 9999)")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--png-compression", type=int, default=3)
    ap.add_argument("--reconnect-sec", type=float, default=2.0)
    ap.add_argument("--accel-hz", type=int, default=63)
    ap.add_argument("--gyro-hz", type=int, default=200)

    ap.add_argument("--ppg-sr", type=int, default=200)
    ap.add_argument("--ppg-pw", type=int, default=118)
    ap.add_argument("--ppg-adc", type=int, default=4096)
    ap.add_argument("--ppg-avg", type=int, default=1)
    ap.add_argument("--ppg-poll-ms", type=float, default=5.0)
    ap.add_argument("--ppg-max-batch", type=int, default=32)
    ap.add_argument("--ppg-no-rate-print", action="store_true")

    args = ap.parse_args()

    app = PiAllInOneSender(args)
    try:
        app.start()
        print("[ALL] Sending RealSense+Depth+IMU and PPG. Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping...")
    finally:
        app.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
