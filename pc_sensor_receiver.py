#!/usr/bin/env python3
"""
pc_sensor_receiver.py

PC unified receiver:
- PPG via TCPServerReceiver (from ppg_receive.py)
- RealSense RGB + Depth + IMU via len-prefixed pickle packets
- Optional visualization:
    --viz     : OpenCV windows for RGB (+ optional depth)
    --ppg-viz : Matplotlib PPG traces (red/ir)
- Optional rate printing:
    --rate-print : prints PPG Hz and RGB-D fps on receiver end

Reuses code from:
- ppg_receive.py: TCPServerReceiver, PPGWindowBuffer, PlotSink, RateMonitor
- rs_d455_tcp_receiver.py: recv_exact, decode_jpeg, decode_depth_png,
                           depth_to_colormap, overlay_imu_text, overlay_depth_probe
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

# --- MUST import from your provided receiver modules ---
from ppg_receive import (
    TCPServerReceiver,
    PPGWindowBuffer,
    PlotSink,
    RateMonitor,
)

from rs_d455_tcp_receiver import (
    recv_exact,
    decode_jpeg,
    decode_depth_png,
    depth_to_colormap,
    overlay_imu_text,
    overlay_depth_probe,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class PCReceiverConfig:
    listen_ip: str = "0.0.0.0"

    # PPG
    ppg_port: int = 9999
    ppg_poll_hz: float = 500.0

    # RealSense
    rs_port: int = 50000
    rs_accept_timeout_sec: float = 1.0

    # Optional: bounds on internal queues (0 = unbounded)
    max_ppg_queue: int = 0
    max_rs_queue: int = 0

class LabeledRateMonitor(RateMonitor):
    """
    Small wrapper to reuse RateMonitor but print a stream label so RX logs are not ambiguous.
    """
    def __init__(self, label: str, interval_sec: float = 2.0):
        super().__init__(interval_sec=float(interval_sec))
        self.label = str(label)

    def add(self, n: int) -> None:
        # Copy of RateMonitor.add() behavior, but labeled print.
        self._count += int(n)
        now = time.time()
        dt = now - self._t0
        if dt >= self.interval:
            rate = (self._count / dt) if dt > 0 else 0.0
            print(f"[RX RATE][{self.label}] {rate:.1f} Hz ({self._count} samples / {dt:.2f} s)")
            self._count = 0


# ---------------------------------------------------------------------
# Internal thread-safe buffers
# ---------------------------------------------------------------------

class _ThreadSafeQueue:
    """Minimal bounded/unbounded FIFO with condition variable."""

    def __init__(self, maxlen: int = 0):
        self._maxlen = int(maxlen) if maxlen else 0
        self._q: List[Any] = []
        self._cv = threading.Condition()

    def put(self, item: Any) -> None:
        with self._cv:
            if self._maxlen and len(self._q) >= self._maxlen:
                self._q.pop(0)  # drop oldest
            self._q.append(item)
            self._cv.notify()

    def get_all(self) -> List[Any]:
        with self._cv:
            items = self._q
            self._q = []
            return items

    def size(self) -> int:
        with self._cv:
            return len(self._q)


# ---------------------------------------------------------------------
# RealSense TCP server receiver (OOP wrapper around rs_d455_tcp_receiver helpers)
# ---------------------------------------------------------------------

@dataclass
class RealSenseFramePacket:
    t: float
    rgb_bgr: Optional[np.ndarray]              # HxWx3 uint8
    depth_u16: Optional[np.ndarray]            # HxW uint16 (aligned to color)
    depth_scale: float                         # meters per depth unit
    imu: Dict[str, Any]                        # e.g. {"accel":[...], "gyro":[...]}
    raw: Dict[str, Any]                        # original dict


class RealSenseTCPServerReceiver:
    """
    Accepts a single sender at a time. Reconnect-friendly.

    Packet format:
      [4-byte big endian length][pickle(dict)]
    """

    def __init__(self, listen_ip: str, port: int, accept_timeout_sec: float = 1.0):
        self.listen_ip = listen_ip
        self.port = int(port)
        self.accept_timeout_sec = float(accept_timeout_sec)

        self._srv: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._client_addr: Optional[Tuple[str, int]] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._srv is not None:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.listen_ip, self.port))
        srv.listen(1)
        srv.settimeout(self.accept_timeout_sec)
        self._srv = srv

    def stop(self) -> None:
        self._close_client()
        if self._srv is not None:
            try:
                self._srv.close()
            except Exception:
                pass
        self._srv = None

    def _close_client(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = None
            self._client_addr = None

    def _accept_if_needed(self) -> None:
        if self._srv is None:
            self.start()

        with self._lock:
            if self._client is not None:
                return

        try:
            conn, addr = self._srv.accept()  # type: ignore[union-attr]
        except socket.timeout:
            return

        conn.settimeout(5.0)
        with self._lock:
            self._client = conn
            self._client_addr = (addr[0], addr[1])
        print(f"[RS][TCP] Client connected: {addr[0]}:{addr[1]}")

    def poll_one(self) -> Optional[RealSenseFramePacket]:
        self._accept_if_needed()

        with self._lock:
            conn = self._client

        if conn is None:
            return None

        try:
            header = recv_exact(conn, 4)
            (length,) = struct.unpack("!I", header)

            payload = recv_exact(conn, length)
            packet = pickle.loads(payload)

            t = float(packet.get("t", time.time()))
            imu = packet.get("imu", {}) or {}

            jpeg_bytes = packet.get("jpeg", b"")
            depth_png = packet.get("depth_png", None)
            depth_scale = float(packet.get("depth_scale", 0.001))

            rgb = decode_jpeg(jpeg_bytes)
            depth_u16 = decode_depth_png(depth_png) if depth_png is not None else None

            return RealSenseFramePacket(
                t=t,
                rgb_bgr=rgb,
                depth_u16=depth_u16,
                depth_scale=depth_scale,
                imu=imu,
                raw=packet,
            )

        except socket.timeout:
            return None
        except EOFError:
            print("[RS][TCP] Client disconnected")
            self._close_client()
            return None
        except Exception as e:
            print(f"[RS][TCP] Receiver error: {e}")
            self._close_client()
            return None


# ---------------------------------------------------------------------
# Unified PC receiver
# ---------------------------------------------------------------------

class PCSensorReceiver:
    """
    Receives:
      - PPG samples: (ts, red, ir, src_ip)
      - RealSense packets: RealSenseFramePacket

    Optional:
      - Maintains a PPGWindowBuffer for matplotlib trace viz
      - Prints PPG Hz and RS fps using RateMonitor (from ppg_receive.py)
    """

    def __init__(
        self,
        cfg: PCReceiverConfig = PCReceiverConfig(),
        on_ppg: Optional[Callable[[float, int, int, str], None]] = None,
        on_rs: Optional[Callable[[RealSenseFramePacket], None]] = None,
    ):
        self.cfg = cfg
        self.on_ppg = on_ppg
        self.on_rs = on_rs

        self._ppg_rx = TCPServerReceiver(self.cfg.listen_ip, int(self.cfg.ppg_port))
        self._rs_rx = RealSenseTCPServerReceiver(
            self.cfg.listen_ip,
            int(self.cfg.rs_port),
            accept_timeout_sec=self.cfg.rs_accept_timeout_sec,
        )

        self._ppg_q = _ThreadSafeQueue(maxlen=self.cfg.max_ppg_queue)
        self._rs_q = _ThreadSafeQueue(maxlen=self.cfg.max_rs_queue)

        self._stop = threading.Event()
        self._t_ppg: Optional[threading.Thread] = None
        self._t_rs: Optional[threading.Thread] = None

        self._latest_lock = threading.Lock()
        self._latest_ppg: Optional[Tuple[float, int, int, str]] = None
        self._latest_rs: Optional[RealSenseFramePacket] = None

        # ---- PPG trace buffer (reuses ppg_receive.py types) ----
        self._ppg_window: Optional[PPGWindowBuffer] = None

        # ---- Rate monitors (reuses RateMonitor from ppg_receive.py) ----
        self._ppg_rate: Optional[RateMonitor] = None
        self._rs_rate: Optional[RateMonitor] = None

        # ---- Piggyback PPG dedupe ----
        # If RS packets carry ppg_latest, we accept only increasing timestamps.
        self._last_piggy_ppg_ts: float = -1.0

    # -----------------------
    # Public helpers
    # -----------------------

    def enable_ppg_window(self, window_size: int) -> None:
        """Enable a thread-safe PPGWindowBuffer for --ppg-viz."""
        self._ppg_window = PPGWindowBuffer(maxlen=int(window_size))

    def get_ppg_window(self) -> Optional[PPGWindowBuffer]:
        return self._ppg_window

    def enable_rate_printing(self, interval_sec: float = 2.0) -> None:
        """
        Enable printing:
          - PPG: samples/sec (Hz)
          - RS: frames/sec (fps)

        Uses labeled RateMonitor to disambiguate streams.
        """
        self._ppg_rate = LabeledRateMonitor("PPG", interval_sec=float(interval_sec))
        self._rs_rate = LabeledRateMonitor("RGBD", interval_sec=float(interval_sec))

    def print_rates_now(self) -> None:
        """
        If you want a one-shot print call (instead of continuous),
        you can extend this to compute from internal timestamps. For now,
        continuous printing is handled by RateMonitor when enabled.
        """
        # RateMonitor prints automatically when dt >= interval; nothing to do here.
        pass

    # -----------------------
    # Threads
    # -----------------------

    def _ppg_loop(self) -> None:
        sleep_s = 1.0 / max(1.0, float(self.cfg.ppg_poll_hz))
        print(f"[PPG][TCP] Listening on {self.cfg.listen_ip}:{self.cfg.ppg_port}")

        while not self._stop.is_set():
            samples = self._ppg_rx.poll(max_records=2000)
            if samples:
                # rate (reuse RateMonitor)
                if self._ppg_rate is not None:
                    self._ppg_rate.add(len(samples))

                for ts, red, ir, src_ip in samples:
                    self._ppg_q.put((ts, red, ir, src_ip))
                    with self._latest_lock:
                        self._latest_ppg = (ts, red, ir, src_ip)

                    # maintain plotting window if enabled
                    if self._ppg_window is not None:
                        self._ppg_window.append(ts, red, ir)

                    if self.on_ppg is not None:
                        try:
                            self.on_ppg(ts, red, ir, src_ip)
                        except Exception as e:
                            print(f"[PPG] on_ppg callback error: {e}")

            time.sleep(sleep_s)

    def _rs_loop(self) -> None:
        self._rs_rx.start()
        print(f"[RS][TCP] Listening on {self.cfg.listen_ip}:{self.cfg.rs_port}")

        while not self._stop.is_set():
            pkt = self._rs_rx.poll_one()
            if pkt is None:
                continue

            # rate (reuse RateMonitor)
            if self._rs_rate is not None:
                self._rs_rate.add(1)

            self._rs_q.put(pkt)
            with self._latest_lock:
                self._latest_rs = pkt

            if self.on_rs is not None:
                try:
                    self.on_rs(pkt)
                except Exception as e:
                    print(f"[RS] on_rs callback error: {e}")

            # ---- Piggyback PPG handling ----
            # If the sender includes packet["ppg_latest"] = (ts, red, ir),
            # inject it into the same PPG pathways (queue + plot buffer + rate).
            try:
                pig = pkt.raw.get("ppg_latest", None)
            except Exception:
                pig = None

            if pig is not None:
                try:
                    ts, red, ir = pig
                    ts = float(ts)
                    red = int(red)
                    ir = int(ir)
                except Exception:
                    ts = None

                if ts is not None and ts > self._last_piggy_ppg_ts:
                    self._last_piggy_ppg_ts = ts

                    # Update PPG rate using 1 sample per RGBD packet piggyback
                    if self._ppg_rate is not None:
                        self._ppg_rate.add(1)

                    # Put into PPG queue so downstream consumer sees it
                    src_ip = "piggyback"
                    self._ppg_q.put((ts, red, ir, src_ip))
                    with self._latest_lock:
                        self._latest_ppg = (ts, red, ir, src_ip)

                    # Feed plot window if enabled
                    if self._ppg_window is not None:
                        self._ppg_window.append(ts, red, ir)

    # -----------------------
    # Core API
    # -----------------------

    def start(self) -> None:
        if self._t_ppg and self._t_ppg.is_alive():
            return
        self._stop.clear()

        self._t_ppg = threading.Thread(target=self._ppg_loop, daemon=True)
        self._t_rs = threading.Thread(target=self._rs_loop, daemon=True)

        self._t_ppg.start()
        self._t_rs.start()

    def stop(self) -> None:
        self._stop.set()

        if self._t_ppg:
            self._t_ppg.join(timeout=2.0)
        if self._t_rs:
            self._t_rs.join(timeout=2.0)

        try:
            self._rs_rx.stop()
        except Exception:
            pass

    def get_ppg(self) -> List[Tuple[float, int, int, str]]:
        return self._ppg_q.get_all()

    def get_rs(self) -> List[RealSenseFramePacket]:
        return self._rs_q.get_all()

    def latest_ppg(self) -> Optional[Tuple[float, int, int, str]]:
        with self._latest_lock:
            return self._latest_ppg

    def latest_rs(self) -> Optional[RealSenseFramePacket]:
        with self._latest_lock:
            return self._latest_rs


# ---------------------------------------------------------------------
# Visualization (OpenCV + Matplotlib)
# ---------------------------------------------------------------------

def start_ppg_matplotlib_animation(window: PPGWindowBuffer, interval_ms: int = 50):
    """
    Reuses PlotSink logic from ppg_receive.py, but starts it non-blocking so it can
    coexist with OpenCV windows (or headless loop).
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    sink = PlotSink(window)

    # Reuse the same animate function logic
    anim = animation.FuncAnimation(
        sink.fig, sink._animate, interval=int(interval_ms), blit=False,
        cache_frame_data=False
    )

    # non-blocking show
    plt.show(block=False)
    return sink, anim


def run_visual_loops(
    rx: PCSensorReceiver,
    show_rgb: bool,
    show_depth: bool,
    probe_center: bool,
    ppg_viz: bool,
    ppg_interval_ms: int,
) -> None:
    """
    Main-thread visual loop:
      - OpenCV: cv2.imshow + waitKey
      - Matplotlib: plt.pause to keep GUI responsive

    ESC closes OpenCV windows; Ctrl+C also exits.
    """
    import matplotlib.pyplot as plt

    if show_rgb:
        cv2.namedWindow("RealSense D455 RGB", cv2.WINDOW_NORMAL)
    if show_depth:
        cv2.namedWindow("RealSense D455 Depth (colormap)", cv2.WINDOW_NORMAL)

    sink = None
    anim = None
    if ppg_viz:
        w = rx.get_ppg_window()
        if w is None:
            raise RuntimeError("PPG window not enabled but ppg_viz requested.")
        sink, anim = start_ppg_matplotlib_animation(w, interval_ms=ppg_interval_ms)

    print("[VIZ] Running visualisation. ESC closes OpenCV windows.")
    try:
        while True:
            pkt = rx.latest_rs()

            if pkt is not None and pkt.rgb_bgr is not None:
                frame = pkt.rgb_bgr.copy()

                overlay_imu_text(frame, pkt.imu)

                if pkt.depth_u16 is not None and probe_center:
                    probe_xy = (frame.shape[1] // 2, frame.shape[0] // 2)
                    overlay_depth_probe(frame, pkt.depth_u16, pkt.depth_scale, probe_xy)

                if show_rgb:
                    cv2.imshow("RealSense D455 RGB", frame)

                if show_depth and pkt.depth_u16 is not None:
                    depth_vis = depth_to_colormap(pkt.depth_u16, pkt.depth_scale)
                    cv2.imshow("RealSense D455 Depth (colormap)", depth_vis)

            # OpenCV key handling
            if show_rgb or show_depth:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

            # keep matplotlib responsive if enabled
            if ppg_viz:
                plt.pause(0.001)

            time.sleep(0.005)

    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PC unified receiver for PPG + RealSense (RGB/Depth/IMU) over TCP.")
    ap.add_argument("--listen-ip", default="0.0.0.0", help="IP to bind TCP servers to (default: 0.0.0.0)")
    ap.add_argument("--ppg-port", type=int, default=9999, help="PPG TCP port (default: 9999)")
    ap.add_argument("--rs-port", type=int, default=50000, help="RealSense TCP port (default: 50000)")

    # RealSense visualization toggles
    ap.add_argument("--viz", action="store_true", help="Enable OpenCV visualisation windows (ESC to quit).")
    ap.add_argument("--no-rgb", action="store_true", help="Disable RGB window (only relevant with --viz).")
    ap.add_argument("--show-depth", action="store_true", help="Show depth colormap window if depth is present.")
    ap.add_argument("--no-depth-probe", action="store_true", help="Disable center depth probe overlay (with --viz).")

    # PPG trace visualization (requested)
    ap.add_argument("--ppg-viz", action="store_true", help="Enable Matplotlib live PPG traces.")
    ap.add_argument("--ppg-window", type=int, default=1500, help="PPG plot window size in samples (default: 1500).")
    ap.add_argument("--ppg-interval-ms", type=int, default=50, help="PPG plot refresh interval (ms).")

    # Rate printing (requested)
    ap.add_argument("--rate-print", action="store_true", help="Print PPG sampling rate (Hz) and RGB-D fps.")
    ap.add_argument("--rate-interval", type=float, default=2.0, help="Rate print interval seconds (default: 2.0).")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    cfg = PCReceiverConfig(
        listen_ip=args.listen_ip,
        ppg_port=args.ppg_port,
        rs_port=args.rs_port,
    )

    rx = PCSensorReceiver(cfg)

    if args.ppg_viz:
        rx.enable_ppg_window(window_size=args.ppg_window)

    if args.rate_print:
        rx.enable_rate_printing(interval_sec=args.rate_interval)

    rx.start()

    try:
        # If any visualization is enabled, run combined GUI loop in main thread
        if args.viz or args.ppg_viz:
            show_rgb = bool(args.viz) and not bool(args.no_rgb)
            show_depth = bool(args.viz) and bool(args.show_depth)
            probe_center = bool(args.viz) and not bool(args.no_depth_probe)

            run_visual_loops(
                rx,
                show_rgb=show_rgb,
                show_depth=show_depth,
                probe_center=probe_center,
                ppg_viz=bool(args.ppg_viz),
                ppg_interval_ms=int(args.ppg_interval_ms),
            )
        else:
            # Headless mode: just keep running (rates print automatically if enabled)
            while True:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping...")
    finally:
        rx.stop()


if __name__ == "__main__":
    main()

