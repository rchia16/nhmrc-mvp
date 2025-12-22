#!/usr/bin/env python3
"""
rs_d455_raw_udp_receiver.py (library-like split; compatible with rs_d455_raw_udp_sender.py)

This version is verified against the protocol in /mnt/data/rs_d455_raw_udp_sender.py:
- CHUNK_MAGIC=b"RSC0", CHUNK_HDR_FMT="!4sIHHHH"
- FRAME_MAGIC=b"RSR0", FRAME_HDR_FMT exactly matches sender
- CFMT_BGR8=1, DFMT_Z16=1 (IMPORTANT: sender uses 1, not 0)
- The header field after (c_len, d_len) is an IMU MASK (uint8), NOT an IMU blob length.
  Sender does NOT append an IMU blob; accel/gyro are in the header.

Provides:
- Reassembly (chunk reassembly)
- parse_raw_frame(payload) -> dict
- RealSenseRawUDPReceiver (headless receiver; callback or polling)
- RealSenseRawUDPViewer (optional OpenCV viewer)
- main(): tiny wiring for quick tests
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import cv2


# ---------------- wire format (MUST match rs_d455_raw_udp_sender.py) ----------------
CHUNK_MAGIC = b"RSC0"
CHUNK_HDR_FMT = "!4sIHHHH"
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

FRAME_MAGIC = b"RSR0"
FRAME_HDR_FMT = (
    "!4sBBHI d f "
    "HHIB3x "
    "HHIB3x "
    "II "
    "B3x fff fff"
)
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_IMU = 1 << 1

# IMPORTANT: must match sender constants
CFMT_BGR8 = 1   # uint8, HxWx3 (sender sets to 1)
DFMT_Z16 = 1    # uint16, HxW  (sender sets to 1)


# ---------------- utility: visualization helpers ----------------
def depth_to_colormap(depth_u16: np.ndarray, depth_scale: float = 0.001) -> np.ndarray:
    d_m = depth_u16.astype(np.float32) * float(depth_scale)
    d_m = np.clip(d_m, 0.0, 5.0)
    d8 = (255.0 * (d_m / 5.0)).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def overlay_text(frame: np.ndarray, lines) -> np.ndarray:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return frame


# ---------------- core: reassembly + parsing ----------------
class Reassembly:
    __slots__ = ("cnt", "parts", "t0", "got")

    def __init__(self, cnt: int):
        self.cnt = int(cnt)
        self.parts = [None] * self.cnt
        self.got = 0
        self.t0 = time.time()

    def add(self, idx: int, frag: bytes):
        idx = int(idx)
        if idx < 0 or idx >= self.cnt:
            return
        if self.parts[idx] is None:
            self.parts[idx] = frag
            self.got += 1

    def complete(self) -> bool:
        return self.got == self.cnt

    def build(self) -> bytes:
        return b"".join(self.parts)


def parse_raw_frame(payload: bytes) -> Optional[Dict[str, Any]]:
    """Parse one complete frame payload (after UDP chunk reassembly)."""
    if len(payload) < FRAME_HDR_SIZE:
        return None

    (
        magic, ver, flags, _rsv, fseq, t_sender, depth_scale,
        cw, ch, cstride, cfmt,
        dw, dh, dstride, dfmt,
        c_len, d_len,
        imu_mask,          # NOTE: uint8 mask (1=accel present, 2=gyro present) from sender
        ax, ay, az,
        gx, gy, gz
    ) = struct.unpack(FRAME_HDR_FMT, payload[:FRAME_HDR_SIZE])

    if magic != FRAME_MAGIC:
        return None

    cw, ch, cstride = int(cw), int(ch), int(cstride)
    dw, dh, dstride = int(dw), int(dh), int(dstride)

    off = FRAME_HDR_SIZE

    # color bytes
    c_len = int(c_len)
    if off + c_len > len(payload):
        return None
    c_bytes = payload[off:off + c_len]
    off += c_len

    # depth bytes (optional)
    d_bytes = b""
    d_len = int(d_len)
    if flags & FLAG_HAS_DEPTH:
        if off + d_len > len(payload):
            return None
        d_bytes = payload[off:off + d_len]
        off += d_len

    # IMPORTANT: sender does NOT append any IMU blob; accel/gyro are in header only.

    # Color decode (raw BGR8)
    if int(cfmt) != CFMT_BGR8:
        return None
    color = np.frombuffer(c_bytes, dtype=np.uint8)

    expected_stride = ch * cstride
    if expected_stride == len(c_bytes):
        color = color.reshape((ch, cstride))
        # BGR8: cw*3 bytes per pixel row
        color = color[:, : cw * 3].reshape((ch, cw, 3))
    else:
        expected_tight = ch * cw * 3
        if expected_tight != len(c_bytes):
            return None
        color = color.reshape((ch, cw, 3))

    # Depth decode (raw Z16)
    depth = None
    if flags & FLAG_HAS_DEPTH:
        if int(dfmt) != DFMT_Z16:
            return None
        depth_arr = np.frombuffer(d_bytes, dtype=np.uint16)

        per_row_u16 = dstride // 2  # stride in bytes -> u16 count
        expected_stride_u16 = dh * per_row_u16
        if expected_stride_u16 * 2 == len(d_bytes) and per_row_u16 > 0:
            depth = depth_arr.reshape((dh, per_row_u16))[:, :dw]
        else:
            expected_tight_u16 = dh * dw
            if expected_tight_u16 * 2 != len(d_bytes):
                return None
            depth = depth_arr.reshape((dh, dw))

    imu = None
    if flags & FLAG_HAS_IMU:
        imu = {
            "mask": int(imu_mask),  # 1 accel, 2 gyro, 3 both
            "accel": (float(ax), float(ay), float(az)) if (int(imu_mask) & 1) else None,
            "gyro": (float(gx), float(gy), float(gz)) if (int(imu_mask) & 2) else None,
        }

    return {
        "ver": int(ver),
        "flags": int(flags),
        "fseq": int(fseq),
        "t_sender": float(t_sender),
        "depth_scale": float(depth_scale),
        "cw": cw, "ch": ch, "cstride": cstride,
        "dw": dw, "dh": dh, "dstride": dstride,
        "color": color,
        "depth": depth,
        "imu": imu,
    }


# ---------------- library: receiver ----------------
FrameCallback = Callable[[Dict[str, Any]], None]


@dataclass
class ReceiverStats:
    frames_ok: int = 0
    frames_drop: int = 0
    fps: float = 0.0
    last_latency_ms: float = 0.0


class RealSenseRawUDPReceiver:
    """
    Headless receiver (library-style).

    - start(): runs in background thread
    - run_forever(): blocking receive loop
    - get_latest(): latest parsed pkt
    - optional on_frame callback invoked per parsed frame (same receiver thread)

    Designed to be imported by your PC orchestrator.
    """

    def __init__(
        self,
        listen_ip: str = "0.0.0.0",
        port: int = 50010,
        timeout_ms: int = 200,
        max_inflight: int = 8,
        rcvbuf_bytes: int = 1 << 22,
        on_frame: Optional[FrameCallback] = None,
    ):
        self.listen_ip = str(listen_ip)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.max_inflight = int(max_inflight)
        self.rcvbuf_bytes = int(rcvbuf_bytes)
        self.on_frame = on_frame

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest_lock = threading.Lock()
        self._latest_pkt: Optional[Dict[str, Any]] = None

        self._stats_lock = threading.Lock()
        self._stats = ReceiverStats()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._latest_lock:
            return self._latest_pkt

    def get_stats(self) -> ReceiverStats:
        with self._stats_lock:
            return ReceiverStats(**self._stats.__dict__)

    def run_forever(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.listen_ip, self.port))
        sock.settimeout(0.5)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_bytes)
        except Exception:
            pass

        frames: Dict[int, Reassembly] = {}
        newest_complete: Optional[bytes] = None

        timeout_s = self.timeout_ms / 1000.0
        t0 = time.time()
        last_stat = t0

        try:
            while not self._stop_evt.is_set():
                now = time.time()

                # drop old inflight frames
                dead = [seq for seq, r in frames.items() if (now - r.t0) > timeout_s]
                if dead:
                    for seq in dead:
                        frames.pop(seq, None)
                    with self._stats_lock:
                        self._stats.frames_drop += len(dead)

                # recv
                try:
                    data, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    data = None
                except OSError:
                    break

                if data and len(data) >= CHUNK_HDR_SIZE:
                    magic, seq, idx, cnt, frag_len, _res = struct.unpack(
                        CHUNK_HDR_FMT, data[:CHUNK_HDR_SIZE]
                    )
                    if magic == CHUNK_MAGIC:
                        frag = data[CHUNK_HDR_SIZE:CHUNK_HDR_SIZE + frag_len]

                        # cap inflight
                        if len(frames) > self.max_inflight:
                            overflow = len(frames) - self.max_inflight
                            for old_seq in sorted(frames.keys())[:overflow]:
                                frames.pop(old_seq, None)
                            with self._stats_lock:
                                self._stats.frames_drop += max(0, overflow)

                        r = frames.get(seq)
                        if r is None:
                            r = Reassembly(cnt=cnt)
                            frames[seq] = r
                        r.add(idx, frag)

                        if r.complete():
                            newest_complete = r.build()
                            frames.pop(seq, None)
                            with self._stats_lock:
                                self._stats.frames_ok += 1

                # parse + publish latest
                if newest_complete is not None:
                    payload = newest_complete
                    newest_complete = None

                    pkt = parse_raw_frame(payload)
                    if pkt is None:
                        with self._stats_lock:
                            self._stats.frames_drop += 1
                    else:
                        try:
                            latency_ms = (time.time() - float(pkt["t_sender"])) * 1000.0
                        except Exception:
                            latency_ms = 0.0
                        with self._stats_lock:
                            self._stats.last_latency_ms = float(latency_ms)

                        with self._latest_lock:
                            self._latest_pkt = pkt

                        if self.on_frame is not None:
                            try:
                                self.on_frame(pkt)
                            except Exception:
                                # keep receiver robust
                                pass

                # stats update
                if (now - last_stat) >= 1.0:
                    with self._stats_lock:
                        dt = max(1e-6, (now - t0))
                        self._stats.fps = float(self._stats.frames_ok) / dt
                    last_stat = now

        finally:
            try:
                sock.close()
            except Exception:
                pass


# ---------------- library: viewer ----------------
class RealSenseRawUDPViewer:
    """
    Optional OpenCV viewer that polls a receiver.
    ESC to exit.
    """

    def __init__(
        self,
        show_rgb: bool = True,
        show_depth: bool = False,
        rgb_window_name: str = "RS UDP RAW Color",
        depth_window_name: str = "RS UDP RAW Depth",
    ):
        self.show_rgb = bool(show_rgb)
        self.show_depth = bool(show_depth)
        self.rgb_window_name = str(rgb_window_name)
        self.depth_window_name = str(depth_window_name)

    def run(self, receiver: RealSenseRawUDPReceiver, poll_hz: float = 120.0):
        period = 1.0 / max(1.0, float(poll_hz))
        try:
            while True:
                pkt = receiver.get_latest()
                if pkt is not None:
                    stats = receiver.get_stats()
                    self._render(pkt, stats)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                time.sleep(period)
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render(self, pkt: Dict[str, Any], stats: ReceiverStats):
        if self.show_rgb and pkt.get("color") is not None:
            frame = pkt["color"].copy()
            imu_present = pkt.get("imu") is not None
            frame = overlay_text(
                frame,
                [
                    f"seq={pkt.get('fseq')}  latency={stats.last_latency_ms:.1f}ms",
                    f"color={pkt.get('cw')}x{pkt.get('ch')}  depth={'yes' if pkt.get('depth') is not None else 'no'}  imu={'yes' if imu_present else 'no'}",
                    f"fps~{stats.fps:.1f}  ok={stats.frames_ok} drop={stats.frames_drop}",
                ],
            )
            cv2.imshow(self.rgb_window_name, frame)

        if self.show_depth and pkt.get("depth") is not None:
            depth_vis = depth_to_colormap(pkt["depth"], depth_scale=pkt.get("depth_scale", 0.001))
            cv2.imshow(self.depth_window_name, depth_vis)


# ---------------- tiny main for testing ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--timeout-ms", type=int, default=200)
    ap.add_argument("--max-inflight", type=int, default=8)
    ap.add_argument("--show-depth", action="store_true")
    ap.add_argument("--no-rgb", action="store_true")
    args = ap.parse_args()

    receiver = RealSenseRawUDPReceiver(
        listen_ip=args.listen_ip,
        port=args.port,
        timeout_ms=args.timeout_ms,
        max_inflight=args.max_inflight,
        on_frame=None,  # viewer polls
    )
    receiver.start()

    viewer = RealSenseRawUDPViewer(
        show_rgb=(not args.no_rgb),
        show_depth=args.show_depth,
    )

    try:
        viewer.run(receiver)
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()

