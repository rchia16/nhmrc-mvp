#!/usr/bin/env python3
"""
UDP receiver for RealSense RGB JPEG + Depth PNG + IMU, with chunk reassembly.

Design:
- Collect chunks by seq until complete, then decode/display
- Drop incomplete frames after timeout
- Latest-wins display: if multiple complete frames exist, show newest
"""

import argparse
import socket
import struct
import time
from typing import Optional, Tuple

import numpy as np
import cv2


# ---------------- wire format (must match sender) ----------------
CHUNK_MAGIC = b"RSC0"
CHUNK_HDR_FMT = "!4sIHHHH"
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

FRAME_MAGIC = b"RSF0"
FRAME_HDR_FMT = "!4sBBHIdII B3x fff fff"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_IMU = 1 << 1


def decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_depth_png(depth_png: bytes) -> Optional[np.ndarray]:
    if not depth_png:
        return None
    arr = np.frombuffer(depth_png, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    return depth


def depth_to_colormap(depth_u16: np.ndarray, depth_scale: float = 0.001) -> np.ndarray:
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, 0.0, 6.0)
    depth_8u = (depth_m / 6.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - depth_8u, cv2.COLORMAP_TURBO)


def overlay_hud(frame: np.ndarray, lines) -> None:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24


class Reassembly:
    __slots__ = ("cnt", "parts", "lens", "t0", "got", "total_len")
    def __init__(self, cnt: int):
        self.cnt = cnt
        self.parts = [None] * cnt
        self.lens = [0] * cnt
        self.got = 0
        self.total_len = 0
        self.t0 = time.time()

    def add(self, idx: int, frag: bytes):
        if idx < 0 or idx >= self.cnt:
            return
        if self.parts[idx] is None:
            self.parts[idx] = frag
            self.lens[idx] = len(frag)
            self.got += 1
            self.total_len += len(frag)

    def complete(self) -> bool:
        return self.got == self.cnt

    def build(self) -> bytes:
        return b"".join(self.parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--show-depth", action="store_true")
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="Used only for depth colormap visualization")
    ap.add_argument("--timeout-ms", type=int, default=200, help="Drop incomplete frames after this time")
    args = ap.parse_args()

    show_rgb = not args.no_rgb
    show_depth = bool(args.show_depth)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_ip, int(args.port)))
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    except Exception:
        pass
    sock.settimeout(0.5)

    cv2.setUseOptimized(True)

    frames = {}  # seq -> Reassembly
    newest_complete = None
    newest_seq = None

    # stats
    t0 = time.time()
    rx_chunks = 0
    rx_frames_ok = 0
    rx_frames_drop = 0
    last_stat = t0
    fps = 0.0

    print(f"[UDP] Listening on {args.listen_ip}:{args.port}")

    try:
        while True:
            # Receive a burst
            try:
                data, _addr = sock.recvfrom(65535)
                if (rx_chunks % 200) == 0:
                    print(f"[RX] got {len(data)} bytes from {_addr}")
            except socket.timeout:
                data = None

            now = time.time()

            # Cleanup old/incomplete
            timeout_s = args.timeout_ms / 1000.0
            dead = []
            for seq, r in frames.items():
                if (now - r.t0) > timeout_s:
                    dead.append(seq)
            for seq_dead in dead:
                frames.pop(seq_dead, None)
                rx_frames_drop += 1
                if (rx_frames_drop % 10) == 0:
                    print(f"[DROP] incomplete frames dropped={rx_frames_drop}"\
                          f" (latest drop seq={seq_dead})")

            if not data:
                # still display latest complete if available
                pass
            else:
                if len(data) < CHUNK_HDR_SIZE:
                    continue
                magic, seq, idx, cnt, frag_len, _res = struct.unpack(CHUNK_HDR_FMT, data[:CHUNK_HDR_SIZE])
                if magic != CHUNK_MAGIC:
                    continue
                frag = data[CHUNK_HDR_SIZE:CHUNK_HDR_SIZE + frag_len]
                rx_chunks += 1

                MAX_INFLIGHT = 8
                if len(frames) > MAX_INFLIGHT:
                    # drop oldest inflight sequences
                    for old_seq in sorted(frames.keys())[:-MAX_INFLIGHT]:
                        frames.pop(old_seq, None)
                        rx_frames_drop += 1

                # Debug: show seq/cnt occasionally
                if (rx_chunks % 200) == 0:
                    print(f"[RX] chunks={rx_chunks} latest_seq={seq} idx={idx}/{cnt}")


                r = frames.get(seq)
                if r is None:
                    r = Reassembly(cnt=cnt)
                    frames[seq] = r
                r.add(idx, frag)

                if r.complete():
                    print(f"[OK] completed frame seq={seq} chunks={cnt} bytes={r.total_len}")

                    payload = r.build()
                    frames.pop(seq, None)

                    # Parse frame header
                    if len(payload) < FRAME_HDR_SIZE:
                        rx_frames_drop += 1
                    else:
                        (fmagic, ver, flags, _rsv, fseq, t_sender,
                         jpeg_len, depth_len, imu_mask,
                         ax, ay, az, gx, gy, gz) = struct.unpack(FRAME_HDR_FMT, payload[:FRAME_HDR_SIZE])
                        if fmagic != FRAME_MAGIC:
                            rx_frames_drop += 1
                        else:
                            # newest-wins
                            newest_complete = payload
                            newest_seq = fseq
                            rx_frames_ok += 1

            # Display newest complete (drop older without decoding)
            if newest_complete is not None:
                payload = newest_complete
                newest_complete = None

                (fmagic, ver, flags, _rsv, fseq, t_sender,
                 jpeg_len, depth_len, imu_mask,
                 ax, ay, az, gx, gy, gz) = struct.unpack(FRAME_HDR_FMT, payload[:FRAME_HDR_SIZE])

                off = FRAME_HDR_SIZE
                jpeg_bytes = payload[off:off + jpeg_len]
                off += jpeg_len
                depth_bytes = payload[off:off + depth_len] if (flags & FLAG_HAS_DEPTH) else b""

                frame = decode_jpeg(jpeg_bytes) if show_rgb else None
                depth_u16 = decode_depth_png(depth_bytes) if (show_depth and depth_bytes) else None

                # One-way latency estimate (requires clock sync)
                lat_ms = (time.time() - float(t_sender)) * 1000.0

                if frame is not None:
                    hud = [
                        f"seq={fseq}",
                        f"FPS(avg)={fps:.1f}",
                        f"chunks={rx_chunks}  ok={rx_frames_ok}  dropped={rx_frames_drop}",
                        f"one-way latency*={lat_ms:.1f} ms",
                        "*Requires Pi/PC clock sync (NTP/Chrony)",
                    ]
                    if imu_mask & 1:
                        hud.append(f"ACC: {ax:+.3f} {ay:+.3f} {az:+.3f}")
                    if imu_mask & 2:
                        hud.append(f"GYR: {gx:+.3f} {gy:+.3f} {gz:+.3f}")

                    overlay_hud(frame, hud)
                    cv2.imshow("RS UDP RGB", frame)

                if show_depth and (depth_u16 is not None):
                    depth_vis = depth_to_colormap(depth_u16, depth_scale=args.depth_scale)
                    cv2.imshow("RS UDP Depth", depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            # stats update
            dt = now - last_stat
            if dt >= 1.0:
                fps = (rx_frames_ok / (now - t0)) if (now > t0) else 0.0
                last_stat = now

    finally:
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()

