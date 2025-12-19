#!/usr/bin/env python3
"""
UDP receiver for RealSense RAW RGB (BGR8) + RAW Depth (Z16) + IMU, with chunk reassembly.

- Reassembles chunked UDP frames by seq
- Drops incomplete frames after timeout
- Latest-wins display
- No JPEG/PNG decoding; just reshape raw buffers
"""

import argparse
import socket
import struct
import time
from typing import Optional

import numpy as np
import cv2


# ---------------- wire format (must match sender) ----------------
CHUNK_MAGIC = b"RSC0"
CHUNK_HDR_FMT = "!4sIHHHH"
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

FRAME_MAGIC = b"RSR0"
FRAME_HDR_FMT = "!4sBBHI d f " \
                "HHIB3x " \
                "HHIB3x " \
                "II " \
                "B3x fff fff"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_IMU   = 1 << 1

CFMT_BGR8 = 1
DFMT_Z16  = 1


def depth_to_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
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
    __slots__ = ("cnt", "parts", "t0", "got", "total_len")
    def __init__(self, cnt: int):
        self.cnt = cnt
        self.parts = [None] * cnt
        self.got = 0
        self.total_len = 0
        self.t0 = time.time()

    def add(self, idx: int, frag: bytes):
        if idx < 0 or idx >= self.cnt:
            return
        if self.parts[idx] is None:
            self.parts[idx] = frag
            self.got += 1
            self.total_len += len(frag)

    def complete(self) -> bool:
        return self.got == self.cnt

    def build(self) -> bytes:
        return b"".join(self.parts)


def parse_raw_frame(payload: bytes):
    if len(payload) < FRAME_HDR_SIZE:
        return None

    (magic, ver, flags, _rsv, fseq, t_sender, depth_scale,
     cw, ch, cstride, cfmt,
     dw, dh, dstride, dfmt,
     c_len, d_len,
     imu_mask,
     ax, ay, az, gx, gy, gz) = struct.unpack(FRAME_HDR_FMT, payload[:FRAME_HDR_SIZE])

    if magic != FRAME_MAGIC:
        return None
    if ver != 1:
        return None
    if cfmt != CFMT_BGR8:
        return None
    if (flags & FLAG_HAS_DEPTH) and (dfmt != DFMT_Z16):
        return None

    off = FRAME_HDR_SIZE
    end_c = off + int(c_len)
    if end_c > len(payload):
        return None
    c_bytes = payload[off:end_c]
    off = end_c

    d_bytes = b""
    if flags & FLAG_HAS_DEPTH:
        end_d = off + int(d_len)
        if end_d > len(payload):
            return None
        d_bytes = payload[off:end_d]

    # Reconstruct numpy arrays without decode
    # Color: uint8 HxWx3 (BGR)
    color = np.frombuffer(c_bytes, dtype=np.uint8)
    expected_c = int(ch) * int(cstride)
    if expected_c != len(c_bytes):
        # Fallback: try tight packing (HxWx3)
        expected_tight = int(ch) * int(cw) * 3
        if expected_tight != len(c_bytes):
            return None
        color = color.reshape((int(ch), int(cw), 3))
    else:
        # Stride-aware view: reshape as rows, then slice to width*3
        color = color.reshape((int(ch), int(cstride)))
        color = color[:, : int(cw) * 3].reshape((int(ch), int(cw), 3))

    depth = None
    if flags & FLAG_HAS_DEPTH:
        depth_arr = np.frombuffer(d_bytes, dtype=np.uint16)
        expected_d = int(dh) * (int(dstride) // 2)  # stride in bytes -> u16 count
        if expected_d * 2 != len(d_bytes):
            # Fallback: tight packing HxW
            expected_tight = int(dh) * int(dw)
            if expected_tight * 2 != len(d_bytes):
                return None
            depth = depth_arr.reshape((int(dh), int(dw)))
        else:
            depth_arr = depth_arr.reshape((int(dh), int(dstride) // 2))
            depth = depth_arr[:, : int(dw)]

    imu = None
    if flags & FLAG_HAS_IMU:
        imu = {"accel": None, "gyro": None}
        if imu_mask & 1:
            imu["accel"] = (ax, ay, az)
        if imu_mask & 2:
            imu["gyro"] = (gx, gy, gz)

    return {
        "seq": int(fseq),
        "t_sender": float(t_sender),
        "depth_scale": float(depth_scale),
        "color": color,
        "depth": depth,
        "imu": imu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--show-depth", action="store_true")
    ap.add_argument("--no-rgb", action="store_true")
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

    # stats
    t0 = time.time()
    rx_chunks = 0
    rx_frames_ok = 0
    rx_frames_drop = 0
    last_stat = t0
    fps = 0.0

    print(f"[UDP RAW] Listening on {args.listen_ip}:{args.port}")

    try:
        while True:
            # Receive one datagram
            try:
                data, addr = sock.recvfrom(65535)
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

            if data:
                if len(data) >= CHUNK_HDR_SIZE:
                    magic, seq, idx, cnt, frag_len, _res = struct.unpack(CHUNK_HDR_FMT, data[:CHUNK_HDR_SIZE])
                    if magic == CHUNK_MAGIC:
                        frag = data[CHUNK_HDR_SIZE:CHUNK_HDR_SIZE + frag_len]
                        rx_chunks += 1

                        MAX_INFLIGHT = 8
                        if len(frames) > MAX_INFLIGHT:
                            for old_seq in sorted(frames.keys())[:-MAX_INFLIGHT]:
                                frames.pop(old_seq, None)
                                rx_frames_drop += 1

                        r = frames.get(seq)
                        if r is None:
                            r = Reassembly(cnt=cnt)
                            frames[seq] = r
                        r.add(idx, frag)

                        if r.complete():
                            payload = r.build()
                            frames.pop(seq, None)
                            newest_complete = payload
                            rx_frames_ok += 1

            # Display newest complete (latest-wins)
            if newest_complete is not None:
                payload = newest_complete
                newest_complete = None

                pkt = parse_raw_frame(payload)
                if pkt is None:
                    rx_frames_drop += 1
                else:
                    lat_ms = (time.time() - float(pkt["t_sender"])) * 1000.0

                    if show_rgb and pkt["color"] is not None:
                        frame = pkt["color"].copy()  # safe for HUD overlay
                        hud = [
                            f"seq={pkt['seq']}",
                            f"FPS(avg)={fps:.1f}",
                            f"chunks={rx_chunks} ok={rx_frames_ok} dropped={rx_frames_drop}",
                            f"one-way latency*={lat_ms:.1f} ms",
                            "*Requires Pi/PC clock sync (NTP/Chrony)",
                        ]
                        imu = pkt.get("imu") or {}
                        if imu.get("accel") is not None:
                            ax, ay, az = imu["accel"]
                            hud.append(f"ACC: {ax:+.3f} {ay:+.3f} {az:+.3f}")
                        if imu.get("gyro") is not None:
                            gx, gy, gz = imu["gyro"]
                            hud.append(f"GYR: {gx:+.3f} {gy:+.3f} {gz:+.3f}")

                        overlay_hud(frame, hud)
                        cv2.imshow("RS UDP RAW RGB", frame)

                    if show_depth and pkt["depth"] is not None:
                        depth_vis = depth_to_colormap(pkt["depth"], depth_scale=pkt["depth_scale"])
                        cv2.imshow("RS UDP RAW Depth", depth_vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

            # stats update
            if (now - last_stat) >= 1.0:
                fps = (rx_frames_ok / (now - t0)) if (now > t0) else 0.0
                last_stat = now

    finally:
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()

