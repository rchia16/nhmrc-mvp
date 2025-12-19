#!/usr/bin/env python3
"""
RealSense D455 -> UDP low-latency streamer (RGB JPEG + Depth PNG + IMU).

Key properties:
- UDP chunking (no TCP head-of-line blocking)
- Receiver reconstructs frames; incomplete frames are dropped
- “latest wins”: old frames can be discarded for low latency
"""

import argparse
import socket
import struct
import time
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
import pyrealsense2 as rs

from imu_reader import IMUReader


# ---------------- UDP framing ----------------
# Chunk header:
#   magic(4)='RSC0' | seq(u32) | idx(u16) | cnt(u16) | frag_len(u16) | reserved(u16)
CHUNK_MAGIC = b"RSC0"
CHUNK_HDR_FMT = "!4sIHHHH"
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

# Frame header at start of reconstructed payload:
#   magic(4)='RSF0' | ver(u8) | flags(u8) | reserved(u16)
#   seq(u32) | t_sender(f64)
#   jpeg_len(u32) | depth_len(u32)
#   imu_mask(u8) | pad(3)
#   accel_xyz(f32*3) | gyro_xyz(f32*3)
FRAME_MAGIC = b"RSF0"
FRAME_HDR_FMT = "!4sBBHIdII B3x fff fff"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_IMU = 1 << 1


def build_frame_payload(
    seq: int,
    t_sender: float,
    jpeg_bytes: bytes,
    depth_bytes: bytes,
    accel: Optional[Tuple[float, float, float]],
    gyro: Optional[Tuple[float, float, float]],
) -> bytes:
    flags = 0
    if depth_bytes:
        flags |= FLAG_HAS_DEPTH
    if (accel is not None) or (gyro is not None):
        flags |= FLAG_HAS_IMU

    imu_mask = 0
    ax = ay = az = 0.0
    gx = gy = gz = 0.0
    if accel is not None:
        imu_mask |= 1
        ax, ay, az = accel
    if gyro is not None:
        imu_mask |= 2
        gx, gy, gz = gyro

    hdr = struct.pack(
        FRAME_HDR_FMT,
        FRAME_MAGIC,
        1,                      # ver
        flags,
        0,                      # reserved
        int(seq),
        float(t_sender),
        int(len(jpeg_bytes)),
        int(len(depth_bytes)),
        int(imu_mask),
        float(ax), float(ay), float(az),
        float(gx), float(gy), float(gz),
    )
    return hdr + jpeg_bytes + depth_bytes


def send_chunked(sock: socket.socket, dst, seq: int, payload: bytes, mtu_payload: int):
    # payload is the reconstructed frame bytes (frame header + blobs)
    max_frag = max(256, int(mtu_payload) - CHUNK_HDR_SIZE)
    cnt = (len(payload) + max_frag - 1) // max_frag
    if cnt > 65535:
        raise RuntimeError("Frame too large to chunk (cnt>65535)")

    for idx in range(cnt):
        off = idx * max_frag
        frag = payload[off : off + max_frag]
        chdr = struct.pack(
            CHUNK_HDR_FMT,
            CHUNK_MAGIC,
            int(seq),
            int(idx),
            int(cnt),
            int(len(frag)),
            0,
        )
        sock.sendto(chdr + frag, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc-ip", required=True, help="PC receiver IP")
    ap.add_argument("--pc-port", type=int, default=50010, help="PC receiver UDP port")
    ap.add_argument("--color-w", type=int, default=640)
    ap.add_argument("--color-h", type=int, default=480)
    ap.add_argument("--depth-w", type=int, default=640)
    ap.add_argument("--depth-h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--jpeg-quality", type=int, default=60)
    ap.add_argument("--no-depth", action="store_true", help="Send RGB only (lowest latency)")
    ap.add_argument("--no-align", action="store_true", help="Do not align depth to color (faster)")
    ap.add_argument("--mtu-payload", type=int, default=1200, help="Max UDP datagram size (bytes)")
    args = ap.parse_args()

    dst = (args.pc_ip, int(args.pc_port))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    except Exception:
        pass

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.depth_w, args.depth_h, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    dev = profile.get_device()

    imu_reader = IMUReader(dev, accel_hz=250, gyro_hz=400)

    align = None
    if (not args.no_depth) and (not args.no_align):
        align = rs.align(rs.stream.color)

    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    print(f"[UDP] Sending to {dst[0]}:{dst[1]}  fps={args.fps}  jpegQ={args.jpeg_quality}  depth={'off' if args.no_depth else 'on'}  align={'off' if args.no_align else 'on'}")
    print(f"[DEPTH] scale={depth_scale} m/unit  (note: scale not transmitted in this minimal UDP format)")

    seq = 0
    last_log = time.time()
    sent_frames = 0
    sent_bytes = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()

            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            depth_frame = None
            if not args.no_depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

            t_sender = time.time()

            # IMU latest
            accel, gyro = imu_reader.get_latest()

            # Color -> JPEG
            color_image = np.asanyarray(color_frame.get_data())
            ok, enc = cv2.imencode(".jpg", color_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                continue
            jpeg_bytes = enc.tobytes()

            depth_bytes = b""
            if depth_frame is not None:
                depth_image = np.asanyarray(depth_frame.get_data())  # uint16
                # Fastest “good enough” is PNG with compression=0 (still CPU but avoids huge raw)
                okd, encd = cv2.imencode(".png", depth_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                if okd:
                    depth_bytes = encd.tobytes()

            payload = build_frame_payload(
                seq=seq,
                t_sender=t_sender,
                jpeg_bytes=jpeg_bytes,
                depth_bytes=depth_bytes,
                accel=accel,
                gyro=gyro,
            )

            send_chunked(sock, dst, seq=seq, payload=payload, mtu_payload=args.mtu_payload)

            seq = (seq + 1) & 0xFFFFFFFF
            sent_frames += 1
            sent_bytes += len(payload)

            now = time.time()
            if now - last_log >= 2.0:
                mbps = (sent_bytes * 8) / (now - last_log) / 1e6
                fps = sent_frames / (now - last_log)
                print(f"[TX] fps={fps:.1f}  approx_payload_mbps={mbps:.2f}")
                sent_frames = 0
                sent_bytes = 0
                last_log = now

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            imu_reader.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

