#!/usr/bin/env python3
"""
RealSense D455 -> UDP RAW streamer (RGB BGR8 + Depth Z16 + IMU).

Updates implemented:
B) Removed huge-copy hot path on the Pi
   - No .tobytes()
   - No hdr + bytes + bytes concatenation
   - Uses memoryview() and socket.sendmsg() scatter/gather to send slices without copying.

C) Pipeline on the Pi
   - Thread 1: RealSense capture -> pushes latest frameset into a "latest-only" queue
   - Thread 2: Packetize + send -> pulls latest frameset and transmits; drops stale frames.

Notes:
- No rs.align on the Pi.
- No JPEG/PNG encoding on the Pi.
- RAW bandwidth can be high; tune resolution/FPS and consider jumbo frames.

Tested assumptions:
- color stream is BGR8
- depth stream is Z16
"""

import argparse
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pyrealsense2 as rs

from imu_reader import IMUReader


# ---------------- Chunking wire format ----------------
# Datagram header:
#   magic(4)='RSC0' | seq(u32) | idx(u16) | cnt(u16) | frag_len(u16) | reserved(u16)
CHUNK_MAGIC = b"RSC0"
CHUNK_HDR_FMT = "!4sIHHHH"
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT)

# Frame header (start of reconstructed payload):
#   magic(4)='RSR0' | ver(u8) | flags(u8) | reserved(u16)
#   seq(u32) | t_sender(f64) | depth_scale(f32)
#   cw(u16) ch(u16) cstride(u32) cfmt(u8) pad(3)
#   dw(u16) dh(u16) dstride(u32) dfmt(u8) pad(3)
#   c_len(u32) d_len(u32)
#   imu_mask(u8) pad(3)
#   accel_xyz(f32*3) | gyro_xyz(f32*3)
FRAME_MAGIC = b"RSR0"
FRAME_HDR_FMT = "!4sBBHI d f " \
                "HHIB3x " \
                "HHIB3x " \
                "II " \
                "B3x fff fff"
FRAME_HDR_SIZE = struct.calcsize(FRAME_HDR_FMT)

FLAG_HAS_DEPTH = 1 << 0
FLAG_HAS_IMU   = 1 << 1

# format codes
CFMT_BGR8 = 1      # uint8, HxWx3
DFMT_Z16  = 1      # uint16, HxW


@dataclass
class CapturedPacket:
    """A unit of work handed from capture thread -> send thread."""
    frameset: rs.composite_frame
    t_sender: float
    accel: Optional[Tuple[float, float, float]]
    gyro: Optional[Tuple[float, float, float]]


class LatestOnlyQueue:
    """
    A lightweight 'latest wins' handoff.
    - Producer overwrites the latest item.
    - Consumer waits for an item and then takes it.
    - Stale items are dropped automatically.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._item: Optional[CapturedPacket] = None
        self._closed = False

    def put_latest(self, item: CapturedPacket) -> None:
        with self._cond:
            self._item = item
            self._cond.notify()

    def get_latest(self, timeout: Optional[float] = None) -> Optional[CapturedPacket]:
        with self._cond:
            if self._item is None and not self._closed:
                self._cond.wait(timeout=timeout)
            item = self._item
            self._item = None
            return item

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


def pack_frame_header(
    seq: int,
    t_sender: float,
    depth_scale: float,
    cw: int, ch: int, cstride: int,
    dw: int, dh: int, dstride: int,
    c_len: int, d_len: int,
    accel: Optional[Tuple[float, float, float]],
    gyro: Optional[Tuple[float, float, float]],
) -> bytes:
    flags = 0
    if d_len > 0:
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

    return struct.pack(
        FRAME_HDR_FMT,
        FRAME_MAGIC,
        1,                  # ver
        flags,
        0,                  # reserved
        int(seq),
        float(t_sender),
        float(depth_scale),

        int(cw), int(ch), int(cstride), int(CFMT_BGR8),
        int(dw), int(dh), int(dstride), int(DFMT_Z16),

        int(c_len), int(d_len),

        int(imu_mask),
        float(ax), float(ay), float(az),
        float(gx), float(gy), float(gz),
    )


def _frag_from_segments(segments, offset: int, length: int):
    """
    Return a list of memoryviews that cover [offset, offset+length)
    across multiple segments, without copying.
    segments: list[memoryview]
    """
    out = []
    remaining = length
    off = offset
    for seg in segments:
        seg_len = len(seg)
        if off >= seg_len:
            off -= seg_len
            continue
        take = min(remaining, seg_len - off)
        out.append(seg[off:off+take])
        remaining -= take
        off = 0
        if remaining <= 0:
            break
    return out


def send_frame_chunked_sendmsg(
    sock: socket.socket,
    dst,
    seq: int,
    frame_hdr: bytes,
    color_buf: memoryview,
    depth_buf: Optional[memoryview],
    mtu_payload: int,
):
    """
    Chunk logical payload [frame_hdr | color_buf | depth_buf] into UDP datagrams.

    Uses sendmsg([chunk_hdr_bytes, frag_part1, frag_part2, ...]) to avoid
    concatenating large bytes objects.
    """
    hdr_mv = memoryview(frame_hdr)
    segments = [hdr_mv, color_buf]
    if depth_buf is not None:
        segments.append(depth_buf)

    total_len = sum(len(s) for s in segments)
    max_frag = max(256, int(mtu_payload) - CHUNK_HDR_SIZE)
    cnt = (total_len + max_frag - 1) // max_frag
    if cnt > 65535:
        raise RuntimeError("Frame too large to chunk (cnt>65535). Reduce resolution/FPS or increase MTU.")

    for idx in range(cnt):
        off = idx * max_frag
        frag_len = min(max_frag, total_len - off)

        frag_parts = _frag_from_segments(segments, off, frag_len)

        # datagram header is tiny; bytes is fine
        chdr = struct.pack(
            CHUNK_HDR_FMT,
            CHUNK_MAGIC,
            int(seq),
            int(idx),
            int(cnt),
            int(frag_len),
            0,
        )

        # Scatter/gather: avoid chdr + frag copy
        sock.sendmsg([chdr, *frag_parts], [], 0, dst)


def capture_loop(pipeline: rs.pipeline, imu_reader: IMUReader, q: LatestOnlyQueue, stop_evt: threading.Event, want_depth: bool):
    """
    Thread 1: Capture frames from RealSense and push latest into queue.
    """
    while not stop_evt.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        depth_frame = None
        if want_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

        accel, gyro = imu_reader.get_latest()
        pkt = CapturedPacket(
            frameset=frames,
            t_sender=time.time(),
            accel=accel,
            gyro=gyro,
        )
        q.put_latest(pkt)


def sender_loop(
    sock: socket.socket,
    dst,
    q: LatestOnlyQueue,
    stop_evt: threading.Event,
    depth_scale: float,
    mtu_payload: int,
    want_depth: bool,
):
    """
    Thread 2: Pull latest frameset and transmit over UDP.
    """
    seq = 0
    last_log = time.time()
    sent_frames = 0
    sent_bits = 0

    while not stop_evt.is_set():
        pkt = q.get_latest(timeout=0.5)
        if pkt is None:
            continue

        frames = pkt.frameset
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        depth_frame = frames.get_depth_frame() if want_depth else None
        if want_depth and not depth_frame:
            continue

        # Pull raw buffers as memoryviews (zero-copy view of pyrealsense2 buffer)
        cw = int(color_frame.get_width())
        ch = int(color_frame.get_height())
        cstride = int(color_frame.get_stride_in_bytes())
        c_len = cstride * ch
        color_buf = memoryview(color_frame.get_data()).cast('B')
        # Ensure our buffer length matches what we advertise in the header
        if len(color_buf) != c_len:
            # Fallback to "tight" assumption for BGR8 (no row padding)
            c_len = cw * ch * 3
        color_buf = color_buf[:c_len]
        dw = dh = dstride = d_len = 0
        depth_buf = None
        if want_depth and depth_frame is not None:
            dw = int(depth_frame.get_width())
            dh = int(depth_frame.get_height())
            dstride = int(depth_frame.get_stride_in_bytes())
            d_len = dstride * dh
            depth_buf = memoryview(depth_frame.get_data()).cast('B')
            if len(depth_buf) != d_len:
                # tight Z16 (no row padding)
                d_len = dw * dh * 2
            depth_buf = depth_buf[:d_len]

        frame_hdr = pack_frame_header(
            seq=seq,
            t_sender=pkt.t_sender,
            depth_scale=depth_scale,
            cw=cw, ch=ch, cstride=cstride,
            dw=dw, dh=dh, dstride=dstride,
            c_len=c_len, d_len=d_len,
            accel=pkt.accel,
            gyro=pkt.gyro,
        )

        send_frame_chunked_sendmsg(
            sock=sock,
            dst=dst,
            seq=seq,
            frame_hdr=frame_hdr,
            color_buf=color_buf[:c_len],
            depth_buf=(depth_buf[:d_len] if depth_buf is not None else None),
            mtu_payload=mtu_payload,
        )

        seq = (seq + 1) & 0xFFFFFFFF
        sent_frames += 1
        sent_bits += (FRAME_HDR_SIZE + c_len + d_len) * 8

        now = time.time()
        if now - last_log >= 2.0:
            fps = sent_frames / (now - last_log)
            mbps = (sent_bits / (now - last_log)) / 1e6
            print(f"[TX] fps={fps:.1f}  approx_payload_mbps={mbps:.2f}")
            sent_frames = 0
            sent_bits = 0
            last_log = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc-ip", required=True, help="PC receiver IP")
    ap.add_argument("--pc-port", type=int, default=50010, help="PC receiver UDP port")

    ap.add_argument("--color-w", type=int, default=640) # 640
    ap.add_argument("--color-h", type=int, default=480) # 480
    ap.add_argument("--depth-w", type=int, default=480) # 640
    ap.add_argument("--depth-h", type=int, default=270) # 480
    ap.add_argument("--fps", type=int, default=15) # 30

    ap.add_argument("--no-depth", action="store_true", help="Send RGB only (lowest bandwidth)")
    ap.add_argument("--mtu-payload", type=int, default=1400, help="Max UDP datagram size (bytes)")
    args = ap.parse_args()

    dst = (args.pc_ip, int(args.pc_port))
    want_depth = not args.no_depth

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
    except Exception:
        pass

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, args.fps)
    if want_depth:
        config.enable_stream(rs.stream.depth, args.depth_w, args.depth_h, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    dev = profile.get_device()

    imu_reader = IMUReader(dev, accel_hz=250, gyro_hz=400)

    depth_scale = 0.001
    try:
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:
        pass

    print(f"[UDP RAW] Sending to {dst[0]}:{dst[1]}  fps={args.fps}  depth={'off' if not want_depth else 'on'}")
    print(f"[DEPTH] scale={depth_scale} m/unit (transmitted in header)")
    print(f"[UDP] mtu_payload={args.mtu_payload} (try ~1450 for MTU1500, ~8800 for jumbo 9000)")
    print("[NOTE] No alignment, no encoding; sender uses sendmsg() with memoryview slices (minimal copies).")

    q = LatestOnlyQueue()
    stop_evt = threading.Event()

    t_cap = threading.Thread(
        target=capture_loop,
        args=(pipeline, imu_reader, q, stop_evt, want_depth),
        daemon=True,
    )
    t_send = threading.Thread(
        target=sender_loop,
        args=(sock, dst, q, stop_evt, depth_scale, int(args.mtu_payload), want_depth),
        daemon=True,
    )

    t_cap.start()
    t_send.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_evt.set()
        q.close()
        try:
            t_cap.join(timeout=1.0)
            t_send.join(timeout=1.0)
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            imu_reader.stop()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

