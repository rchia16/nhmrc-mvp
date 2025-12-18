#!/usr/bin/env python3
import argparse
import socket
import struct
import pickle
from typing import Optional, Tuple

import numpy as np
import cv2


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise EOFError."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("Socket closed while receiving data")
        data += chunk
    return data


def decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_depth_png(depth_png: bytes) -> Optional[np.ndarray]:
    """
    Decode a PNG-encoded depth image.
    Expected output: uint16 (z16) depth image aligned to color.
    """
    if not depth_png:
        return None
    arr = np.frombuffer(depth_png, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    return depth


def depth_to_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    Make a viewable depth colormap (BGR) from uint16 depth.
    Uses meters conversion to get a nicer dynamic range.
    """
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, 0.0, 6.0)
    depth_8u = (depth_m / 6.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - depth_8u, cv2.COLORMAP_TURBO)


def overlay_imu_text(frame: np.ndarray, imu: dict) -> None:
    text_lines = []
    accel = imu.get("accel")
    gyro = imu.get("gyro")

    if accel is not None:
        text_lines.append(f"ACC: {accel[0]:+.3f}, {accel[1]:+.3f}, {accel[2]:+.3f}")
    if gyro is not None:
        text_lines.append(f"GYR: {gyro[0]:+.3f}, {gyro[1]:+.3f}, {gyro[2]:+.3f}")

    y = 20
    for line in text_lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 20


def overlay_depth_probe(
    frame: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    probe_xy: Tuple[int, int],
) -> None:
    """Overlay depth (meters) at a given pixel position."""
    h, w = depth_u16.shape[:2]
    x, y = probe_xy
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    d_u = int(depth_u16[y, x])
    d_m = float(d_u) * float(depth_scale)

    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"Depth@({x},{y}): {d_m:.2f} m",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def handle_client(conn: socket.socket, show_depth: bool, show_rgb: bool) -> None:
    print("Client connected.")
    try:
        while True:
            header = recv_exact(conn, 4)
            (length,) = struct.unpack("!I", header)

            payload = recv_exact(conn, length)
            packet = pickle.loads(payload)

            jpeg_bytes = packet.get("jpeg")
            imu = packet.get("imu", {})

            depth_png = packet.get("depth_png")  # PNG-encoded uint16
            depth_scale = float(packet.get("depth_scale", 0.001))

            frame = decode_jpeg(jpeg_bytes)

            depth_u16 = None
            if depth_png is not None:
                depth_u16 = decode_depth_png(depth_png)

            if frame is None:
                print("Received frame could not be decoded.")
                continue

            overlay_imu_text(frame, imu)

            if depth_u16 is not None:
                probe_xy = (frame.shape[1] // 2, frame.shape[0] // 2)
                overlay_depth_probe(frame, depth_u16, depth_scale, probe_xy)

            if show_rgb:
                cv2.imshow("RealSense D455 RGB", frame)

            if show_depth and depth_u16 is not None:
                depth_vis = depth_to_colormap(depth_u16, depth_scale)
                cv2.imshow("RealSense D455 Depth (colormap)", depth_vis)

            if show_rgb or (show_depth and depth_u16 is not None):
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed, closing viewer.")
                    break

    except EOFError:
        print("Client disconnected.")
    except Exception as e:
        print(f"Receiver error: {e}")
    finally:
        conn.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Receive RealSense D455 RGB (+ optional depth + IMU) over TCP.")
    ap.add_argument("--listen-ip", default="0.0.0.0", help="IP to bind the TCP server to (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=50000, help="TCP port to listen on (default: 50000)")
    ap.add_argument("--no-rgb", action="store_true", help="Do not display the RGB window")
    ap.add_argument("--show-depth", action="store_true", help="If depth is present, display a depth colormap window")
    args = ap.parse_args()

    host = args.listen_ip
    port = args.port
    show_rgb = not args.no_rgb
    show_depth = bool(args.show_depth)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"Listening on {host}:{port} ...")
    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"Incoming connection from {addr}")
            handle_client(conn, show_depth=show_depth, show_rgb=show_rgb)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: shutting down server.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
