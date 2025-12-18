#!/usr/bin/env python3
import argparse
import socket
import struct
import pickle
import threading
import time
from collections import deque

import numpy as np
import cv2
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient

from yolo_sofa import SpatialSoundHeadphoneYOLO


# -----------------------------
# Helpers
# -----------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("Socket closed while receiving data")
        data += chunk
    return data


def decode_depth_png(depth_png: bytes) -> np.ndarray | None:
    """Decode a PNG-encoded uint16 depth image (aligned to color)."""
    if not depth_png:
        return None
    arr = np.frombuffer(depth_png, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16, copy=False)
    return depth


def depth_at_pixel_m(depth_u16: np.ndarray, depth_scale: float, x_px: int, y_px: int) -> float | None:
    """Return depth in meters at (x_px, y_px) with bounds check."""
    if depth_u16 is None:
        return None
    h, w = depth_u16.shape[:2]
    x = int(max(0, min(w - 1, int(x_px))))
    y = int(max(0, min(h - 1, int(y_px))))
    d_u = int(depth_u16[y, x])
    # Common invalid values: 0
    if d_u <= 0:
        return None
    return float(d_u) * float(depth_scale)


def prune_older_than(buf: deque, newest_t: float, window_sec: float):
    cutoff = newest_t - window_sec
    while buf and buf[0][0] < cutoff:
        buf.popleft()


# -----------------------------
# 1-minute rolling buffers
# -----------------------------
class RollingBuffer1Min:
    def __init__(self, window_sec: float = 60.0):
        self.window_sec = float(window_sec)
        self._lock = threading.Lock()
        self.imu = deque()   # (t, accel_tuple_or_None, gyro_tuple_or_None)
        self.ppg = deque()   # (t, red, ir)

    def add_imu(self, t: float, accel, gyro):
        with self._lock:
            self.imu.append((t, accel, gyro))
            prune_older_than(self.imu, t, self.window_sec)

    def add_ppg(self, t: float, red: int, ir: int):
        with self._lock:
            self.ppg.append((t, red, ir))
            prune_older_than(self.ppg, t, self.window_sec)

    def snapshot(self):
        with self._lock:
            return list(self.imu), list(self.ppg)


# -----------------------------
# RealSense TCP receiver + YOLO + OSC sender
# (mirrors yolo_realsense.py logic, but frames come over TCP)
# -----------------------------
class RealSenseYOLOFromTCP:
    def __init__(
        self,
        listen_ip: str,
        port: int,
        model_path: str,
        osc_ip: str,
        osc_port: int,
        conf_threshold: float = 0.3,
        show_debug: bool = False,
    ):
        self.listen_ip = listen_ip
        self.port = int(port)
        self.conf_threshold = float(conf_threshold)
        self.show_debug = bool(show_debug)

        self.model = YOLO(model_path)
        self.cls_names = self.model.names
        self.osc_client = SimpleUDPClient(osc_ip, int(osc_port))

        # Stability gating (same intent as yolo_realsense.py)
        self.cid_stability_threshold = 0.2
        self.position_bucket_count = 10
        self.cid_stability_reset_time = 1.0
        self.cid_position_state = {}  # (cid, bucket) -> {"first_seen": t0, "last_seen": t_last}

        self.frame_index = 0

        self._srv = None
        self._running = False
        self._thread = None

        self.on_packet = None  # callback(packet_dict)

    def _x_to_bucket(self, x_norm: float) -> int:
        x_clamped = max(0.0, min(1.0, float(x_norm)))
        b = int(x_clamped * self.position_bucket_count)
        return min(b, self.position_bucket_count - 1)

    def _is_cid_position_stable(self, cid: int, x_norm: float, now: float) -> bool:
        bucket = self._x_to_bucket(x_norm)
        key = (cid, bucket)
        state = self.cid_position_state.get(key)

        if state is None:
            self.cid_position_state[key] = {"first_seen": now, "last_seen": now}
            return False

        if now - state["last_seen"] > self.cid_stability_reset_time:
            self.cid_position_state[key] = {"first_seen": now, "last_seen": now}
            return False

        state["last_seen"] = now
        return (now - state["first_seen"]) >= self.cid_stability_threshold

    def _serve(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.listen_ip, self.port))
        self._srv.listen(1)

        print(f"[RS TCP] Listening on {self.listen_ip}:{self.port} ...")

        while self._running:
            conn, addr = self._srv.accept()
            print(f"[RS TCP] Incoming connection from {addr}")
            try:
                self._handle_client(conn)
            except Exception as e:
                print(f"[RS TCP] Client handler error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    def _handle_client(self, conn: socket.socket):
        while self._running:
            # packet format is identical to rs_d455_tcp_receiver.py
            header = recv_exact(conn, 4)
            (length,) = struct.unpack("!I", header)
            payload = recv_exact(conn, length)
            packet = pickle.loads(payload)

            # Let caller log IMU, etc.
            if self.on_packet:
                self.on_packet(packet)

            # Decode JPEG -> frame
            jpeg_bytes = packet.get("jpeg")
            depth_png = packet.get("depth_png")
            depth_scale = float(packet.get("depth_scale", 0.001))
            if not jpeg_bytes:
                continue
            jpg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            depth_u16 = None
            if depth_png is not None:
                depth_u16 = decode_depth_png(depth_png)

            self.frame_index += 1
            now = time.time()

            h, w, _ = frame.shape

            # YOLO inference (mirrors yolo_realsense.py structure)
            results = self.model(frame, verbose=False)
            result = results[0]

            stable_candidates = {}  # cls_id -> best detection

            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue

                    cls_id = int(box.cls[0])
                    cls_name = self.cls_names.get(cls_id, str(cls_id))

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    x_norm = float(cx / w)
                    y_norm = float(cy / h)

                    # Stability gating
                    if not self._is_cid_position_stable(cls_id, x_norm, now):
                        continue

                    # Depth at center pixel (meters), if depth is available (aligned to color)
                    depth_m = None
                    if depth_u16 is not None:
                        depth_m = depth_at_pixel_m(depth_u16, depth_scale, int(cx), int(cy))

                    prev = stable_candidates.get(cls_id)
                    if (prev is None) or (conf > prev["conf"]):
                        stable_candidates[cls_id] = dict(
                            conf=conf, x_norm=x_norm, y_norm=y_norm, depth_m=depth_m, cls_name=cls_name
                        )

                    if self.show_debug:
                        label = f"{cls_id}:{conf:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Send scene messages via OSC (same message layout as yolo_realsense.py)
            if stable_candidates:
                for cls_id, cand in stable_candidates.items():
                    msg = [
                        float(cand["x_norm"]),
                        cand["cls_name"],
                        float(cand["y_norm"]),
                        cand["depth_m"],          # None if not provided
                        int(self.frame_index),
                    ]
                    self.osc_client.send_message("/yolo", msg)

            if self.show_debug:
                cv2.imshow("RealSense TCP YOLO", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("[RS TCP] ESC pressed, closing viewer window.")
                    self.show_debug = False
                    cv2.destroyAllWindows()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._srv:
                self._srv.close()
        except Exception:
            pass


# -----------------------------
# PPG TCP receiver (same !dii format as ppg_receive.py)
# -----------------------------
class PPGTCPReceiver:
    PACK_FMT = "!dii"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    def __init__(self, listen_ip: str, port: int):
        self.listen_ip = listen_ip
        self.port = int(port)
        self._srv = None
        self._running = False
        self._thread = None
        self.on_sample = None  # callback(ts, red, ir, src_ip)

    def _serve(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.listen_ip, self.port))
        self._srv.listen(1)
        print(f"[PPG TCP] Listening on {self.listen_ip}:{self.port} ...")

        while self._running:
            conn, addr = self._srv.accept()
            src_ip = addr[0]
            print(f"[PPG TCP] Connected: {addr}")
            buf = bytearray()
            try:
                while self._running:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while len(buf) >= self.PACK_SIZE:
                        rec = bytes(buf[:self.PACK_SIZE])
                        del buf[:self.PACK_SIZE]
                        ts, red, ir = struct.unpack(self.PACK_FMT, rec)
                        if self.on_sample:
                            self.on_sample(ts, red, ir, src_ip)
            except Exception as e:
                print(f"[PPG TCP] Receiver error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print("[PPG TCP] Disconnected.")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._srv:
                self._srv.close()
        except Exception:
            pass


# -----------------------------
# Online processing hook
# -----------------------------
def online_process(buffers: RollingBuffer1Min):
    """
    Placeholder: run every ~1s. Put your real-time signal processing here.
    You get last 60s of IMU and PPG.
    """
    imu, ppg = buffers.snapshot()
    # Example: just print counts
    # print(f"[BUF] imu={len(imu)} ppg={len(ppg)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs-listen-ip", default="0.0.0.0")
    ap.add_argument("--rs-port", type=int, default=50000)
    ap.add_argument("--ppg-listen-ip", default="0.0.0.0")
    ap.add_argument("--ppg-port", type=int, default=9999)

    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--osc-ip", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=6969)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--show-debug", action="store_true")

    args = ap.parse_args()

    # 1-minute buffers
    buffers = RollingBuffer1Min(window_sec=60.0)

    # Start sonification engine (OSC server + BRIR playback)
    ss = SpatialSoundHeadphoneYOLO(verbose=False)
    sound_thread = threading.Thread(target=ss.start, daemon=True)
    sound_thread.start()

    # RealSense TCP -> YOLO -> OSC
    rs = RealSenseYOLOFromTCP(
        listen_ip=args.rs_listen_ip,
        port=args.rs_port,
        model_path=args.model,
        osc_ip=args.osc_ip,
        osc_port=args.osc_port,
        conf_threshold=args.conf,
        show_debug=args.show_debug,
    )

    def on_rs_packet(packet: dict):
        t = float(packet.get("t", time.time()))
        imu = packet.get("imu", {}) or {}
        accel = imu.get("accel")
        gyro = imu.get("gyro")
        buffers.add_imu(t, accel, gyro)

    rs.on_packet = on_rs_packet
    rs.start()

    # PPG TCP receiver -> buffer
    ppg = PPGTCPReceiver(listen_ip=args.ppg_listen_ip, port=args.ppg_port)

    def on_ppg(ts, red, ir, src_ip):
        buffers.add_ppg(float(ts), int(red), int(ir))

    ppg.on_sample = on_ppg
    ppg.start()

    # Online processing loop
    try:
        while True:
            online_process(buffers)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: shutting down...")
    finally:
        rs.stop()
        ppg.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    # python main_sonify_live.py --rs-port 50000 \
    #   --ppg-port 9999 --model yolov8n.pt
    main()

