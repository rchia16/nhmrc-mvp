#!/usr/bin/env python3
import socket
import threading
import struct
import pickle
import time

import numpy as np
import pyrealsense2 as rs
import cv2

class IMUReader:
    """Reads accel/gyro directly from the Motion Module sensor via callback."""
    def __init__(self, device: rs.device, accel_hz=250, gyro_hz=400):
        self.lock = threading.Lock()
        self.accel = None
        self.gyro = None

        # Find a sensor that actually has accel/gyro profiles (don't rely on name)
        self.motion_sensor = None
        for s in device.query_sensors():
            has_accel = False
            has_gyro = False
            for p in s.get_stream_profiles():
                if p.stream_type() == rs.stream.accel:
                    has_accel = True
                elif p.stream_type() == rs.stream.gyro:
                    has_gyro = True
            if has_accel and has_gyro:
                self.motion_sensor = s
                break

        if self.motion_sensor is None:
            # Helpful debug: list sensor names we did see
            names = []
            for s in device.query_sensors():
                try:
                    names.append(s.get_info(rs.camera_info.name))
                except Exception:
                    names.append("<unknown>")
            raise RuntimeError(f"No accel+gyro sensor found. Sensors seen: {names}")

        # Select stream profiles that match supported rates
        accel_prof = None
        gyro_prof = None
        for p in self.motion_sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.accel and p.format() == rs.format.motion_xyz32f and p.fps() == accel_hz:
                accel_prof = p
            if p.stream_type() == rs.stream.gyro and p.format() == rs.format.motion_xyz32f and p.fps() == gyro_hz:
                gyro_prof = p

        if accel_prof is None or gyro_prof is None:
            raise RuntimeError(f"Could not find accel@{accel_hz} and gyro@{gyro_hz} motion profiles")

        self.motion_sensor.open([accel_prof, gyro_prof])
        self.motion_sensor.start(self._cb)

    def _cb(self, f):
        if not f.is_motion_frame():
            return
        m = f.as_motion_frame()
        data = m.get_motion_data()
        st = m.get_profile().stream_type()
        with self.lock:
            if st == rs.stream.accel:
                self.accel = (data.x, data.y, data.z)
            elif st == rs.stream.gyro:
                self.gyro = (data.x, data.y, data.z)

    def get_latest(self):
        with self.lock:
            return self.accel, self.gyro

    def stop(self):
        try:
            self.motion_sensor.stop()
        except Exception:
            pass
        try:
            self.motion_sensor.close()
        except Exception:
            pass


class RealSenseD455TCPSender:
    """
    Streams color video + IMU (accel + gyro) over TCP.

    Packet format:
        [4-byte big-endian length][pickled dict]

    dict has:
        {
            "t": float (timestamp),
            "jpeg": bytes (encoded color frame),
            "depth_png": bytes (PNG-encoded uint16 depth frame, aligned to color),
            "depth_scale": float (meters per depth unit),
            "depth_wh": (w, h),
            "imu": {
                "accel": (ax, ay, az) or None,
                "gyro": (gx, gy, gz) or None,
            }
        }
    """

    def __init__(
        self,
        host: str,
        port: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 80,
    ):
        self.host = host      # desktop PC IP
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self.sock = None
        self.running = False
        self.thread = None

        # --- RealSense setup (similar to your YOLO pipeline) ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.config.enable_stream(rs.stream.depth, self.width,
                                  self.height, rs.format.z16, self.fps)

        # Align depth to color so depth pixels correspond to the color image
        self.align = rs.align(rs.stream.color)

        # Depth scale (meters per unit)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"Depth scale: {self.depth_scale} m/unit")


        self.profile = self.pipeline.start(self.config)
        print("RealSense D455 started (color + depth).")

        dev = self.profile.get_device()
        self.imu_reader = IMUReader(dev, accel_hz=250, gyro_hz=400)
        print("RealSense D455 started (accel + gyro).")

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------
    def _connect_socket(self):
        """Connect to the desktop TCP server."""
        while True:
            try:
                print(f"Connecting to {self.host}:{self.port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = sock
                print("Connected to desktop.")
                return
            except OSError as e:
                print(f"Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

    def _send_packet(self, payload: bytes):
        """Send length-prefixed payload over TCP."""
        if self.sock is None:
            return
        try:
            header = struct.pack("!I", len(payload))
            self.sock.sendall(header + payload)
        except OSError as e:
            print(f"Socket send error: {e}. Closing socket.")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    # ------------------------------------------------------------------
    # Main streaming loop (runs in a thread)
    # ------------------------------------------------------------------
    def _stream_loop(self):
        self.running = True
        self._connect_socket()

        try:
            while self.running:
                # Wait for frames (color + possible IMU frames in the set)
                frames = self.pipeline.wait_for_frames()
                if frames is None:
                    continue

                # Align depth to color
                frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue


                # Collect IMU samples (accel + gyro) from this frameset
                accel_data = None
                gyro_data = None

                # Get latest IMU sample (independent of frameset)
                accel_data, gyro_data = (None, None)
                if getattr(self, "imu_reader", None) is not None:
                    accel_data, gyro_data = self.imu_reader.get_latest()
                    # optional debug
                    # if accel_data or gyro_data:
                    #     print("accel:", accel_data, "gyro:", gyro_data)

                # Convert color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())  # uint16 (z16)
                # Depth compression: 16-bit PNG (lossless)
                # 0..9 (higher = smaller/slower)
                png_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]   
                ok_d, enc_d = cv2.imencode(".png", depth_image, png_params)
                if not ok_d:
                    print("Failed to encode depth as PNG.")
                    continue
                depth_png_bytes = enc_d.tobytes()


                # JPEG encode to shrink bandwidth
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                ok, enc = cv2.imencode(".jpg", color_image, encode_params)
                if not ok:
                    print("Failed to encode frame as JPEG.")
                    continue
                jpeg_bytes = enc.tobytes()

                packet = {
                    "t": time.time(),
                    "jpeg": jpeg_bytes,
                    "depth_png": depth_png_bytes,
                    "depth_scale": self.depth_scale,
                    "depth_wh": (int(self.width), int(self.height)),
                    "imu": {
                        "accel": accel_data,
                        "gyro": gyro_data,
                    },
                }

                payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
                if self.sock is None:
                    self._connect_socket()
                self._send_packet(payload)

        except Exception as e:
            print(f"Error in streaming loop: {e}")
        finally:
            self.running = False
            print("Stopping RealSense pipeline...")
            self.pipeline.stop()

            if getattr(self, "imu_reader", None) is not None:
                try:
                    self.imu_reader.stop()
                except Exception:
                    pass

            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
            print("Sender shutdown complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


def main():
    # CHANGE THIS to your desktop PC's IP address
    desktop_ip = "192.168.0.194"
    port = 50000

    sender = RealSenseD455TCPSender(
        host=desktop_ip,
        port=port,
        width=640,
        height=480,
        fps=30,
        jpeg_quality=80,
    )

    sender.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping sender...")
    finally:
        sender.stop()


if __name__ == "__main__":
    main()

