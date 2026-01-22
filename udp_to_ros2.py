#!/usr/bin/env python3
"""
udp_to_ros2.py

Bridge: rs_d455_raw_udp_receiver.py (UDP RAW) -> ROS2 topics

Publishes:
  /rs/color        sensor_msgs/Image  (bgr8)
  /rs/depth        sensor_msgs/Image  (16UC1) optional
  /rs/imu          sensor_msgs/Imu    optional
  /rs/meta         std_msgs/String    compact JSON metadata
  /rs/depth_scale  std_msgs/Float32
  /rs/t_sender     std_msgs/Float64   epoch seconds
  /rs/fseq         std_msgs/UInt32

Design note:
- UDP receiver runs in a background thread.
- ROS2 timer publishes the latest packet from the receiver to avoid publishing directly
  from a non-ROS thread.
"""

import argparse
import json
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, Float64, UInt32, String

from rs_d455_raw_udp_receiver import RealSenseRawUDPReceiver, FLAG_HAS_DEPTH, FLAG_HAS_IMU


def _epoch_to_ros_stamp(node: Node, t_epoch: float):
    sec = int(t_epoch)
    nsec = int((t_epoch - sec) * 1e9)
    stamp = node.get_clock().now().to_msg()
    stamp.sec = sec
    stamp.nanosec = nsec
    return stamp


class RsUdpToRos2Bridge(Node):
    def __init__(
        self,
        listen_ip: str,
        port: int,
        timeout_ms: int,
        max_inflight: int,
        rcvbuf_bytes: int,
        publish_hz: float,
        frame_id_color: str,
        frame_id_depth: str,
        frame_id_imu: str,
    ):
        super().__init__("rs_d455_udp_to_ros2_bridge")

        # --- ROS publishers ---
        self.pub_color = self.create_publisher(Image, "/rs/color", 10)
        self.pub_depth = self.create_publisher(Image, "/rs/depth", 10)
        self.pub_imu = self.create_publisher(Imu, "/rs/imu", 50)

        self.pub_meta = self.create_publisher(String, "/rs/meta", 10)
        self.pub_depth_scale = self.create_publisher(Float32, "/rs/depth_scale", 10)
        self.pub_t_sender = self.create_publisher(Float64, "/rs/t_sender", 10)
        self.pub_fseq = self.create_publisher(UInt32, "/rs/fseq", 10)

        self.frame_id_color = frame_id_color
        self.frame_id_depth = frame_id_depth
        self.frame_id_imu = frame_id_imu

        # --- UDP receiver (background thread) ---
        self.receiver = RealSenseRawUDPReceiver(
            listen_ip=listen_ip,
            port=port,
            timeout_ms=timeout_ms,
            max_inflight=max_inflight,
            rcvbuf_bytes=rcvbuf_bytes,
            on_frame=None,  # we publish via ROS timer
        )
        self.receiver.start()

        # --- publish timer ---
        period = 1.0 / max(1.0, float(publish_hz))
        self.timer = self.create_timer(period, self._tick)

        self._last_fseq_published: Optional[int] = None

        self.get_logger().info(
            f"UDP->ROS2 bridge running. Listening on {listen_ip}:{port}, publishing ~{publish_hz} Hz."
        )

    def destroy_node(self):
        try:
            self.receiver.stop()
        except Exception:
            pass
        super().destroy_node()

    def _tick(self):
        pkt = self.receiver.get_latest()
        if pkt is None:
            return

        fseq = int(pkt.get("fseq", 0))
        # Avoid republishing same frame over and over if publish_hz > UDP fps
        if self._last_fseq_published == fseq:
            return
        self._last_fseq_published = fseq

        flags = int(pkt.get("flags", 0))
        t_sender = float(pkt.get("t_sender", 0.0))
        depth_scale = float(pkt.get("depth_scale", 0.001))

        color = pkt.get("color", None)   # np array HxWx3 BGR
        depth = pkt.get("depth", None)   # np array HxW uint16 or None
        imu = pkt.get("imu", None)       # dict or None

        if color is None:
            return

        # --- scalar topics ---
        m_scale = Float32()
        m_scale.data = depth_scale
        self.pub_depth_scale.publish(m_scale)

        m_t = Float64()
        m_t.data = t_sender
        self.pub_t_sender.publish(m_t)

        m_seq = UInt32()
        m_seq.data = fseq
        self.pub_fseq.publish(m_seq)

        # --- meta topic (compact JSON) ---
        meta = {
            "ver": int(pkt.get("ver", 1)),
            "flags": flags,
            "fseq": fseq,
            "t_sender": t_sender,
            "depth_scale": depth_scale,
            "cw": int(pkt.get("cw", color.shape[1])),
            "ch": int(pkt.get("ch", color.shape[0])),
            "cstride": int(pkt.get("cstride", color.shape[1] * 3)),
            "dw": int(pkt.get("dw", depth.shape[1] if depth is not None else 0)),
            "dh": int(pkt.get("dh", depth.shape[0] if depth is not None else 0)),
            "dstride": int(pkt.get("dstride", (depth.shape[1] * 2) if depth is not None else 0)),
        }
        m_meta = String()
        m_meta.data = json.dumps(meta, separators=(",", ":"))
        self.pub_meta.publish(m_meta)

        # --- publish color image ---
        color = np.ascontiguousarray(color)
        ch, cw = color.shape[:2]

        img_c = Image()
        img_c.header.stamp = _epoch_to_ros_stamp(self, t_sender if t_sender else time.time())
        img_c.header.frame_id = self.frame_id_color
        img_c.height = ch
        img_c.width = cw
        img_c.encoding = "bgr8"
        img_c.is_bigendian = 0
        img_c.step = cw * 3
        img_c.data = color.tobytes()
        self.pub_color.publish(img_c)

        # --- publish depth image (if present) ---
        if (flags & FLAG_HAS_DEPTH) and depth is not None:
            depth = np.ascontiguousarray(depth)
            dh, dw = depth.shape[:2]

            img_d = Image()
            img_d.header.stamp = img_c.header.stamp
            img_d.header.frame_id = self.frame_id_depth
            img_d.height = dh
            img_d.width = dw
            img_d.encoding = "16UC1"
            img_d.is_bigendian = 0
            img_d.step = dw * 2
            img_d.data = depth.tobytes()
            self.pub_depth.publish(img_d)

        # --- publish IMU (if present) ---
        if (flags & FLAG_HAS_IMU) and imu is not None:
            imu_msg = Imu()
            imu_msg.header.stamp = img_c.header.stamp
            imu_msg.header.frame_id = self.frame_id_imu

            # Orientation unknown
            imu_msg.orientation_covariance[0] = -1.0

            # Use mask semantics from your receiver:
            # mask bit1: accel present, bit2: gyro present
            mask = int(imu.get("mask", 0))

            if mask & 2 and imu.get("gyro") is not None:
                gx, gy, gz = imu["gyro"]
                imu_msg.angular_velocity.x = float(gx)
                imu_msg.angular_velocity.y = float(gy)
                imu_msg.angular_velocity.z = float(gz)
            else:
                imu_msg.angular_velocity_covariance[0] = -1.0

            if mask & 1 and imu.get("accel") is not None:
                ax, ay, az = imu["accel"]
                imu_msg.linear_acceleration.x = float(ax)
                imu_msg.linear_acceleration.y = float(ay)
                imu_msg.linear_acceleration.z = float(az)
            else:
                imu_msg.linear_acceleration_covariance[0] = -1.0

            self.pub_imu.publish(imu_msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--timeout-ms", type=int, default=200)
    ap.add_argument("--max-inflight", type=int, default=8)
    ap.add_argument("--rcvbuf-bytes", type=int, default=(1 << 22))

    ap.add_argument("--publish-hz", type=float, default=120.0, help="ROS publish loop rate (not UDP rate).")

    ap.add_argument("--frame-id-color", default="realsense_color")
    ap.add_argument("--frame-id-depth", default="realsense_depth")
    ap.add_argument("--frame-id-imu", default="realsense_imu")

    args = ap.parse_args()

    rclpy.init()
    node = RsUdpToRos2Bridge(
        listen_ip=args.listen_ip,
        port=args.port,
        timeout_ms=args.timeout_ms,
        max_inflight=args.max_inflight,
        rcvbuf_bytes=args.rcvbuf_bytes,
        publish_hz=args.publish_hz,
        frame_id_color=args.frame_id_color,
        frame_id_depth=args.frame_id_depth,
        frame_id_imu=args.frame_id_imu,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

