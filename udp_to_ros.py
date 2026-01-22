#!/usr/bin/env python3
"""
rs_d455_udp_to_ros_bridge.py

Bridge: rs_d455_raw_udp_receiver.py (UDP RAW) -> ROS (rospy) topics

Publishes:
  /rs/color        sensor_msgs/Image  (bgr8)
  /rs/depth        sensor_msgs/Image  (16UC1) optional
  /rs/imu          sensor_msgs/Imu    optional
  /rs/meta         std_msgs/String    compact JSON metadata
  /rs/depth_scale  std_msgs/Float32
  /rs/t_sender     std_msgs/Float64   epoch seconds
  /rs/fseq         std_msgs/UInt32

Notes:
- UDP receiver runs in a background thread.
- ROS loop publishes latest packet at a chosen rate.
"""

import argparse
import json
import time
from typing import Optional

import numpy as np
import rospy

from config import deep_get, load_config

from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, Float64, UInt32, String

from rs_d455_raw_udp_receiver import RealSenseRawUDPReceiver, FLAG_HAS_DEPTH, FLAG_HAS_IMU


class RsUdpToRosBridge:
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
        # --- ROS publishers ---
        self.pub_color = rospy.Publisher("/rs/color", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("/rs/depth", Image, queue_size=1)
        self.pub_imu = rospy.Publisher("/rs/imu", Imu, queue_size=10)

        self.pub_meta = rospy.Publisher("/rs/meta", String, queue_size=10)
        self.pub_depth_scale = rospy.Publisher("/rs/depth_scale", Float32, queue_size=10)
        self.pub_t_sender = rospy.Publisher("/rs/t_sender", Float64, queue_size=10)
        self.pub_fseq = rospy.Publisher("/rs/fseq", UInt32, queue_size=10)

        self.frame_id_color = frame_id_color
        self.frame_id_depth = frame_id_depth
        self.frame_id_imu = frame_id_imu

        # --- UDP receiver ---
        self.receiver = RealSenseRawUDPReceiver(
            listen_ip=listen_ip,
            port=port,
            timeout_ms=timeout_ms,
            max_inflight=max_inflight,
            rcvbuf_bytes=rcvbuf_bytes,
            on_frame=None,  # publish from ROS thread/loop
        )
        self.receiver.start()

        self.publish_hz = float(publish_hz)
        self._last_fseq_published: Optional[int] = None

        rospy.loginfo(
            f"[UDP->ROS] Listening on {listen_ip}:{port}, publishing ~{self.publish_hz:.1f} Hz"
        )

    def shutdown(self):
        try:
            self.receiver.stop()
        except Exception:
            pass

    def spin(self):
        rate = rospy.Rate(max(1e-3, self.publish_hz))
        while not rospy.is_shutdown():
            pkt = self.receiver.get_latest()
            if pkt is None:
                rate.sleep()
                continue

            fseq = int(pkt.get("fseq", 0))
            # Avoid republishing identical frame if publish_hz > UDP fps
            if self._last_fseq_published == fseq:
                rate.sleep()
                continue
            self._last_fseq_published = fseq

            flags = int(pkt.get("flags", 0))
            t_sender = float(pkt.get("t_sender", 0.0))
            depth_scale = float(pkt.get("depth_scale", 0.001))

            color = pkt.get("color", None)   # np array HxWx3 BGR
            depth = pkt.get("depth", None)   # np array HxW uint16 or None
            imu = pkt.get("imu", None)       # dict or None

            if color is None:
                rate.sleep()
                continue

            # Use sender time if valid, else now
            if t_sender and t_sender > 0:
                stamp = rospy.Time.from_sec(t_sender)
            else:
                stamp = rospy.Time.now()

            # --- publish scalars ---
            self.pub_depth_scale.publish(Float32(data=depth_scale))
            self.pub_t_sender.publish(Float64(data=t_sender))
            self.pub_fseq.publish(UInt32(data=fseq))

            # --- publish meta ---
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
            self.pub_meta.publish(String(data=json.dumps(meta, separators=(",", ":"))))

            # --- publish color image ---
            color = np.ascontiguousarray(color)
            ch, cw = color.shape[:2]

            msg_c = Image()
            msg_c.header.stamp = stamp
            msg_c.header.frame_id = self.frame_id_color
            msg_c.height = ch
            msg_c.width = cw
            msg_c.encoding = "bgr8"
            msg_c.is_bigendian = 0
            msg_c.step = cw * 3
            msg_c.data = color.tobytes()
            self.pub_color.publish(msg_c)

            # --- publish depth image ---
            if (flags & FLAG_HAS_DEPTH) and depth is not None:
                depth = np.ascontiguousarray(depth)
                dh, dw = depth.shape[:2]

                msg_d = Image()
                msg_d.header.stamp = stamp
                msg_d.header.frame_id = self.frame_id_depth
                msg_d.height = dh
                msg_d.width = dw
                msg_d.encoding = "16UC1"
                msg_d.is_bigendian = 0
                msg_d.step = dw * 2
                msg_d.data = depth.tobytes()
                self.pub_depth.publish(msg_d)

            # --- publish IMU ---
            if (flags & FLAG_HAS_IMU) and imu is not None:
                imu_msg = Imu()
                imu_msg.header.stamp = stamp
                imu_msg.header.frame_id = self.frame_id_imu

                # Orientation unknown
                imu_msg.orientation_covariance[0] = -1.0

                # Your receiver uses mask semantics:
                # bit0 (1): accel present, bit1 (2): gyro present
                mask = int(imu.get("mask", 0))

                if (mask & 2) and (imu.get("gyro") is not None):
                    gx, gy, gz = imu["gyro"]
                    imu_msg.angular_velocity.x = float(gx)
                    imu_msg.angular_velocity.y = float(gy)
                    imu_msg.angular_velocity.z = float(gz)
                else:
                    imu_msg.angular_velocity_covariance[0] = -1.0

                if (mask & 1) and (imu.get("accel") is not None):
                    ax, ay, az = imu["accel"]
                    imu_msg.linear_acceleration.x = float(ax)
                    imu_msg.linear_acceleration.y = float(ay)
                    imu_msg.linear_acceleration.z = float(az)
                else:
                    imu_msg.linear_acceleration_covariance[0] = -1.0

                self.pub_imu.publish(imu_msg)

            rate.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50010)
    ap.add_argument("--timeout-ms", type=int, default=600)
    ap.add_argument("--max-inflight", type=int, default=64)
    ap.add_argument("--rcvbuf-bytes", type=int, default=(1 << 22))
    ap.add_argument("--publish-hz", type=float, default=120.0)

    ap.add_argument("--frame-id-color", default="realsense_color")
    ap.add_argument("--frame-id-depth", default="realsense_depth")
    ap.add_argument("--frame-id-imu", default="realsense_imu")
    args = ap.parse_args()

    rospy.init_node("rs_d455_udp_to_ros_bridge", anonymous=False)

    bridge = RsUdpToRosBridge(
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
    rospy.on_shutdown(bridge.shutdown)
    bridge.spin()


if __name__ == "__main__":
    main()

