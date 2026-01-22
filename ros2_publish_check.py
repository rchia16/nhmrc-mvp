#!/usr/bin/env python3
"""
rs_ros2_publish_check.py

ROS2 publishing checker for the UDP->ROS2 bridge topics:
  /rs/color        sensor_msgs/Image (expected bgr8)
  /rs/depth        sensor_msgs/Image (expected 16UC1) optional
  /rs/imu          sensor_msgs/Imu   optional
  /rs/meta         std_msgs/String   JSON metadata
  /rs/depth_scale  std_msgs/Float32
  /rs/t_sender     std_msgs/Float64  epoch seconds
  /rs/fseq         std_msgs/UInt32

What it checks:
- Messages arrive within a time window
- Encodings match expectations
- Frame dims match meta (if meta present)
- fseq monotonicity + detects drops / out-of-order
- Estimates FPS for each stream
- Estimates latency using t_sender (epoch) vs local wall-clock
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, Float64, UInt32, String


@dataclass
class TopicStats:
    last_t: float = 0.0
    count: int = 0
    count_interval: int = 0
    last_print_t: float = field(default_factory=time.time)

    def tick(self):
        now = time.time()
        self.last_t = now
        self.count += 1
        self.count_interval += 1

    def fps_since_last_print(self, now: float) -> float:
        dt = max(1e-6, now - self.last_print_t)
        fps = self.count_interval / dt
        self.count_interval = 0
        self.last_print_t = now
        return fps


class RsPublishChecker(Node):
    def __init__(
        self,
        expect_depth: bool,
        expect_imu: bool,
        window_s: float,
        print_every_s: float,
        require_meta_match: bool,
    ):
        super().__init__("rs_publish_checker")

        # Best-effort QoS is often used for image streams; compatible with many publishers.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.expect_depth = expect_depth
        self.expect_imu = expect_imu
        self.window_s = float(window_s)
        self.print_every_s = float(print_every_s)
        self.require_meta_match = require_meta_match

        # Latest values
        self.meta: Dict[str, Any] = {}
        self.depth_scale: Optional[float] = None
        self.t_sender: Optional[float] = None
        self.fseq: Optional[int] = None

        self.last_fseq: Optional[int] = None
        self.drops: int = 0
        self.out_of_order: int = 0

        self.stats: Dict[str, TopicStats] = {
            "color": TopicStats(),
            "depth": TopicStats(),
            "imu": TopicStats(),
            "meta": TopicStats(),
            "depth_scale": TopicStats(),
            "t_sender": TopicStats(),
            "fseq": TopicStats(),
        }

        # Subscriptions
        self.create_subscription(Image, "/rs/color", self._on_color, qos)
        self.create_subscription(Image, "/rs/depth", self._on_depth, qos)
        self.create_subscription(Imu, "/rs/imu", self._on_imu, qos)

        self.create_subscription(String, "/rs/meta", self._on_meta, 10)
        self.create_subscription(Float32, "/rs/depth_scale", self._on_depth_scale, 10)
        self.create_subscription(Float64, "/rs/t_sender", self._on_t_sender, 10)
        self.create_subscription(UInt32, "/rs/fseq", self._on_fseq, 10)

        self.start_t = time.time()
        self.last_report_t = self.start_t

        self.timer = self.create_timer(0.2, self._report_tick)

        self.get_logger().info(
            f"Checking /rs/* publishing for {self.window_s:.1f}s "
            f"(expect_depth={self.expect_depth}, expect_imu={self.expect_imu}, require_meta_match={self.require_meta_match})"
        )

    # ---------------- callbacks ----------------

    def _on_meta(self, msg: String):
        self.stats["meta"].tick()
        try:
            self.meta = json.loads(msg.data)
        except Exception:
            self.meta = {}

    def _on_depth_scale(self, msg: Float32):
        self.stats["depth_scale"].tick()
        self.depth_scale = float(msg.data)

    def _on_t_sender(self, msg: Float64):
        self.stats["t_sender"].tick()
        self.t_sender = float(msg.data)

    def _on_fseq(self, msg: UInt32):
        self.stats["fseq"].tick()
        self.fseq = int(msg.data)

        if self.last_fseq is not None:
            if self.fseq < self.last_fseq:
                self.out_of_order += 1
            else:
                gap = self.fseq - self.last_fseq
                if gap > 1:
                    self.drops += (gap - 1)
        self.last_fseq = self.fseq

    def _on_color(self, msg: Image):
        self.stats["color"].tick()

        # Encoding sanity
        enc = (msg.encoding or "").lower()
        if enc not in ("bgr8", "rgb8"):
            self.get_logger().warn(f"/rs/color unexpected encoding={msg.encoding}")

        # Basic size sanity
        if msg.width <= 0 or msg.height <= 0:
            self.get_logger().warn(f"/rs/color invalid size {msg.width}x{msg.height}")

        # Optional: validate against meta
        if self.require_meta_match and self.meta:
            cw = self.meta.get("cw")
            ch = self.meta.get("ch")
            if cw is not None and int(cw) != int(msg.width):
                self.get_logger().warn(f"/rs/color width mismatch meta cw={cw} msg.width={msg.width}")
            if ch is not None and int(ch) != int(msg.height):
                self.get_logger().warn(f"/rs/color height mismatch meta ch={ch} msg.height={msg.height}")

    def _on_depth(self, msg: Image):
        self.stats["depth"].tick()

        enc = (msg.encoding or "").lower()
        if enc != "16uc1":
            self.get_logger().warn(f"/rs/depth unexpected encoding={msg.encoding}")

        if msg.width <= 0 or msg.height <= 0:
            self.get_logger().warn(f"/rs/depth invalid size {msg.width}x{msg.height}")

        if self.require_meta_match and self.meta:
            dw = self.meta.get("dw")
            dh = self.meta.get("dh")
            if dw is not None and int(dw) != int(msg.width):
                self.get_logger().warn(f"/rs/depth width mismatch meta dw={dw} msg.width={msg.width}")
            if dh is not None and int(dh) != int(msg.height):
                self.get_logger().warn(f"/rs/depth height mismatch meta dh={dh} msg.height={msg.height}")

    def _on_imu(self, msg: Imu):
        self.stats["imu"].tick()
        # Not much to validate besides that messages arrive.
        # If orientation covariance[0] == -1, orientation is unknown (fine).

    # ---------------- reporting / pass-fail ----------------

    def _topic_ok(self, key: str, required: bool) -> bool:
        if not required:
            return True
        last = self.stats[key].last_t
        return (time.time() - last) <= self.window_s and self.stats[key].count > 0

    def _report_tick(self):
        now = time.time()
        if now - self.last_report_t < self.print_every_s:
            return
        self.last_report_t = now

        # FPS snapshots
        fps_color = self.stats["color"].fps_since_last_print(now)
        fps_depth = self.stats["depth"].fps_since_last_print(now)
        fps_imu = self.stats["imu"].fps_since_last_print(now)

        fps_meta = self.stats["meta"].fps_since_last_print(now)
        fps_seq = self.stats["fseq"].fps_since_last_print(now)
        fps_ts = self.stats["t_sender"].fps_since_last_print(now)

        # Latency estimate
        latency_ms = None
        if self.t_sender and self.t_sender > 0:
            latency_ms = (time.time() - float(self.t_sender)) * 1000.0

        self.get_logger().info(
            "Rates: "
            f"color={fps_color:.1f}fps depth={fps_depth:.1f}fps imu={fps_imu:.1f}fps | "
            f"meta={fps_meta:.1f} fseq={fps_seq:.1f} t_sender={fps_ts:.1f} | "
            f"drops={self.drops} ooo={self.out_of_order}"
            + (f" latency~{latency_ms:.1f}ms" if latency_ms is not None else "")
        )

        # Required topic checks
        required = {
            "color": True,
            "meta": True,
            "depth_scale": True,
            "t_sender": True,
            "fseq": True,
            "depth": self.expect_depth,
            "imu": self.expect_imu,
        }
        ok = True
        for k, req in required.items():
            if not self._topic_ok(k, req):
                ok = False
                self.get_logger().warn(f"Missing/stale topic: {k} (required={req})")

        # End condition: after window, exit with pass/fail
        if (now - self.start_t) >= self.window_s:
            if ok:
                self.get_logger().info("✅ PASS: Required topics published within window.")
                raise SystemExit(0)
            else:
                self.get_logger().error("❌ FAIL: One or more required topics missing/stale.")
                raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-s", type=float, default=10.0, help="How long to observe before PASS/FAIL.")
    ap.add_argument("--print-every-s", type=float, default=1.0, help="Status print interval.")
    ap.add_argument("--expect-depth", action="store_true", help="Fail if /rs/depth does not arrive.")
    ap.add_argument("--expect-imu", action="store_true", help="Fail if /rs/imu does not arrive.")
    ap.add_argument("--require-meta-match", action="store_true", help="Warn if meta dims don't match msg dims.")
    args = ap.parse_args()

    rclpy.init()
    node = RsPublishChecker(
        expect_depth=args.expect_depth,
        expect_imu=args.expect_imu,
        window_s=args.window_s,
        print_every_s=args.print_every_s,
        require_meta_match=args.require_meta_match,
    )
    try:
        rclpy.spin(node)
    except SystemExit as e:
        # Convert to clean shutdown
        node.destroy_node()
        rclpy.shutdown()
        raise e
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

