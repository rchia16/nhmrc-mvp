#!/usr/bin/env python3
"""
rs_ros_publish_check.py

ROS (rospy) checker for the UDP->ROS bridge topics:
  /rs/color        sensor_msgs/Image (expected bgr8)
  /rs/depth        sensor_msgs/Image (expected 16UC1) optional
  /rs/imu          sensor_msgs/Imu   optional
  /rs/meta         std_msgs/String   JSON metadata
  /rs/depth_scale  std_msgs/Float32
  /rs/t_sender     std_msgs/Float64  epoch seconds
  /rs/fseq         std_msgs/UInt32

Checks:
- required topics received within window
- encodings sanity
- fseq monotonic + drops/out-of-order
- latency estimate using t_sender
- simple FPS estimates
Exits:
- 0 on PASS
- 2 on FAIL
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import rospy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, Float64, UInt32, String


@dataclass
class TopicStats:
    last_wall_t: float = 0.0
    count_total: int = 0
    count_interval: int = 0
    last_print_t: float = field(default_factory=time.time)

    def tick(self):
        now = time.time()
        self.last_wall_t = now
        self.count_total += 1
        self.count_interval += 1

    def fps_since_last(self, now: float) -> float:
        dt = max(1e-6, now - self.last_print_t)
        fps = self.count_interval / dt
        self.count_interval = 0
        self.last_print_t = now
        return fps


class RsPublishChecker:
    def __init__(self, expect_depth: bool, expect_imu: bool, window_s: float, print_every_s: float, require_meta_match: bool):
        self.expect_depth = expect_depth
        self.expect_imu = expect_imu
        self.window_s = float(window_s)
        self.print_every_s = float(print_every_s)
        self.require_meta_match = require_meta_match

        self.meta: Dict[str, Any] = {}
        self.depth_scale: Optional[float] = None
        self.t_sender: Optional[float] = None
        self.fseq: Optional[int] = None

        self.last_fseq: Optional[int] = None
        self.drops = 0
        self.out_of_order = 0

        self.stats: Dict[str, TopicStats] = {
            "color": TopicStats(),
            "depth": TopicStats(),
            "imu": TopicStats(),
            "meta": TopicStats(),
            "depth_scale": TopicStats(),
            "t_sender": TopicStats(),
            "fseq": TopicStats(),
        }

        rospy.Subscriber("/rs/color", Image, self._on_color, queue_size=1)
        rospy.Subscriber("/rs/depth", Image, self._on_depth, queue_size=1)
        rospy.Subscriber("/rs/imu", Imu, self._on_imu, queue_size=10)

        rospy.Subscriber("/rs/meta", String, self._on_meta, queue_size=10)
        rospy.Subscriber("/rs/depth_scale", Float32, self._on_depth_scale, queue_size=10)
        rospy.Subscriber("/rs/t_sender", Float64, self._on_t_sender, queue_size=10)
        rospy.Subscriber("/rs/fseq", UInt32, self._on_fseq, queue_size=10)

        self.start_wall = time.time()
        self.last_report_wall = self.start_wall

        rospy.loginfo(
            f"[CHECK] window={self.window_s:.1f}s expect_depth={self.expect_depth} expect_imu={self.expect_imu} require_meta_match={self.require_meta_match}"
        )

    # ---------- callbacks ----------

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
        enc = (msg.encoding or "").lower()
        if enc not in ("bgr8", "rgb8"):
            rospy.logwarn(f"[CHECK] /rs/color unexpected encoding={msg.encoding}")

        if msg.width <= 0 or msg.height <= 0:
            rospy.logwarn(f"[CHECK] /rs/color invalid size {msg.width}x{msg.height}")

        if self.require_meta_match and self.meta:
            cw = self.meta.get("cw")
            ch = self.meta.get("ch")
            if cw is not None and int(cw) != int(msg.width):
                rospy.logwarn(f"[CHECK] /rs/color width mismatch meta cw={cw} msg.width={msg.width}")
            if ch is not None and int(ch) != int(msg.height):
                rospy.logwarn(f"[CHECK] /rs/color height mismatch meta ch={ch} msg.height={msg.height}")

    def _on_depth(self, msg: Image):
        self.stats["depth"].tick()
        enc = (msg.encoding or "").lower()
        if enc != "16uc1":
            rospy.logwarn(f"[CHECK] /rs/depth unexpected encoding={msg.encoding}")

        if msg.width <= 0 or msg.height <= 0:
            rospy.logwarn(f"[CHECK] /rs/depth invalid size {msg.width}x{msg.height}")

        if self.require_meta_match and self.meta:
            dw = self.meta.get("dw")
            dh = self.meta.get("dh")
            if dw is not None and int(dw) != int(msg.width):
                rospy.logwarn(f"[CHECK] /rs/depth width mismatch meta dw={dw} msg.width={msg.width}")
            if dh is not None and int(dh) != int(msg.height):
                rospy.logwarn(f"[CHECK] /rs/depth height mismatch meta dh={dh} msg.height={msg.height}")

    def _on_imu(self, msg: Imu):
        self.stats["imu"].tick()

    # ---------- checking ----------

    def _topic_ok(self, key: str, required: bool) -> bool:
        if not required:
            return True
        s = self.stats[key]
        if s.count_total <= 0:
            return False
        return (time.time() - s.last_wall_t) <= self.window_s

    def step(self) -> Optional[int]:
        now = time.time()
        if (now - self.last_report_wall) >= self.print_every_s:
            self.last_report_wall = now

            fps_color = self.stats["color"].fps_since_last(now)
            fps_depth = self.stats["depth"].fps_since_last(now)
            fps_imu = self.stats["imu"].fps_since_last(now)
            fps_meta = self.stats["meta"].fps_since_last(now)
            fps_seq = self.stats["fseq"].fps_since_last(now)
            fps_ts = self.stats["t_sender"].fps_since_last(now)

            latency_ms = None
            if self.t_sender and self.t_sender > 0:
                latency_ms = (time.time() - float(self.t_sender)) * 1000.0

            rospy.loginfo(
                f"[CHECK] rates color={fps_color:.1f} depth={fps_depth:.1f} imu={fps_imu:.1f} | "
                f"meta={fps_meta:.1f} fseq={fps_seq:.1f} t_sender={fps_ts:.1f} | "
                f"drops={self.drops} ooo={self.out_of_order}"
                + (f" latency~{latency_ms:.1f}ms" if latency_ms is not None else "")
            )

        # End-of-window PASS/FAIL
        if (now - self.start_wall) >= self.window_s:
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
                    rospy.logerr(f"[CHECK] Missing/stale topic: {k} (required={req})")

            if ok:
                rospy.loginfo("[CHECK] ✅ PASS: Required topics received within window.")
                return 0
            else:
                rospy.logerr("[CHECK] ❌ FAIL: One or more required topics missing/stale.")
                return 2

        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--print-every-s", type=float, default=1.0)
    ap.add_argument("--expect-depth", action="store_true")
    ap.add_argument("--expect-imu", action="store_true")
    ap.add_argument("--require-meta-match", action="store_true")
    args = ap.parse_args()

    rospy.init_node("rs_ros_publish_check", anonymous=False)

    checker = RsPublishChecker(
        expect_depth=args.expect_depth,
        expect_imu=args.expect_imu,
        window_s=args.window_s,
        print_every_s=args.print_every_s,
        require_meta_match=args.require_meta_match,
    )

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rc = checker.step()
        if rc is not None:
            # Exit with code (use sys.exit so caller can detect success/fail)
            sys.exit(rc)
        rate.sleep()


if __name__ == "__main__":
    main()

