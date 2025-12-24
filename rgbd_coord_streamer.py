#!/usr/bin/env python3
"""
py_main.py

Lightweight OSC publisher that converts incoming RealSense RGB-D frames into
YOLO detections and streams the positional metadata expected by
DepthAwareSpatialSound.

It listens for UDP RealSense frames (sent by rs_d455_raw_udp_sender.py), runs
YOLO on the colour image, samples depth at each detection centre, and sends the
following OSC message per detection:

    /yolo [x_norm, sound_key, y_norm, depth_m, frame_index]

Where:
    * x_norm, y_norm are normalised [0, 1] coordinates of the detection centre
    * sound_key is the YOLO label, used by DepthAwareSpatialSound to pick a
      sound
    * depth_m is the depth at the centre pixel (metres); omitted if no depth
    * frame_index increments per incoming frame

DepthAwareSpatialSound will turn these into spherical coordinates and queue the
corresponding sound for playback.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, Optional
from pythonosc import udp_client
from ultralytics import YOLO

from config import deep_get, load_config
from rs_d455_raw_udp_receiver import RealSenseRawUDPReceiver


class YoloDepthOSCStreamer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model = YOLO(deep_get(cfg, "yolo.model"))
        self.conf = float(deep_get(cfg, "yolo.conf", 0.3))

        target_ip = str(deep_get(cfg, "network.pi_ip", "127.0.0.1"))
        target_port = int(deep_get(cfg, "ports.audio_udp", 40100))
        self.osc_client = udp_client.SimpleUDPClient(target_ip, target_port)

        self.frame_index = 0

    def on_frame(self, pkt: Dict):
        color = pkt.get("color")
        depth = pkt.get("depth")
        depth_scale = float(pkt.get("depth_scale", 0.001))

        if color is None:
            return

        h, w = color.shape[:2]
        res = self.model(color, verbose=False)[0]

        if res.boxes is None:
            self.frame_index += 1
            return

        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < self.conf:
                continue

            label = self.model.names[int(box.cls[0])]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            x_norm = max(0.0, min(1.0, cx / w))
            y_norm = max(0.0, min(1.0, cy / h))

            depth_m: Optional[float] = None
            if depth is not None:
                dh, dw = depth.shape[:2]
                cx_d = int(max(0, min(dw - 1, cx * dw / w)))
                cy_d = int(max(0, min(dh - 1, cy * dh / h)))
                d_raw = float(depth[cy_d, cx_d])
                if d_raw > 0:
                    depth_m = d_raw * depth_scale

            msg = [float(x_norm), label, float(y_norm), depth_m, int(self.frame_index)]
            self.osc_client.send_message("/yolo", msg)

        self.frame_index += 1


class App:
    def __init__(self, cfg: Dict, rs_rx: RealSenseRawUDPReceiver | None = None):
        self.cfg = cfg
        self.stop_evt = threading.Event()
        self.streamer = YoloDepthOSCStreamer(cfg)

        self._own_rx = rs_rx is None
        if rs_rx is None:
            rs_rx = RealSenseRawUDPReceiver(
                listen_ip=str(deep_get(cfg, "network.pc_listen_ip", "0.0.0.0")),
                port=int(deep_get(cfg, "ports.rs_udp", 50010)),
                timeout_ms=int(deep_get(cfg, "realsense.rs_timeout_ms", 200)),
                max_inflight=int(deep_get(cfg, "realsense.max_inflight", 8)),
                on_frame=self.streamer.on_frame,
            )
        else:
            # share an existing receiver (pc_main) without rebinding the socket
            rs_rx.on_frame = self.streamer.on_frame

        self.rx = rs_rx
        self._thread: threading.Thread | None = None

    def _run(self):
        self.rx.start()

        try:
            while not self.stop_evt.is_set():
                time.sleep(0.2)
        finally:
            if self._own_rx:
                try:
                    self.rx.stop()
                except Exception:
                    pass

    def start(self, background: bool = False):
        if background:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        else:
            self._run()

    def stop(self):
        self.stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./streaming_config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    app = App(cfg)
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()
