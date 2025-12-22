#!/usr/bin/env python3
"""
pc_receive_sensors_yolo_sonify_send_audio.py

PC-side orchestrator (library-aligned):

- Receives RealSense RAW RGB-D + IMU over UDP using:
    RealSenseRawUDPReceiver (from rs_d455_raw_udp_receiver.py)
- Receives PPG via TCP using ppg_receive.py
- Runs YOLO on RGB frames
- Performs SOFA BRIR spatialisation using yolo_sofa.py
- Sends spatialised WAV bytes to Raspberry Pi via UDP
  using bth_audio_manager.py protocol

Run:
  python3 pc_receive_sensors_yolo_sonify_send_audio.py --config nhmrc_streaming_config.yaml
"""

from __future__ import annotations

import argparse
import io
import time
import random
import struct
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
import soundfile as sf
from scipy import signal
from ultralytics import YOLO

# ---------------- REQUIRED PROJECT IMPORTS ----------------

from ppg_receive import TCPServerReceiver
import bth_audio_manager as bam
from yolo_sofa import SpatialSoundHeadphoneYOLO

# ✅ NEW: import updated receiver library
from rs_d455_raw_udp_receiver import (
    RealSenseRawUDPReceiver,
    ReceiverStats,
)

# ---------------- CONFIG HELPERS ----------------

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------- PPG RECEIVER ----------------

class PPGReceiverThread:
    def __init__(self, listen_ip: str, port: int, poll_hz: float):
        self.rx = TCPServerReceiver(listen_ip, port)
        self.poll_hz = poll_hz
        self.latest: Optional[Tuple[float, int, int, str]] = None
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        period = 1.0 / max(1.0, self.poll_hz)
        while not self._stop.is_set():
            samples = self.rx.poll(max_records=2000)
            if samples:
                self.latest = samples[-1]
            time.sleep(period)


# ---------------- UDP AUDIO BYTES SENDER ----------------

class UDPAudioBytesSender(bam.UDPAudioFileSender):
    """
    Sends in-memory WAV bytes using same protocol as UDPAudioFileSender.
    """

    def send_bytes(self, filename: str, wav_bytes: bytes, repeat_chunk0: int = 2):
        file_id = random.randint(1, 0xFFFFFFFF)
        chunks = self._build_chunks(filename, wav_bytes, self.chunk_payload_bytes)

        for idx in [0] * repeat_chunk0 + list(range(1, len(chunks))):
            payload = chunks[idx]
            hdr = struct.pack(
                bam.HDR_FMT,
                bam.MAGIC,
                file_id,
                idx,
                len(chunks),
                len(payload),
                0,
            )
            self.sock.sendto(hdr + payload, self.target)
            if self.inter_packet_sleep_s > 0:
                time.sleep(self.inter_packet_sleep_s)


# ---------------- SOFA SPATIALISER ----------------

class SofaSpatialiser:
    """
    Wrapper around SpatialSoundHeadphoneYOLO for offline BRIR convolution.
    """

    def __init__(self, sofa_path: str, image_width: float, verbose: bool = False):
        self.engine = SpatialSoundHeadphoneYOLO(
            sofa_file_path=sofa_path,
            image_width=image_width,
            verbose=verbose,
        )
        self.fs = int(self.engine.BRIR_samplerate)

    @staticmethod
    def _ensure_mono(x: np.ndarray) -> np.ndarray:
        return x if x.ndim == 1 else x.mean(axis=1)

    def spatialise_file(self, mono_path: str, az_deg: float, r: float) -> bytes:
        audio, fs_src = sf.read(mono_path, dtype="float32", always_2d=False)
        audio = self._ensure_mono(audio)

        if fs_src != self.fs:
            audio = signal.resample_poly(audio, self.fs, fs_src)

        stereo = self.engine.sound_process(
            data=[az_deg, 0.0, r],
            audio=audio,
            BRIRs=self.engine.BRIRs,
            sourcePositions=self.engine.BRIR_sourcePositions,
        )

        peak = np.max(np.abs(stereo))
        if peak > 1.0:
            stereo /= peak

        buf = io.BytesIO()
        sf.write(buf, stereo, self.fs, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ---------------- YOLO + SONIFICATION ----------------

class YoloSofaSonifier:
    def __init__(
        self,
        cfg: Dict,
        audio_sender: UDPAudioBytesSender,
        spatialiser: SofaSpatialiser,
    ):
        self.model = YOLO(deep_get(cfg, "yolo.model"))
        self.conf = float(deep_get(cfg, "yolo.conf", 0.3))

        self.audio_sender = audio_sender
        self.spatialiser = spatialiser

        self.sound_map = deep_get(cfg, "sonification.sound_map", {})
        self.default_sound = deep_get(cfg, "sonification.default_sound")

        self.last_play = {}

    def process(self, pkt: Dict):
        rgb = pkt["color"]
        depth = pkt.get("depth")
        depth_scale = pkt.get("depth_scale", 0.001)

        h, w = rgb.shape[:2]
        res = self.model(rgb, verbose=False)[0]

        if res.boxes is None:
            return

        now = time.time()

        for box in res.boxes:
            if float(box.conf[0]) < self.conf:
                continue

            label = self.model.names[int(box.cls[0])]
            mono_path = self.sound_map.get(label, self.default_sound)
            if not mono_path:
                continue

            cx = int(box.xyxy[0][0] + box.xyxy[0][2]) // 2
            x_norm = cx / w
            az = -90.0 + 180.0 * x_norm

            r = 1.0
            if depth is not None:
                d = depth[int(box.xyxy[0][1]), cx]
                if d > 0:
                    r = float(d) * depth_scale

            if now - self.last_play.get(label, 0) < 0.5:
                continue
            self.last_play[label] = now

            wav_bytes = self.spatialiser.spatialise_file(mono_path, az, r)
            self.audio_sender.send_bytes(f"{label}.wav", wav_bytes)

            print(f"[SOFA] {label} az={az:.1f} r={r:.2f}m bytes={len(wav_bytes)}")


# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./streaming_config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # ---------- Receiver ----------
    rs_rx = RealSenseRawUDPReceiver(
        listen_ip=deep_get(cfg, "network.pc_listen_ip", "0.0.0.0"),
        port=int(deep_get(cfg, "ports.rs_udp")),
        timeout_ms=int(deep_get(cfg, "realsense.rs_timeout_ms", 200)),
        max_inflight=int(deep_get(cfg, "realsense.max_inflight", 8)),
    )
    rs_rx.start()

    # ---------- PPG ----------
    ppg_rx = PPGReceiverThread(
        listen_ip=deep_get(cfg, "network.pc_listen_ip", "0.0.0.0"),
        port=int(deep_get(cfg, "ports.ppg_tcp")),
        poll_hz=float(deep_get(cfg, "ppg.pc_poll_hz", 500.0)),
    )
    ppg_rx.start()

    # ---------- Audio ----------
    audio_sender = UDPAudioBytesSender(
        target_ip=deep_get(cfg, "network.pi_ip"),
        target_port=int(deep_get(cfg, "ports.audio_udp")),
        chunk_payload_bytes=int(deep_get(cfg, "audio.send_chunk_bytes", 1200)),
        inter_packet_sleep_s=float(deep_get(cfg, "audio.send_inter_packet_sleep_s", 0.0)),
    )

    spatialiser = SofaSpatialiser(
        sofa_path=deep_get(cfg, "sofa.file"),
        image_width=float(deep_get(cfg, "realsense.color_w", 640)),
    )

    sonifier = YoloSofaSonifier(cfg, audio_sender, spatialiser)

    print("[PC] Running: RS → YOLO → SOFA → UDP → Pi")

    try:
        while True:
            pkt = rs_rx.get_latest()
            if pkt:
                sonifier.process(pkt)
            time.sleep(1.0 / float(deep_get(cfg, "yolo.yolo_hz", 10)))
    except KeyboardInterrupt:
        print("\n[PC] Shutdown")
    finally:
        rs_rx.stop()
        ppg_rx.stop()


if __name__ == "__main__":
    main()

