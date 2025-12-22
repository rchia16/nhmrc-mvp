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
import os
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
from config import deep_get, load_config

# ✅ NEW: import updated receiver library
from rs_d455_raw_udp_receiver import (
    RealSenseRawUDPReceiver,
    ReceiverStats,
)

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
    Thin wrapper around bth_audio_manager.UDPAudioFileSender that:
      - rate-limits per label + globally (prevents receiver "inflight" pile-up)
      - uses the base-class send_bytes() (supports pacing/buffers if you
      applied the bth_audio_manager diff)
    """

    def __init__(
        self,
        *args,
        per_key_ms: int = 400,
        global_ms: int = 120,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.per_key_ms = int(per_key_ms)
        self.global_ms = int(global_ms)
        self.last_sent_ms: Dict[str, int] = {}
        self.global_last_ms: int = 0

    def should_send(self, key: str, now_ms: int) -> bool:
        if self.global_ms > 0 and (now_ms - self.global_last_ms) < self.global_ms:
            return False
        t = self.last_sent_ms.get(key, 0)
        if self.per_key_ms > 0 and (now_ms - t) < self.per_key_ms:
            return False
        self.last_sent_ms[key] = now_ms
        self.global_last_ms = now_ms
        return True

    def send_wav_bytes(
        self,
        filename: str,
        wav_bytes: bytes,
        repeat_chunk0: int = 4,
        auto_pace_big_files: bool = True,
    ):
        # Prefer the improved base-class method (added in the bth_audio_manager diff).
        if hasattr(super(), "send_bytes"):
            return super().send_bytes(
                filename=filename,
                file_bytes=wav_bytes,
                repeat_chunk0=repeat_chunk0,
                auto_pace_big_files=auto_pace_big_files,
            )

        # Fallback: older bth_audio_manager without send_bytes() support.
        file_id = random.randint(1, 0xFFFFFFFF)
        chunks = self._build_chunks(os.path.basename(filename), wav_bytes, self.chunk_payload_bytes)
        idx_order = [0] * max(1, int(repeat_chunk0)) + list(range(1, len(chunks)))
        for idx in idx_order:
            payload = chunks[idx]
            hdr = struct.pack(
                bam.HDR_FMT,
                bam.MAGIC,
                int(file_id),
                int(idx),
                int(len(chunks)),
                int(len(payload)),
                0,
            )
            self.sock.sendto(hdr + payload, self.target)
            # crude pacing helps big SOFA WAVs survive UDP on busy links
            if self.inter_packet_sleep_s > 0:
                time.sleep(self.inter_packet_sleep_s)
            elif auto_pace_big_files and len(wav_bytes) >= 64 * 1024:
                time.sleep(0.0008)


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
        self.per_label_cooldown_s = float(
            deep_get(cfg, "audio.per_label_cooldown_s", 0.5))
        self.repeat_chunk0 = int(deep_get(cfg, "audio.send_repeat_chunk0", 4))

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

            # --- compute bbox center in RGB space ---
            x1, y1, x2, y2 = box.xyxy[0]
            cx_rgb = int((x1 + x2) / 2)
            cy_rgb = int((y1 + y2) / 2)

            # azimuth still based on RGB width
            x_norm = cx_rgb / w
            az = -90.0 + 180.0 * x_norm

            r = 1.0
            if depth is not None:
                dh, dw = depth.shape[:2]

                # map RGB coords -> depth coords
                cx_d = int(cx_rgb * dw / w)
                cy_d = int(cy_rgb * dh / h)

                # clamp to valid indices
                cx_d = max(0, min(dw - 1, cx_d))
                cy_d = max(0, min(dh - 1, cy_d))

                d = depth[cy_d, cx_d]
                if d > 0:
                    r = float(d) * depth_scale

            # cx = int(box.xyxy[0][0] + box.xyxy[0][2]) // 2
            # x_norm = cx / w
            # az = -90.0 + 180.0 * x_norm

            # r = 1.0
            # if depth is not None:
            #     d = depth[int(box.xyxy[0][1]), cx]
            #     if d > 0:
            #         r = float(d) * depth_scale

            # 1) cooldown (existing)
            if now - self.last_play.get(label, 0) < self.per_label_cooldown_s:
                continue
            self.last_play[label] = now

            # 2) UDP flood protection (new): per-label + global limiter
            now_ms = int(now * 1000)
            if not self.audio_sender.should_send(label, now_ms):
                continue

            wav_bytes = self.spatialiser.spatialise_file(mono_path, az, r)
            self.audio_sender.send_wav_bytes(f"{label}.wav", wav_bytes,
                                             repeat_chunk0=self.repeat_chunk0)

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
        # pacing (strongly recommended for big SOFA WAVs)
        inter_packet_sleep_s=float(deep_get(cfg, "audio.send_inter_packet_sleep_s", 0.0)),
        # new: avoid receiver overload
        per_key_ms=int(deep_get(cfg, "audio.per_key_ms", 400)),
        global_ms=int(deep_get(cfg, "audio.global_ms", 120)),
        # if you applied the bth_audio_manager diff, these will be honored:
        sndbuf_bytes=int(deep_get(cfg, "audio.send_sndbuf_bytes", 8 << 20)),
        min_inter_file_gap_s=float(deep_get(cfg, "audio.min_inter_file_gap_s", 0.15)),
    )

    spatialiser = SofaSpatialiser(
        sofa_path=deep_get(cfg, "sonification.sofa"),
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

