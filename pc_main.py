#!/usr/bin/env python3
"""
pc_receive_sensors_yolo_sonify_send_audio.py

PC-side orchestrator (library-aligned):

- Receives RealSense RAW RGB-D + IMU over UDP using:
    RealSenseRawUDPReceiver (from rs_d455_raw_udp_receiver.py)
- Receives PPG via TCP using ppg_receive.py
- Runs YOLO on RGB frames
- Performs SOFA BRIR spatialisation using yolo_sofa.py
- LONG-TERM MODE (recommended):
    Streams continuous PCM16 frames over UDP with a tiny header (udp_pcm_stream.py)
- LEGACY MODE:
    Sends spatialised WAV bytes to Raspberry Pi via UDP using bth_audio_manager.py protocol

Run:
  python3 pc_main.py --config streaming_config.yaml
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

# ✅ import updated receiver library
from rs_d455_raw_udp_receiver import (
    RealSenseRawUDPReceiver,
    ReceiverStats,
)

# ✅ OPTIONAL: long-term PCM streaming sender
# Requires udp_pcm_stream.py (the module you created earlier) to be present.
try:
    from udp_pcm_stream import PCMStreamUDPSender, PCMStreamParams
except Exception:
    PCMStreamUDPSender = None  # type: ignore
    PCMStreamParams = None  # type: ignore


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


# ---------------- UDP AUDIO BYTES SENDER (LEGACY) ----------------

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


# ---------------- PCM STREAMING HELPERS ----------------

def wav_bytes_to_pcm16_bytes(wav_bytes: bytes) -> bytes:
    """
    Decode WAV bytes into interleaved int16 PCM bytes.
    Ensures 2 channels by upmixing mono.
    """
    bio = io.BytesIO(wav_bytes)
    audio, _sr = sf.read(bio, dtype="int16", always_2d=True)
    if audio.shape[1] == 1:
        audio = audio.repeat(2, axis=1)  # mono -> stereo
    return audio.tobytes()


def pcm16_chunker(pcm_bytes: bytes, frame_bytes: int):
    """
    Yield fixed-size PCM frames. Any remainder is dropped.
    """
    n = (len(pcm_bytes) // frame_bytes) * frame_bytes
    for i in range(0, n, frame_bytes):
        yield pcm_bytes[i:i + frame_bytes]


# ---------------- SOFA SPATIALISER ----------------

class SofaSpatialiser:
    """
    Wrapper around SpatialSoundHeadphoneYOLO for offline BRIR convolution.
    """

    def __init__(
        self,
        sofa_path: str,
        image_width: float,
        verbose: bool = False,
        out_sr: int = 16000,
        clip_s: float = 0.35,
        fade_ms: float = 8.0,
        max_peak: float = 0.95,
    ):
        self.engine = SpatialSoundHeadphoneYOLO(
            sofa_file_path=sofa_path,
            image_width=image_width,
            verbose=verbose,
        )
        # BRIR sample rate (SOFA dataset / engine internal)
        self.fs_brir = int(self.engine.BRIR_samplerate)

        # Output constraints to reduce UDP payload:
        # - short clips
        # - resample to 16kHz
        # - 16-bit PCM WAV
        self.out_sr = int(out_sr)
        self.clip_s = float(clip_s)
        self.fade_ms = float(fade_ms)
        self.max_peak = float(max_peak)

    @staticmethod
    def _ensure_mono(x: np.ndarray) -> np.ndarray:
        return x if x.ndim == 1 else x.mean(axis=1)

    @staticmethod
    def _apply_fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
        """Short fade-in/out to avoid clicks when clipping."""
        n = int(sr * (fade_ms / 1000.0))
        if n <= 1 or x.size <= 2 * n:
            return x
        w = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= w
        x[-n:] *= w[::-1]
        return x

    def spatialise_file(self, mono_path: str, az_deg: float, r: float) -> bytes:
        """
        Offline pipeline:
          1) read mono
          2) BRIR convolve for (az,r)
          3) trim
          4) resample to out_sr
          5) fade
          6) normalize to max_peak
          7) write PCM_16 WAV bytes
        """
        # 1) Load source mono
        src, fs = sf.read(mono_path, dtype="float32", always_2d=False)
        src = self._ensure_mono(np.asarray(src, dtype=np.float32))
        if fs != self.fs_brir:
            src = signal.resample_poly(src, self.fs_brir, fs).astype(np.float32, copy=False)

        # 2) Spatialise via your engine
        # [az_deg, el_deg, r]
        data = [float(az_deg), 0., float(r) ]
        stereo = self.engine.sound_process(
            data,
            src,
            BRIRs=self.engine.BRIRs,
            sourcePositions=self.engine.BRIR_sourcePositions,
        )

        stereo = np.asarray(stereo, dtype=np.float32)
        if stereo.ndim == 1:
            stereo = np.stack([stereo, stereo], axis=1)
        elif stereo.ndim == 2 and stereo.shape[0] == 2 and stereo.shape[1] != 2:
            stereo = stereo.T

        # 3) Trim after convolution tail
        if self.clip_s > 0:
            stereo = stereo[: int(self.fs_brir * self.clip_s), :]

        # 4) Resample to output SR
        if self.out_sr > 0 and self.out_sr != self.fs_brir:
            left = signal.resample_poly(stereo[:, 0], self.out_sr, self.fs_brir)
            right = signal.resample_poly(stereo[:, 1], self.out_sr, self.fs_brir)
            stereo = np.stack([left, right], axis=1).astype(np.float32, copy=False)

        # 5) Fade
        stereo[:, 0] = self._apply_fade(stereo[:, 0], self.out_sr, self.fade_ms)
        stereo[:, 1] = self._apply_fade(stereo[:, 1], self.out_sr, self.fade_ms)

        # 6) Normalize/clamp
        peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
        if peak > 0:
            stereo *= min(1.0, self.max_peak / peak)
        stereo = np.clip(stereo, -1.0, 1.0)

        # 7) Encode WAV bytes (PCM16)
        buf = io.BytesIO()
        sf.write(buf, stereo, int(self.out_sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ---------------- YOLO + SONIFICATION ----------------

class YoloSofaSonifier:
    def __init__(
        self,
        cfg: Dict,
        audio_sender: UDPAudioBytesSender,
        spatialiser: SofaSpatialiser,
        pcm_stream: Optional[object] = None,
        pcm_params: Optional[object] = None,
    ):
        self.cfg = cfg
        self.model = YOLO(deep_get(cfg, "yolo.model"))
        self.conf = float(deep_get(cfg, "yolo.conf", 0.3))

        self.audio_sender = audio_sender
        self.spatialiser = spatialiser

        # Long-term streaming path (preferred)
        self.pcm_stream = pcm_stream
        self.pcm_params = pcm_params

        self.sound_map = deep_get(cfg, "sonification.sound_map", {})
        self.default_sound = deep_get(cfg, "sonification.default_sound")

        self.last_play = {}
        self.per_label_cooldown_s = float(deep_get(cfg, "audio.per_label_cooldown_s", 0.5))
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

            # bbox center in RGB
            x1, y1, x2, y2 = box.xyxy[0]
            cx_rgb = int((x1 + x2) / 2)
            cy_rgb = int((y1 + y2) / 2)

            # azimuth from RGB width
            x_norm = cx_rgb / max(1, w)
            az = -90.0 + 180.0 * x_norm

            # range from depth
            r = 1.0
            if depth is not None:
                dh, dw = depth.shape[:2]
                cx_d = int(cx_rgb * dw / max(1, w))
                cy_d = int(cy_rgb * dh / max(1, h))
                cx_d = max(0, min(dw - 1, cx_d))
                cy_d = max(0, min(dh - 1, cy_d))
                d = depth[cy_d, cx_d]
                if d > 0:
                    r = float(d) * depth_scale

            # cooldown
            if now - self.last_play.get(label, 0) < self.per_label_cooldown_s:
                continue
            self.last_play[label] = now

            # UDP flood protection (legacy sender limiter; fine to keep)
            now_ms = int(now * 1000)
            if not self.audio_sender.should_send(label, now_ms):
                continue

            wav_bytes = self.spatialiser.spatialise_file(mono_path, az, r)

            # ---- LONG-TERM MODE: stream PCM frames ----
            if self.pcm_stream is not None and self.pcm_params is not None:
                frame_bytes = int(getattr(self.pcm_params, "frame_bytes"))
                pcm_bytes = wav_bytes_to_pcm16_bytes(wav_bytes)
                for frame in pcm16_chunker(pcm_bytes, frame_bytes):
                    x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                    print("peak", np.max(np.abs(x)), "rms", np.sqrt(np.mean(x*x)))
                    self.pcm_stream.send_frame(frame)
            else:
                # ---- LEGACY MODE: send WAV as file chunks ----
                self.audio_sender.send_wav_bytes(
                    f"{label}.wav",
                    wav_bytes,
                    repeat_chunk0=self.repeat_chunk0
                )

            print(f"[SOFA] {label} az={az:.1f} r={r:.2f}m "\
                  f"bytes={len(wav_bytes)} dt={time.time() - now}")


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

    # ---------- Audio (legacy sender stays for fallback + throttling) ----------
    audio_sender = UDPAudioBytesSender(
        target_ip=deep_get(cfg, "network.pi_ip"),
        target_port=int(deep_get(cfg, "ports.audio_udp")),
        chunk_payload_bytes=int(deep_get(cfg, "audio.send_chunk_bytes", 1200)),
        inter_packet_sleep_s=float(deep_get(cfg, "audio.send_inter_packet_sleep_s", 0.0)),
        per_key_ms=int(deep_get(cfg, "audio.per_key_ms", 400)),
        global_ms=int(deep_get(cfg, "audio.global_ms", 120)),
        sndbuf_bytes=int(deep_get(cfg, "audio.send_sndbuf_bytes", 8 << 20)),
        min_inter_file_gap_s=float(deep_get(cfg, "audio.min_inter_file_gap_s", 0.15)),
    )

    spatialiser = SofaSpatialiser(
        sofa_path=deep_get(cfg, "sonification.sofa"),
        image_width=float(deep_get(cfg, "realsense.color_w", 640)),
        out_sr=int(deep_get(cfg, "sonification.out_sr", 16000)),
        clip_s=float(deep_get(cfg, "sonification.clip_s", 0.35)),
        fade_ms=float(deep_get(cfg, "sonification.fade_ms", 8.0)),
        max_peak=float(deep_get(cfg, "sonification.max_peak", 0.95)),
    )

    # ---------- PCM streaming (long-term recommended) ----------
    pcm_stream = None
    pcm_params = None
    if bool(deep_get(cfg, "audio.stream.enabled", False)):
        if PCMStreamUDPSender is None or PCMStreamParams is None:
            raise RuntimeError(
                "audio.stream.enabled is true, but udp_pcm_stream.py could not be imported."
            )

        pcm_params = PCMStreamParams(
            sample_rate=int(deep_get(cfg, "audio.stream.sample_rate", deep_get(cfg, "sonification.out_sr", 16000))),
            channels=int(deep_get(cfg, "audio.stream.channels", 2)),
            sample_width_bytes=2,  # int16
            frame_ms=int(deep_get(cfg, "audio.stream.frame_ms", 20)),
            jitter_ms=int(deep_get(cfg, "audio.stream.jitter_ms", 120)),      # receiver uses this; harmless here
            max_jitter_ms=int(deep_get(cfg, "audio.stream.max_jitter_ms", 400)),
            bind_ip="0.0.0.0",
            mtu_payload=int(deep_get(cfg, "audio.stream.mtu_payload", 1400)),
        )

        # This streaming design expects: 1 UDP packet == 1 PCM frame
        if int(getattr(pcm_params, "frame_bytes")) > int(getattr(pcm_params, "mtu_payload")):
            raise ValueError(
                f"PCM frame_bytes={getattr(pcm_params,'frame_bytes')} exceeds "
                f"mtu_payload={getattr(pcm_params,'mtu_payload')}. "
                f"Reduce audio.stream.frame_ms or increase audio.stream.mtu_payload."
            )

        pcm_stream = PCMStreamUDPSender(
            host=str(deep_get(cfg, "network.pi_ip")),
            port=int(deep_get(cfg, "audio.stream.port", 50030)),
            params=pcm_params,
            sndbuf_bytes=int(deep_get(cfg, "audio.stream.sndbuf_bytes", 8 * 1024 * 1024)),
            debug=bool(deep_get(cfg, "audio.debug_udp", False)),
        )
        print(
            f"[PCM][TX] enabled -> {deep_get(cfg,'network.pi_ip')}:{deep_get(cfg,'audio.stream.port',50030)} "
            f"sr={getattr(pcm_params,'sample_rate')} ch={getattr(pcm_params,'channels')} "
            f"frame_ms={getattr(pcm_params,'frame_ms')} frame_bytes={getattr(pcm_params,'frame_bytes')}"
        )

    sonifier = YoloSofaSonifier(cfg, audio_sender, spatialiser, pcm_stream=pcm_stream, pcm_params=pcm_params)

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
        if pcm_stream is not None:
            try:
                pcm_stream.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

