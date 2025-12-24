#!/usr/bin/env python3
"""
pc_main.py

PC-side orchestrator (library-aligned):

- Receives RealSense RAW RGB-D + IMU over UDP using:
    RealSenseRawUDPReceiver (from rs_d455_raw_udp_receiver.py)
- Receives PPG via TCP using ppg_receive.py
- Runs YOLO on RGB frames
- Streams detections via OSC to a local SOFA/BT spatialiser
  (DepthAwareSpatialSound), keeping all spatialisation on the PC

Run:
  python pc_main.py --config nhmrc_streaming_config.yaml
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
from dataclasses import dataclass

import numpy as np
import yaml
import soundfile as sf
from scipy import signal
from ultralytics import YOLO
from pythonosc import udp_client

# ---------------- REQUIRED PROJECT IMPORTS ----------------

from ppg_receive import TCPServerReceiver
import bth_audio_manager as bam
from yolo_sofa import SpatialSoundHeadphoneYOLO, DepthAwareSpatialSound
from config import deep_get, load_config

# import updated receiver library
from rs_d455_raw_udp_receiver import (
    RealSenseRawUDPReceiver,
    ReceiverStats,
)

from rgbd_coord_streamer import App as VisAudioStreamer


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

def wav_bytes_to_pcm16_frames(wav_bytes: bytes, frame_bytes: int):
    """
    Decode WAV bytes to int16 interleaved stereo PCM frames of size frame_bytes.
    Any remainder is dropped (for streaming).
    """
    bio = io.BytesIO(wav_bytes)
    audio, sr = sf.read(bio, dtype="int16", always_2d=True)
    # audio shape: (N, C)
    if audio.shape[1] == 1:
        # upmix mono -> stereo
        audio = audio.repeat(2, axis=1)
    pcm = audio.tobytes()
    n = (len(pcm) // frame_bytes) * frame_bytes
    for i in range(0, n, frame_bytes):
        yield pcm[i:i + frame_bytes]


# ---------------- SOFA SPATIALISER ----------------
class YoloOSCStreamer:
    """Send YOLO detections to a local OSC spatialiser on the PC."""

    def __init__(self, cfg: Dict, osc_host: str, osc_port: int):
        self.cfg = cfg
        self.model = YOLO(deep_get(cfg, "yolo.model")).to('cuda')
        self.conf = float(deep_get(cfg, "yolo.conf", 0.3))
        self.osc = udp_client.SimpleUDPClient(osc_host, int(osc_port))
        self.frame_id = 0
        self.max_det = int(deep_get(cfg, "yolo.max_det", 10))
        self._timing_samples = []
        self._timing_report_every = 10  # frames

    def process(self, pkt: Dict):
        rgb = pkt["color"]
        depth = pkt.get("depth")
        depth_scale = pkt.get("depth_scale", 0.001)

        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        res = self.model(rgb, verbose=False, max_det=self.max_det)[0]
        t1 = time.perf_counter()

        # If YOLO returns nothing (e.g., empty frame), still account for the
        # inference time so the periodic profiler can emit a line.
        if res.boxes is None or len(res.boxes) == 0:
            self._record_timing(t1 - t0, 0.0)
            return

        frame_id = self.frame_id
        self.frame_id += 1

        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < self.conf:
                continue

            label = self.model.names[int(box.cls[0])]

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx_rgb = int((x1 + x2) / 2)
            cy_rgb = int((y1 + y2) / 2)
            y_norm = cy_rgb / float(max(1.0, h))

            r = 1.0
            if depth is not None:
                dh, dw = depth.shape[:2]
                cx_d = int(cx_rgb * dw / w)
                cy_d = int(cy_rgb * dh / h)
                cx_d = max(0, min(dw - 1, cx_d))
                cy_d = max(0, min(dh - 1, cy_d))
                d = depth[cy_d, cx_d]
                if d > 0:
                    r = float(d) * depth_scale

            self.osc.send_message("/yolo", [cx_rgb, label, y_norm, r, frame_id])

        # Basic profiling so we can pinpoint OSC slowdown sources.
        # Model inference dominates when OSC streams feel sluggish.
        t2 = time.perf_counter()
        self._record_timing(t1-t0, t2-t1)

    def _record_timing(self, inference_s:float, osc_s:float):
        """Collect and periodically print YOLO→OSC timing averages."""
        self._timing_samples.append((inference_s, osc_s))
        if len(self._timing_samples) >= self._timing_report_every:
            inf_avg = sum(s[0] for s in self._timing_samples) / len(self._timing_samples)
            osc_avg = sum(s[1] for s in self._timing_samples) / len(self._timing_samples)
            print(
                f"[YOLO→OSC] avg inference={inf_avg*1000:.1f} ms, "
                f"avg osc/packaging={osc_avg*1000:.1f} ms over {len(self._timing_samples)} frames"
            )
            self._timing_samples.clear()



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

    def _trim_clip(self, audio:np.ndarray, sr:int, clip_s_ovr:float=None) -> np.ndarray:
        """Keep only the first clip_s seconds (or entire signal if shorter)."""
        if clip_s_ovr is not None:
            clip_s = clip_s_ovr
        else:
            clip_s = self.clip_s

        if clip_s <= 0:
            return audio
        n = int(sr * clip_s)
        if n <= 0:
            return audio
        return audio[:n]

    def spatialise_array(
        self,
        mono_path: str,
        az_deg: float,
        r: float,
        clip_override_s: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Spatialise and return float32 stereo array + sample rate.
        Optional clip_override_s lets callers force a shorter windowed render.
        """
        clip_s = self.clip_s if clip_override_s is None else float(clip_override_s)

        audio, fs_src = sf.read(mono_path, dtype="float32", always_2d=False)
        audio = self._ensure_mono(audio)

        # 1) Trim early to reduce convolution cost
        audio = self._trim_clip(audio, fs_src)
        audio = self._trim_clip(audio, fs_src, clip_s_ovr=clip_s)

        # 2) Resample source to BRIR sample rate for convolution
        if fs_src != self.fs_brir:
            audio = signal.resample_poly(audio, self.fs_brir,
                                         fs_src).astype(np.float32, copy=False)
 

        stereo = self.engine.sound_process(
            data=[az_deg, 0.0, r],
            audio=audio,
            BRIRs=self.engine.BRIRs,
            sourcePositions=self.engine.BRIR_sourcePositions,
        )

        # Ensure shape (N,2)
        stereo = np.asarray(stereo, dtype=np.float32)
        if stereo.ndim == 1:
            stereo = np.stack([stereo, stereo], axis=1)
        elif stereo.ndim == 2 and stereo.shape[0] == 2 and stereo.shape[1] != 2:
            # sometimes returned as (2,N)
            stereo = stereo.T

        # 3) Trim again after convolution (BRIR can extend tail)
        stereo = stereo[: int(self.fs_brir * clip_s), :] if clip_s > 0 else stereo

        # 4) Downsample to 16kHz to shrink payload
        if self.out_sr > 0 and self.out_sr != self.fs_brir:
            left = signal.resample_poly(stereo[:, 0], self.out_sr, self.fs_brir)
            right = signal.resample_poly(stereo[:, 1], self.out_sr, self.fs_brir)
            stereo = np.stack([left, right], axis=1).astype(np.float32, copy=False)

        # 5) Fade-in/out to avoid clicks
        stereo[:, 0] = self._apply_fade(stereo[:, 0], self.out_sr, self.fade_ms)
        stereo[:, 1] = self._apply_fade(stereo[:, 1], self.out_sr, self.fade_ms)

        # 6) Normalize and keep some headroom, then clamp
        peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
        if peak > 0:
            stereo *= min(1.0, self.max_peak / peak)
        stereo = np.clip(stereo, -1.0, 1.0)

        return stereo.astype(np.float32, copy=False), int(self.out_sr)

    def spatialise_file(self, mono_path: str, az_deg: float, r: float) -> bytes:
        stereo, sr = self.spatialise_array(mono_path, az_deg, r)
        buf = io.BytesIO()
        # 7) 16-bit PCM WAV (small + universally playable)
        # sf.write(buf, stereo, int(self.out_sr), format="WAV", subtype="PCM_16")
        sf.write(buf, stereo, int(sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ---------------- SCENE MIXDOWN ----------------


@dataclass
class SceneAudioEvent:
    label: str
    conf: float
    area: float
    az_deg: float
    r: float
    mono_path: str


class SceneAudioWindowMixer:
    """
    Collect detections over a short window, spatialise them, and send a single
    mixed clip (WAV or packetised PCM) to the Pi.
    """
    def __init__(
        self,
        spatialiser: SofaSpatialiser,
        audio_sender: UDPAudioBytesSender,
        pcm_stream: Optional[object],
        pcm_params: Optional[object],
        window_ms: int = 200,
        max_objects: int = 3,
        repeat_chunk0: int = 1,
    ):
        self.spatialiser = spatialiser
        self.audio_sender = audio_sender
        self.pcm_stream = pcm_stream
        self.pcm_params = pcm_params
        self.window_ms = int(window_ms)
        self.max_objects = int(max_objects)
        self.repeat_chunk0 = int(repeat_chunk0)

        self.events: List[SceneAudioEvent] = []
        self.window_start = time.time()

        # Mix and transmit in the background so rendering one window doesn't
        # block the detector loop that is accumulating the next window.
        self._stop_evt = threading.Event()
        self._mix_queue: Queue[list[SceneAudioEvent]] = Queue(maxsize=2)
        self._worker = threading.Thread(target=self._mix_worker, daemon=True)
        self._worker.start()

    def _mix_worker(self):
        while not self._stop_evt.is_set():
            try:
                events = self._mix_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                wav_bytes = self._mix_events(events)
                if wav_bytes:
                    self._send_audio(wav_bytes)
            finally:
                self._mix_queue.task_done()

    def _enqueue_mix(self, events: list[SceneAudioEvent]):
        try:
            self._mix_queue.put_nowait(events)
        except Full:
            # Drop the oldest in-flight mix so the latest scene can be rendered
            # while the previous one plays.
            try:
                _ = self._mix_queue.get_nowait()
            except Empty:
                pass
            try:
                self._mix_queue.put_nowait(events)
            except Full:
                # If another thread raced us, just drop this window.
                pass    

    def add_event(self, event: SceneAudioEvent):
        self.events.append(event)

    def _mix_events(self, events: List[SceneAudioEvent]) -> Optional[bytes]:
        if not events:
            return None

        # prioritize by confidence * area (bigger / more certain objects win)
        ordered = sorted(events, key=lambda e: (e.conf * max(e.area, 1e-6)), reverse=True)
        selected = ordered[: self.max_objects]

        # fixed window length in samples
        window_s = self.window_ms / 1000.0
        stereo_mix: Optional[np.ndarray] = None
        sr_out: Optional[int] = None

        for ev in selected:
            stereo, sr = self.spatialiser.spatialise_array(
                ev.mono_path, ev.az_deg, ev.r, clip_override_s=window_s
            )
            target_samples = int(sr * window_s)
            stereo = stereo[:target_samples]
            if stereo.shape[0] < target_samples:
                pad = np.zeros((target_samples - stereo.shape[0], 2), dtype=np.float32)
                stereo = np.vstack([stereo, pad])

            if stereo_mix is None:
                stereo_mix = stereo
                sr_out = sr
            else:
                # ensure same length
                n = min(stereo_mix.shape[0], stereo.shape[0])
                stereo_mix[:n, :] += stereo[:n, :]

        if stereo_mix is None or sr_out is None:
            return None

        # normalize to avoid clipping when summing
        peak = float(np.max(np.abs(stereo_mix))) if stereo_mix.size else 0.0
        if peak > 1.0:
            stereo_mix /= peak
        stereo_mix = np.clip(stereo_mix, -1.0, 1.0)

        buf = io.BytesIO()
        sf.write(buf, stereo_mix, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _send_audio(self, wav_bytes: bytes):
        # Prefer continuous PCM streaming when available
        if self.pcm_stream is not None and self.pcm_params is not None:
            frame_bytes = int(getattr(self.pcm_params, "frame_bytes"))
            for frame in wav_bytes_to_pcm16_frames(wav_bytes, frame_bytes):
                self.pcm_stream.send_frame(frame)
        else:
            self.audio_sender.send_wav_bytes(
                "scene_mix.wav",
                wav_bytes,
                repeat_chunk0=self.repeat_chunk0,
            )

    def maybe_flush(self, now: Optional[float] = None):
        now = time.time() if now is None else now
        if (now - self.window_start) * 1000.0 < self.window_ms:
            return
        self.flush(now)

    def flush(self, now: Optional[float] = None):
        now = time.time() if now is None else now
        events = list(self.events)
        self.events = []
        self.window_start = now
        if events:
            self._enqueue_mix(events)

    def stop(self):
        self._stop_evt.set()
        try:
            while not self._mix_queue.empty():
                try:
                    _ = self._mix_queue.get_nowait()
                    self._mix_queue.task_done()
                except Empty:
                    break
        except Exception:
            pass
        self._worker.join(timeout=1.0)



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
        self.model = YOLO(deep_get(cfg, "yolo.model")).to('cuda')
        self.conf = float(deep_get(cfg, "yolo.conf", 0.3))
        self.max_det = int(deep_get(cfg, 'yolo.max_det', 10))

        self.audio_sender = audio_sender
        self.spatialiser = spatialiser

        # Long-term streaming path (if enabled)
        self.pcm_stream = pcm_stream
        self.pcm_params = pcm_params

        self.sound_map = deep_get(cfg, "sonification.sound_map", {})
        self.default_sound = deep_get(cfg, "sonification.default_sound")

        self.last_play = {}
        self.per_label_cooldown_s = float(
            deep_get(cfg, "audio.per_label_cooldown_s", 0.5))
        self.repeat_chunk0 = int(deep_get(cfg, "audio.send_repeat_chunk0", 4))

        self.scene_mixer = SceneAudioWindowMixer(
            spatialiser=self.spatialiser,
            audio_sender=self.audio_sender,
            pcm_stream=self.pcm_stream,
            pcm_params=self.pcm_params,
            window_ms=int(deep_get(cfg, "audio.scene.window_ms", 200)),
            max_objects=int(deep_get(cfg, "audio.scene.max_objects", 3)),
            repeat_chunk0=self.repeat_chunk0,
        )

    def process(self, pkt: Dict):
        rgb = pkt["color"]
        depth = pkt.get("depth")
        depth_scale = pkt.get("depth_scale", 0.001)

        h, w = rgb.shape[:2]
        res = self.model(rgb, verbose=False, max_det=self.max_det)[0]

        if res.boxes is None:
            self.scene_mixer.maybe_flush(time.time())
            return

        now = time.time()
        self.scene_mixer.maybe_flush(now)
        pending_events: List[SceneAudioEvent] = []

        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < self.conf:
                continue

            label = self.model.names[int(box.cls[0])]
            mono_path = self.sound_map.get(label, self.default_sound)
            if not mono_path:
                continue

            # --- compute bbox center in RGB space ---
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
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

            # 1) cooldown (existing)
            if now - self.last_play.get(label, 0) < self.per_label_cooldown_s:
                continue
            self.last_play[label] = now

            area = float((x2 - x1) * (y2 - y1)) / float(max(1.0, w * h))
            pending_events.append(
                SceneAudioEvent(
                    label=label,
                    conf=conf,
                    area=area,
                    az_deg=az,
                    r=r,
                    mono_path=mono_path,
                )
            )

        for ev in pending_events:
            self.scene_mixer.add_event(ev)

        # Send one clip per window (packetised PCM preferred)
        self.scene_mixer.maybe_flush(time.time())

        if pending_events:
            labels = ", ".join([ev.label for ev in pending_events])
            print(f"[SOFA] window queued: {labels} (n={len(pending_events)})")



# ---------------- MAIN ----------------
def _run_pc_spatialiser(cfg: Dict, stop_evt: threading.Event):
    """Run the OSC-driven SOFA spatialiser on the PC (with BT output)."""

    sofa_path = deep_get(cfg, "sonification.sofa", "./sofa-lib/BRIR_HATS_3degree_for_glasses.sofa")
    image_width = float(deep_get(cfg, "realsense.color_w", 640))
    osc_port = int(deep_get(cfg, "ports.audio_udp", 40100))
    bt_connect_timeout_s = deep_get(cfg, "audio.bt_connect_timeout_s", 20.0)

    output_blocksize = deep_get(cfg, "audio.output_blocksize")
    output_latency_s = deep_get(cfg, "audio.output_latency_s")
    if output_blocksize is not None:
        try:
            output_blocksize = int(output_blocksize)
        except (TypeError, ValueError):
            output_blocksize = None
    if output_latency_s is not None:
        try:
            output_latency_s = float(output_latency_s)
        except (TypeError, ValueError):
            output_latency_s = None

    app = DepthAwareSpatialSound(
        sofa_file_path=sofa_path,
        image_width=image_width,
        osc_port=osc_port,
        verbose=True,
        bt_mac=str(deep_get(cfg, "audio.bt_mac", "")) or None,
        bt_pair=bool(deep_get(cfg, "audio.pair", False)),
        bt_trust=bool(deep_get(cfg, "audio.trust", True)),
        bt_connect_timeout_s=float(bt_connect_timeout_s),
        output_blocksize=output_blocksize,
        output_latency_s=output_latency_s,
    )

    thread = threading.Thread(target=app.start, daemon=True)
    thread.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    finally:
        try:
            app.OSCserver.server_close()
        except Exception:
            pass


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

    # ---------- PPG ----------
    ppg_rx = PPGReceiverThread(
        listen_ip=deep_get(cfg, "network.pc_listen_ip", "0.0.0.0"),
        port=int(deep_get(cfg, "ports.ppg_tcp")),
        poll_hz=float(deep_get(cfg, "ppg.pc_poll_hz", 500.0)),
    )
    ppg_rx.start()

    # ---------- Audio / OSC spatialisation on PC ----------
    stop_evt = threading.Event()
    spatial_thread = threading.Thread(
        target=_run_pc_spatialiser,
        args=(cfg, stop_evt),
        daemon=True,
    )
    spatial_thread.start()    

    osc_port = int(deep_get(cfg, "ports.audio_udp", 40100))
    sonifier: Optional[YoloOSCStreamer] = None
    sonifier = YoloOSCStreamer(cfg, "127.0.0.1", osc_port)    

    print("[PC] Running: RS → YOLO → OSC → BT (local)")

    # av_stream = VisAudioStreamer(cfg, rs_rx=rs_rx)

    try:
        # Run the RGB-D visual/audio streamer in the background (shares the
        # same UDP socket)
        # av_stream.start(background=True)
        rs_rx.start()
        while True:
            pkt = rs_rx.get_latest()
            if pkt and sonifier is not None:
                sonifier.process(pkt)
            time.sleep(1.0 / float(deep_get(cfg, "yolo.yolo_hz", 10)))
    except KeyboardInterrupt:
        print("\n[PC] Shutdown")
    finally:
        stop_evt.set()
        # av_stream.stop()
        rs_rx.stop()
        ppg_rx.stop()


if __name__ == "__main__":
    main()

