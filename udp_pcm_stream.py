#!/usr/bin/env python3
"""
udp_pcm_stream.py

Low-latency PCM-over-UDP streaming with:
- tiny fixed header
- receiver jitter buffer
- continuous ALSA playback via aplay stdin

Protocol (network byte order):
  magic(4) = b'PCMS'
  version(1) = 1
  stream_id(u32)
  seq(u32)            monotonically increasing per stream_id
  timestamp_us(u64)   sender time (optional, for debug/metrics)
  payload_len(u16)
  flags(u8)           reserved (0)
Total header size: 24 bytes
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict


_HDR_FMT = "!4sBIIQHB"  # 4+1+4+4+8+2+1 = 24 bytes
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_MAGIC = b"PCMS"
_VERSION = 1


def _now_us() -> int:
    return int(time.time() * 1_000_000)


@dataclass
class PCMStreamParams:
    sample_rate: int = 16000
    channels: int = 2
    sample_width_bytes: int = 2  # int16
    frame_ms: int = 20           # 20ms packets are a good default
    jitter_ms: int = 120         # target receiver buffer before starting playout
    max_jitter_ms: int = 400     # hard cap for buffer growth
    bind_ip: str = "0.0.0.0"
    mtu_payload: int = 1400      # payload only (header excluded)


    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.channels * self.sample_width_bytes

    @property
    def jitter_frames(self) -> int:
        return max(1, int(self.jitter_ms / self.frame_ms))

    @property
    def max_jitter_frames(self) -> int:
        return max(self.jitter_frames + 1, int(self.max_jitter_ms / self.frame_ms))


class PCMStreamUDPSender:
    def __init__(
        self,
        host: str,
        port: int,
        params: PCMStreamParams,
        stream_id: Optional[int] = None,
        sndbuf_bytes: int = 8 * 1024 * 1024,
    ):
        self.host = host
        self.port = port
        self.params = params
        self.stream_id = int(stream_id if stream_id is not None else (time.time() * 1000) % (2**32))
        self.seq = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf_bytes)
        except OSError:
            pass

        # sanity: keep one packet per frame (recommended)
        if self.params.frame_bytes > self.params.mtu_payload:
            raise ValueError(
                f"frame_bytes={self.params.frame_bytes} exceeds mtu_payload={self.params.mtu_payload}. "
                f"Reduce frame_ms or mtu_payload."
            )

    def send_frame(self, pcm_frame: bytes):
        """Send exactly one PCM frame (int16 interleaved stereo)"""
        if len(pcm_frame) != self.params.frame_bytes:
            raise ValueError(f"Expected {self.params.frame_bytes} bytes, got {len(pcm_frame)}")

        hdr = struct.pack(
            _HDR_FMT,
            _MAGIC,
            _VERSION,
            self.stream_id,
            self.seq,
            _now_us(),
            len(pcm_frame),
            0,
        )
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.sock.sendto(hdr + pcm_frame, (self.host, self.port))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class _JitterBuffer:
    """
    Stores packets keyed by seq. Plays out in-order from expected_seq.
    Missing packets => return None and caller can insert silence.
    """
    def __init__(self, params: PCMStreamParams):
        self.params = params
        self.lock = threading.Lock()
        self.buf: Dict[int, bytes] = {}
        self.expected_seq: Optional[int] = None
        self.started = False

    def push(self, seq: int, payload: bytes):
        with self.lock:
            # initialize expected sequence on first packet
            if self.expected_seq is None:
                self.expected_seq = seq

            # store
            self.buf[seq] = payload

            # cap buffer to avoid unbounded growth
            if len(self.buf) > self.params.max_jitter_frames:
                # drop oldest seqs below expected_seq if they piled up
                if self.expected_seq is not None:
                    drop_before = (self.expected_seq - self.params.max_jitter_frames) & 0xFFFFFFFF
                    # best-effort prune: remove far-behind keys
                    to_del = []
                    for k in self.buf.keys():
                        # not perfect wrap-safe ordering, but good enough for normal runs
                        if k < drop_before:
                            to_del.append(k)
                    for k in to_del:
                        self.buf.pop(k, None)

    def ready_to_start(self) -> bool:
        with self.lock:
            return len(self.buf) >= self.params.jitter_frames and (self.expected_seq is not None)

    def pop_next(self) -> Optional[bytes]:
        with self.lock:
            if self.expected_seq is None:
                return None
            payload = self.buf.pop(self.expected_seq, None)
            self.expected_seq = (self.expected_seq + 1) & 0xFFFFFFFF
            return payload


class PCMStreamUDPReceiver:
    def __init__(self, port: int, params: PCMStreamParams, rcvbuf_bytes: int = 8 * 1024 * 1024):
        self.port = port
        self.params = params
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
        except OSError:
            pass
        self.sock.bind((self.params.bind_ip, self.port))
        self.sock.settimeout(0.5)

        self.jb = _JitterBuffer(params)
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._rx_loop, daemon=True)

        # stats
        self.pkts = 0
        self.bad = 0
        self.last_seq: Optional[int] = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        try:
            self.sock.close()
        except Exception:
            pass

    def _rx_loop(self):
        while not self.stop_evt.is_set():
            try:
                data, _addr = self.sock.recvfrom(_HDR_SIZE + self.params.mtu_payload + 64)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < _HDR_SIZE:
                self.bad += 1
                continue

            try:
                magic, ver, stream_id, seq, ts_us, payload_len, flags = struct.unpack(_HDR_FMT, data[:_HDR_SIZE])
            except struct.error:
                self.bad += 1
                continue

            if magic != _MAGIC or ver != _VERSION:
                self.bad += 1
                continue

            payload = data[_HDR_SIZE:_HDR_SIZE + payload_len]
            if len(payload) != payload_len:
                self.bad += 1
                continue

            if payload_len != self.params.frame_bytes:
                # enforce fixed frame size in this design
                self.bad += 1
                continue

            self.pkts += 1
            self.last_seq = seq
            self.jb.push(seq, payload)


class ALSAContinuousPlayer:
    """
    Keeps an 'aplay' subprocess open and feeds PCM frames to stdin continuously.
    This avoids start/stop clicks and works well with A2DP sinks (if default device routes to BT).
    """
    def __init__(self, params: PCMStreamParams, alsa_device: Optional[str] = None):
        self.params = params
        self.alsa_device = alsa_device
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "aplay",
            "-q",
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(self.params.sample_rate),
            "-c", str(self.params.channels),
        ]
        if self.alsa_device:
            cmd.extend(["-D", self.alsa_device])

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def write(self, pcm_frame: bytes):
        if not self.proc or not self.proc.stdin:
            return
        try:
            self.proc.stdin.write(pcm_frame)
        except BrokenPipeError:
            # output device disappeared; caller should restart
            raise

    def stop(self):
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


class PCMStreamPlayerApp:
    """
    Orchestrates:
      UDP receiver -> jitter buffer -> continuous ALSA player
    """
    def __init__(self, listen_port: int, params: PCMStreamParams, alsa_device: Optional[str] = None, debug: bool = True):
        self.listen_port = listen_port
        self.params = params
        self.alsa_device = alsa_device
        self.debug = debug

        self.rx = PCMStreamUDPReceiver(port=self.listen_port, params=self.params)
        self.player = ALSAContinuousPlayer(params=self.params, alsa_device=self.alsa_device)

        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._play_loop, daemon=True)

    def start(self):
        if self.debug:
            print(f"[PCM][RX] bind={self.params.bind_ip}:{self.listen_port} "
                  f"sr={self.params.sample_rate} ch={self.params.channels} frame={self.params.frame_ms}ms "
                  f"jitter={self.params.jitter_ms}ms")
        self.rx.start()
        self.player.start()
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        self.rx.stop()
        self.player.stop()

    def _play_loop(self):
        silence = b"\x00" * self.params.frame_bytes
        started = False
        last_stat = time.time()
        underruns = 0

        while not self.stop_evt.is_set():
            if not started:
                if self.rx.jb.ready_to_start():
                    started = True
                    if self.debug:
                        print(f"[PCM][PLAY] starting playout (buffered {len(self.rx.jb.buf)} frames)")
                else:
                    time.sleep(0.005)
                    continue

            payload = self.rx.jb.pop_next()
            if payload is None:
                underruns += 1
                payload = silence

            try:
                self.player.write(payload)
            except BrokenPipeError:
                # try restarting aplay (e.g. BT sink reset)
                if self.debug:
                    print("[PCM][ALSA] Broken pipe, restarting aplay...")
                self.player.stop()
                time.sleep(0.2)
                self.player.start()

            # pace playout exactly to frame rate
            time.sleep(self.params.frame_ms / 1000.0)

            if self.debug and (time.time() - last_stat) > 2.0:
                last_stat = time.time()
                print(f"[PCM][STAT] pkts={self.rx.pkts} bad={self.rx.bad} "
                      f"buf={len(self.rx.jb.buf)} underruns={underruns}")

