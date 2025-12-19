#!/usr/bin/env python3
"""
Raspberry Pi unified sensor streamer:
- MAX30102 PPG -> TCP (records packed as !dii)
- RealSense D455 RGB + Depth + IMU -> TCP (len-prefixed pickle dict)

PC side receivers (existing):
- PPG: ppg_receive.py (TCPServerReceiver / PPGPacketCodec) or main_sonify_live.py (PPGTCPReceiver)
- RealSense: rs_d455_tcp_receiver.py or main_sonify_live.py (RealSenseYOLOFromTCP)
"""

import time
import os
import sys
import signal
import subprocess
from dataclasses import dataclass
from typing import Optional

# --- PPG pieces (from ppg_stream.py) ---
from ppg_stream import (
    RingBuffer,
    RateMonitor,
    SampleSpooler,
    TCPPPGSender,
    MAX30102PPGStream,
)

# --- RealSense sender (from rs_d455_tcp_sender.py) ---
from rs_d455_tcp_sender import RealSenseD455TCPSender

# RPi.GPIO cleanup is used in ppg_stream.py; keeping import here for safe shutdown.
from RPi import GPIO


@dataclass
class PPGConfig:
    pc_ip: str
    port: int = 9999
    log_dir: str = 'logs/'
    rotate_seconds: int = 3600
    fsync_every: int = 0
    rate_print: bool = True
    force_poll: bool = False
    no_data_timeout: float = 5.0
    poll_sleep_ms: float = 5.0
    # Run PPG streaming in a separate OS process to avoid GIL/CPU starvation from RealSense encoding.
    use_subprocess: bool = True
    # Path to ppg_stream.py (relative to this file or absolute).
    ppg_script: str = "ppg_stream.py"
    ring_maxlen: int = 60 * 60 * 60  # 1 hour @ 1 Hz; but we store tuples; adjust as desired


@dataclass
class RealSenseConfig:
    pc_ip: str
    port: int = 50000
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 480
    depth_height: int = 270
    fps: int = 30
    jpeg_quality: int = 80
    piggyback_ppg_latest: bool = True


class RaspberryPiSensorStreamer:
    """
    One class to start/stop both:
      - PPG TCP sender (MAX30102)
      - RealSense TCP sender (RGB + Depth + IMU)

    This intentionally keeps the **wire formats identical** to your existing PC receivers:
      - PPG: struct.pack("!dii", ts, red, ir)
      - RealSense: [4-byte length][pickle.dumps(dict)]
    """

    def __init__(
        self,
        ppg: Optional[PPGConfig] = None,
        rs: Optional[RealSenseConfig] = None,
    ):
        self.ppg_cfg = ppg
        self.rs_cfg = rs

        self._ppg_sender: Optional[TCPPPGSender] = None
        self._ppg_spooler: Optional[SampleSpooler] = None
        self._ppg_stream: Optional[MAX30102PPGStream] = None
        self._ppg_rate: Optional[RateMonitor] = None
        self._ppg_buffer: Optional[RingBuffer] = None
        self._ppg_proc: Optional[subprocess.Popen] = None

        self._rs_sender: Optional[RealSenseD455TCPSender] = None

        self._running = False

    # ---------------------------
    # Construction helpers
    # ---------------------------

    def _start_ppg_subprocess(self) -> None:
        # Already running?
        if self._ppg_proc is not None and self._ppg_proc.poll() is None:
            return

        script = getattr(self.ppg_cfg, "ppg_script", "ppg_stream.py")
        # Resolve relative path next to this file
        if not os.path.isabs(script):
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)

        cmd = [
            sys.executable, script,
            "--send-ip", str(self.ppg_cfg.pc_ip),
            "--send-port", str(int(self.ppg_cfg.port)),
            "--no-plot",
            "--log-dir", str(self.ppg_cfg.log_dir),
            "--rotate-seconds", str(int(self.ppg_cfg.rotate_seconds)),
            "--fsync-every", str(int(self.ppg_cfg.fsync_every)),
            "--no-data-timeout", str(float(self.ppg_cfg.no_data_timeout)),
            "--poll-sleep-ms", str(float(self.ppg_cfg.poll_sleep_ms)),
        ]

        if bool(self.ppg_cfg.rate_print):
            cmd.append("--rate-print")
        if bool(self.ppg_cfg.force_poll):
            cmd.append("--force-poll")

        # Start in a new session so we can SIGTERM the whole group.
        self._ppg_proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=None,  # inherit console
            stderr=None,
            start_new_session=True,
        )
        print(f"[PPG] Subprocess streaming -> {self.ppg_cfg.pc_ip}:{self.ppg_cfg.port} "
              f"(pid={self._ppg_proc.pid})")


    def _build_ppg(self):
        if self.ppg_cfg is None:
            return

        # If enabled, run PPG streaming as a *separate process* so it remains reliable
        # even when RealSense encoding/sending is slow and holds the GIL.
        # Default: False
        if bool(getattr(self.ppg_cfg, "use_subprocess", False)):
            self._start_ppg_subprocess()
            return

        self._ppg_rate = RateMonitor(
            interval_sec=2.0, 
            enabled=bool(self.ppg_cfg.rate_print))
        self._ppg_buffer = RingBuffer(
            maxlen=int(self.ppg_cfg.ring_maxlen)
        )

        self._ppg_sender = TCPPPGSender(
            self.ppg_cfg.pc_ip,
            int(self.ppg_cfg.port)
        )

        self._ppg_spooler = SampleSpooler(
            log_dir=self.ppg_cfg.log_dir,
            rotate_every_seconds=int(self.ppg_cfg.rotate_seconds),
            forward_sender=self._ppg_sender,
            fsync_every=int(self.ppg_cfg.fsync_every),
        )

        self._ppg_stream = MAX30102PPGStream(
            buffer=self._ppg_buffer,
            spooler=self._ppg_spooler,
            rate_monitor=self._ppg_rate,
            force_poll=bool(self.ppg_cfg.force_poll),
            no_data_timeout=float(self.ppg_cfg.no_data_timeout),
            poll_sleep_ms=float(self.ppg_cfg.poll_sleep_ms),
        )

    def _build_rs(self):
        if self.rs_cfg is None:
            return

        # Optional: piggyback the latest PPG sample into every RGB-D packet.
        # This guarantees at least one PPG sample arrives per RGB-D frame on the PC,
        # even if the dedicated PPG TCP stream gets stalled under heavy load.
        ppg_latest_fn = None
        if bool(getattr(self.rs_cfg, "piggyback_ppg_latest", False)) and self._ppg_buffer is not None:
            ppg_latest_fn = self._ppg_buffer.latest

        self._rs_sender = RealSenseD455TCPSender(
            host=self.rs_cfg.pc_ip,
            port=int(self.rs_cfg.port),
            color_width=int(self.rs_cfg.color_width),
            color_height=int(self.rs_cfg.color_height),
            depth_width=int(self.rs_cfg.depth_width),
            depth_height=int(self.rs_cfg.depth_height),
            fps=int(self.rs_cfg.fps),
            jpeg_quality=int(self.rs_cfg.jpeg_quality),
            ppg_latest_fn=ppg_latest_fn,
        )

    # ---------------------------
    # Public API
    # ---------------------------
    def start(self):
        if self._running:
            return
        self._running = True

        # Build components
        if self.ppg_cfg is not None:
            self._build_ppg()
        if self.rs_cfg is not None:
            self._build_rs()

        # Start PPG chain: sender -> spooler -> stream
        if self._ppg_sender is not None:
            self._ppg_sender.start()
            print(f"[PPG] TCP streaming -> " \
                  f"{self._ppg_sender.host}:{self._ppg_sender.port}")
        if self._ppg_spooler is not None:
            self._ppg_spooler.start()
            print(f"[PPG] Logging -> {self._ppg_spooler.log_dir}")
        if self._ppg_stream is not None:
            self._ppg_stream.start()
            print("[PPG] MAX30102 stream started")

        # Start RealSense stream thread
        if self._rs_sender is not None:
            self._rs_sender.start()
            print(f"[RS] TCP streaming -> "\
                  f"{self.rs_cfg.pc_ip}:{self.rs_cfg.port}")

    def stop(self):
        self._running = False

        # Stop PPG subprocess if used
        if self._ppg_proc is not None:
            try:
                if self._ppg_proc.poll() is None:
                    os.killpg(self._ppg_proc.pid, signal.SIGTERM)
                    self._ppg_proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(self._ppg_proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            finally:
                self._ppg_proc = None

        # Stop RealSense first (camera resources)
        if self._rs_sender is not None:
            try:
                self._rs_sender.stop()
            except Exception as e:
                print(f"[RS] stop error: {e}")
            self._rs_sender = None

        # Stop PPG stream chain
        if self._ppg_stream is not None:
            try:
                self._ppg_stream.stop()
            except Exception as e:
                print(f"[PPG] stream stop error: {e}")
            self._ppg_stream = None

        # GPIO cleanup (MAX30102 uses GPIO interrupts)
        try:
            GPIO.cleanup()
        except Exception:
            pass

        if self._ppg_spooler is not None:
            try:
                self._ppg_spooler.stop()
            except Exception as e:
                print(f"[PPG] spooler stop error: {e}")
            self._ppg_spooler = None

        if self._ppg_sender is not None:
            try:
                self._ppg_sender.stop()
            except Exception as e:
                print(f"[PPG] sender stop error: {e}")
            self._ppg_sender = None

        print("[ALL] shutdown complete")

    def run_forever(self, sleep_sec: float = 1.0):
        self.start()
        try:
            while True:
                time.sleep(float(sleep_sec))
        except KeyboardInterrupt:
            print("KeyboardInterrupt: stopping...")
        finally:
            self.stop()


def main():
    # Example usage:
    #   python3 rpi_sensor_streamer.py --edit PC IPs below

    pc_ip = "192.168.0.194"  # <-- set to your PC on the same network

    streamer = RaspberryPiSensorStreamer(
        ppg=PPGConfig(pc_ip=pc_ip,
                      port=9999,
                      rate_print=True,
                      force_poll=True),
        rs=RealSenseConfig(pc_ip=pc_ip,
                           port=50000,
                           color_width=640,
                           color_height=480,
                           depth_width=480,
                           depth_height=270,
                           fps=30,
                           jpeg_quality=80),
    )
    streamer.run_forever()


if __name__ == "__main__":
    main()

