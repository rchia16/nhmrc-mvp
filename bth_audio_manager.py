#!/usr/bin/env python3
"""
bth_audio_manager.py
Single script that can run in either:
  - receiver mode (Raspberry Pi): connect BT speaker, receive chunked UDP "audio files", play them
  - sender mode (PC): chunk + send an audio file over UDP using the matching protocol

Examples
  # (Pi) receive + play to BT speaker
  python3 bth_audio_manager.py recv --bt-mac AA:BB:CC:DD:EE:FF --udp-port 40100 --pair

  # (PC) send a file to the Pi
  python3 bth_audio_manager.py send --pi-ip 192.168.50.2 --pi-port 40100 --file ./ding.wav

UDP protocol (matches both modes)
- Each UDP datagram:
    MAGIC(4)='AUD0'
    file_id(u32)
    chunk_idx(u16)
    chunk_cnt(u16)
    payload_len(u16)
    reserved(u16)
    payload bytes...

- File reassembly:
    Receiver concatenates payloads in idx order 0..chunk_cnt-1
- Chunk 0 payload begins with:
    name_len(u16) + name(bytes utf-8) + file_bytes...
- Other chunks: just file bytes slices (continued stream)

Notes
- UDP is lossy; for best results keep files short or add pacing (--sleep) / smaller chunks.
- Playback uses external tools; ensure one exists on Pi: ffplay (ffmpeg) or mpg123 or aplay.
"""

import argparse
import os
import queue
import random
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

MAGIC = b"AUD0"
HDR_FMT = "!4sIHHHH"  # magic, file_id, idx, cnt, payload_len, reserved
HDR_SIZE = struct.calcsize(HDR_FMT)


# ---------------------- Bluetooth control (Pi / recv mode) ----------------------

class BluetoothSpeakerManager:
    """
    Manages connecting to a BT speaker via `bluetoothctl`.
    Assumes your Pi is already set up so that BT audio sinks work once connected.
    """

    def __init__(self, mac: str, pair: bool = False, trust: bool = True, connect_timeout_s: float = 20.0):
        self.mac = mac
        self.pair = bool(pair)
        self.trust = bool(trust)
        self.connect_timeout_s = float(connect_timeout_s)

    @staticmethod
    def _run_btctl(commands: List[str], timeout_s: float = 25.0) -> Tuple[int, str]:
        p = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        script = "\n".join(commands) + "\n"
        try:
            out, _ = p.communicate(script, timeout=timeout_s)
            return p.returncode, out
        except subprocess.TimeoutExpired:
            p.kill()
            out = ""
            try:
                out = p.stdout.read() if p.stdout else ""
            except Exception:
                pass
            return 124, out

    def ensure_connected(self) -> None:
        cmds = [
            "power on",
            "agent on",
            "default-agent",
        ]
        if self.trust:
            cmds.append(f"trust {self.mac}")
        if self.pair:
            cmds.append(f"pair {self.mac}")
        cmds.append(f"connect {self.mac}")
        cmds.append("quit")

        rc, out = self._run_btctl(cmds, timeout_s=self.connect_timeout_s)
        print("[BT] bluetoothctl output:\n" + out.strip())
        if rc == 124:
            raise RuntimeError("[BT] bluetoothctl timed out (is bluetoothd running?)")

    def is_connected(self) -> bool:
        """Return True if bluetoothctl reports the device as connected."""
        rc, out = self._run_btctl([f"info {self.mac}", "quit"], timeout_s=10.0)
        if rc == 124:
            print("[BT] connection check timed out")
            return False
        print("[BT] info output:\n" + out.strip())
        return "Connected: yes" in out

    def disconnect(self) -> None:
        rc, out = self._run_btctl([f"disconnect {self.mac}", "quit"], timeout_s=10.0)
        print("[BT] disconnect output:\n" + out.strip())


# ---------------------- UDP file reassembly (Pi / recv mode) ----------------------

@dataclass
class InFlightFile:
    cnt: int
    parts: Dict[int, bytes]
    t0: float


class UDPFileReceiver:
    """
    Receives chunked UDP “file” messages, reassembles them, and emits (filename, bytes).
    """

    def __init__(
        self,
        listen_ip: str,
        port: int,
        out_queue: "queue.Queue[Tuple[str, bytes]]",
        # Large SOFA WAVs often need longer than 1s to fully arrive on a busy link.
        timeout_s: float = 3.0,
        # Avoid holding many incomplete files; "latest wins" playback prefers fewer inflight.
        max_inflight: int = 2,
        # Bigger socket buffer reduces drops during bursts.
        rcvbuf_bytes: int = 8 << 20,
        debug: bool = False,
    ):
        self.listen_ip = listen_ip
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.max_inflight = int(max_inflight)
        self.out_queue = out_queue
        self.debug = bool(debug)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.listen_ip, self.port))
        self._sock.settimeout(0.5)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf_bytes))
        except Exception:
            pass

        self._running = False
        self._th: Optional[threading.Thread] = None
        self._inflight: Dict[int, InFlightFile] = {}

        self.rx_chunks = 0
        self.rx_files_ok = 0
        self.rx_files_drop = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        print(f"[UDP] Listening on {self.listen_ip}:{self.port}")

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    def _drop_old_inflight_if_needed(self):
        if len(self._inflight) <= self.max_inflight:
            return
        items = sorted(self._inflight.items(), key=lambda kv: kv[1].t0)
        for file_id, _inf in items[: max(0, len(items) - self.max_inflight)]:
            self._inflight.pop(file_id, None)
            self.rx_files_drop += 1

    @staticmethod
    def _parse_completed(parts: Dict[int, bytes], cnt: int) -> Optional[Tuple[str, bytes]]:
        payload = b"".join(parts[i] for i in range(cnt))
        if len(payload) < 2:
            return None
        name_len = struct.unpack("!H", payload[:2])[0]
        if len(payload) < 2 + name_len:
            return None
        name = payload[2 : 2 + name_len].decode("utf-8", errors="replace")
        data = payload[2 + name_len :]
        name = os.path.basename(name) or f"audio_{int(time.time())}.bin"
        return name, data

    def _loop(self):
        last_stat = time.time()

        while self._running:
            now = time.time()

            # timeout inflight
            dead = [fid for fid, inf in self._inflight.items() if (now - inf.t0) > self.timeout_s]
            for fid in dead:
                self._inflight.pop(fid, None)
                self.rx_files_drop += 1

            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                data = None
            except OSError:
                break

            if data:
                if len(data) < HDR_SIZE:
                    continue
                magic, file_id, idx, cnt, payload_len, _rsv = struct.unpack(HDR_FMT, data[:HDR_SIZE])
                if magic != MAGIC:
                    continue

                frag = data[HDR_SIZE : HDR_SIZE + payload_len]
                self.rx_chunks += 1

                inf = self._inflight.get(file_id)
                if inf is None:
                    inf = InFlightFile(cnt=cnt, parts={}, t0=time.time())
                    self._inflight[file_id] = inf
                    self._drop_old_inflight_if_needed()

                if inf.cnt != cnt:
                    inf.cnt = cnt
                    inf.parts.clear()
                    inf.t0 = time.time()

                if idx not in inf.parts:
                    inf.parts[idx] = frag

                if self.debug and (self.rx_chunks % 200 == 0):
                    print(f"[UDP] chunks={self.rx_chunks} from={addr} file_id={file_id} idx={idx}/{cnt}")

                if len(inf.parts) == inf.cnt:
                    try:
                        parsed = self._parse_completed(inf.parts, inf.cnt)
                        if parsed is not None:
                            self.out_queue.put(parsed)
                            self.rx_files_ok += 1
                        else:
                            self.rx_files_drop += 1
                    finally:
                        self._inflight.pop(file_id, None)

            if (time.time() - last_stat) >= 2.0:
                print(f"[UDP] ok={self.rx_files_ok} drop={self.rx_files_drop} inflight={len(self._inflight)}")
                last_stat = time.time()


# ---------------------- Audio playback (Pi / recv mode) ----------------------

class AudioPlayer:
    """
    Plays audio files via external tools. Priority: ffplay -> mpg123 -> aplay
    """

    def __init__(self, work_dir: str = "/tmp/udp_audio", keep_files: bool = False):
        self.work_dir = work_dir
        self.keep_files = bool(keep_files)
        os.makedirs(self.work_dir, exist_ok=True)

        self._player = self._choose_player()
        if self._player is None:
            raise RuntimeError("No audio player found. Install ffmpeg (ffplay) or mpg123 or alsa-utils (aplay).")
        print(f"[AUDIO] Using player: {self._player[0]}")

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _choose_player():
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
        if shutil.which("mpg123"):
            return ["mpg123", "-q"]
        if shutil.which("aplay"):
            return ["aplay", "-q"]
        return None

    def stop(self):
        with self._lock:
            if self._proc and (self._proc.poll() is None):
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._proc = None

    def play_bytes(self, filename: str, data: bytes):
        safe_name = os.path.basename(filename)
        path = os.path.join(self.work_dir, f"{int(time.time()*1000)}_{safe_name}")
        with open(path, "wb") as f:
            f.write(data)

        with self._lock:
            if self._proc and (self._proc.poll() is None):
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            try:
                self._proc = subprocess.Popen(self._player + [path])
                print(f"[AUDIO] Playing: {path} ({len(data)} bytes)")
            except Exception as e:
                print(f"[AUDIO] Failed to start player: {e}")

        if not self.keep_files:
            threading.Thread(target=self._del_later, args=(path,), daemon=True).start()

    @staticmethod
    def _del_later(path: str, delay_s: float = 10.0):
        time.sleep(delay_s)
        try:
            os.remove(path)
        except Exception:
            pass


# ---------------------- Receiver App (Pi / recv mode) ----------------------

class UDPAudioBluetoothReceiverApp:
    def __init__(
        self,
        bt_mac: str,
        udp_listen_ip: str,
        udp_port: int,
        pair: bool,
        trust: bool,
        reconnect_every_s: float = 5.0,
        reassembly_timeout_s: float = 3.0,
        max_inflight: int = 2,
        rcvbuf_bytes: int = 8 << 20,
        keep_files: bool = False,
        debug_udp: bool = False,
    ):
        self.bt = BluetoothSpeakerManager(bt_mac, pair=pair, trust=trust)
        self.q: "queue.Queue[Tuple[str, bytes]]" = queue.Queue(maxsize=8)
        self.rx = UDPFileReceiver(
            listen_ip=udp_listen_ip,
            port=udp_port,
            out_queue=self.q,
            timeout_s=reassembly_timeout_s,
            max_inflight=max_inflight,
            rcvbuf_bytes=rcvbuf_bytes,
            debug=debug_udp,
        )
        self.player = AudioPlayer(keep_files=keep_files)

        self.reconnect_every_s = float(reconnect_every_s)
        self._running = False
        self._worker: Optional[threading.Thread] = None

    def start(self):
        self._running = True

        try:
            self.bt.ensure_connected()
        except Exception as e:
            print(f"[BT] initial connect failed: {e}")

        self.rx.start()

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()

        # threading.Thread(target=self._bt_reconnect_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self.rx.stop()
        self.player.stop()
        try:
            self.bt.disconnect()
        except Exception:
            pass

    def _bt_reconnect_loop(self):
        while self._running:
            try:
                self.bt.ensure_connected()
            except Exception as e:
                print(f"[BT] reconnect attempt failed: {e}")
            time.sleep(self.reconnect_every_s)

    def _playback_loop(self):
        while self._running:
            try:
                name, data = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            # latest-wins: drain queued files and play newest
            while True:
                try:
                    n2, d2 = self.q.get_nowait()
                    name, data = n2, d2
                except queue.Empty:
                    break

            self.player.play_bytes(name, data)

    def run_forever(self):
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[APP] KeyboardInterrupt")
        finally:
            self.stop()


# ---------------------- Sender (PC / send mode) ----------------------

class UDPAudioFileSender:
    def __init__(
        self,
        target_ip: str,
        target_port: int,
        chunk_payload_bytes: int = 1200,
        inter_packet_sleep_s: float = 0.0,
        ttl: int = 64,
        dscp: int = 0,   # 0..63 (optional QoS)
        bind_ip: str = "",
        sndbuf_bytes: int = 8 << 20,
        min_inter_file_gap_s: float = 0.0,
    ):
        self.target = (target_ip, int(target_port))
        self.chunk_payload_bytes = int(chunk_payload_bytes)
        if self.chunk_payload_bytes <= 0:
            raise ValueError("chunk_payload_bytes must be > 0")
        self.inter_packet_sleep_s = float(inter_packet_sleep_s)
        self.min_inter_file_gap_s = float(min_inter_file_gap_s)
        self._last_file_sent_t = 0.0
        self._send_lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if bind_ip:
            self.sock.bind((bind_ip, 0))

        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(sndbuf_bytes))
        except Exception:
            pass

        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, int(ttl))
        except OSError:
            pass
        if 0 <= dscp <= 63:
            try:
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, int(dscp) << 2)
            except OSError:
                pass

    @staticmethod
    def _build_chunks(filename: str, file_bytes: bytes, chunk_payload_bytes: int) -> List[bytes]:
        base = os.path.basename(filename).encode("utf-8", errors="replace")
        if len(base) > 65535:
            base = base[:65535]
        prefix = struct.pack("!H", len(base)) + base
        all_bytes = prefix + file_bytes

        chunks: List[bytes] = []
        for i in range(0, len(all_bytes), chunk_payload_bytes):
            chunks.append(all_bytes[i : i + chunk_payload_bytes])
        return chunks

    def send_bytes(
        self,
        filename: str,
        file_bytes: bytes,
        file_id: Optional[int] = None,
        repeat_chunk0: int = 4,
        auto_pace_big_files: bool = True,
    ):
        """
        Send in-memory bytes as a chunked UDP "file".
        - repeat_chunk0: sends the first chunk multiple times (contains filename prefix)
        - auto_pace_big_files: if payload is large, enable a small default inter-packet sleep
          to reduce receiver drops under load.
        """
        b = file_bytes
 

        if file_id is None:
            file_id = random.randint(1, 0xFFFFFFFF)

        chunks = self._build_chunks(os.path.basename(filename), b, self.chunk_payload_bytes)
        cnt = len(chunks)
        if cnt == 0:
            raise ValueError("File is empty (no chunks to send).")
        if cnt > 0xFFFF:
            raise ValueError(f"Too many chunks ({cnt}) for u16 fields. Increase --chunk-bytes.")

        # If we are sending large SOFA WAVs (hundreds of KB), a tiny pacing
        # greatly reduces drops.
        inter_sleep = self.inter_packet_sleep_s
        if auto_pace_big_files and inter_sleep <= 0.0 and len(b) >= 64 * 1024:
            inter_sleep = 0.0008  # ~0.8ms; tune 0.0003..0.002 if needed

        print(f"[SEND] {filename} -> {self.target[0]}:{self.target[1]}")
        print(f"[SEND] bytes={len(b)}  chunk_payload={self.chunk_payload_bytes}  chunks={cnt}  file_id={file_id}")

        # Send chunk0 multiple times to reduce chance of losing filename prefix.
        idx_order = [0] * max(1, int(repeat_chunk0)) + list(range(1, cnt))

        with self._send_lock:
            # Optional gap between whole files (prevents piling up many
            # inflight file_ids on receiver)
            if self.min_inter_file_gap_s > 0:
                dt_gap = time.time() - self._last_file_sent_t
                if dt_gap < self.min_inter_file_gap_s:
                    time.sleep(self.min_inter_file_gap_s - dt_gap)

            t0 = time.time()
            for idx in idx_order:
                payload = chunks[idx]
                hdr = struct.pack(HDR_FMT, MAGIC, int(file_id), int(idx),
                                  int(cnt), int(len(payload)), 0)
                self.sock.sendto(hdr + payload, self.target)
                if inter_sleep > 0:
                    time.sleep(inter_sleep)
            self._last_file_sent_t = time.time()

        dt = time.time() - t0
        pps = len(idx_order) / dt if dt > 0 else float("inf")
        print(f"[SEND] done in {dt:.3f}s  packets={len(idx_order)}  ~{pps:.1f} pkt/s")

    def send_file(self, path: str, file_id: Optional[int] = None, repeat_chunk0: int = 4):
        with open(path, "rb") as f:
            b = f.read()
        self.send_bytes(os.path.basename(path), b, file_id=file_id,
                        repeat_chunk0=repeat_chunk0)


# ---------------------- CLI ----------------------

def _cmd_recv(args: argparse.Namespace) -> int:
    app = UDPAudioBluetoothReceiverApp(
        bt_mac=args.bt_mac,
        udp_listen_ip=args.udp_ip,
        udp_port=args.udp_port,
        pair=args.pair,
        trust=not args.no_trust,
        reconnect_every_s=args.reconnect_every,
        reassembly_timeout_s=args.timeout,
        max_inflight=args.max_inflight,
        rcvbuf_bytes=args.rcvbuf,
        keep_files=args.keep_files,
        debug_udp=args.debug_udp,
    )
    app.run_forever()
    return 0


def _cmd_send(args: argparse.Namespace) -> int:
    sender = UDPAudioFileSender(
        target_ip=args.pi_ip,
        target_port=args.pi_port,
        chunk_payload_bytes=args.chunk_bytes,
        inter_packet_sleep_s=args.sleep,
        ttl=args.ttl,
        dscp=args.dscp,
        bind_ip=args.bind_ip,
        sndbuf_bytes=args.sndbuf,
        min_inter_file_gap_s=args.file_gap,
    )
    sender.send_file(args.file, repeat_chunk0=args.repeat_chunk0)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="bth_audio_manager.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # recv (Pi)
    ap_r = sub.add_parser("recv", help="Raspberry Pi: connect BT speaker, receive UDP audio files, play them")
    ap_r.add_argument("--bt-mac", required=True, help="Bluetooth speaker MAC "\
                      "address (AA:BB:CC:DD:EE:FF)", default="28:FA:19:0F:E9:CF")
    ap_r.add_argument("--udp-ip", default="0.0.0.0", help="UDP bind IP (default: 0.0.0.0)")
    ap_r.add_argument("--udp-port", type=int, default=40100, help="UDP port to receive audio files")
    ap_r.add_argument("--pair", action="store_true", help="Try to pair with the device (first-time setup)")
    ap_r.add_argument("--no-trust", action="store_true", help="Do not trust the device")
    ap_r.add_argument("--reconnect-every", type=float, default=5.0, help="Seconds between BT reconnect attempts")
    ap_r.add_argument("--timeout", type=float, default=3.0, help="""Seconds
                      to wait for missing UDP chunks before dropping a
                      file""")
    ap_r.add_argument("--max-inflight", type=int, default=2, help="""Max
                      concurrent incomplete files to track (default: 2)""")
    ap_r.add_argument("--rcvbuf", type=int, default=8 << 20,
                       help="UDP receive socket buffer bytes (default: 8MiB)")
    ap_r.add_argument("--keep-files", action="store_true", help="Keep received files on disk (debug)")
    ap_r.add_argument("--debug-udp", action="store_true", help="Extra UDP debug prints")
    ap_r.set_defaults(func=_cmd_recv)

    # send (PC)
    ap_s = sub.add_parser("send", help="PC: chunk + send an audio file over UDP to the Pi receiver")
    ap_s.add_argument("--pi-ip", required=True, help="Raspberry Pi IP (receiver)")
    ap_s.add_argument("--pi-port", type=int, default=40100, help="Receiver UDP port")
    ap_s.add_argument("--file", required=True, help="Path to audio file (wav/mp3/ogg etc.)")
    ap_s.add_argument("--chunk-bytes", type=int, default=1200, help="Payload bytes per UDP packet (safe for MTU1500)")
    ap_s.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between packets (try 0.0005 if drops)")
    ap_s.add_argument("--repeat-chunk0", type=int, default=4, help="""How many
                      times to send chunk0 (filename prefix) (default: 4)""")
    ap_s.add_argument("--ttl", type=int, default=64)
    ap_s.add_argument("--dscp", type=int, default=0, help="DSCP 0..63 (QoS), optional")
    ap_s.add_argument("--bind-ip", default="", help="Optional local bind IP")
    ap_s.add_argument("--sndbuf", type=int, default=8 << 20, help="""UDP send
                      socket buffer bytes (default: 8MiB)""")
    ap_s.add_argument("--file-gap", type=float, default=0.0, help="""Minimum
                      seconds between whole files (rate limit). Try 0.15""")
    ap_s.set_defaults(func=_cmd_send)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

