import argparse
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PPGPacketCodec:
    PACK_FMT = "!dii"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    @classmethod
    def decode(cls, data: bytes):
        if len(data) < cls.PACK_SIZE:
            return None
        return struct.unpack(cls.PACK_FMT, data[:cls.PACK_SIZE])  # (ts, red, ir)


class PPGWindowBuffer:
    """Thread-safe sliding window for plotting."""
    def __init__(self, maxlen: int):
        self.ts = deque(maxlen=maxlen)
        self.red = deque(maxlen=maxlen)
        self.ir = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, ts: float, red: int, ir: int):
        with self._lock:
            self.ts.append(ts)
            self.red.append(red)
            self.ir.append(ir)

    def snapshot(self):
        with self._lock:
            return (list(self.ts), list(self.red), list(self.ir))


class TCPServerReceiver:
    """
    TCP server that accepts a single sender at a time.
    If the sender disconnects, it accepts a new connection (reconnect friendly).
    """
    PACK_SIZE = PPGPacketCodec.PACK_SIZE

    def __init__(self, listen_ip: str, port: int):
        self.listen_ip = listen_ip
        self.port = int(port)

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.listen_ip, self.port))
        self._srv.listen(1)
        self._srv.settimeout(1.0)

        self._client = None
        self._client_addr = None
        self._buf = bytearray()

    def _close_client(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = None
        self._client_addr = None
        self._buf.clear()

    def _accept_if_needed(self):
        if self._client is not None:
            return
        try:
            c, addr = self._srv.accept()
        except socket.timeout:
            return
        c.settimeout(1.0)
        self._client = c
        self._client_addr = addr
        print(f"[TCP] Client connected: {addr[0]}:{addr[1]}")

    def poll(self, max_records: int = 2000):
        """
        Return up to max_records decoded samples.
        Each sample: (ts, red, ir, src_ip)
        """
        out = []
        self._accept_if_needed()
        if self._client is None:
            return out

        try:
            chunk = self._client.recv(4096)
            if not chunk:
                # disconnect
                print("[TCP] Client disconnected")
                self._close_client()
                return out
            self._buf.extend(chunk)
        except socket.timeout:
            return out
        except Exception:
            self._close_client()
            return out

        # parse complete records
        while len(out) < max_records and len(self._buf) >= self.PACK_SIZE:
            rec = bytes(self._buf[:self.PACK_SIZE])
            del self._buf[:self.PACK_SIZE]
            decoded = PPGPacketCodec.decode(rec)
            if decoded is None:
                continue
            ts, red, ir = decoded
            out.append((ts, red, ir, self._client_addr[0]))

        return out


class RateMonitor:
    def __init__(self, interval_sec: float = 2.0):
        self.interval = float(interval_sec)
        self._count = 0
        self._t0 = time.time()

    def add(self, n: int):
        self._count += int(n)
        now = time.time()
        dt = now - self._t0
        if dt >= self.interval:
            print(f"[RX RATE] {self._count/dt:.1f} Hz ({self._count} samples / {dt:.2f} s)")
            self._count = 0
            self._t0 = now


class PrinterSink:
    """Throttled console printing."""
    def __init__(self, every_seconds: float = 0.2):
        self.every_seconds = float(every_seconds)
        self._last = 0.0

    def consume(self, ts: float, red: int, ir: int, src_ip: str):
        now = time.time()
        if now - self._last >= self.every_seconds:
            print(f"from {src_ip}  t={ts:.3f}  red={red}  ir={ir}")
            self._last = now


class PlotSink:
    """Matplotlib plot driven by a shared window buffer."""
    def __init__(self, window: PPGWindowBuffer):
        self.window = window
        self.fig, self.ax = plt.subplots()
        self.red_line, = self.ax.plot([], [], "r-")
        self.ir_line, = self.ax.plot([], [], "b-")
        self.ax.set_title("PPG Receiver (UDP)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("PPG value")

    def _animate(self, _frame):
        _ts, red, ir = self.window.snapshot()
        if len(red) < 2:
            return self.red_line, self.ir_line

        x = np.arange(len(red))
        red_arr = np.asarray(red, dtype=np.int32)
        ir_arr = np.asarray(ir, dtype=np.int32)

        self.red_line.set_data(x, red_arr)
        self.ir_line.set_data(x, ir_arr)

        self.ax.set_xlim(0, len(red))
        ymin = int(min(red_arr.min(), ir_arr.min()))
        ymax = int(max(red_arr.max(), ir_arr.max()))
        if ymin == ymax:
            ymax = ymin + 1
        self.ax.set_ylim(ymin, ymax)

        return self.red_line, self.ir_line

    def show(self, interval_ms: int = 50):
        self.anim = animation.FuncAnimation(
            self.fig, self._animate, interval=interval_ms, blit=False)
        plt.show()



class ReceiverApp:
    def __init__(self, listen_ip: str, port: int, window_size: int, plot: bool,
                 print_every: float):
        self.receiver = TCPServerReceiver(listen_ip, port)
        self.window = PPGWindowBuffer(window_size)
        self.plot_enabled = plot
        self.printer = PrinterSink(every_seconds=print_every)
        self.plotter = PlotSink(self.window) if plot else None
        self._stop = threading.Event()
        self.rx_rate = RateMonitor(interval_sec=2.0)


    def _ingest_loop(self):
        while not self._stop.is_set():
            packets = self.receiver.poll(max_records=2000)
            self.rx_rate.add(len(packets))
            for ts, red, ir, src_ip in packets:
                self.window.append(ts, red, ir)
                if not self.plot_enabled:
                    self.printer.consume(ts, red, ir, src_ip)
            time.sleep(0.001)

    def run(self):
        print(f"[TCP] Listening on {self.receiver.listen_ip}:{self.receiver.port}")
        if self.plot_enabled:
            t = threading.Thread(target=self._ingest_loop, daemon=True)
            t.start()
            try:
                self.plotter.show(interval_ms=50)
            finally:
                self._stop.set()
        else:
            self._ingest_loop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen-ip", type=str, default="0.0.0.0",
                   help="IP to bind to (0.0.0.0 listens on all interfaces).")
    p.add_argument("--port", type=int, default=9999, help="UDP port to listen on.")
    p.add_argument("--plot", action="store_true", help="Show live plot.")
    p.add_argument("--window", type=int, default=2000, help="Plot window size (samples).")
    p.add_argument("--print-every", type=float, default=0.2,
                   help="Seconds between printed samples (0.2 -> ~5 lines/sec).")
    args = p.parse_args()

    ReceiverApp(args.listen_ip, args.port, args.window, args.plot,
                args.print_every).run()


if __name__ == "__main__":
    main()

