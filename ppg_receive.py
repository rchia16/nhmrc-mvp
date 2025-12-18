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


class UDPReceiver:
    """Non-blocking UDP receiver."""
    def __init__(self, listen_ip: str, port: int):
        self.listen_ip = listen_ip
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.port))
        self.sock.setblocking(False)

    def poll(self, max_reads: int = 2000):
        """Drain up to max_reads packets; yields (ts, red, ir, addr_ip)."""
        out = []
        for _ in range(max_reads):
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            decoded = PPGPacketCodec.decode(data)
            if decoded is None:
                continue
            ts, red, ir = decoded
            out.append((ts, red, ir, addr[0]))
        return out


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
        animation.FuncAnimation(self.fig, self._animate, interval=interval_ms, blit=False)
        plt.show()


class ReceiverApp:
    def __init__(self, listen_ip: str, port: int, window_size: int, plot: bool):
        self.receiver = UDPReceiver(listen_ip, port)
        self.window = PPGWindowBuffer(window_size)
        self.plot_enabled = plot
        self.printer = PrinterSink(every_seconds=0.2)
        self.plotter = PlotSink(self.window) if plot else None

    def _ingest_loop(self):
        """Run until plot window closes (or forever in print mode)."""
        while True:
            packets = self.receiver.poll(max_reads=2000)
            for ts, red, ir, src_ip in packets:
                self.window.append(ts, red, ir)
                if not self.plot_enabled:
                    self.printer.consume(ts, red, ir, src_ip)

            # light sleep to avoid pegging a core
            time.sleep(0.001)

    def run(self):
        print(f"[UDP] Listening on {self.receiver.listen_ip}:{self.receiver.port}")
        if self.plot_enabled:
            # ingest in background while matplotlib owns the main thread
            t = threading.Thread(target=self._ingest_loop, daemon=True)
            t.start()
            self.plotter.show(interval_ms=50)
        else:
            self._ingest_loop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen-ip", type=str, default="0.0.0.0",
                   help="IP to bind to (0.0.0.0 listens on all interfaces).")
    p.add_argument("--port", type=int, default=9999, help="UDP port to listen on.")
    p.add_argument("--plot", action="store_true", help="Show live plot.")
    p.add_argument("--window", type=int, default=2000, help="Plot window size (samples).")
    args = p.parse_args()

    ReceiverApp(args.listen_ip, args.port, args.window, args.plot).run()


if __name__ == "__main__":
    main()

