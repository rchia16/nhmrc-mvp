import argparse
import socket
import struct
import threading
import queue
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import max30102
from RPi import GPIO


class RingBuffer:
    """Thread-safe ring buffer for (ts, red, ir) samples."""
    def __init__(self, maxlen: int):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item):
        with self._lock:
            self._buf.append(item)

    def drain(self):
        """Return all items and clear."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
        return items

    def snapshot(self):
        with self._lock:
            return list(self._buf)


class UDPPPGSender:
    """
    UDP sender with a background worker thread.
    Packet format: !dii (timestamp float64, red int32, ir int32)
    """
    PACK_FMT = "!dii"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    def __init__(self, host: str, port: int, max_queue: int = 4096):
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._q = queue.Queue(maxsize=max_queue)
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def enqueue(self, ts: float, red: int, ir: int):
        if not self._running:
            return
        try:
            self._q.put_nowait((float(ts), int(red), int(ir)))
        except queue.Full:
            # Drop rather than block acquisition
            pass

    def _run(self):
        while self._running:
            item = self._q.get()
            if item is None:
                continue
            ts, red, ir = item
            try:
                pkt = struct.pack(self.PACK_FMT, ts, red, ir)
                self._sock.sendto(pkt, (self.host, self.port))
            except Exception:
                # keep going on transient network errors
                pass


class MAX30102PPGStream:
    """Owns the MAX30102 device and GPIO interrupt callback."""
    def __init__(self, buffer: RingBuffer, udp_sender: UDPPPGSender | None = None):
        self.buffer = buffer
        self.udp_sender = udp_sender
        self.m = max30102.MAX30102()
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        GPIO.add_event_detect(self.m.interrupt, GPIO.FALLING, callback=self._gpio_callback)

    def stop(self):
        if not self._running:
            return
        GPIO.remove_event_detect(self.m.interrupt)
        self._running = False

    def _gpio_callback(self, channel):
        """
        Keep callback FAST: read i2c, append, enqueue for UDP, return.
        """
        data = self.m.i2c_thread_func()
        if data is None:
            return
        # data = (timestamp, red, ir)
        self.buffer.append(data)
        if self.udp_sender is not None:
            ts, red, ir = data
            self.udp_sender.enqueue(ts, red, ir)


class RealtimePlotter:
    """Matplotlib animation that plots last N samples from drained batches."""
    def __init__(self, sample_window: int, drain_fn):
        self.sample_window = int(sample_window)
        self.drain_fn = drain_fn

        self.red = np.full(self.sample_window, -1, dtype=np.int32)
        self.ir = np.full(self.sample_window, -1, dtype=np.int32)

        self.fig, self.ax = plt.subplots()

    @staticmethod
    def _slide_append(arr: np.ndarray, new: np.ndarray):
        n = len(new)
        if n <= 0:
            return
        if n >= len(arr):
            arr[:] = new[-len(arr):]
            return
        arr[:-n] = arr[n:]
        arr[-n:] = new

    def _animate(self, _frame):
        batch = self.drain_fn()
        if batch:
            _, red, ir = zip(*batch)
            self._slide_append(self.red, np.asarray(red, dtype=np.int32))
            self._slide_append(self.ir, np.asarray(ir, dtype=np.int32))

        self.ax.cla()
        x = np.arange(self.sample_window)
        self.ax.plot(x, self.red, "r-")
        self.ax.plot(x, self.ir, "b-")
        self.ax.set_title("MAX30102 Realtime PPG (local) + UDP stream")
        self.ax.set_xlabel("Sample index")
        self.ax.set_ylabel("PPG value")

    def show(self, interval_ms: int = 50):
        animation.FuncAnimation(self.fig, self._animate, interval=interval_ms, blit=False)
        plt.show()


class App:
    def __init__(self, send_ip: str | None, send_port: int, plot_window: int):
        self.buffer = RingBuffer(maxlen=60 * 60 * 60)  # ~1 hour @ 60Hz, adjust as needed
        self.sender = UDPPPGSender(send_ip, send_port) if send_ip else None
        self.stream = MAX30102PPGStream(self.buffer, udp_sender=self.sender)
        self.plotter = RealtimePlotter(plot_window, drain_fn=self.buffer.drain)

    def run(self):
        try:
            if self.sender:
                self.sender.start()
                print(f"[UDP] Streaming enabled -> {self.sender.host}:{self.sender.port}")
            self.stream.start()
            self.plotter.show(interval_ms=50)
        finally:
            self.stream.stop()
            GPIO.cleanup()
            if self.sender:
                self.sender.stop()
            print("Exiting")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--send-ip", type=str, default=None,
                   help="Receiver IP on WiFi (e.g., your laptop IP). If omitted, no UDP streaming.")
    p.add_argument("--send-port", type=int, default=9999, help="Receiver UDP port.")
    p.add_argument("--plot-window", type=int, default=2000, help="Samples to show in local plot.")
    args = p.parse_args()

    App(args.send_ip, args.send_port, args.plot_window).run()


if __name__ == "__main__":
    main()

