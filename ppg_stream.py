import argparse
import socket
import struct
import threading
import queue
from collections import deque
import os
import time
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import max30102
from RPi import GPIO

class RateMonitor:
    """
    Counts samples and periodically prints effective sample rate.
    """
    def __init__(self, interval_sec: float = 2.0, enabled: bool = False):
        self.interval = float(interval_sec)
        self.enabled = enabled
        self._count = 0
        self._t0 = time.time()

    def add(self, n: int):
        if not self.enabled:
            return

        self._count += int(n)
        now = time.time()
        dt = now - self._t0

        if dt >= self.interval:
            rate = self._count / dt
            print(f"[RATE] {rate:.1f} Hz ({self._count} samples / {dt:.2f} s)")
            self._count = 0
            self._t0 = now


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

class SampleSpooler:
    PACK_FMT = "!dii"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    def __init__(
        self,
        log_dir: str,
        rotate_every_seconds: int = 3600,
        forward_sender=None,
        fsync_every: int = 0,
    ):
        self.log_dir = log_dir
        self.rotate_every = timedelta(seconds=rotate_every_seconds)
        self.forward_sender = forward_sender
        self.fsync_every = fsync_every

        self.q = queue.SimpleQueue()
        self._running = False
        self._thread = None
        self._fh = None
        self._count = 0
        self._next_rotate = None

    def _open_new_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.log_dir, f"ppg_{ts}.bin")
        self._fh = open(path, "ab", buffering=0)
        self._next_rotate = datetime.now() + self.rotate_every
        print(f"[LOG] New file: {path}")

    def start(self):
        self._open_new_file()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.q.put(None)
        if self._thread:
            self._thread.join(timeout=2)
        if self._fh:
            self._fh.close()

    def enqueue(self, sample):
        self.q.put(sample)

    def _run(self):
        while self._running:
            item = self.q.get()
            if item is None:
                continue

            if datetime.now() >= self._next_rotate:
                self._fh.close()
                self._open_new_file()

            ts, red, ir = item
            self._fh.write(struct.pack(self.PACK_FMT, ts, red, ir))
            self._count += 1

            if self.forward_sender:
                self.forward_sender.enqueue(ts, red, ir)



class TCPPPGSender:
    """
    TCP sender with auto-reconnect.
    Sends packed records !dii (16 bytes each).
    """
    PACK_FMT = "!dii"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    def __init__(self, host: str, port: int, reconnect_sec: float = 2.0):
        self.host = host
        self.port = int(port)
        self.reconnect_sec = float(reconnect_sec)

        self._q = queue.SimpleQueue()
        self._running = False
        self._thread = None
        self._sock = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._q.put(None)
        self._close_sock()

    def enqueue(self, ts: float, red: int, ir: int):
        if not self._running:
            return
        self._q.put((float(ts), int(red), int(ir)))

    def _close_sock(self):
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def _connect(self):
        self._close_sock()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.host, self.port))
        s.settimeout(None)
        self._sock = s
        print(f"[TCP] Connected -> {self.host}:{self.port}")

    def _run(self):
        while self._running:
            # Ensure connection
            if self._sock is None:
                try:
                    self._connect()
                except Exception as e:
                    print(f"[TCP] Connect failed: {e}. Retrying in {self.reconnect_sec}s")
                    time.sleep(self.reconnect_sec)
                    continue

            item = self._q.get()
            if item is None:
                continue

            ts, red, ir = item
            pkt = struct.pack(self.PACK_FMT, ts, red, ir)

            try:
                # sendall guarantees full write or raises
                self._sock.sendall(pkt)
            except Exception as e:
                print(f"[TCP] Send failed: {e}. Reconnecting...")
                self._close_sock()
                # loop will reconnect and continue



class MAX30102PPGStream:
    def __init__(self, buffer: RingBuffer, spooler: SampleSpooler,
                 rate_monitor: RateMonitor, force_poll: bool = False,
                 no_data_timeout: float = 5.0, poll_sleep_ms: float=5.0):
        self.buffer = buffer
        self.spooler = spooler
        self.rate_monitor = rate_monitor
        self.force_poll = force_poll
        self.no_data_timeout = float(no_data_timeout)
        self.poll_sleep_ms = poll_sleep_ms

        self.m = max30102.MAX30102()
        self.m.setup(
            led_mode=0x03,
            sample_rate=200,
            pulse_width=118,
            adc_range=4096,
            fifo_average=1,
            fifo_rollover=False,
            fifo_a_full=15,
        )

        self._running = False
        self._poll_thread = None
        self._last_sample_time = time.time()
        # self._drain_event = threading.Event()
        # self._drain_thread = None


    def start(self):
        if self._running:
            return
        self._running = True

        if self.force_poll:
            print("[PPG] Using FIFO polling mode (force)")
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
        else:
            # Start a FIFO drain worker
            self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
            self._drain_thread.start()

            # Trigger on BOTH to avoid missing transitions
            GPIO.add_event_detect(self.m.interrupt, GPIO.BOTH, callback=self._irq_callback, bouncetime=1)

            # watchdog stays
            self._poll_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._poll_thread.start()

    def _irq_callback(self, channel):
        # Keep GPIO callback ultra-fast: just wake the drain thread
        self._drain_event.set()

    def _drain_loop(self):
        while self._running:
            # Wait until an interrupt occurs (or wake periodically)
            self._drain_event.wait(timeout=0.5)
            self._drain_event.clear()

            try:
                # Drain until FIFO empty
                while True:
                    batch = self.m.i2c_thread_func(max_batch=32, require_ppg_rdy=False)
                    if not batch:
                        break
                    self._handle_batch(batch)
            except Exception as e:
                print(f"[DRAIN] ERROR: {e}")


    def stop(self):
        if not self._running:
            return
        if not self.force_poll:
            GPIO.remove_event_detect(self.m.interrupt)
        self._drain_event.set()
        self._running = False


    def _handle_batch(self, batch):
        if not batch:
            return
        self._last_sample_time = time.time()
        self.rate_monitor.add(len(batch))
        for sample in batch:
            self.buffer.append(sample)
            self.spooler.enqueue(sample)

    def _gpio_callback(self, channel):
        try:
            batch = self.m.i2c_thread_func(max_batch=32, require_ppg_rdy=False)
            self._handle_batch(batch)
        except Exception as e:
            print(f"[GPIO_CB] ERROR: {e}")

    def _poll_loop(self):
        # Poll FIFO regardless of interrupts. Great for debugging.
        while self._running:
            try:
                batch = self.m.i2c_thread_func(max_batch=32, require_ppg_rdy=False)
                self._handle_batch(batch)
            except Exception as e:
                print(f"[POLL] ERROR: {e}")
            time.sleep(self.poll_sleep_ms/1000.0) # adjust if you push SR very high

    def _watchdog_loop(self):
        # If no samples arrive for N seconds, print useful diagnostics
        while self._running:
            if (time.time() - self._last_sample_time) > self.no_data_timeout:
                try:
                    level = GPIO.input(self.m.interrupt)
                    n = self.m.get_data_present()
                    regs = self.m.dump_regs()
                    print(f"[PPG][NO DATA] INT level={level} FIFO_samples={n} regs={regs}")
                except Exception as e:
                    print(f"[PPG][NO DATA] diagnostics failed: {e}")
                self._last_sample_time = time.time()
            time.sleep(0.2)


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
        self.ax.set_title("MAX30102 Realtime PPG (local) + TCP stream")
        self.ax.set_xlabel("Sample index")
        self.ax.set_ylabel("PPG value")

    def show(self, interval_ms: int = 50):
        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


class App:
    def __init__(
        self,
        send_ip,
        send_port,
        plot_window,
        no_plot,
        log_dir,
        rotate_every_seconds,
        fsync_every,
        rate_print,
        force_poll,
        no_data_timeout,
        poll_sleep_ms,
    ):
        self.no_plot = no_plot

        self.rate_monitor = RateMonitor(
            interval_sec=2.0,
            enabled=rate_print
        )

        self.buffer = RingBuffer(maxlen=60 * 60 * 60)

        self.sender = TCPPPGSender(send_ip, send_port) if send_ip else None


        self.spooler = SampleSpooler(
            log_dir=log_dir,
            rotate_every_seconds=rotate_every_seconds,
            forward_sender=self.sender,
            fsync_every=fsync_every
        )

        self.stream = MAX30102PPGStream(
            self.buffer,
            spooler=self.spooler,
            rate_monitor=self.rate_monitor,
            force_poll=force_poll,
            no_data_timeout=no_data_timeout,
            poll_sleep_ms=poll_sleep_ms,
        )


        self.plotter = None if no_plot else RealtimePlotter(
            plot_window, drain_fn=self.buffer.drain)


    def run(self):
        try:
            if self.sender:
                self.sender.start()
                print(f"[TCP] Streaming enabled -> {self.sender.host}:{self.sender.port}")

            self.spooler.start()
            print(f"[LOG] Writing lossless log to {self.spooler.log_dir}")

            self.stream.start()

            if self.no_plot:
                print("[INFO] Running headless (no plot)")
                while True:
                    time.sleep(1)
            else:
                self.plotter.show(interval_ms=50)

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")

        finally:
            self.stream.stop()
            GPIO.cleanup()
            self.spooler.stop()
            if self.sender:
                self.sender.stop()
            print("Exiting")



def main():
    # python3 ppg_stream.py \
    #   --no-plot \
    #   --rate-print \
    #   --send-ip 192.168.0.194 \
    #   --send-port 9999


    p = argparse.ArgumentParser()
    p.add_argument("--send-ip", type=str, default=None,
                   help="Receiver IP on WiFi (e.g., your laptop IP)."\
                   "If omitted, no TCP streaming.")
    p.add_argument("--send-port", type=int, default=9999,
                   help="Receiver TCP port.")
    p.add_argument("--plot-window", type=int, default=2000,
                   help="Samples to show in local plot.")
    p.add_argument("--no-plot", action="store_true", help="Run headless (no matplotlib, no GUI)")
    p.add_argument("--log-dir", type=str, default="logs/",
                   help="Binary append-only log on the Pi (lossless).")
    p.add_argument("--rotate-seconds", type=int, default=3600,
                   help="Duration to create new log file in seconds.")
    p.add_argument("--fsync-every", type=int, default=0,
                   help="Call fsync every N samples (0 = never; faster but less crash-durable).")
    p.add_argument("--rate-print",
                   action="store_true",
                   help="Print effective sampling rate every ~2 seconds"
                  )
    p.add_argument("--force-poll", action="store_true",
               help="Ignore GPIO interrupts and poll FIFO in a loop (debug / headless robustness).")
    p.add_argument("--no-data-timeout", type=float, default=5.0,
                   help="Print diagnostics if no samples arrive for this many seconds.")
    p.add_argument("--poll-sleep-ms", type=float, default=5.0,
               help="Polling sleep interval in ms (only used with --force-poll).")



    args = p.parse_args()

    App(
        args.send_ip,
        args.send_port,
        args.plot_window,
        args.no_plot,
        args.log_dir,
        args.rotate_seconds,
        args.fsync_every,
        args.rate_print,
        args.force_poll,
        args.no_data_timeout,
        args.poll_sleep_ms,
    ).run()



if __name__ == "__main__":
    main()

