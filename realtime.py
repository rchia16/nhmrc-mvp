import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque
from functools import partial

import threading
import max30102
import hrcalc

from RPi import GPIO

# How many samples to keep in memory
BUFFER_SIZE = int(1 * 60 * 60)  # 1 min * 60 sec * 60 Hz

# Global plotting buffers (optional; for live plots)
red_buffer = np.full(BUFFER_SIZE, -1, dtype=np.int32)
ir_buffer = np.full(BUFFER_SIZE, -1, dtype=np.int32)

fig, ax = plt.subplots()
ir_line, = ax.plot([], [], 'b-')
red_line, = ax.plot([], [], 'r-')


def update_buffer(buffer: np.ndarray, new_data: np.ndarray):
    """
    Slide data in 'buffer' to the left and append 'new_data' at the end.
    Newest samples end up at the highest indices.
    """
    N = len(new_data)
    if N == 0:
        return buffer
    if N >= len(buffer):
        buffer[:] = new_data[-len(buffer):]
        return buffer

    buffer[:-N] = buffer[N:]
    buffer[-N:] = new_data
    return buffer


class PPGStream:
    """
    Owns the MAX30102 device, the data buffer, and the GPIO interrupt callback.

    Public methods:
        start()       – begin listening on the interrupt pin
        stop()        – stop listening
        get_all()     – get all buffered samples (optionally clear)
        get_latest(n) – peek at last n samples

    Each sample is a tuple: (timestamp, red, ir)
    """

    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.m = max30102.MAX30102()
        self.running = False

    # ---------- internal helpers ----------

    def _add_sample(self, sample):
        with self.lock:
            self.buffer.append(sample)

    def _gpio_callback(self, channel):
        """
        Callback invoked by RPi.GPIO when a falling edge occurs on the interrupt pin.
        """
        data = self.m.i2c_thread_func()
        if data is not None:
            # i2c_thread_func returns (timestamp, red, ir)
            self._add_sample(data)

    # ---------- public API ----------

    def start(self):
        """
        Enable edge detection on the sensor's interrupt line.
        """
        if self.running:
            return
        self.running = True
        GPIO.add_event_detect(
            self.m.interrupt,
            GPIO.FALLING,
            callback=self._gpio_callback
        )

    def stop(self):
        """
        Disable edge detection and stop streaming.
        """
        if not self.running:
            return
        GPIO.remove_event_detect(self.m.interrupt)
        self.running = False

    def get_all(self, clear=True):
        """
        Return all samples currently in the buffer.

        Parameters
        ----------
        clear : bool
            If True, clear the buffer after reading.

        Returns
        -------
        list[(timestamp, red, ir)]
        """
        with self.lock:
            items = list(self.buffer)
            if clear:
                self.buffer.clear()
        return items

    def get_latest(self, n):
        """
        Return the last n samples without clearing.

        Returns
        -------
        list[(timestamp, red, ir)]
        """
        with self.lock:
            return list(self.buffer)[-n:]


# ---------- plotting / processing ----------

def animate(frame_idx, stream: PPGStream):
    """
    Matplotlib animation callback.

    Pulls new data from the stream, updates buffers, and redraws.
    """
    batch = stream.get_all(clear=True)
    if batch:
        # batch is a list of (timestamp, red, ir)
        _, red, ir = zip(*batch)
        red_arr = np.asarray(red, dtype=np.int32)
        ir_arr = np.asarray(ir, dtype=np.int32)

        update_buffer(red_buffer, red_arr)
        update_buffer(ir_buffer, ir_arr)

        # Example: compute HR/SpO2 over the most recent window
        # You can adapt this as needed:
        # hr, spo2 = hrcalc.calc_hr_and_spo2(ir_arr, red_arr)
        # print("HR:", hr, "SpO2:", spo2)

    ax.cla()
    ax.plot(np.arange(BUFFER_SIZE), red_buffer, 'r-')
    ax.plot(np.arange(BUFFER_SIZE), ir_buffer, 'b-')
    ax.set_xlabel("Sample index")
    ax.set_ylabel("PPG value")
    ax.set_title("MAX30102 Realtime PPG")
    return ir_line, red_line


def main():
    stream = PPGStream()
    stream.start()

    # Set up animation (50 ms between frames)
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(stream,),
        interval=50,
        blit=False
    )

    try:
        plt.show()
    finally:
        # Make sure we always clean up GPIO
        stream.stop()
        GPIO.cleanup()
        print("Exiting")


if __name__ == '__main__':
    main()
