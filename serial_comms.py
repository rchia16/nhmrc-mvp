#!/usr/bin/env python3

import time
import serial


SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 1_000_000

# ============================================================
# Change this between 0.5 and 2.0 seconds
# ============================================================

PACKET_PERIOD_S = 1.0


def stream_packets(
    port=SERIAL_PORT,
    baud=SERIAL_BAUD,
    packet_period_s=PACKET_PERIOD_S,
):
    if not 0.5 <= packet_period_s <= 2.0:
        raise ValueError(
            "packet_period_s must be between 0.5 and 2.0"
        )

    ser = serial.Serial(
        port,
        baudrate=baud,
        timeout=0.05,
    )

    # Arduino commonly resets when USB serial opens.
    time.sleep(2.0)

    packet = {
        "imu": [],
        "ppg": [],
        "mag": [],
    }

    packet_start = time.monotonic()

    while True:

        raw = ser.readline()

        if raw:
            try:
                line = raw.decode(
                    "ascii",
                    errors="strict"
                ).strip()

            except UnicodeDecodeError:
                continue

            if not line:
                continue

            # Diagnostic line from Arduino
            if line.startswith("#"):
                print(line)
                continue

            fields = line.split(",")

            try:

                # ------------------------------------------
                # IMU
                # ------------------------------------------
                if fields[0] == "I" and len(fields) == 10:

                    packet["imu"].append({
                        "t_us": int(fields[1]),

                        "accel_seq": int(fields[2]),
                        "gyro_seq": int(fields[3]),

                        "ax": float(fields[4]),
                        "ay": float(fields[5]),
                        "az": float(fields[6]),

                        "gx": float(fields[7]),
                        "gy": float(fields[8]),
                        "gz": float(fields[9]),
                    })


                # ------------------------------------------
                # PPG
                # ------------------------------------------
                elif fields[0] == "P" and len(fields) == 5:

                    packet["ppg"].append({
                        "t_us": int(fields[1]),
                        "seq": int(fields[2]),
                        "red": int(fields[3]),
                        "ir": int(fields[4]),
                    })


                # ------------------------------------------
                # Magnetometer
                # ------------------------------------------
                elif fields[0] == "M" and len(fields) == 6:

                    packet["mag"].append({
                        "t_us": int(fields[1]),
                        "seq": int(fields[2]),

                        "mx": float(fields[3]),
                        "my": float(fields[4]),
                        "mz": float(fields[5]),
                    })

            except (ValueError, IndexError):
                # Ignore incomplete/corrupted serial line
                pass


        # ==================================================
        # Yield one complete time packet
        # ==================================================

        now = time.monotonic()

        if now - packet_start >= packet_period_s:

            packet["packet_period_s"] = packet_period_s

            packet["host_time"] = time.time()

            yield packet

            packet = {
                "imu": [],
                "ppg": [],
                "mag": [],
            }

            packet_start = now


if __name__ == "__main__":

    for packet in stream_packets(
        packet_period_s=PACKET_PERIOD_S
    ):

        print(
            "packet:",
            f"IMU={len(packet['imu'])}",
            f"PPG={len(packet['ppg'])}",
            f"MAG={len(packet['mag'])}",
        )

        # `packet` is now your downstream streaming variable.
        #
        # Examples:
        #
        # send_to_model(packet)
        # socket.send(packet)
        # queue.put(packet)
        # save_packet(packet)
