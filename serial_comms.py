import argparse
import struct
import time

import serial
import serial.tools.list_ports


# ============================================================
# PROTOCOL V2
# ============================================================

MAGIC = b"\xA5\x5A"

PROTOCOL_VERSION = 2

PKT_IMU = 1
PKT_STATS = 2

IMU_ACCEL = 1
IMU_GYRO = 2
IMU_QUAT = 3


# ------------------------------------------------------------
# Packet structure
#
# uint8   magic0
# uint8   magic1
# uint8   version
# uint8   packet_type
#
# uint32  packet_sequence
#
# uint64  sensor_timestamp_us
# uint32  host_timestamp_us
#
# uint8   sensor_sequence
# uint8   sensor_type
# uint8   sensor_status
# uint8   n_values
#
# 16-byte payload
#
# uint16 CRC
# ------------------------------------------------------------

HEADER_FMT = "<BBBBIQIBBBB"

HEADER_SIZE = struct.calcsize(
    HEADER_FMT
)

PAYLOAD_SIZE = 16

CRC_SIZE = 2

PACKET_SIZE = (
    HEADER_SIZE
    + PAYLOAD_SIZE
    + CRC_SIZE
)

assert PACKET_SIZE == 42


# ============================================================
# CRC16-CCITT
# ============================================================

def crc16_ccitt(data: bytes) -> int:

    crc = 0xFFFF

    for byte in data:

        crc ^= byte << 8

        for _ in range(8):

            if crc & 0x8000:

                crc = (
                    (crc << 1) ^ 0x1021
                ) & 0xFFFF

            else:

                crc = (
                    crc << 1
                ) & 0xFFFF

    return crc


# ============================================================
# SERIAL PORT
# ============================================================

def list_ports():

    print("Available ports:")

    for p in (
        serial.tools.list_ports.comports()
    ):

        print(
            f"  {p.device:8s} "
            f"{p.description}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="COM3",
        help="Serial port, e.g. COM3",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=500000,
    )

    parser.add_argument(
        "--print-imu",
        action="store_true",
        help="Print every IMU packet",
    )

    args = parser.parse_args()


    list_ports()

    print()
    print(
        f"Opening {args.port} "
        f"at {args.baud} baud"
    )


    ser = serial.Serial(
        args.port,
        args.baud,
        timeout=0.1,
    )


    # Opening Mega usually resets it.
    print(
        "Waiting for Mega reset/startup..."
    )

    time.sleep(2.0)

    # Remove startup ASCII and anything
    # left from the reset transition.
    ser.reset_input_buffer()

    print("Binary acquisition started.")
    print(
        f"Protocol V2 packet size: "
        f"{PACKET_SIZE} bytes"
    )
    print()


    buf = bytearray()


    # --------------------------------------------------------
    # Host-side diagnostics
    # --------------------------------------------------------

    crc_errors = 0
    framing_errors = 0

    packet_gaps = 0

    last_packet_sequence = None


    # Counts between STATS packets
    sensor_counts = {
        IMU_ACCEL: 0,
        IMU_GYRO: 0,
        IMU_QUAT: 0,
    }

    last_stats_walltime = time.perf_counter()


    # Track BNO timestamp deltas independently
    last_sensor_timestamp = {}


    try:

        while True:

            chunk = ser.read(
                ser.in_waiting or 1
            )

            if chunk:
                buf.extend(chunk)


            # ----------------------------------------------
            # Decode all complete packets in buffer
            # ----------------------------------------------

            while len(buf) >= PACKET_SIZE:

                pos = buf.find(MAGIC)

                # No magic marker available
                if pos < 0:

                    # Preserve final byte in case it
                    # is first byte of magic.
                    if (
                        len(buf) > 0
                        and buf[-1] == MAGIC[0]
                    ):

                        buf[:] = buf[-1:]

                    else:

                        buf.clear()

                    break


                # Throw away garbage before frame
                if pos > 0:

                    framing_errors += pos

                    del buf[:pos]


                if len(buf) < PACKET_SIZE:
                    break


                frame = bytes(
                    buf[:PACKET_SIZE]
                )


                # ------------------------------------------
                # CRC
                # ------------------------------------------

                received_crc = (
                    frame[-2]
                    | (frame[-1] << 8)
                )

                calculated_crc = (
                    crc16_ccitt(
                        frame[:-2]
                    )
                )


                if (
                    calculated_crc
                    != received_crc
                ):

                    crc_errors += 1

                    # Drop one byte and search
                    # for magic again.
                    del buf[0]

                    continue


                # Valid frame
                del buf[:PACKET_SIZE]


                # ------------------------------------------
                # Header
                # ------------------------------------------

                header = struct.unpack_from(
                    HEADER_FMT,
                    frame,
                    0,
                )

                (
                    magic0,
                    magic1,
                    version,
                    packet_type,

                    packet_sequence,

                    sensor_timestamp_us,
                    host_timestamp_us,

                    sensor_sequence,
                    sensor_type,
                    sensor_status,
                    n_values,

                ) = header


                if version != PROTOCOL_VERSION:

                    print(
                        "WARNING: unsupported "
                        f"protocol version {version}"
                    )

                    continue


                # ------------------------------------------
                # UART packet sequence integrity
                # ------------------------------------------

                if (
                    last_packet_sequence
                    is not None
                ):

                    expected = (
                        last_packet_sequence + 1
                    ) & 0xFFFFFFFF

                    if (
                        packet_sequence
                        != expected
                    ):

                        missed = (
                            packet_sequence
                            - expected
                        ) & 0xFFFFFFFF

                        packet_gaps += missed

                        print(
                            "UART PACKET GAP:",
                            f"expected={expected}",
                            f"received={packet_sequence}",
                            f"missing={missed}",
                        )


                last_packet_sequence = (
                    packet_sequence
                )


                payload_offset = HEADER_SIZE


                # ==================================================
                # IMU PACKET
                # ==================================================

                if packet_type == PKT_IMU:

                    values = struct.unpack_from(
                        "<4f",
                        frame,
                        payload_offset,
                    )

                    sensor_counts[
                        sensor_type
                    ] = (
                        sensor_counts.get(
                            sensor_type,
                            0,
                        )
                        + 1
                    )


                    # ----------------------------------------------
                    # BNO timestamp delta
                    # ----------------------------------------------

                    dt_us = None
                    rate_hz = None

                    previous = (
                        last_sensor_timestamp.get(
                            sensor_type
                        )
                    )

                    if previous is not None:

                        dt_us = (
                            sensor_timestamp_us
                            - previous
                        )

                        if dt_us > 0:

                            rate_hz = (
                                1_000_000.0
                                / dt_us
                            )

                    last_sensor_timestamp[
                        sensor_type
                    ] = sensor_timestamp_us


                    # ----------------------------------------------
                    # Optional per-packet printing
                    # ----------------------------------------------

                    if args.print_imu:

                        if sensor_type == IMU_ACCEL:

                            name = "ACC"

                        elif sensor_type == IMU_GYRO:

                            name = "GYRO"

                        elif sensor_type == IMU_QUAT:

                            name = "QUAT"

                        else:

                            name = (
                                f"UNKNOWN"
                                f"({sensor_type})"
                            )


                        value_text = " ".join(
                            f"{v:.6f}"
                            for v in values[:n_values]
                        )


                        if rate_hz is None:

                            rate_text = "-"

                        else:

                            rate_text = (
                                f"{rate_hz:.1f}Hz"
                            )


                        print(
                            f"{name:5s} "
                            f"pkt={packet_sequence:8d} "
                            f"bseq={sensor_sequence:3d} "
                            f"bt={sensor_timestamp_us:12d} "
                            f"ht={host_timestamp_us:10d} "
                            f"status={sensor_status} "
                            f"rate={rate_text:>8s} "
                            f"{value_text}"
                        )


                # ==================================================
                # STATS PACKET
                # ==================================================

                elif packet_type == PKT_STATS:

                    (
                        overflow,
                        max_depth,
                        current_depth,
                        bno_resets,

                    ) = struct.unpack_from(
                        "<4I",
                        frame,
                        payload_offset,
                    )


                    now = time.perf_counter()

                    elapsed = (
                        now
                        - last_stats_walltime
                    )

                    if elapsed <= 0:
                        elapsed = 1.0


                    acc_hz = (
                        sensor_counts.get(
                            IMU_ACCEL,
                            0
                        )
                        / elapsed
                    )

                    gyro_hz = (
                        sensor_counts.get(
                            IMU_GYRO,
                            0
                        )
                        / elapsed
                    )

                    quat_hz = (
                        sensor_counts.get(
                            IMU_QUAT,
                            0
                        )
                        / elapsed
                    )


                    print(
                        "STATS",
                        f"pkt={packet_sequence}",
                        f"overflow={overflow}",
                        f"max_depth={max_depth}",
                        f"current={current_depth}",
                        f"resets={bno_resets}",
                        f"acc={acc_hz:.1f}Hz",
                        f"gyro={gyro_hz:.1f}Hz",
                        f"quat={quat_hz:.1f}Hz",
                        f"crc_errors={crc_errors}",
                        f"packet_gaps={packet_gaps}",
                    )


                    sensor_counts = {
                        IMU_ACCEL: 0,
                        IMU_GYRO: 0,
                        IMU_QUAT: 0,
                    }

                    last_stats_walltime = now


                else:

                    print(
                        "Unknown packet type:",
                        packet_type
                    )


    except KeyboardInterrupt:

        print()
        print("Stopping acquisition.")


    finally:

        ser.close()

        print(
            "Serial port closed."
        )

        print(
            "Final diagnostics:"
        )

        print(
            f"  CRC errors:   {crc_errors}"
        )

        print(
            f"  framing bytes discarded: "
            f"{framing_errors}"
        )

        print(
            f"  UART packet gaps: "
            f"{packet_gaps}"
        )


if __name__ == "__main__":
    main()

# import struct
# import serial

# PORT = "/dev/ttyACM0"
# # PORT = "COM3"
# BAUD = 500000

# MAGIC = b"\xA5\x5A"

# PACKET_SIZE = 30

# FMT = "<BBBBHIBB4fH"

# ser = serial.Serial(
#     PORT,
#     BAUD,
#     timeout=0.1,
# )


# def crc16_ccitt(data: bytes) -> int:
#     crc = 0xFFFF

#     for byte in data:
#         crc ^= byte << 8

#         for _ in range(8):
#             if crc & 0x8000:
#                 crc = ((crc << 1) ^ 0x1021) & 0xFFFF
#             else:
#                 crc = (crc << 1) & 0xFFFF

#     return crc


# buf = bytearray()

# while True:

#     chunk = ser.read(4096)

#     if chunk:
#         buf.extend(chunk)

#     while len(buf) >= PACKET_SIZE:

#         # Find packet boundary.
#         pos = buf.find(MAGIC)

#         if pos < 0:
#             buf.clear()
#             break

#         if pos > 0:
#             del buf[:pos]

#         if len(buf) < PACKET_SIZE:
#             break

#         frame = bytes(buf[:PACKET_SIZE])

#         fields = struct.unpack(FMT, frame)

#         (
#             magic0,
#             magic1,
#             version,
#             packet_type,
#             sequence,
#             timestamp_us,
#             sensor_type,
#             n_values,
#             v0,
#             v1,
#             v2,
#             v3,
#             received_crc,
#         ) = fields

#         calculated_crc = crc16_ccitt(
#             frame[:-2]
#         )

#         if calculated_crc != received_crc:

#             # Lost synchronization or corrupt packet.
#             del buf[0]
#             continue

#         del buf[:PACKET_SIZE]

#         if packet_type == 1:

#             if sensor_type == 1:
#                 print(
#                     "ACC",
#                     sequence,
#                     timestamp_us,
#                     v0, v1, v2,
#                 )

#             elif sensor_type == 2:
#                 print(
#                     "GYRO",
#                     sequence,
#                     timestamp_us,
#                     v0, v1, v2,
#                 )

#             elif sensor_type == 3:
#                 print(
#                     "QUAT",
#                     sequence,
#                     timestamp_us,
#                     v0, v1, v2, v3,
#                 )

#         elif packet_type == 2:

#             # Same 16 bytes need interpreting as integers.
#             stats = struct.unpack_from(
#                 "<4I",
#                 frame,
#                 12
#             )

#             print(
#                 "STATS:",
#                 "overflow =", stats[0],
#                 "max depth =", stats[1],
#                 "current =", stats[2],
#                 "tx =", stats[3],
#             )

