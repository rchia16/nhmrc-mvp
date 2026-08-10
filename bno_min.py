import time

import RPi.GPIO as GPIO
import spidev


# Pin configuration (BCM numbering).
INT_PIN = 23   # Physical pin 16
RST_PIN = 24   # Physical pin 18
CS_PIN = 5     # Physical pin 29
WAKE_PIN = 6   # Physical pin 31 (BNO085 PS0/WAKE)

SPI_SPEED_HZ = 1_000_000
INT_TIMEOUT_S = 0.5
MAX_PACKET_LENGTH = 4096


GPIO.setmode(GPIO.BCM)
GPIO.setup(INT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(CS_PIN, GPIO.OUT, initial=GPIO.HIGH)
# PS0 must be high during reset (with PS1 strapped high) to select SPI.
GPIO.setup(WAKE_PIN, GPIO.OUT, initial=GPIO.HIGH)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = SPI_SPEED_HZ
spi.mode = 0b11
try:
    # D5 is the logical CS; the disconnected CE0 line must not frame packets.
    spi.no_cs = True
except (AttributeError, OSError) as exc:
    print(f"Warning: could not disable kernel CE0: {exc}")

sequence_numbers = [0] * 6


def wait_for_int(timeout_s=INT_TIMEOUT_S):
    """Wait for active-low INT without resetting the sensor."""
    deadline = time.monotonic() + timeout_s
    while GPIO.input(INT_PIN) == GPIO.HIGH:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.0005)
    return True


def reset_sensor():
    """Reset with PS0/WAKE high so the BNO085 starts in SPI mode."""
    GPIO.output(WAKE_PIN, GPIO.HIGH)
    GPIO.output(CS_PIN, GPIO.HIGH)
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.05)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.75)


def read_packet(timeout_s=0.1):
    """Read one complete SHTP packet while D5 remains asserted."""
    if not wait_for_int(timeout_s):
        return None

    GPIO.output(CS_PIN, GPIO.LOW)
    try:
        header = spi.readbytes(4)
        if len(header) != 4:
            raise RuntimeError(
                f"Short SPI header: expected 4 bytes, got {len(header)}"
            )

        raw_length = header[0] | (header[1] << 8)
        continuation = bool(raw_length & 0x8000)
        packet_length = raw_length & 0x7FFF
        channel = header[2]
        sequence = header[3]

        if packet_length == 0:
            return None
        if packet_length < 4 or packet_length > MAX_PACKET_LENGTH:
            raise RuntimeError(
                f"Invalid SHTP header length={packet_length} "
                f"channel={channel} sequence={sequence} raw={header}"
            )
        if channel >= len(sequence_numbers):
            raise RuntimeError(
                f"Invalid SHTP channel={channel} length={packet_length} "
                f"sequence={sequence} raw={header}"
            )

        payload_length = packet_length - 4
        payload = spi.readbytes(payload_length) if payload_length else []
        if len(payload) != payload_length:
            raise RuntimeError(
                f"Short SPI payload: expected {payload_length} bytes, "
                f"got {len(payload)}"
            )
    finally:
        GPIO.output(CS_PIN, GPIO.HIGH)

    if continuation:
        raise RuntimeError(
            "Unexpected SHTP continuation packet; reset or power-cycle the BNO085"
        )

    return channel, sequence, payload


def send_packet(channel, payload):
    """Wake the BNO085 and send one host-to-device SHTP packet."""
    if not 0 <= channel < len(sequence_numbers):
        raise ValueError(f"Invalid SHTP channel {channel}")

    sequence = sequence_numbers[channel]
    packet_length = len(payload) + 4
    packet = [
        packet_length & 0xFF,
        (packet_length >> 8) & 0x7F,
        channel,
        sequence,
        *payload,
    ]

    # After boot, PS0 becomes active-low WAKE. INT acknowledges host access.
    GPIO.output(WAKE_PIN, GPIO.LOW)
    try:
        if not wait_for_int():
            raise TimeoutError(
                f"BNO085 did not acknowledge WAKE for channel {channel}"
            )

        GPIO.output(CS_PIN, GPIO.LOW)
        try:
            spi.xfer2(packet)
        finally:
            GPIO.output(CS_PIN, GPIO.HIGH)
    finally:
        GPIO.output(WAKE_PIN, GPIO.HIGH)

    sequence_numbers[channel] = (sequence + 1) % 256
    print(
        f"Sent SHTP packet channel={channel} sequence={sequence} "
        f"length={packet_length}"
    )


def drain_startup_packets():
    """Drain complete startup packets until the device has been idle for 100 ms."""
    print("Clearing startup packets...")
    deadline = time.monotonic() + 2.0
    idle_since = None
    drained = 0

    while time.monotonic() < deadline:
        packet = read_packet(timeout_s=0.05)
        if packet is None:
            if idle_since is None:
                idle_since = time.monotonic()
            if drained and time.monotonic() - idle_since >= 0.1:
                break
            continue

        idle_since = None
        channel, sequence, payload = packet
        drained += 1
        print(
            f" Startup packet {drained}: channel={channel} "
            f"sequence={sequence} length={len(payload) + 4}"
        )

    if drained == 0:
        raise RuntimeError(
            "No BNO085 startup packets received; check INT, CS, SPI mode, "
            "PS0/WAKE, PS1, and RESET wiring"
        )
    print(f"Startup drain complete: {drained} packets")


def enable_rotation_vector():
    """Enable rotation-vector report 0x05 at 50 Hz."""
    interval_us = 20_000
    payload = [
        0xFD,  # Set Feature Command
        0x05,  # Rotation Vector
        0x00,  # Feature flags
        0x00,
        0x00,  # Change sensitivity
        interval_us & 0xFF,
        (interval_us >> 8) & 0xFF,
        (interval_us >> 16) & 0xFF,
        (interval_us >> 24) & 0xFF,
        0x00,
        0x00,
        0x00,
        0x00,  # Batch interval
        0x00,
        0x00,
        0x00,
        0x00,  # Sensor-specific configuration
    ]
    print("Enabling rotation vector at 50 Hz...")
    send_packet(2, payload)


def decode_rotation_vector(payload):
    """Return an (i, j, k, real) quaternion from a channel-3 payload."""
    start = 5 if payload and payload[0] == 0xFB else 0
    for offset in range(start, max(start, len(payload) - 11)):
        if payload[offset] != 0x05 or offset + 12 > len(payload):
            continue

        values = [
            int.from_bytes(payload[index:index + 2], "little", signed=True)
            for index in range(offset + 4, offset + 12, 2)
        ]
        scale = 1.0 / (1 << 14)
        return tuple(value * scale for value in values)
    return None


def main():
    print("Resetting BNO085...")
    reset_sensor()
    drain_startup_packets()
    enable_rotation_vector()
    print("\nStreaming data. Press Ctrl+C to stop.\n")

    packet_count = 0
    quaternion_count = 0
    last_status = time.monotonic()

    while True:
        packet = read_packet()
        if packet is not None:
            channel, sequence, payload = packet
            packet_count += 1

            if channel == 3:
                quaternion = decode_rotation_vector(payload)
                if quaternion is not None:
                    quaternion_count += 1
                    raw_i, raw_j, raw_k, raw_real = quaternion
                    print(
                        f"Quat I:{raw_i:+.4f} J:{raw_j:+.4f} "
                        f"K:{raw_k:+.4f} Real:{raw_real:+.4f}",
                        end="\r",
                        flush=True,
                    )

        now = time.monotonic()
        if now - last_status >= 2.0:
            print(
                f"\nStatus: packets={packet_count} quaternions={quaternion_count} "
                f"INT={'LOW' if GPIO.input(INT_PIN) == GPIO.LOW else 'HIGH'}"
            )
            last_status = now


try:
    main()
except KeyboardInterrupt:
    print("\nStopping stream.")
finally:
    GPIO.output(WAKE_PIN, GPIO.HIGH)
    GPIO.output(CS_PIN, GPIO.HIGH)
    spi.close()
    GPIO.cleanup()
