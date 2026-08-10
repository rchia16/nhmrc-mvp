import time
import spidev
import RPi.GPIO as GPIO

# Pin Configuration (BCM Numbering)
INT_PIN = 23  # Physical Pin 16
RST_PIN = 24  # Physical Pin 18
CS_PIN = 5    # Physical Pin 29

GPIO.setmode(GPIO.BCM)
GPIO.setup(INT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RST_PIN, GPIO.OUT)
GPIO.setup(CS_PIN, GPIO.OUT)
GPIO.output(CS_PIN, GPIO.HIGH)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000  # 1 MHz
spi.mode = 0b11             # SPI Mode 3

def reset_sensor():
    """Performs a clean hardware reset sequence"""
    GPIO.output(RST_PIN, GPIO.LOW)
    time.sleep(0.05)
    GPIO.output(RST_PIN, GPIO.HIGH)
    time.sleep(0.4)

def read_packet():
    """Waits for INT pin to go LOW, handles CS, and reads full packet"""
    if GPIO.input(INT_PIN) == GPIO.HIGH:
        timeout = time.time() + 0.1
        while GPIO.input(INT_PIN) == GPIO.HIGH:
            if time.time() > timeout:
                return None, None

    GPIO.output(CS_PIN, GPIO.LOW)
    header = spi.readbytes(4)
    packet_length = ((header[1] & 0x7F) << 8) | header[0]
    channel = header[2]

    if packet_length <= 4 or packet_length > 512:
        GPIO.output(CS_PIN, GPIO.HIGH)
        return None, None

    payload = spi.readbytes(packet_length - 4)
    GPIO.output(CS_PIN, GPIO.HIGH)
    return channel, payload

def send_packet(channel, payload):
    """Sends a packet using the exact SHTP wake-on-CS pattern"""
    packet_length = len(payload) + 4
    header = [
        packet_length & 0xFF,
        (packet_length >> 8) & 0xFF,
        channel,
        0x00  # Sequence number
    ]
    full_packet = header + payload

    # Assert CS LOW to prompt the sensor to wake up
    GPIO.output(CS_PIN, GPIO.LOW)

    # Wait briefly for acknowledgement
    timeout = time.time() + 0.05
    while GPIO.input(INT_PIN) == GPIO.HIGH:
        if time.time() > timeout:
            break

    spi.xfer2(full_packet)
    GPIO.output(CS_PIN, GPIO.HIGH)
    time.sleep(0.02)

def enable_rotation_vector():
    """Sends a perfectly padded SHTP command to stream Rotation Vector (0x05) at 50Hz"""
    print("Sending configuration command...")

    # Crucial Fix: This payload must match the exact 17-byte layout required by the firmware
    payload = [
        0xFD,        # 0: Command ID (Set Feature Command)
        0x05,        # 1: Feature ID (Rotation Vector)
        0x00,        # 2: Report Flags (0x00)
        0x00, 0x00,  # 3-4: Change Sensitivity (0)
        0x20, 0x4E, 0x00, 0x00, # 5-8: Report Interval (20,000 microseconds = 50Hz)
        0x00, 0x00, 0x00, 0x00, # 9-12: Batch Interval (0)
        0x00, 0x00, 0x00, 0x00  # 13-16: Sensor-specific config padding (Must be 4 bytes!)
    ]
    send_packet(2, payload)

try:
    print("Resetting BNO085...")
    reset_sensor()

    print("Clearing bootloader startup packets...")
    # Safely flush out the boot headers
    for _ in range(15):
        chan, pay = read_packet()
        if chan is not None:
            print(f" Cleared packet from channel {chan} (Size: {len(pay)+4})")
        time.sleep(0.01)

    enable_rotation_vector()
    print("\nStreaming Data! Press Ctrl+C to stop.\n")

    while True:
        channel, payload = read_packet()

        if channel == 3 and payload is not None:
            # Look through the payload for the base Report ID data blocks
            for i in range(len(payload) - 10):
                # 0x05 is the input report ID for Rotation Vector data
                if payload[i] == 0x05:
                    try:
                        # Raw 2-byte little endian values
                        # In Channel 3 input reports, the real data payload starts 4 bytes past report header index
                        raw_i = int.from_bytes(payload[i+4:i+6], byteorder='little', signed=True)
                        raw_j = int.from_bytes(payload[i+6:i+8], byteorder='little', signed=True)
                        raw_k = int.from_bytes(payload[i+8:i+10], byteorder='little', signed=True)
                        raw_real = int.from_bytes(payload[i+10:i+12], byteorder='little', signed=True)

                        scale = 1.0 / (1 << 14) # 14-bit fixed point multiplier conversion
                        print(f"Quat -> I:{raw_i*scale:+.4f} J:{raw_j*scale:+.4f} K:{raw_k*scale:+.4f} Real:{raw_real*scale:+.4f}      ", end='\r')
                    except IndexError:
                        pass
        time.sleep(0.002)

except KeyboardInterrupt:
    print("\nStopping stream.")
finally:
    spi.close()
    GPIO.cleanup()

