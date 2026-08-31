import serial
import time

# PORT = "/dev/ttyACM1"   # change if needed
PORT = "COM9"   # change if needed

ser = serial.Serial(PORT, 500000, timeout=2)

ser.dtr = True
ser.rts = True

time.sleep(0.5)

data = ser.read(500)

print("Received bytes:", len(data))
print("Magic position:", data.find(b"\xA5\x5A"))
print("First bytes:", data[:100].hex(" "))

ser.close()
