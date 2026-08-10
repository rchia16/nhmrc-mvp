import serial
import time

# Adjust port mapping if your Arduino connects on /dev/ttyACM1
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1.0)
time.sleep(2) # Give the Mega a brief moment to reset on connection

print("Starting combined data logger...")

with open("sensor_log.csv", "w") as f:
    f.write("Quat_Real,Quat_I,Quat_J,Quat_K,Heart_Red,Heart_IR\n") # File header
    
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "DATA_ERR" not in line and line:
                    print(f"Logged: {line}")
                    f.write(f"{line}\n")
                    f.flush() # Force write data to disk immediately
    except KeyboardInterrupt:
        print("\nLogging stopped.")
    finally:
        ser.close()

