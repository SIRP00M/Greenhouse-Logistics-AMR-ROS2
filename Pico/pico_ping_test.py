import serial
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

print(f"[INFO] Opening {PORT}...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

time.sleep(2)

# ล้างข้อมูลเก่าที่ Pico อาจส่งมาตอน boot
ser.reset_input_buffer()

print("[OK] Pico connected")

while True:
    print("PC  -> PING")

    ser.write(b"PING\n")

    response = ser.readline().decode("utf-8", errors="ignore").strip()

    if response:
        print(f"PICO -> {response}")
    else:
        print("PICO -> [NO RESPONSE]")

    print()

    time.sleep(1)

