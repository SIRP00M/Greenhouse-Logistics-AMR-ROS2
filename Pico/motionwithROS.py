from machine import Pin, PWM
import sys
import uselect
import time

# ============================================================
# MOTOR CONFIG
# ============================================================

# RPWM, LPWM
MOTOR_PINS = [
    (2, 3),   # M1 Front Left
    (4, 5),   # M2 Front Right
    (6, 7),   # M3 Rear Left
    (8, 9),   # M4 Rear Right
]

# ถ้ามอเตอร์ตัวไหนหมุนกลับทิศ
# เปลี่ยน 1 -> -1
MOTOR_DIR = [
    1,   # Front Left
    1,   # Front Right
    1,   # Rear Left
    1,   # Rear Right
]

PWM_FREQ = 20000
MAX_COMMAND = 1000

# ถ้าไม่ได้รับคำสั่งจาก ROS เกิน 500 ms -> STOP
FAILSAFE_MS = 500


# ============================================================
# PWM SETUP
# ============================================================

motors = []

for rpwm_pin, lpwm_pin in MOTOR_PINS:
    rpwm = PWM(Pin(rpwm_pin))
    lpwm = PWM(Pin(lpwm_pin))

    rpwm.freq(PWM_FREQ)
    lpwm.freq(PWM_FREQ)

    rpwm.duty_u16(0)
    lpwm.duty_u16(0)

    motors.append((rpwm, lpwm))


# ============================================================
# MOTOR FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def set_motor(index, command):
    command = int(command)
    command = clamp(command, -MAX_COMMAND, MAX_COMMAND)

    command *= MOTOR_DIR[index]

    rpwm, lpwm = motors[index]

    duty = int(abs(command) / MAX_COMMAND * 65535)

    if command > 0:
        rpwm.duty_u16(duty)
        lpwm.duty_u16(0)

    elif command < 0:
        rpwm.duty_u16(0)
        lpwm.duty_u16(duty)

    else:
        rpwm.duty_u16(0)
        lpwm.duty_u16(0)


def set_all_motors(fl, fr, rl, rr):
    set_motor(0, fl)
    set_motor(1, fr)
    set_motor(2, rl)
    set_motor(3, rr)


def stop_all():
    set_all_motors(0, 0, 0, 0)


# ============================================================
# USB SERIAL
# ============================================================

poll = uselect.poll()
poll.register(sys.stdin, uselect.POLLIN)

last_command_time = time.ticks_ms()

stop_all()

print("PICO_MOTOR_READY")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    events = poll.poll(20)

    if events:

        line = sys.stdin.readline()

        if line:
            line = line.strip()

            if line == "PING":
                print("PONG")

            elif line == "STOP":
                stop_all()
                last_command_time = time.ticks_ms()

            elif line.startswith("MOTOR"):
                try:
                    parts = line.split()

                    if len(parts) == 5:

                        fl = int(parts[1])
                        fr = int(parts[2])
                        rl = int(parts[3])
                        rr = int(parts[4])

                        set_all_motors(
                            fl,
                            fr,
                            rl,
                            rr
                        )

                        last_command_time = time.ticks_ms()

                except Exception as e:
                    stop_all()
                    print("ERR:", e)

    # ========================================================
    # FAILSAFE
    # ========================================================

    now = time.ticks_ms()

    if time.ticks_diff(now, last_command_time) > FAILSAFE_MS:
        stop_all()

    time.sleep_ms(5)
