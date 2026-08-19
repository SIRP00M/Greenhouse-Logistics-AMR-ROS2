from machine import Pin, PWM
import time

PWM_FREQ = 1000
DUTY = 65535   # 100%


# =========================
# Motor setup
# =========================

m1_a = PWM(Pin(2))
m1_b = PWM(Pin(3))

m2_a = PWM(Pin(4))
m2_b = PWM(Pin(5))

m3_a = PWM(Pin(6))
m3_b = PWM(Pin(7))

m4_a = PWM(Pin(8))
m4_b = PWM(Pin(9))


motors = [
    m1_a, m1_b,
    m2_a, m2_b,
    m3_a, m3_b,
    m4_a, m4_b
]

for pwm in motors:
    pwm.freq(PWM_FREQ)
    pwm.duty_u16(0)


# =========================
# STOP
# =========================

def stop():

    for pwm in motors:
        pwm.duty_u16(0)


# =========================
# FORWARD
# =========================

def forward():

    # M1 Rear Left
    m1_a.duty_u16(DUTY)
    m1_b.duty_u16(0)

    # M2 Rear Right
    m2_a.duty_u16(0)
    m2_b.duty_u16(DUTY)

    # M3 Front Left
    m3_a.duty_u16(DUTY)
    m3_b.duty_u16(0)

    # M4 Front Right
    m4_a.duty_u16(0)
    m4_b.duty_u16(DUTY)


# =========================
# REVERSE
# =========================

def reverse():

    # M1 Rear Left
    m1_a.duty_u16(0)
    m1_b.duty_u16(DUTY)

    # M2 Rear Right
    m2_a.duty_u16(DUTY)
    m2_b.duty_u16(0)

    # M3 Front Left
    m3_a.duty_u16(0)
    m3_b.duty_u16(DUTY)

    # M4 Front Right
    m4_a.duty_u16(DUTY)
    m4_b.duty_u16(0)


# =========================
# TEST
# =========================

try:

    stop()

    print("ALL MOTORS FORWARD")
    forward()

    time.sleep(3)

    print("STOP")
    stop()

    time.sleep(2)

    print("ALL MOTORS REVERSE")
    reverse()

    time.sleep(3)

    print("STOP")
    stop()

except KeyboardInterrupt:

    stop()

finally:

    stop()
