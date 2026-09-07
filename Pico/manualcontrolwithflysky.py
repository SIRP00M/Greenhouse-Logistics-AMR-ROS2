from machine import Pin, PWM, UART
from time import sleep_ms, ticks_ms, ticks_diff


# ============================================================
# CONFIG
# ============================================================

PWM_FREQ = 1000

# ความแรงสูงสุด
# 14000 = ~21%
# 32000 = ~49%
# 40000 = ~61%
# 50000 = ~76%
# 65535 = 100%
MAX_PWM = 45000

# Soft start
# ยิ่งน้อย = ออกตัวนุ่ม
# ยิ่งมาก = ตอบสนองเร็ว
ACCEL_STEP = 0.04

# Deadzone ของคันโยก
DEADBAND = 60

# ถ้า iBUS หายเกิน 300ms ให้หยุดทันที
FAILSAFE_MS = 300


# ============================================================
# RC CENTER CALIBRATION
# ค่าที่เราอ่านได้จากรีโมตตัวนี้
# ============================================================

CENTER_CH1 = 1503
CENTER_CH2 = 1582
CENTER_CH4 = 1497


# ============================================================
# DIRECTION
#
# ถ้าทิศไหนกลับ เปลี่ยน 1 เป็น -1
# ============================================================

X_SIGN = 1      # Strafe ซ้าย/ขวา
Y_SIGN = 1      # หน้า/หลัง
R_SIGN = 1      # หมุน


# ============================================================
# MOTOR CLASS
# ============================================================

class Motor:

    def __init__(self, rpwm_pin, lpwm_pin, invert=False):

        self.rpwm = PWM(Pin(rpwm_pin))
        self.lpwm = PWM(Pin(lpwm_pin))

        self.rpwm.freq(PWM_FREQ)
        self.lpwm.freq(PWM_FREQ)

        self.invert = invert

        self.stop()


    def drive(self, value):

        # จำกัด -1.0 ถึง +1.0
        value = max(-1.0, min(1.0, value))

        if self.invert:
            value = -value

        duty = int(abs(value) * MAX_PWM)


        # Forward
        if value > 0:

            self.rpwm.duty_u16(duty)
            self.lpwm.duty_u16(0)


        # Backward
        elif value < 0:

            self.rpwm.duty_u16(0)
            self.lpwm.duty_u16(duty)


        # Stop
        else:

            self.stop()


    def stop(self):

        self.rpwm.duty_u16(0)
        self.lpwm.duty_u16(0)


# ============================================================
# MOTOR PIN
#
# Motor 1 = Front Left
# Motor 2 = Front Right
# Motor 3 = Rear Left
# Motor 4 = Rear Right
# ============================================================

FL = Motor(
    2,
    3,
    invert=False
)

FR = Motor(
    4,
    5,
    invert=False
)

RL = Motor(
    6,
    7,
    invert=False
)

RR = Motor(
    8,
    9,
    invert=False
)


# ============================================================
# CURRENT MOTOR SPEED
# ใช้สำหรับ Soft Start
# ============================================================

current_fl = 0.0
current_fr = 0.0
current_rl = 0.0
current_rr = 0.0


# ============================================================
# STOP ALL
# ============================================================

def stop_all():

    global current_fl
    global current_fr
    global current_rl
    global current_rr

    FL.stop()
    FR.stop()
    RL.stop()
    RR.stop()

    current_fl = 0.0
    current_fr = 0.0
    current_rl = 0.0
    current_rr = 0.0


# ============================================================
# FLYSKY iBUS UART
#
# iBUS Signal -> GP1
#
# GP0 = UART TX (ไม่ได้ใช้)
# GP1 = UART RX
# ============================================================

uart = UART(
    0,
    baudrate=115200,
    tx=Pin(0),
    rx=Pin(1),
    bits=8,
    parity=None,
    stop=1,
    rxbuf=256
)


# ============================================================
# iBUS PARSER
# Fixed 32 byte buffer
# ป้องกัน MemoryError แบบที่เจอก่อนหน้า
# ============================================================

frame = bytearray(32)

frame_index = 0


def read_ibus():

    global frame_index

    while uart.any():

        data = uart.read(1)

        if not data:
            return None

        b = data[0]


        # ----------------------------------------------------
        # BYTE 0
        # iBUS frame ต้องเริ่ม 0x20
        # ----------------------------------------------------

        if frame_index == 0:

            if b == 0x20:

                frame[0] = b

                frame_index = 1

            continue


        # ----------------------------------------------------
        # BYTE 1
        # byte สองต้องเป็น 0x40
        # ----------------------------------------------------

        if frame_index == 1:

            if b == 0x40:

                frame[1] = b

                frame_index = 2


            elif b == 0x20:

                # อาจเป็น frame ใหม่
                frame[0] = b

                frame_index = 1


            else:

                frame_index = 0


            continue


        # ----------------------------------------------------
        # รับ byte ที่เหลือ
        # ----------------------------------------------------

        frame[frame_index] = b

        frame_index += 1


        # ----------------------------------------------------
        # Frame ครบ 32 bytes
        # ----------------------------------------------------

        if frame_index >= 32:

            frame_index = 0


            # Checksum จาก FlySky
            received_checksum = (
                frame[30]
                |
                (frame[31] << 8)
            )


            # Checksum ที่ Pico คำนวณเอง
            calculated_checksum = (
                0xFFFF
                -
                sum(frame[:30])
            ) & 0xFFFF


            # Frame เสีย
            if received_checksum != calculated_checksum:

                continue


            # ------------------------------------------------
            # Decode 14 channels
            # ------------------------------------------------

            channels = [0] * 14


            for i in range(14):

                low = frame[2 + i * 2]

                high = frame[3 + i * 2]


                channels[i] = (
                    low
                    |
                    (high << 8)
                )


            return channels


    return None


# ============================================================
# NORMALIZE RC
#
# เปลี่ยน 1000-2000
#
# เป็น
#
# -1.0 ... 0 ... +1.0
# ============================================================

def normalize(value, center):

    delta = value - center


    # Deadzone
    if abs(delta) <= DEADBAND:

        return 0.0


    # --------------------------------------------------------
    # Positive side
    # --------------------------------------------------------

    if delta > 0:

        denominator = 2000 - center

        if denominator <= 0:

            return 0.0

        output = delta / denominator


    # --------------------------------------------------------
    # Negative side
    # --------------------------------------------------------

    else:

        denominator = center - 1000

        if denominator <= 0:

            return 0.0

        output = delta / denominator


    return max(
        -1.0,
        min(1.0, output)
    )


# ============================================================
# SOFT START / RAMP
# ============================================================

def ramp(current, target):

    # ต้องเพิ่มความเร็ว
    if current < target:

        current += ACCEL_STEP

        if current > target:

            current = target


    # ต้องลดความเร็ว
    elif current > target:

        current -= ACCEL_STEP

        if current < target:

            current = target


    return current


# ============================================================
# MECANUM MIXER
#
# x = Strafe
# y = Forward / Backward
# r = Rotate
# ============================================================

def mecanum(x, y, r):

    global current_fl
    global current_fr
    global current_rl
    global current_rr


    # --------------------------------------------------------
    # Mecanum calculation
    # --------------------------------------------------------

    fl = y + x + r

    fr = y - x - r

    rl = y - x + r

    rr = y + x - r


    # --------------------------------------------------------
    # Normalize
    #
    # ถ้ารวมแล้วเกิน 1.0
    # ลดทุกล้อลงตามสัดส่วน
    # --------------------------------------------------------

    maximum = max(
        abs(fl),
        abs(fr),
        abs(rl),
        abs(rr),
        1.0
    )


    fl /= maximum
    fr /= maximum
    rl /= maximum
    rr /= maximum


    # --------------------------------------------------------
    # Soft Start
    # --------------------------------------------------------

    current_fl = ramp(
        current_fl,
        fl
    )

    current_fr = ramp(
        current_fr,
        fr
    )

    current_rl = ramp(
        current_rl,
        rl
    )

    current_rr = ramp(
        current_rr,
        rr
    )


    # --------------------------------------------------------
    # Send to motors
    # --------------------------------------------------------

    FL.drive(current_fl)

    FR.drive(current_fr)

    RL.drive(current_rl)

    RR.drive(current_rr)


# ============================================================
# MAIN
# ============================================================

print()
print("===================================")
print("     FLYSKY MECANUM CONTROL")
print("===================================")
print()
print("GP1 = iBUS RX")
print()
print("CH1 = STRAFE")
print("CH2 = FORWARD / BACKWARD")
print("CH4 = ROTATE")
print()
print("MAX PWM =", MAX_PWM)
print("FAILSAFE =", FAILSAFE_MS, "ms")
print("SOFT START =", ACCEL_STEP)
print()
print("READY!")
print()


stop_all()


last_packet = ticks_ms()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ch = read_ibus()


    # --------------------------------------------------------
    # ได้ iBUS packet
    # --------------------------------------------------------

    if ch:

        last_packet = ticks_ms()


        # ====================================================
        # CHANNEL INPUT
        # ====================================================


        # CH1
        # Right stick Left / Right
        # Mecanum Strafe

        x = normalize(
            ch[0],
            CENTER_CH1
        )

        x *= X_SIGN


        # ----------------------------------------------------


        # CH2
        # Right stick Up / Down
        # Forward / Backward

        y = normalize(
            ch[1],
            CENTER_CH2
        )

        y *= Y_SIGN


        # ----------------------------------------------------


        # CH4
        # Left stick Left / Right
        # Rotate

        r = normalize(
            ch[3],
            CENTER_CH4
        )

        r *= R_SIGN


        # ====================================================
        # DRIVE
        # ====================================================

        mecanum(
            x,
            y,
            r
        )


    # ========================================================
    # FAILSAFE
    #
    # Receiver หลุด
    # รีโมตดับ
    # สาย iBUS หลุด
    #
    # => STOP
    # ========================================================

    if ticks_diff(
        ticks_ms(),
        last_packet
    ) > FAILSAFE_MS:

        stop_all()


    sleep_ms(5)
