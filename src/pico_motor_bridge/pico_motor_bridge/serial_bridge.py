#!/usr/bin/env python3

import time
import serial

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Int64MultiArray


class PicoMotorBridge(Node):

    def __init__(self):

        super().__init__('pico_motor_bridge')

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter(
            'port',
            '/dev/ttyACM0'
        )

        self.declare_parameter(
            'baud',
            115200
        )

        self.declare_parameter(
            'max_command',
            1000
        )

        self.declare_parameter(
            'max_linear_x',
            0.50
        )

        self.declare_parameter(
            'max_linear_y',
            0.50
        )

        self.declare_parameter(
            'max_angular_z',
            1.50
        )

        self.declare_parameter(
            'cmd_timeout',
            0.5
        )

        self.declare_parameter(
            'send_rate',
            20.0
        )

        # ====================================================
        # GET PARAMETERS
        # ====================================================

        self.port = (
            self.get_parameter('port')
            .get_parameter_value()
            .string_value
        )

        self.baud = (
            self.get_parameter('baud')
            .get_parameter_value()
            .integer_value
        )

        self.max_command = (
            self.get_parameter('max_command')
            .get_parameter_value()
            .integer_value
        )

        self.max_linear_x = (
            self.get_parameter('max_linear_x')
            .get_parameter_value()
            .double_value
        )

        self.max_linear_y = (
            self.get_parameter('max_linear_y')
            .get_parameter_value()
            .double_value
        )

        self.max_angular_z = (
            self.get_parameter('max_angular_z')
            .get_parameter_value()
            .double_value
        )

        self.cmd_timeout = (
            self.get_parameter('cmd_timeout')
            .get_parameter_value()
            .double_value
        )

        send_rate = (
            self.get_parameter('send_rate')
            .get_parameter_value()
            .double_value
        )

        # ====================================================
        # SERIAL
        # ====================================================

        self.get_logger().info(
            f'Opening Pico {self.port} @ {self.baud}'
        )

        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=0,
            write_timeout=0.05
        )

        time.sleep(1)

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        self.rx_buffer = ""

        self.get_logger().info(
            'Pico connected'
        )

        # ====================================================
        # MOTOR COMMAND STATE
        # ====================================================

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        self.last_cmd_time = 0.0

        # ====================================================
        # ROS SUBSCRIBER
        # ====================================================

        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # ====================================================
        # ROS PUBLISHER
        # ====================================================

        self.encoder_pub = self.create_publisher(
            Int64MultiArray,
            '/wheel_encoder_ticks',
            10
        )

        # ====================================================
        # TIMERS
        # ====================================================

        self.motor_timer = self.create_timer(
            1.0 / send_rate,
            self.send_motor_command
        )

        # อ่าน serial เร็วกว่ารอบ telemetry
        self.serial_timer = self.create_timer(
            0.01,
            self.read_serial
        )

        self.get_logger().info(
            'Motor + encoder bridge READY'
        )

    # ========================================================
    # CMD VEL
    # ========================================================

    def cmd_vel_callback(self, msg):

        self.target_x = msg.linear.x
        self.target_y = msg.linear.y
        self.target_z = msg.angular.z

        self.last_cmd_time = time.monotonic()

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def clamp(value, minimum, maximum):

        return max(
            minimum,
            min(maximum, value)
        )

    # ========================================================
    # MECANUM MIXING
    # ========================================================

    def calculate_wheels(
        self,
        vx,
        vy,
        wz
    ):

        x = vx / self.max_linear_x
        y = vy / self.max_linear_y
        z = wz / self.max_angular_z

        x = self.clamp(
            x,
            -1.0,
            1.0
        )

        y = self.clamp(
            y,
            -1.0,
            1.0
        )

        z = self.clamp(
            z,
            -1.0,
            1.0
        )

        fl = x - y - z
        fr = x + y + z
        rl = x + y - z
        rr = x - y + z

        largest = max(
            abs(fl),
            abs(fr),
            abs(rl),
            abs(rr),
            1.0
        )

        fl /= largest
        fr /= largest
        rl /= largest
        rr /= largest

        return (
            int(fl * self.max_command),
            int(fr * self.max_command),
            int(rl * self.max_command),
            int(rr * self.max_command)
        )

    # ========================================================
    # MOTOR SEND
    # ========================================================

    def send_motor_command(self):

        now = time.monotonic()

        if (
            self.last_cmd_time == 0.0
            or
            now - self.last_cmd_time
            > self.cmd_timeout
        ):

            vx = 0.0
            vy = 0.0
            wz = 0.0

        else:

            vx = self.target_x
            vy = self.target_y
            wz = self.target_z

        fl, fr, rl, rr = (
            self.calculate_wheels(
                vx,
                vy,
                wz
            )
        )

        command = (
            f"MOTOR "
            f"{fl} "
            f"{fr} "
            f"{rl} "
            f"{rr}\n"
        )

        try:

            self.serial.write(
                command.encode()
            )

        except serial.SerialException as e:

            self.get_logger().error(
                f'Serial write error: {e}'
            )

    # ========================================================
    # SERIAL RECEIVE
    # ========================================================

    def read_serial(self):

        try:

            waiting = self.serial.in_waiting

            if waiting <= 0:
                return

            raw = self.serial.read(
                waiting
            )

            text = raw.decode(
                'utf-8',
                errors='ignore'
            )

            self.rx_buffer += text

            while '\n' in self.rx_buffer:

                line, self.rx_buffer = (
                    self.rx_buffer.split(
                        '\n',
                        1
                    )
                )

                line = line.strip()

                if line:
                    self.process_serial_line(
                        line
                    )

        except serial.SerialException as e:

            self.get_logger().error(
                f'Serial read error: {e}'
            )

    # ========================================================
    # PROCESS PICO MESSAGE
    # ========================================================

    def process_serial_line(self, line):

        if not line.startswith('ENC '):
            return

        parts = line.split()

        if len(parts) != 5:
            return

        try:

            ticks = [
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4])
            ]

        except ValueError:
            return

        msg = Int64MultiArray()

        msg.data = ticks

        self.encoder_pub.publish(
            msg
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop_robot(self):

        try:

            self.serial.write(
                b'STOP\n'
            )

        except Exception:
            pass

    def destroy_node(self):

        self.stop_robot()

        try:
            self.serial.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PicoMotorBridge()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
