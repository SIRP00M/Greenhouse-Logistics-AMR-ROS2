#!/usr/bin/env python3

import time
import serial

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist


class PicoMotorBridge(Node):

    def __init__(self):
        super().__init__('pico_motor_bridge')

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        self.declare_parameter('max_command', 1000)

        # ROS velocity limits
        self.declare_parameter('max_linear_x', 0.50)
        self.declare_parameter('max_linear_y', 0.50)
        self.declare_parameter('max_angular_z', 1.50)

        # ถ้า /cmd_vel หายเกินเวลานี้ -> STOP
        self.declare_parameter('cmd_timeout', 0.5)

        # ส่งไป Pico กี่ครั้งต่อวินาที
        self.declare_parameter('send_rate', 20.0)

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
            f'Connecting Pico: {self.port} @ {self.baud}'
        )

        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=0.05,
            write_timeout=0.05
        )

        time.sleep(1.0)

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        self.get_logger().info('Pico serial connected')

        # ====================================================
        # CMD STATE
        # ====================================================

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        self.last_cmd_time = 0.0

        # ====================================================
        # ROS
        # ====================================================

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        period = 1.0 / send_rate

        self.timer = self.create_timer(
            period,
            self.send_motor_command
        )

        self.get_logger().info(
            'Listening on /cmd_vel'
        )

        self.get_logger().info(
            'Mecanum motor bridge READY'
        )

    # ========================================================
    # /cmd_vel callback
    # ========================================================

    def cmd_vel_callback(self, msg):

        self.target_x = msg.linear.x
        self.target_y = msg.linear.y
        self.target_z = msg.angular.z

        self.last_cmd_time = time.monotonic()

    # ========================================================
    # Clamp
    # ========================================================

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    # ========================================================
    # Mecanum mixing
    # ========================================================

    def calculate_wheels(self, vx, vy, wz):

        # Normalize ROS velocity into -1 .. +1

        x = vx / self.max_linear_x
        y = vy / self.max_linear_y
        z = wz / self.max_angular_z

        x = self.clamp(x, -1.0, 1.0)
        y = self.clamp(y, -1.0, 1.0)
        z = self.clamp(z, -1.0, 1.0)

        # ====================================================
        # MECANUM INVERSE KINEMATICS
        #
        #        FRONT
        #
        #      FL      FR
        #       \      /
        #
        #       /      \
        #      RL      RR
        #
        # ====================================================

        fl = x - y - z
        fr = x + y + z
        rl = x + y - z
        rr = x - y + z

        # Normalize if mixing exceeds 1.0
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

        # Convert to Pico range
        fl = int(fl * self.max_command)
        fr = int(fr * self.max_command)
        rl = int(rl * self.max_command)
        rr = int(rr * self.max_command)

        return fl, fr, rl, rr

    # ========================================================
    # Send command to Pico
    # ========================================================

    def send_motor_command(self):

        now = time.monotonic()

        # No command / command timeout
        if (
            self.last_cmd_time == 0.0
            or now - self.last_cmd_time > self.cmd_timeout
        ):
            vx = 0.0
            vy = 0.0
            wz = 0.0

        else:
            vx = self.target_x
            vy = self.target_y
            wz = self.target_z

        fl, fr, rl, rr = self.calculate_wheels(
            vx,
            vy,
            wz
        )

        command = (
            f'MOTOR {fl} {fr} {rl} {rr}\n'
        )

        try:
            self.serial.write(
                command.encode('utf-8')
            )

        except serial.SerialException as e:
            self.get_logger().error(
                f'Serial error: {e}'
            )

    # ========================================================
    # Shutdown
    # ========================================================

    def stop_robot(self):

        try:
            self.serial.write(
                b'STOP\n'
            )

            time.sleep(0.05)

        except Exception:
            pass

    def destroy_node(self):

        self.get_logger().info(
            'Stopping motors...'
        )

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
