#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Float32MultiArray


class EncoderVelocity(Node):

    def __init__(self):
        super().__init__('encoder_velocity')

        # ====================================================
        # CONFIG
        # ====================================================

        self.declare_parameter(
            'ticks_per_revolution',
            1800.0
        )

        self.ticks_per_rev = (
            self.get_parameter('ticks_per_revolution')
            .get_parameter_value()
            .double_value
        )

        # ====================================================
        # STATE
        # ====================================================

        self.last_ticks = None
        self.last_time = None

        # ====================================================
        # SUBSCRIBER
        # ====================================================

        self.encoder_sub = self.create_subscription(
            Int64MultiArray,
            '/wheel_encoder_ticks',
            self.encoder_callback,
            10
        )

        # ====================================================
        # PUBLISHERS
        # ====================================================

        self.rpm_pub = self.create_publisher(
            Float32MultiArray,
            '/wheel_rpm',
            10
        )

        self.rad_pub = self.create_publisher(
            Float32MultiArray,
            '/wheel_rad_s',
            10
        )

        self.get_logger().info(
            f'Encoder velocity ready | TPR = {self.ticks_per_rev}'
        )

    def encoder_callback(self, msg):

        if len(msg.data) != 4:
            return

        current_ticks = list(msg.data)
        current_time = time.monotonic()

        # ครั้งแรกยังไม่มี delta
        if self.last_ticks is None:

            self.last_ticks = current_ticks
            self.last_time = current_time
            return

        dt = current_time - self.last_time

        if dt <= 0.0:
            return

        # ====================================================
        # DELTA TICKS
        # ====================================================

        delta_ticks = [
            current_ticks[i] - self.last_ticks[i]
            for i in range(4)
        ]

        # ====================================================
        # REVOLUTIONS / SECOND
        # ====================================================

        rev_per_sec = [
            (delta / self.ticks_per_rev) / dt
            for delta in delta_ticks
        ]

        # ====================================================
        # RPM
        # ====================================================

        rpm = [
            rps * 60.0
            for rps in rev_per_sec
        ]

        # ====================================================
        # RAD/S
        # ====================================================

        rad_s = [
            rps * 2.0 * math.pi
            for rps in rev_per_sec
        ]

        # ====================================================
        # PUBLISH
        # ====================================================

        rpm_msg = Float32MultiArray()
        rpm_msg.data = rpm

        rad_msg = Float32MultiArray()
        rad_msg.data = rad_s

        self.rpm_pub.publish(rpm_msg)
        self.rad_pub.publish(rad_msg)

        # update state
        self.last_ticks = current_ticks
        self.last_time = current_time


def main(args=None):

    rclpy.init(args=args)

    node = EncoderVelocity()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
