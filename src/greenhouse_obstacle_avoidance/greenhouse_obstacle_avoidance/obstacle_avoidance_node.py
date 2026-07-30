#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance_node')

        # ระยะควบคุม
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('avoid_distance', 0.65)
        self.declare_parameter('side_distance', 0.30)

        # ความเร็ว
        self.declare_parameter('forward_speed', 0.12)
        self.declare_parameter('slow_speed', 0.06)
        self.declare_parameter('turn_speed', 0.55)

        # การตั้งค่า LiDAR
        self.declare_parameter('front_half_angle_deg', 20.0)
        self.declare_parameter('front_angle_offset_deg', 0.0)
        self.declare_parameter('scan_timeout', 0.70)
        self.declare_parameter('minimum_valid_points', 10)

        self.stop_distance = float(
            self.get_parameter('stop_distance').value
        )
        self.avoid_distance = float(
            self.get_parameter('avoid_distance').value
        )
        self.side_distance = float(
            self.get_parameter('side_distance').value
        )

        self.forward_speed = float(
            self.get_parameter('forward_speed').value
        )
        self.slow_speed = float(
            self.get_parameter('slow_speed').value
        )
        self.turn_speed = float(
            self.get_parameter('turn_speed').value
        )

        self.front_half_angle_deg = float(
            self.get_parameter('front_half_angle_deg').value
        )
        self.front_angle_offset_deg = float(
            self.get_parameter('front_angle_offset_deg').value
        )
        self.scan_timeout = float(
            self.get_parameter('scan_timeout').value
        )
        self.minimum_valid_points = int(
            self.get_parameter('minimum_valid_points').value
        )

        self.latest_scan = None
        self.last_scan_time = None
        self.previous_state = None

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10,
        )

        # ประมวลผล 10 ครั้งต่อวินาที
        self.control_timer = self.create_timer(
            0.10,
            self.control_loop,
        )

        self.get_logger().info(
            'Obstacle avoidance node started'
        )
        self.get_logger().info(
            f'Stop distance: {self.stop_distance:.2f} m'
        )
        self.get_logger().info(
            f'Avoid distance: {self.avoid_distance:.2f} m'
        )
        self.get_logger().info(
            f'Front angle offset: '
            f'{self.front_angle_offset_deg:.1f} degrees'
        )

    def scan_callback(self, scan: LaserScan):
        self.latest_scan = scan
        self.last_scan_time = self.get_clock().now()

    @staticmethod
    def wrap_angle_degrees(angle: float) -> float:
        """แปลงมุมให้อยู่ในช่วง -180 ถึง 180 องศา"""
        return (angle + 180.0) % 360.0 - 180.0

    def get_sector_distances(self, scan: LaserScan):
        front_distances = []
        left_distances = []
        right_distances = []

        valid_points = 0

        for index, distance in enumerate(scan.ranges):

            if not math.isfinite(distance):
                continue

            if distance < scan.range_min:
                continue

            if distance > scan.range_max:
                continue

            valid_points += 1

            angle_radians = (
                scan.angle_min
                + index * scan.angle_increment
            )

            angle_degrees = math.degrees(angle_radians)

            relative_angle = self.wrap_angle_degrees(
                angle_degrees - self.front_angle_offset_deg
            )

            # พื้นที่ด้านหน้า
            if abs(relative_angle) <= self.front_half_angle_deg:
                front_distances.append(distance)

            # พื้นที่ด้านซ้าย
            if 25.0 <= relative_angle <= 85.0:
                left_distances.append(distance)

            # พื้นที่ด้านขวา
            if -85.0 <= relative_angle <= -25.0:
                right_distances.append(distance)

        front = (
            min(front_distances)
            if front_distances
            else math.inf
        )

        left = (
            min(left_distances)
            if left_distances
            else math.inf
        )

        right = (
            min(right_distances)
            if right_distances
            else math.inf
        )

        return front, left, right, valid_points

    def choose_turn_direction(
        self,
        left_distance: float,
        right_distance: float,
    ) -> float:
        """
        angular.z เป็นบวก = เลี้ยวซ้าย
        angular.z เป็นลบ = เลี้ยวขวา
        """

        if left_distance >= right_distance:
            return self.turn_speed

        return -self.turn_speed

    def publish_velocity(
        self,
        linear_x: float,
        angular_z: float,
    ):
        command = Twist()

        command.linear.x = float(linear_x)
        command.angular.z = float(angular_z)

        self.cmd_vel_publisher.publish(command)

    def report_state(
        self,
        state: str,
        front: float = math.inf,
        left: float = math.inf,
        right: float = math.inf,
    ):
        if state == self.previous_state:
            return

        self.previous_state = state

        self.get_logger().info(
            f'State={state} | '
            f'front={front:.2f} m | '
            f'left={left:.2f} m | '
            f'right={right:.2f} m'
        )

    def stop_robot(self, state: str):
        self.publish_velocity(0.0, 0.0)
        self.report_state(state)

    def control_loop(self):
        # ยังไม่ได้รับข้อมูล LiDAR
        if self.latest_scan is None or self.last_scan_time is None:
            self.stop_robot('WAITING_FOR_SCAN')
            return

        current_time = self.get_clock().now()

        scan_age = (
            current_time - self.last_scan_time
        ).nanoseconds / 1_000_000_000.0

        # ข้อมูล LiDAR ขาดหาย
        if scan_age > self.scan_timeout:
            self.stop_robot('SCAN_TIMEOUT')
            return

        front, left, right, valid_points = (
            self.get_sector_distances(self.latest_scan)
        )

        # จำนวนข้อมูลถูกต้องน้อยเกินไป
        if valid_points < self.minimum_valid_points:
            self.stop_robot('INVALID_SCAN')
            return

        # ถูกล้อมหรือพื้นที่แคบมาก
        if (
            front <= self.stop_distance
            and left <= self.stop_distance
            and right <= self.stop_distance
        ):
            self.publish_velocity(0.0, 0.0)

            self.report_state(
                'BLOCKED',
                front,
                left,
                right,
            )
            return

        # วัตถุอยู่ใกล้ด้านหน้ามาก ให้หยุดเดินหน้าและหมุน
        if front <= self.stop_distance:
            turn_command = self.choose_turn_direction(
                left,
                right,
            )

            self.publish_velocity(
                0.0,
                turn_command,
            )

            self.report_state(
                'EMERGENCY_TURN',
                front,
                left,
                right,
            )
            return

        # เริ่มเข้าใกล้วัตถุ ให้เดินช้าและเลี้ยวหลบ
        if front <= self.avoid_distance:
            turn_command = self.choose_turn_direction(
                left,
                right,
            )

            self.publish_velocity(
                self.slow_speed,
                turn_command,
            )

            self.report_state(
                'AVOIDING_FRONT',
                front,
                left,
                right,
            )
            return

        # วัตถุใกล้ด้านซ้าย ให้เบี่ยงไปทางขวา
        if left <= self.side_distance:
            self.publish_velocity(
                self.slow_speed,
                -self.turn_speed * 0.60,
            )

            self.report_state(
                'AVOIDING_LEFT',
                front,
                left,
                right,
            )
            return

        # วัตถุใกล้ด้านขวา ให้เบี่ยงไปทางซ้าย
        if right <= self.side_distance:
            self.publish_velocity(
                self.slow_speed,
                self.turn_speed * 0.60,
            )

            self.report_state(
                'AVOIDING_RIGHT',
                front,
                left,
                right,
            )
            return

        # ด้านหน้าโล่ง
        self.publish_velocity(
            self.forward_speed,
            0.0,
        )

        self.report_state(
            'FORWARD',
            front,
            left,
            right,
        )

    def emergency_stop(self):
        for _ in range(3):
            self.publish_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)

    node = ObstacleAvoidanceNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            'Keyboard interrupt received'
        )

    finally:
        node.emergency_stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
