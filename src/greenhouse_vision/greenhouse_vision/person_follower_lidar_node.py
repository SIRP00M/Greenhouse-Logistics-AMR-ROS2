#!/usr/bin/env python3

import math
import statistics
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from greenhouse_vision_msgs.msg import PersonDetection


class PersonFollowerLidar(Node):

    def __init__(self):
        super().__init__("person_follower_lidar")

        # ====================================================
        # FOLLOW DISTANCE
        # ====================================================

        # ระยะที่อยากให้หุ่นรักษาจากคน
        self.declare_parameter(
            "target_distance",
            1.5,
        )

        # +/- ระยะที่ยอมให้ถือว่าโอเค
        self.declare_parameter(
            "distance_deadband",
            0.20,
        )

        # ====================================================
        # CAMERA -> LIDAR ALIGNMENT
        # ====================================================

        # C922 HFOV starting value
        self.declare_parameter(
            "camera_hfov_deg",
            70.0,
        )

        # LiDAR ของเราหน้า-หลังกลับ
        # ดังนั้น default = 180 degrees
        self.declare_parameter(
            "lidar_angle_offset_deg",
            180.0,
        )

        # sector รอบ target ที่เอา LiDAR มาหาค่า distance
        self.declare_parameter(
            "target_sector_deg",
            6.0,
        )

        # ====================================================
        # CAMERA TRACKING
        # ====================================================

        self.declare_parameter(
            "horizontal_deadband",
            0.07,
        )

        # คนต้องอยู่ใกล้กลางภาพพอ ถึงจะเดินหน้า
        self.declare_parameter(
            "forward_enable_error",
            0.25,
        )

        self.declare_parameter(
            "kp_angular",
            1.5,
        )

        self.declare_parameter(
            "max_angular",
            0.80,
        )

        # ====================================================
        # DISTANCE CONTROL
        # ====================================================

        self.declare_parameter(
            "kp_distance",
            0.35,
        )

        # จากที่เราทดสอบ:
        # 0.05 m/s ไม่ขยับ
        # ~0.20 m/s เริ่มขยับ
        self.declare_parameter(
            "min_linear",
            0.20,
        )

        self.declare_parameter(
            "max_linear",
            0.30,
        )

        # ====================================================
        # SAFETY
        # ====================================================

        # วัตถุด้านหน้าต่ำกว่านี้ -> STOP
        self.declare_parameter(
            "hard_stop_distance",
            0.45,
        )

        # +/- องศาด้านหน้าหุ่น
        self.declare_parameter(
            "front_safety_deg",
            25.0,
        )

        self.declare_parameter(
            "min_confidence",
            0.35,
        )

        self.declare_parameter(
            "detection_timeout",
            0.50,
        )

        self.declare_parameter(
            "scan_timeout",
            0.40,
        )

        self.declare_parameter(
            "control_rate",
            20.0,
        )

        # ====================================================
        # READ PARAMETERS
        # ====================================================

        self.target_distance = float(
            self.get_parameter(
                "target_distance"
            ).value
        )

        self.distance_deadband = float(
            self.get_parameter(
                "distance_deadband"
            ).value
        )

        self.camera_hfov = math.radians(
            float(
                self.get_parameter(
                    "camera_hfov_deg"
                ).value
            )
        )

        self.lidar_offset = math.radians(
            float(
                self.get_parameter(
                    "lidar_angle_offset_deg"
                ).value
            )
        )

        self.target_sector = math.radians(
            float(
                self.get_parameter(
                    "target_sector_deg"
                ).value
            )
        )

        self.horizontal_deadband = float(
            self.get_parameter(
                "horizontal_deadband"
            ).value
        )

        self.forward_enable_error = float(
            self.get_parameter(
                "forward_enable_error"
            ).value
        )

        self.kp_angular = float(
            self.get_parameter(
                "kp_angular"
            ).value
        )

        self.max_angular = float(
            self.get_parameter(
                "max_angular"
            ).value
        )

        self.kp_distance = float(
            self.get_parameter(
                "kp_distance"
            ).value
        )

        self.min_linear = float(
            self.get_parameter(
                "min_linear"
            ).value
        )

        self.max_linear = float(
            self.get_parameter(
                "max_linear"
            ).value
        )

        self.hard_stop_distance = float(
            self.get_parameter(
                "hard_stop_distance"
            ).value
        )

        self.front_safety_sector = math.radians(
            float(
                self.get_parameter(
                    "front_safety_deg"
                ).value
            )
        )

        self.min_confidence = float(
            self.get_parameter(
                "min_confidence"
            ).value
        )

        self.detection_timeout = float(
            self.get_parameter(
                "detection_timeout"
            ).value
        )

        self.scan_timeout = float(
            self.get_parameter(
                "scan_timeout"
            ).value
        )

        self.control_rate = float(
            self.get_parameter(
                "control_rate"
            ).value
        )

        # ====================================================
        # STATE
        # ====================================================

        self.person_detected = False

        self.confidence = 0.0
        self.horizontal_error = 0.0

        self.last_detection_time = 0.0

        self.scan = None
        self.last_scan_time = 0.0

        self.last_log_time = 0.0

        # ====================================================
        # ROS
        # ====================================================

        self.person_sub = self.create_subscription(
            PersonDetection,
            "/person_detection",
            self.person_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.distance_pub = self.create_publisher(
            Float32,
            "/person_distance",
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop,
        )

        # ====================================================
        # STARTUP LOG
        # ====================================================

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            " CAMERA + LIDAR PERSON FOLLOWER READY"
        )

        self.get_logger().info(
            f" Target distance : "
            f"{self.target_distance:.2f} m"
        )

        self.get_logger().info(
            f" Camera HFOV     : "
            f"{math.degrees(self.camera_hfov):.1f} deg"
        )

        self.get_logger().info(
            f" LiDAR offset    : "
            f"{math.degrees(self.lidar_offset):.1f} deg"
        )

        self.get_logger().info(
            f" Linear speed    : "
            f"{self.min_linear:.2f} - "
            f"{self.max_linear:.2f} m/s"
        )

        self.get_logger().info(
            "=========================================="
        )

    # ========================================================
    # PERSON CALLBACK
    # ========================================================

    def person_callback(self, msg):

        self.last_detection_time = time.monotonic()

        if (
            not msg.detected
            or msg.confidence < self.min_confidence
        ):
            self.person_detected = False
            return

        self.person_detected = True

        self.confidence = float(
            msg.confidence
        )

        self.horizontal_error = float(
            msg.horizontal_error
        )

    # ========================================================
    # LIDAR CALLBACK
    # ========================================================

    def scan_callback(self, msg):

        self.scan = msg

        self.last_scan_time = (
            time.monotonic()
        )

    # ========================================================
    # UTIL
    # ========================================================

    @staticmethod
    def clamp(
        value,
        minimum,
        maximum,
    ):

        return max(
            minimum,
            min(maximum, value),
        )

    @staticmethod
    def angle_difference(
        angle_a,
        angle_b,
    ):

        return math.atan2(
            math.sin(
                angle_a - angle_b
            ),
            math.cos(
                angle_a - angle_b
            ),
        )

    # ========================================================
    # VALID LIDAR RANGE
    # ========================================================

    def valid_range(self, value):

        if self.scan is None:
            return False

        if not math.isfinite(value):
            return False

        minimum = max(
            float(self.scan.range_min),
            0.10,
        )

        maximum = min(
            float(self.scan.range_max),
            8.0,
        )

        return (
            minimum
            <= value
            <= maximum
        )

    # ========================================================
    # GET RANGES IN SECTOR
    # ========================================================

    def ranges_in_sector(
        self,
        center_angle,
        half_width,
    ):

        if self.scan is None:
            return []

        values = []

        angle = (
            self.scan.angle_min
        )

        for distance in self.scan.ranges:

            difference = abs(
                self.angle_difference(
                    angle,
                    center_angle,
                )
            )

            if (
                difference <= half_width
                and self.valid_range(distance)
            ):

                values.append(
                    float(distance)
                )

            angle += (
                self.scan.angle_increment
            )

        return values

    # ========================================================
    # CAMERA ERROR -> LIDAR TARGET ANGLE
    # ========================================================

    def get_target_lidar_angle(self):

        # Detector:
        #
        # left image  = negative
        # right image = positive
        #
        # ROS LaserScan:
        #
        # + angle = CCW / left
        #
        # ดังนั้นต้องกลับ sign
        #
        # จากนั้น + 180deg เพราะ LiDAR
        # mount กลับหน้า-หลัง

        camera_angle = (
            -self.horizontal_error
            * (self.camera_hfov / 2.0)
        )

        lidar_angle = (
            camera_angle
            + self.lidar_offset
        )

        # Normalize -pi .. +pi
        lidar_angle = math.atan2(
            math.sin(lidar_angle),
            math.cos(lidar_angle),
        )

        return lidar_angle

    # ========================================================
    # PERSON DISTANCE FROM LIDAR
    # ========================================================

    def get_person_distance(self):

        target_angle = (
            self.get_target_lidar_angle()
        )

        values = self.ranges_in_sector(
            target_angle,
            self.target_sector,
        )

        if len(values) < 2:
            return None

        # เรียงจากใกล้ -> ไกล
        values.sort()

        # ใช้กลุ่มจุดใกล้สุด
        # เพื่อไม่เอากำแพงด้านหลัง person มาปน
        nearest_count = min(
            7,
            len(values),
        )

        nearest_group = (
            values[:nearest_count]
        )

        distance = (
            statistics.median(
                nearest_group
            )
        )

        return float(distance)

    # ========================================================
    # FRONT OBSTACLE DISTANCE
    # ========================================================

    def get_front_obstacle_distance(self):

        # IMPORTANT:
        #
        # LiDAR mount กลับ 180 deg
        #
        # ดังนั้น "หน้าหุ่น"
        # ไม่ใช่ LaserScan angle 0
        #
        # แต่คือ self.lidar_offset

        values = self.ranges_in_sector(
            self.lidar_offset,
            self.front_safety_sector,
        )

        if not values:
            return None

        return min(values)

    # ========================================================
    # STOP
    # ========================================================

    def publish_stop(self):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    # ========================================================
    # PUBLISH COMMAND
    # ========================================================

    def publish_command(
        self,
        linear,
        angular,
    ):

        cmd = Twist()

        cmd.linear.x = float(linear)
        cmd.linear.y = 0.0
        cmd.angular.z = float(angular)

        self.cmd_pub.publish(cmd)

    # ========================================================
    # MAIN CONTROL LOOP
    # ========================================================

    def control_loop(self):

        now = time.monotonic()

        # ====================================================
        # CAMERA FAILSAFE
        # ====================================================

        if (
            not self.person_detected
            or
            self.last_detection_time == 0.0
            or
            now - self.last_detection_time
            > self.detection_timeout
        ):

            self.publish_stop()

            self.log_state(
                "NO PERSON",
                None,
                0.0,
                0.0,
            )

            return

        # ====================================================
        # LIDAR FAILSAFE
        # ====================================================

        if (
            self.scan is None
            or
            self.last_scan_time == 0.0
            or
            now - self.last_scan_time
            > self.scan_timeout
        ):

            self.publish_stop()

            self.log_state(
                "NO LIDAR",
                None,
                0.0,
                0.0,
            )

            return

        # ====================================================
        # FRONT SAFETY
        # ====================================================

        front_distance = (
            self.get_front_obstacle_distance()
        )

        if (
            front_distance is not None
            and
            front_distance
            < self.hard_stop_distance
        ):

            self.publish_stop()

            self.log_state(
                "HARD STOP",
                front_distance,
                0.0,
                0.0,
            )

            return

        # ====================================================
        # CAMERA ANGULAR CONTROL
        # ====================================================

        x_error = (
            self.horizontal_error
        )

        if (
            abs(x_error)
            <= self.horizontal_deadband
        ):

            angular = 0.0

        else:

            angular = (
                -self.kp_angular
                * x_error
            )

        angular = self.clamp(
            angular,
            -self.max_angular,
            self.max_angular,
        )

        # ====================================================
        # GET PERSON DISTANCE
        # ====================================================

        person_distance = (
            self.get_person_distance()
        )

        # LiDAR หา target ไม่เจอ
        if person_distance is None:

            # ยังใช้ camera หมุนหาคนได้
            # แต่ไม่อนุญาตให้เดินหน้า

            linear = 0.0

            self.publish_command(
                linear,
                angular,
            )

            self.log_state(
                "SEARCH RANGE",
                None,
                linear,
                angular,
            )

            return

        # ====================================================
        # PUBLISH PERSON DISTANCE
        # ====================================================

        distance_msg = Float32()

        distance_msg.data = float(
            person_distance
        )

        self.distance_pub.publish(
            distance_msg
        )

        # ====================================================
        # DISTANCE CONTROL
        # ====================================================

        distance_error = (
            person_distance
            - self.target_distance
        )

        linear = 0.0

        state = "DIST OK"

        # ----------------------------------------------------
        # Person too far -> follow
        # ----------------------------------------------------

        if (
            distance_error
            > self.distance_deadband
        ):

            # คนต้องอยู่ใกล้กลางภาพก่อน
            if (
                abs(x_error)
                <= self.forward_enable_error
            ):

                raw_linear = (
                    self.kp_distance
                    * distance_error
                )

                # compensate physical motor deadzone
                linear = max(
                    self.min_linear,
                    raw_linear,
                )

                linear = min(
                    linear,
                    self.max_linear,
                )

                state = "FOLLOW"

            else:

                # คนอยู่ข้างมาก
                # หมุนก่อน ไม่เดินหน้า

                linear = 0.0

                state = "TURN"

        # ----------------------------------------------------
        # At target / too close
        # ----------------------------------------------------

        else:

            linear = 0.0

            state = "DIST OK"

        # ====================================================
        # SEND CMD_VEL
        # ====================================================

        self.publish_command(
            linear,
            angular,
        )

        self.log_state(
            state,
            person_distance,
            linear,
            angular,
        )

    # ========================================================
    # LOG
    # ========================================================

    def log_state(
        self,
        state,
        distance,
        linear,
        angular,
    ):

        now = time.monotonic()

        if (
            now - self.last_log_time
            < 0.5
        ):
            return

        if distance is None:

            distance_text = "---"

        else:

            distance_text = (
                f"{distance:.2f}m"
            )

        target_angle_deg = math.degrees(
            self.get_target_lidar_angle()
        )

        self.get_logger().info(
            f"{state:12s} | "
            f"x={self.horizontal_error:+.3f} | "
            f"lidar_angle={target_angle_deg:+.1f}deg | "
            f"d={distance_text} | "
            f"v={linear:.2f} | "
            f"w={angular:+.2f}"
        )

        self.last_log_time = now

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(self):

        self.publish_stop()

        return super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PersonFollowerLidar()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.publish_stop()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()
