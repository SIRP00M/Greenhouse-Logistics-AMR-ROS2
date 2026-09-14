#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from greenhouse_vision_msgs.msg import PersonDetection


class PersonFollowerNode(Node):

    def __init__(self):
        super().__init__("person_follower")

        # ====================================================
        # PARAMETERS
        # ====================================================

        # area_ratio ที่ต้องการเมื่อคนอยู่ระยะตามเป้าหมาย
        # ค่านี้ต้อง tune กับ C922 จริงภายหลัง
        self.declare_parameter("target_area_ratio", 0.12)

        # Deadband ป้องกันหุ่นส่ายไปมา
        self.declare_parameter("horizontal_deadband", 0.08)
        self.declare_parameter("area_deadband", 0.015)

        # Controller gain
        self.declare_parameter("kp_angular", 0.90)
        self.declare_parameter("kp_linear", 1.20)

        # Safety limits
        self.declare_parameter("max_linear", 0.15)
        self.declare_parameter("max_angular", 0.60)

        # ถ้าคนออกข้างมากกว่าเท่านี้
        # ให้หยุดเดินหน้าแล้วหมุนอย่างเดียว
        self.declare_parameter("rotate_only_error", 0.35)

        # detection หายเกินเวลานี้ -> STOP
        self.declare_parameter("detection_timeout", 0.40)

        # confidence ต่ำกว่านี้ไม่ตาม
        self.declare_parameter("min_confidence", 0.35)

        # controller frequency
        self.declare_parameter("control_rate", 20.0)

        # smoothing:
        # 1.0 = ไม่ smooth
        # ต่ำลง = smooth มากขึ้น
        self.declare_parameter("filter_alpha", 0.35)

        # ====================================================
        # GET PARAMETERS
        # ====================================================

        self.target_area_ratio = float(
            self.get_parameter("target_area_ratio").value
        )

        self.horizontal_deadband = float(
            self.get_parameter("horizontal_deadband").value
        )

        self.area_deadband = float(
            self.get_parameter("area_deadband").value
        )

        self.kp_angular = float(
            self.get_parameter("kp_angular").value
        )

        self.kp_linear = float(
            self.get_parameter("kp_linear").value
        )

        self.max_linear = float(
            self.get_parameter("max_linear").value
        )

        self.max_angular = float(
            self.get_parameter("max_angular").value
        )

        self.rotate_only_error = float(
            self.get_parameter("rotate_only_error").value
        )

        self.detection_timeout = float(
            self.get_parameter("detection_timeout").value
        )

        self.min_confidence = float(
            self.get_parameter("min_confidence").value
        )

        self.control_rate = float(
            self.get_parameter("control_rate").value
        )

        self.filter_alpha = float(
            self.get_parameter("filter_alpha").value
        )

        # ====================================================
        # STATE
        # ====================================================

        self.target_detected = False
        self.target_confidence = 0.0

        self.horizontal_error = 0.0
        self.area_ratio = 0.0

        self.filtered_horizontal = 0.0
        self.filtered_area = 0.0

        self.filter_initialized = False

        self.last_detection_time = 0.0
        self.last_log_time = 0.0

        # ====================================================
        # ROS
        # ====================================================

        self.detection_sub = self.create_subscription(
            PersonDetection,
            "/person_detection",
            self.detection_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop,
        )

        self.get_logger().info(
            "Person follower READY"
        )

        self.get_logger().info(
            f"target_area={self.target_area_ratio:.3f} | "
            f"max_linear={self.max_linear:.2f} m/s | "
            f"max_angular={self.max_angular:.2f} rad/s"
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(
            minimum,
            min(maximum, value),
        )

    # ========================================================
    # DETECTION CALLBACK
    # ========================================================

    def detection_callback(self, msg):

        self.last_detection_time = time.monotonic()

        if (
            not msg.detected
            or msg.confidence < self.min_confidence
        ):
            self.target_detected = False
            return

        self.target_detected = True
        self.target_confidence = float(msg.confidence)

        self.horizontal_error = float(
            msg.horizontal_error
        )

        self.area_ratio = float(
            msg.area_ratio
        )

        # ====================================================
        # EMA FILTER
        # ====================================================

        if not self.filter_initialized:

            self.filtered_horizontal = (
                self.horizontal_error
            )

            self.filtered_area = (
                self.area_ratio
            )

            self.filter_initialized = True

        else:

            a = self.filter_alpha

            self.filtered_horizontal = (
                a * self.horizontal_error
                +
                (1.0 - a)
                * self.filtered_horizontal
            )

            self.filtered_area = (
                a * self.area_ratio
                +
                (1.0 - a)
                * self.filtered_area
            )

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
    # CONTROL LOOP
    # ========================================================

    def control_loop(self):

        now = time.monotonic()

        # ====================================================
        # FAILSAFE
        # ====================================================

        detection_age = (
            now - self.last_detection_time
            if self.last_detection_time > 0.0
            else float("inf")
        )

        if (
            not self.target_detected
            or detection_age > self.detection_timeout
        ):
            self.publish_stop()

            if now - self.last_log_time >= 1.0:

                self.get_logger().info(
                    "NO TARGET -> STOP"
                )

                self.last_log_time = now

            return

        # ====================================================
        # ANGULAR CONTROL
        #
        # Detector:
        # left image  = negative
        # right image = positive
        #
        # ROS:
        # +angular.z = turn left
        #
        # therefore:
        # angular = -Kp * error
        # ====================================================

        x_error = self.filtered_horizontal

        if abs(x_error) <= self.horizontal_deadband:

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
        # DISTANCE CONTROL
        #
        # area small -> person far -> forward
        # area target -> stop
        # area too large -> STOP
        #
        # ตอนนี้ไม่ถอยหลังเอง
        # ====================================================

        area_error = (
            self.target_area_ratio
            -
            self.filtered_area
        )

        if area_error <= self.area_deadband:

            linear = 0.0

        else:

            linear = (
                self.kp_linear
                * area_error
            )

        linear = self.clamp(
            linear,
            0.0,
            self.max_linear,
        )

        # ====================================================
        # TURN FIRST
        #
        # ถ้าคนอยู่ข้างมาก
        # อย่าวิ่งเฉียงเข้าใส่
        # ====================================================

        if abs(x_error) >= self.rotate_only_error:

            linear = 0.0

        else:

            # ยิ่ง target ไม่อยู่กลาง
            # ยิ่งลดความเร็วเดินหน้า
            heading_scale = max(
                0.0,
                1.0 - abs(x_error),
            )

            linear *= heading_scale

        # ====================================================
        # PUBLISH CMD_VEL
        # ====================================================

        cmd = Twist()

        cmd.linear.x = float(linear)
        cmd.linear.y = 0.0
        cmd.angular.z = float(angular)

        self.cmd_pub.publish(cmd)

        # ====================================================
        # STATUS LOG
        # ====================================================

        if now - self.last_log_time >= 0.5:

            self.get_logger().info(
                f"conf={self.target_confidence:.2f} | "
                f"x={x_error:+.3f} | "
                f"area={self.filtered_area:.3f} | "
                f"cmd_v={linear:.3f} | "
                f"cmd_w={angular:+.3f}"
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

    node = PersonFollowerNode()

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
