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

        # ระยะที่ต้องการ ใช้ area ratio จาก bounding box
        self.declare_parameter("target_area_ratio", 0.12)
        self.declare_parameter("area_deadband", 0.015)

        # horizontal tracking
        self.declare_parameter("horizontal_deadband", 0.07)

        # ต้องอยู่ใกล้กลางภาพแค่ไหนถึงอนุญาตให้เดินหน้า
        self.declare_parameter("forward_enable_error", 0.20)

        # controller gains
        self.declare_parameter("kp_linear", 4.0)
        self.declare_parameter("kp_angular", 1.5)

        # สำคัญ:
        # จากการทดสอบจริง ฐานเริ่มขยับประมาณ 0.20 m/s
        self.declare_parameter("min_linear", 0.20)
        self.declare_parameter("max_linear", 0.25)

        self.declare_parameter("max_angular", 0.80)

        # target safety
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("detection_timeout", 0.40)

        # controller
        self.declare_parameter("control_rate", 20.0)

        # smoothing
        self.declare_parameter("filter_alpha", 0.35)

        # ====================================================
        # GET PARAMETERS
        # ====================================================

        self.target_area_ratio = float(
            self.get_parameter("target_area_ratio").value
        )

        self.area_deadband = float(
            self.get_parameter("area_deadband").value
        )

        self.horizontal_deadband = float(
            self.get_parameter("horizontal_deadband").value
        )

        self.forward_enable_error = float(
            self.get_parameter("forward_enable_error").value
        )

        self.kp_linear = float(
            self.get_parameter("kp_linear").value
        )

        self.kp_angular = float(
            self.get_parameter("kp_angular").value
        )

        self.min_linear = float(
            self.get_parameter("min_linear").value
        )

        self.max_linear = float(
            self.get_parameter("max_linear").value
        )

        self.max_angular = float(
            self.get_parameter("max_angular").value
        )

        self.min_confidence = float(
            self.get_parameter("min_confidence").value
        )

        self.detection_timeout = float(
            self.get_parameter("detection_timeout").value
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

        self.detected = False

        self.confidence = 0.0
        self.horizontal_error = 0.0
        self.area_ratio = 0.0

        self.filtered_x = 0.0
        self.filtered_area = 0.0

        self.filter_initialized = False

        self.last_detection_time = 0.0
        self.last_log_time = 0.0

        # ====================================================
        # ROS
        # ====================================================

        self.subscription = self.create_subscription(
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

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop,
        )

        self.get_logger().info(
            "===================================="
        )
        self.get_logger().info(
            " PERSON FOLLOWER READY"
        )
        self.get_logger().info(
            f" target_area = {self.target_area_ratio:.3f}"
        )
        self.get_logger().info(
            f" forward speed = "
            f"{self.min_linear:.2f} - "
            f"{self.max_linear:.2f} m/s"
        )
        self.get_logger().info(
            "===================================="
        )

    # ========================================================
    # UTIL
    # ========================================================

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(
            minimum,
            min(maximum, value),
        )

    # ========================================================
    # DETECTION
    # ========================================================

    def detection_callback(self, msg):

        self.last_detection_time = time.monotonic()

        if (
            not msg.detected
            or msg.confidence < self.min_confidence
        ):
            self.detected = False
            return

        self.detected = True

        self.confidence = float(
            msg.confidence
        )

        self.horizontal_error = float(
            msg.horizontal_error
        )

        self.area_ratio = float(
            msg.area_ratio
        )

        # ----------------------------------------------------
        # EMA smoothing
        # ----------------------------------------------------

        if not self.filter_initialized:

            self.filtered_x = (
                self.horizontal_error
            )

            self.filtered_area = (
                self.area_ratio
            )

            self.filter_initialized = True

        else:

            a = self.filter_alpha

            self.filtered_x = (
                a * self.horizontal_error
                +
                (1.0 - a) * self.filtered_x
            )

            self.filtered_area = (
                a * self.area_ratio
                +
                (1.0 - a) * self.filtered_area
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0

        self.cmd_pub.publish(msg)

    # ========================================================
    # CONTROL
    # ========================================================

    def control_loop(self):

        now = time.monotonic()

        # ----------------------------------------------------
        # FAILSAFE
        # ----------------------------------------------------

        if self.last_detection_time == 0.0:
            self.stop()
            return

        age = (
            now - self.last_detection_time
        )

        if (
            not self.detected
            or age > self.detection_timeout
        ):

            self.stop()

            if now - self.last_log_time > 1.0:

                self.get_logger().info(
                    "NO PERSON -> STOP"
                )

                self.last_log_time = now

            return

        # ====================================================
        # ANGULAR
        # ====================================================

        x_error = self.filtered_x

        if (
            abs(x_error)
            <= self.horizontal_deadband
        ):

            angular = 0.0

        else:

            # left image = negative
            # +angular.z = turn left

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
        # FORWARD DISTANCE
        # ====================================================

        area_error = (
            self.target_area_ratio
            -
            self.filtered_area
        )

        linear = 0.0
        state = "HOLD"

        # ----------------------------------------------------
        # Person far enough -> follow
        # ----------------------------------------------------

        if (
            area_error > self.area_deadband
        ):

            # ------------------------------------------------
            # First align robot with person
            # ------------------------------------------------

            if (
                abs(x_error)
                <= self.forward_enable_error
            ):

                raw_linear = (
                    self.kp_linear
                    * area_error
                )

                # IMPORTANT:
                # force command above physical motor deadzone

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

                # Person not centered enough.
                # Turn first, don't charge sideways.

                linear = 0.0
                state = "TURN"

        # ----------------------------------------------------
        # Person at / closer than target distance
        # ----------------------------------------------------

        else:

            linear = 0.0
            state = "DISTANCE OK"

        # ====================================================
        # PUBLISH
        # ====================================================

        cmd = Twist()

        cmd.linear.x = float(linear)
        cmd.linear.y = 0.0
        cmd.angular.z = float(angular)

        self.cmd_pub.publish(cmd)

        # ====================================================
        # LOG
        # ====================================================

        if (
            now - self.last_log_time
            >= 0.5
        ):

            self.get_logger().info(
                f"{state:11s} | "
                f"x={x_error:+.3f} | "
                f"area={self.filtered_area:.3f} | "
                f"target={self.target_area_ratio:.3f} | "
                f"v={linear:.2f} | "
                f"w={angular:+.2f}"
            )

            self.last_log_time = now

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(self):

        self.stop()

        return super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PersonFollowerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
