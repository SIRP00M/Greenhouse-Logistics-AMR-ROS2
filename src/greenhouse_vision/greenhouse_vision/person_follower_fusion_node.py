#!/usr/bin/env python3

import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

from greenhouse_vision_msgs.msg import PersonDetection


class PersonFollowerFusion(Node):

    def __init__(self):
        super().__init__("person_follower_fusion")

        # ====================================================
        # PERSON FOLLOWING
        # ====================================================

        self.declare_parameter("target_distance", 1.5)
        self.declare_parameter("distance_deadband", 0.20)

        self.declare_parameter("camera_hfov_deg", 70.0)

        # Our C1 is mounted 180 degrees opposite robot front
        self.declare_parameter("lidar_angle_offset_deg", 180.0)

        self.declare_parameter("target_sector_deg", 6.0)

        self.declare_parameter("horizontal_deadband", 0.07)
        self.declare_parameter("forward_enable_error", 0.28)

        self.declare_parameter("kp_angular", 1.5)
        self.declare_parameter("max_angular", 0.80)

        self.declare_parameter("kp_distance", 0.35)

        # Physical mecanum deadzone from our testing
        self.declare_parameter("min_linear", 0.20)
        self.declare_parameter("max_linear", 0.30)

        # ====================================================
        # OBSTACLE AVOIDANCE
        # ====================================================

        # Start avoiding before getting too close
        self.declare_parameter("avoid_trigger_distance", 0.80)

        # Hysteresis: don't quit avoidance immediately
        self.declare_parameter("avoid_release_distance", 1.00)

        # Side obstacle trigger
        # Distance from LiDAR to robot side
        # Set robot_half_width to actual chassis half-width.
        #
        # Example:
        # robot width = 0.40 m -> half width = 0.20 m
        # desired side clearance = 0.15 m
        #
        # trigger = 0.20 + 0.15 = 0.35 m
        self.declare_parameter("robot_half_width", 0.20)
        self.declare_parameter("side_clearance", 0.15)

        # If everything gets THIS close -> stop
        self.declare_parameter("emergency_stop_distance", 0.30)

        # Mecanum strafe command
        self.declare_parameter("avoid_strafe_speed", 0.20)

        # Slight turning while strafing
        self.declare_parameter("avoid_max_angular", 0.35)

        # ====================================================
        # PERSON SEARCH / REACQUIRE
        # ====================================================

        # Keep searching this long after camera loses person
        self.declare_parameter("search_timeout", 6.0)

        self.declare_parameter("search_angular", 0.45)

        # ====================================================
        # FAILSAFE
        # ====================================================

        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("detection_timeout", 0.50)
        self.declare_parameter("scan_timeout", 0.40)

        self.declare_parameter("control_rate", 20.0)

        # ====================================================
        # LOAD PARAMETERS
        # ====================================================

        self.target_distance = float(
            self.get_parameter("target_distance").value
        )

        self.distance_deadband = float(
            self.get_parameter("distance_deadband").value
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
            self.get_parameter("kp_angular").value
        )

        self.max_angular = float(
            self.get_parameter("max_angular").value
        )

        self.kp_distance = float(
            self.get_parameter("kp_distance").value
        )

        self.min_linear = float(
            self.get_parameter("min_linear").value
        )

        self.max_linear = float(
            self.get_parameter("max_linear").value
        )

        self.avoid_trigger = float(
            self.get_parameter(
                "avoid_trigger_distance"
            ).value
        )

        self.avoid_release = float(
            self.get_parameter(
                "avoid_release_distance"
            ).value
        )

        self.robot_half_width = float(
            self.get_parameter(
                "robot_half_width"
            ).value
        )

        self.side_clearance = float(
            self.get_parameter(
                "side_clearance"
            ).value
        )

        self.side_trigger = (
            self.robot_half_width
            + self.side_clearance
        )

        self.emergency_stop = float(
            self.get_parameter(
                "emergency_stop_distance"
            ).value
        )

        self.avoid_strafe = float(
            self.get_parameter(
                "avoid_strafe_speed"
            ).value
        )

        self.avoid_max_angular = float(
            self.get_parameter(
                "avoid_max_angular"
            ).value
        )

        self.search_timeout = float(
            self.get_parameter(
                "search_timeout"
            ).value
        )

        self.search_angular = float(
            self.get_parameter(
                "search_angular"
            ).value
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

        self.horizontal_error = 0.0
        self.confidence = 0.0

        self.last_detection_msg_time = 0.0
        self.last_person_seen_time = 0.0

        # Remember where the person disappeared.
        self.last_seen_error = 0.0

        # +1 -> turn left
        # -1 -> turn right
        self.search_direction = 1.0

        self.scan = None
        self.last_scan_time = 0.0

        # Avoidance state:
        # +1 = strafe LEFT
        # -1 = strafe RIGHT
        #  0 = no avoidance
        self.avoid_direction = 0

        self.last_log_time = 0.0

        # ====================================================
        # ROS
        # ====================================================

        self.create_subscription(
            PersonDetection,
            "/person_detection",
            self.person_callback,
            10,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
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

        self.state_pub = self.create_publisher(
            String,
            "/follow_state",
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop,
        )

        self.get_logger().info(
            "============================================="
        )
        self.get_logger().info(
            " PERSON FOLLOWER + LIDAR AVOIDANCE READY"
        )
        self.get_logger().info(
            f" target distance : {self.target_distance:.2f} m"
        )
        self.get_logger().info(
            f" lidar offset    : "
            f"{math.degrees(self.lidar_offset):.1f} deg"
        )
        self.get_logger().info(
            f" avoid trigger   : {self.avoid_trigger:.2f} m"
        )
        self.get_logger().info(
            "============================================="
        )

    # ========================================================
    # CALLBACKS
    # ========================================================

    def person_callback(self, msg):

        now = time.monotonic()

        self.last_detection_msg_time = now

        valid = (
            msg.detected
            and msg.confidence >= self.min_confidence
        )

        if not valid:
            self.person_detected = False
            return

        self.person_detected = True

        self.confidence = float(
            msg.confidence
        )

        self.horizontal_error = float(
            msg.horizontal_error
        )

        self.last_seen_error = (
            self.horizontal_error
        )

        self.last_person_seen_time = now

        # Detector:
        # left image  = negative -> robot must turn LEFT (+z)
        # right image = positive -> robot must turn RIGHT (-z)

        if self.horizontal_error < -0.03:
            self.search_direction = +1.0

        elif self.horizontal_error > 0.03:
            self.search_direction = -1.0

    def scan_callback(self, msg):

        self.scan = msg
        self.last_scan_time = time.monotonic()

    # ========================================================
    # UTIL
    # ========================================================

    @staticmethod
    def clamp(value, minimum, maximum):

        return max(
            minimum,
            min(maximum, value),
        )

    @staticmethod
    def angle_difference(a, b):

        return math.atan2(
            math.sin(a - b),
            math.cos(a - b),
        )

    def valid_range(self, value):

        if self.scan is None:
            return False

        if not math.isfinite(value):
            return False

        return (
            max(self.scan.range_min, 0.10)
            <= value
            <= min(self.scan.range_max, 8.0)
        )

    # ========================================================
    # ROBOT ANGLE -> RAW LIDAR ANGLE
    # ========================================================

    def robot_to_lidar_angle(
        self,
        robot_angle,
    ):

        angle = (
            robot_angle
            + self.lidar_offset
        )

        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    # ========================================================
    # SECTOR READ
    # ========================================================

    def ranges_in_robot_sector(
        self,
        robot_angle,
        half_width,
    ):

        if self.scan is None:
            return []

        lidar_center = (
            self.robot_to_lidar_angle(
                robot_angle
            )
        )

        values = []

        angle = self.scan.angle_min

        for distance in self.scan.ranges:

            diff = abs(
                self.angle_difference(
                    angle,
                    lidar_center,
                )
            )

            if (
                diff <= half_width
                and self.valid_range(distance)
            ):
                values.append(
                    float(distance)
                )

            angle += self.scan.angle_increment

        return values

    # ========================================================
    # ROBUST NEAREST DISTANCE
    # ========================================================

    def sector_distance(
        self,
        robot_angle_deg,
        half_width_deg,
    ):

        values = self.ranges_in_robot_sector(
            math.radians(robot_angle_deg),
            math.radians(half_width_deg),
        )

        # No returned points usually means open space.
        if not values:

            if self.scan is not None:
                return min(
                    float(self.scan.range_max),
                    8.0,
                )

            return 8.0

        values.sort()

        # Don't trust one single noisy point.
        nearest = values[
            :min(5, len(values))
        ]

        return float(
            statistics.median(nearest)
        )

    # ========================================================
    # OBSTACLE MAP
    # ========================================================

    def obstacle_distances(self):

        # Robot coordinates:
        #
        #           FRONT 0°
        #
        #      +40°        -40°
        #     FL              FR
        #
        # +90° LEFT       RIGHT -90°
        #

        front = self.sector_distance(
            0.0,
            18.0,
        )

        front_left = self.sector_distance(
            +40.0,
            18.0,
        )

        front_right = self.sector_distance(
            -40.0,
            18.0,
        )

        left = self.sector_distance(
            +90.0,
            22.0,
        )

        right = self.sector_distance(
            -90.0,
            22.0,
        )

        # Clearance estimate for choosing which side
        # to go around an obstacle.

        left_clearance = min(
            front_left,
            left,
        )

        right_clearance = min(
            front_right,
            right,
        )

        return {
            "front": front,
            "front_left": front_left,
            "front_right": front_right,
            "left": left,
            "right": right,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
        }

    # ========================================================
    # PERSON DISTANCE
    # ========================================================

    def get_person_distance(self):

        if self.scan is None:
            return None

        # horizontal_error:
        # left image = negative
        #
        # ROS angle:
        # left = positive
        #
        # therefore sign is inverted.

        camera_angle = (
            -self.horizontal_error
            * (self.camera_hfov / 2.0)
        )

        values = self.ranges_in_robot_sector(
            camera_angle,
            self.target_sector,
        )

        if len(values) < 2:
            return None

        values.sort()

        nearest_group = values[
            :min(7, len(values))
        ]

        return float(
            statistics.median(
                nearest_group
            )
        )

    # ========================================================
    # CAMERA ANGULAR CONTROLLER
    # ========================================================

    def tracking_angular(self):

        error = self.horizontal_error

        if (
            abs(error)
            <= self.horizontal_deadband
        ):
            return 0.0

        angular = (
            -self.kp_angular
            * error
        )

        return self.clamp(
            angular,
            -self.max_angular,
            self.max_angular,
        )

    # ========================================================
    # COMMAND
    # ========================================================

    def publish_command(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
    ):

        cmd = Twist()

        cmd.linear.x = float(x)
        cmd.linear.y = float(y)
        cmd.angular.z = float(z)

        self.cmd_pub.publish(cmd)

    def stop(self):

        self.publish_command()

    # ========================================================
    # STATE
    # ========================================================

    def publish_state(self, state):

        msg = String()
        msg.data = state

        self.state_pub.publish(msg)

    # ========================================================
    # OBSTACLE AVOIDANCE DECISION
    # ========================================================

    def update_avoidance(
        self,
        obs,
    ):

        front = obs["front"]

        left_clear = (
            obs["left_clearance"]
        )

        right_clear = (
            obs["right_clearance"]
        )

        left = obs["left"]
        right = obs["right"]

        # ----------------------------------------------------
        # Existing avoidance:
        # don't instantly change direction.
        # ----------------------------------------------------

        if self.avoid_direction != 0:

            if (
                front > self.avoid_release
                and left > self.side_trigger
                and right > self.side_trigger
            ):

                self.avoid_direction = 0

            else:
                return

        # ----------------------------------------------------
        # LEFT too close -> STRAFE RIGHT
        #
        # ROS +Y = left
        # ROS -Y = right
        # ----------------------------------------------------

        if (
            left < self.side_trigger
            and right > left + 0.10
        ):

            self.avoid_direction = -1
            return

        # ----------------------------------------------------
        # RIGHT too close -> STRAFE LEFT
        # ----------------------------------------------------

        if (
            right < self.side_trigger
            and left > right + 0.10
        ):

            self.avoid_direction = +1
            return

        # ----------------------------------------------------
        # FRONT blocked -> choose clearer side
        # ----------------------------------------------------

        if front < self.avoid_trigger:

            if left_clear > right_clear:
                self.avoid_direction = +1
            else:
                self.avoid_direction = -1

    # ========================================================
    # OBSTACLE AVOIDANCE CONTROL
    # ========================================================

    def do_avoidance(
        self,
        obs,
    ):

        if self.avoid_direction == 0:
            return False

        front = obs["front"]

        left_clear = (
            obs["left_clearance"]
        )

        right_clear = (
            obs["right_clearance"]
        )

        # Direction we want to strafe into
        desired_clearance = (
            left_clear
            if self.avoid_direction > 0
            else right_clear
        )

        # If everything is packed around us,
        # don't blindly ram sideways.

        if (
            front < self.emergency_stop
            and desired_clearance
            < self.emergency_stop
        ):

            self.stop()

            self.publish_state(
                "EMERGENCY_STOP"
            )

            self.log(
                "EMERGENCY",
                obs,
                None,
                0.0,
                0.0,
                0.0,
            )

            return True

        # Mecanum:
        # +Y = left
        # -Y = right

        strafe = (
            self.avoid_strafe
            * self.avoid_direction
        )

        # While avoiding we can still gently
        # keep camera pointed at the person.

        angular = 0.0

        if self.person_detected:

            angular = self.clamp(
                self.tracking_angular(),
                -self.avoid_max_angular,
                self.avoid_max_angular,
            )

        state = (
            "AVOID_LEFT"
            if self.avoid_direction > 0
            else "AVOID_RIGHT"
        )

        self.publish_command(
            x=0.0,
            y=strafe,
            z=angular,
        )

        self.publish_state(state)

        self.log(
            state,
            obs,
            None,
            0.0,
            strafe,
            angular,
        )

        return True

    # ========================================================
    # PERSON SEARCH
    # ========================================================

    def search_person(
        self,
        obs,
    ):

        now = time.monotonic()

        if self.last_person_seen_time == 0.0:

            self.stop()
            self.publish_state("NO_PERSON")
            return

        lost_for = (
            now - self.last_person_seen_time
        )

        if lost_for > self.search_timeout:

            self.stop()

            self.publish_state(
                "SEARCH_TIMEOUT"
            )

            self.log(
                "SEARCH TIMEOUT",
                obs,
                None,
                0.0,
                0.0,
                0.0,
            )

            return

        # ----------------------------------------------------
        # If too close to one side while searching,
        # strafe away before rotating.
        # ----------------------------------------------------

        if obs["left"] < self.emergency_stop:

            self.publish_command(
                y=-self.avoid_strafe,
            )

            self.publish_state(
                "SEARCH_ESCAPE_RIGHT"
            )

            return

        if obs["right"] < self.emergency_stop:

            self.publish_command(
                y=+self.avoid_strafe,
            )

            self.publish_state(
                "SEARCH_ESCAPE_LEFT"
            )

            return

        # ----------------------------------------------------
        # Turn toward last known person direction.
        #
        # If they walked behind the robot, continue
        # rotating until camera sees them again.
        # ----------------------------------------------------

        angular = (
            self.search_direction
            * self.search_angular
        )

        self.publish_command(
            z=angular,
        )

        self.publish_state(
            "SEARCH_PERSON"
        )

        self.log(
            "SEARCH",
            obs,
            None,
            0.0,
            0.0,
            angular,
        )

    # ========================================================
    # NORMAL PERSON FOLLOWING
    # ========================================================

    def follow_person(
        self,
        obs,
    ):

        angular = (
            self.tracking_angular()
        )

        distance = (
            self.get_person_distance()
        )

        if distance is None:

            # Keep aligning with camera,
            # but don't move forward blindly.

            self.publish_command(
                z=angular,
            )

            self.publish_state(
                "TRACK_NO_RANGE"
            )

            self.log(
                "NO RANGE",
                obs,
                None,
                0.0,
                0.0,
                angular,
            )

            return

        distance_msg = Float32()
        distance_msg.data = distance

        self.distance_pub.publish(
            distance_msg
        )

        error = (
            distance
            - self.target_distance
        )

        linear = 0.0

        if (
            error > self.distance_deadband
        ):

            # First point robot roughly toward person.

            if (
                abs(self.horizontal_error)
                <= self.forward_enable_error
            ):

                linear = (
                    self.kp_distance
                    * error
                )

                linear = max(
                    self.min_linear,
                    linear,
                )

                linear = min(
                    self.max_linear,
                    linear,
                )

                state = "FOLLOW"

            else:

                linear = 0.0
                state = "TURN_TO_PERSON"

        else:

            state = "DISTANCE_OK"

        self.publish_command(
            x=linear,
            y=0.0,
            z=angular,
        )

        self.publish_state(state)

        self.log(
            state,
            obs,
            distance,
            linear,
            0.0,
            angular,
        )

    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def control_loop(self):

        now = time.monotonic()

        # ====================================================
        # LIDAR FAILSAFE
        # ====================================================

        if (
            self.scan is None
            or
            now - self.last_scan_time
            > self.scan_timeout
        ):

            self.stop()

            self.publish_state(
                "NO_LIDAR"
            )

            return

        obs = self.obstacle_distances()

        # ====================================================
        # UPDATE LOCAL AVOIDANCE
        # ====================================================

        self.update_avoidance(obs)

        if self.do_avoidance(obs):
            return

        # ====================================================
        # CAMERA TARGET VALID?
        # ====================================================

        camera_alive = (
            self.last_detection_msg_time > 0.0
            and
            now - self.last_detection_msg_time
            <= self.detection_timeout
        )

        person_visible = (
            camera_alive
            and self.person_detected
        )

        # ====================================================
        # PERSON VISIBLE
        # ====================================================

        if person_visible:

            self.follow_person(obs)
            return

        # ====================================================
        # PERSON LOST
        # ====================================================

        self.search_person(obs)

    # ========================================================
    # LOG
    # ========================================================

    def log(
        self,
        state,
        obs,
        distance,
        vx,
        vy,
        wz,
    ):

        now = time.monotonic()

        if (
            now - self.last_log_time
            < 0.5
        ):
            return

        if distance is None:
            d_text = "---"
        else:
            d_text = f"{distance:.2f}m"

        self.get_logger().info(
            f"{state:14s} | "
            f"xerr={self.horizontal_error:+.2f} | "
            f"person={d_text} | "
            f"F={obs['front']:.2f} "
            f"L={obs['left']:.2f} "
            f"R={obs['right']:.2f} | "
            f"cmd=({vx:+.2f},"
            f"{vy:+.2f},"
            f"{wz:+.2f})"
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

    node = PersonFollowerFusion()

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
