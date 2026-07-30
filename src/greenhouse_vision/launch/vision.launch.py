#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_path = LaunchConfiguration("model_path")
    confidence_threshold = LaunchConfiguration("confidence_threshold")
    image_size = LaunchConfiguration("image_size")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="yolo11n.pt",
                description="Path to the Ultralytics YOLO model",
            ),

            DeclareLaunchArgument(
                "confidence_threshold",
                default_value="0.40",
                description="Minimum confidence for person detection",
            ),

            DeclareLaunchArgument(
                "image_size",
                default_value="640",
                description="YOLO inference image size",
            ),

            Node(
                package="greenhouse_vision",
                executable="camera_node.py",
                name="camera_node",
                output="screen",
                emulate_tty=True,
            ),

            Node(
                package="greenhouse_vision",
                executable="person_detector.py",
                name="person_detector",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "model_path": model_path,
                        "confidence_threshold": confidence_threshold,
                        "image_size": image_size,
                    }
                ],
            ),
        ]
    )
