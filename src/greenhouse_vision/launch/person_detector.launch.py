#!/usr/bin/env python3

import os

from ament_index_python.packages import (
    get_package_share_directory,
)

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        "greenhouse_vision"
    )

    config_file = os.path.join(
        package_share,
        "config",
        "person_detector.yaml",
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            "OMP_NUM_THREADS",
            "3",
        ),
        SetEnvironmentVariable(
            "OMP_WAIT_POLICY",
            "PASSIVE",
        ),
        SetEnvironmentVariable(
            "GOMP_CPU_AFFINITY",
            "1 2 3",
        ),
        SetEnvironmentVariable(
            "OPENBLAS_NUM_THREADS",
            "1",
        ),
        SetEnvironmentVariable(
            "MKL_NUM_THREADS",
            "1",
        ),
        SetEnvironmentVariable(
            "NUMEXPR_NUM_THREADS",
            "1",
        ),

        Node(
            package="greenhouse_vision",
            executable="person_detector",
            name="person_detector",
            output="screen",
            emulate_tty=True,
            parameters=[config_file],
        ),
    ])
