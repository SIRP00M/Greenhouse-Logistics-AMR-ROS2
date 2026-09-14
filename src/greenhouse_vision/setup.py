from glob import glob
import os

from setuptools import find_packages, setup


package_name = "greenhouse_vision"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(
        exclude=["test"],
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join(
                "share",
                package_name,
                "config",
            ),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="poom",
    maintainer_email="poom@todo.todo",
    description=(
        "YOLO11n NCNN person detection "
        "for greenhouse AMR."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            (
                "person_detector = "
                "greenhouse_vision.person_detector_node:main",
            "person_follower = "
                "greenhouse_vision.person_follower_node:main"
            ),
        ],
    },
)
