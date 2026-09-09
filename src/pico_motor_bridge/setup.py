from setuptools import find_packages, setup

package_name = 'pico_motor_bridge'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='poom',
    maintainer_email='poom@example.com',

    description='ROS 2 USB serial bridge for Raspberry Pi Pico mecanum motor controller',

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'serial_bridge = pico_motor_bridge.serial_bridge:main',
        ],
    },
)
