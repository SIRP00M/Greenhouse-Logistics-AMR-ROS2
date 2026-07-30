from setuptools import find_packages, setup


package_name = 'greenhouse_obstacle_avoidance'


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Poom',
    maintainer_email='poom@example.com',
    description='LiDAR-based obstacle avoidance for greenhouse AMR',
    license='MIT',
    entry_points={
        'console_scripts': [
            (
                'obstacle_avoidance = '
                'greenhouse_obstacle_avoidance.obstacle_avoidance_node:main'
            ),
        ],
    },
)
