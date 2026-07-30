cmake_minimum_required(VERSION 3.8)
project(greenhouse_vision)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate the custom PersonDetection message.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonDetection.msg"
)

# Install the Python ROS 2 nodes as executable scripts.
install(
  PROGRAMS
    greenhouse_vision/camera_node.py
    greenhouse_vision/person_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install ROS 2 launch files.
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
