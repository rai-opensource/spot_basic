# Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Build with: colcon build --packages-select spot_basic
cmake_minimum_required(VERSION 3.22)
project(spot_basic)

find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

########################### Python #################################
ament_python_install_package(spot_basic
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif(BUILD_TESTING)

install(
  DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
