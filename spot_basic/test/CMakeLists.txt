# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
########################### C++ #################################
find_package(ament_cmake_pytest REQUIRED)

########################### Python #################################
ament_add_pytest_test(
    spot_basic_pytest ${CMAKE_CURRENT_SOURCE_DIR}
    # Set the timeout to 12 minutes (720 seconds). The tests have been recently
    # measured to complete within 10 minutes on a normal CI run. When they take
    # longer, it's been because they are hung, and extending the timeout will
    # not actually help them complete.
    TIMEOUT 720
)
