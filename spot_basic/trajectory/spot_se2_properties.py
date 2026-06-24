#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np

# Given Spot's characteristics, it's a bit coarse to try to use these values to estimate how long to spend on
# a pose change, but it makes sense as a provisional solution
MAX_SPEED_FORWARD = 1.6  # Spec sheet number
MAX_SPEED_SIDEWAYS = 1.0  # NOTE just a coarsely tuned choice
MAX_SPEED_ANGULAR = 1.0  # NOTE just a coarsely tuned choice.  0.8 used in bosdyn 'wasd.py'

MAX_SPEED_VEC = np.array(
    [
        MAX_SPEED_FORWARD,
        MAX_SPEED_SIDEWAYS,
        MAX_SPEED_ANGULAR,
    ]
)

# The preceding values are an attempt to represent how fast the robot will actually move.
# The followin values correspond to what the API will reject
COMMAND_LIMIT_SPEED_VEC = np.array([2.0, 2.0, 1.6])

BODY_LENGTH = 1.1
BODY_WIDTH = 0.5
BODY_SHAPE = [
    [BODY_LENGTH / 2, BODY_WIDTH / 2],
    [-BODY_LENGTH / 2, BODY_WIDTH / 2],
    [-BODY_LENGTH / 2, -BODY_WIDTH / 2],
    [BODY_LENGTH / 2, -BODY_WIDTH / 2],
]
