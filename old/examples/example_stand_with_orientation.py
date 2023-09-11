# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse

import rclpy
from bdai_ros2_wrappers.node import NodeWrapper
from spatialmath import SE3, SO3
from spot_utilities.spot_basic_util import create_spot_basic_with_tf

"""
This program makes the robot move to a sequence of
goal poses while standing-in-place, using stand_with_orientation.

To run this program:
   python example_stand_with_orientation.py <robot_name>

"""


def main() -> None:
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    options = parser.parse_args()

    node = NodeWrapper("reproduce_stand_with_orientation_yaw_problem_", spin_thread=True)
    spot = create_spot_basic_with_tf(options.robot, node)

    # Let's define the goal poses in body frame.
    body_t_goals = [
        SE3.Rt(R=SO3.RPY(0, 0, 0), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(0, 0, 45), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(0, 0, -90), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(0, 45, 0), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(0, -90, 0), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(45, 45, 0), t=[0, 0, 0]),
        SE3.Rt(R=SO3.RPY(-90, -45, 0), t=[0, 0, 0]),
    ]

    gravbody_t_body = spot.lookup_a_tform_b(
        spot.FrameHint.GRAV_ALIGNED_BODY.value, spot.FrameHint.BODY.value, wait_for_frames=True
    )
    # we only care about in-place rotations
    gravbody_r_routine = [SO3((gravbody_t_body * body_t_pose).R) for body_t_pose in body_t_goals]

    # this is the height of Spot's body relative to its normal height (0.0 means at normal height)
    body_height = 0.0

    for gravbody_r_orien in gravbody_r_routine:
        spot.stand_with_orientation(gravbody_r_orien, body_height)

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
