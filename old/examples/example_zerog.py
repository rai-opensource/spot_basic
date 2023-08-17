# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse

import rclpy
from spatialmath import SE3, SpatialForce
from spatialmath.spatialvector import SpatialVector
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo starting ZeroG mode.

Use:
    0) have a spot with an arm
    1) start spot driver
    2) make sure there isn't anything within a meter of spot
    3) python relative/path/to/this/script/here robot-name-here
    4) spot will unstow the arm, then be in zero-g mode (low joint stiffness)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("spot_zerog_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())

    spot.initialize_robot()

    spot.stand()

    spot.ready_arm()

    # Send a single Impedance command to set the stiffness
    root_t_task = SE3.Tz(0.3)
    stationary_traj = [(SE3(), 1.0)]

    print("Sending impedance command to set low stiffness -- should not move robot")
    spot.move_end_effector_impedance(
        root_frame_name=spot.FrameHint.BODY,
        root_t_task=root_t_task,
        wrist_t_tool=SE3(),
        trajectory=stationary_traj,
        stiffness=SpatialVector([0, 0, 0, 0, 0, 0]),
        damping=SpatialVector([2.5, 2.5, 2.5, 1.0, 1.0, 1.0]),
        wrench_nm=SpatialForce([0, 0, 0, 0, 0, 0]),
        timeout_sec=2.0,
    )

    rclpy.spin(node)

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
