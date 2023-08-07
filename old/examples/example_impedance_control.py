import argparse
import time

import numpy as np
import rclpy
from spatialmath import SE3, SpatialForce
from spatialmath.spatialvector import SpatialVector
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo calling the arm impedance control.

Use:
    0) have a spot with an arm
    1) start spot driver
    2) make sure there isn't anything within a meter of spot
    3) python relative/path/to/this/script/here robot-name-here
    4) spot will make some circle like trajectories with its end effector
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("spot_impedance_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())

    spot.initialize_robot()

    spot.stand()

    spot.ready_arm()

    root_t_task = SE3.Tx(0.7)
    sweep = np.linspace(0, 2 * np.pi, 50)
    radius = 0.15
    time_scale = 10 / sweep[-1]
    circle_trajectory = [
        (SE3(x=radius * np.cos(angle), y=radius * np.sin(angle), z=0), time_scale * angle) for angle in sweep
    ]

    print("Running stiff impedance control ...")
    spot.move_end_effector_impedance(
        root_frame_name=spot.FrameHint.BODY,
        root_t_task=root_t_task,
        wrist_t_tool=SE3(),
        trajectory=circle_trajectory,
        timeout_sec=circle_trajectory[-1][1] + 0.25,
    )

    print("Running soft impedance control with small wrench...")
    spot.move_end_effector_impedance(
        root_frame_name=spot.FrameHint.BODY,
        root_t_task=root_t_task,
        wrist_t_tool=SE3(),
        trajectory=circle_trajectory,
        stiffness=SpatialVector([100, 100, 0, 20, 20, 20]),
        damping=SpatialVector([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
        wrench_nm=SpatialForce([0, 0, 4, 0, 0, 0]),
        timeout_sec=circle_trajectory[-1][1] + 0.25,
    )

    print("Control finished! Sitting here for 5 more seconds...")
    time.sleep(5.0)

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
