import time

import numpy
import rclpy
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo calling the manipulation api.

Use:
    1) tele op such that some object is center view of the hand camera
    2) start spot driver
    3) execute this script
    4) when I tried this, the arm actually missed picking up a tissue box on the ground, but
    the api request actually executed correctly, so its a win.

"""


def main() -> None:
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("spot_manipulation_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())

    spot.initialize_robot()

    spot.stand()

    spot.ready_arm()
    spot.grasp_along_ray(
        task_frame_t_ray_start=numpy.array([0, 0, 0]),
        task_frame_t_ray_end=numpy.array([1, 0, 0]),
        task_frame_name=str(SpotBasic.FrameHint.HAND.value),
        palm_placement_ratio=0.0,
    )
    print("Grasp attempted, pausing for 5 seconds ...")
    time.sleep(5.0)

    spot.stow_arm()
    time.sleep(1.0)

    print("TADA!")

    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
