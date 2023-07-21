import time

import rclpy
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo gazing at a point in space with the body.

Use:
    1) tele op the robot so that the origin of the vision frame won't be obstructed by spot's own body 
        (the arm might run into the body if you fail to do this?)
    1) start spot driver
    2) execute this script, pass in the robot name
"""


def main() -> None:
    """Command-line interface."""
    import argparse

    ###############################################
    # Boiler plate code
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    parser.add_argument("asynchronous", help="Flag to exercise the async code.", action=argparse.BooleanOptionalAction)
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("gaze_at_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())
    spot.initialize_robot()
    ###############################################

    # gaze at the origin of the vision frame, which should be where the robot started
    target_point = [0, 0, 0]
    if options.asynchronous:
        print("Doing async arm gaze ...")
        spot.arm_gaze_at_async(
            point=target_point,
            frame=spot.FrameHint.VISION,
            result_callback=(lambda x: print("Async arm gaze at complete!")),
        )
    else:
        print("Doing synchronous arm gaze ...")
        spot.arm_gaze_at(point=target_point, frame=spot.FrameHint.VISION)
    time.sleep(5.0)
    print("TADA!")

    ###############################################
    # Boiler plate code
    spot.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    ###############################################


if __name__ == "__main__":
    main()
