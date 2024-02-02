# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import argparse
import time

import bdai_ros2_wrappers.process as ros_process
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo gazing at a point in space with the body.

Use:
    1) tele op the robot so that the origin of the vision frame won't be obstructed by spot's own body
        (the arm might run into the body if you fail to do this?)
    1) start spot driver
    2) execute this script, pass in the robot name
"""


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    parser.add_argument("asynchronous", help="Flag to exercise the async code.", action=argparse.BooleanOptionalAction)
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    ###############################################
    # Boiler plate code
    spot = SpotBasic(args.robot)
    spot.initialize_robot()
    ###############################################

    # gaze at the origin of the vision frame, which should be where the robot started
    target_point = [0, 0, 0]
    if args.asynchronous:
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


if __name__ == "__main__":
    main()
