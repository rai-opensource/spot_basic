#!/usr/bin/env python3
#  Copyright (c) 2023 Boston Dynamics AI Institute Inc.  All rights reserved.
import argparse
import sys

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spot_utilities.spot_basic import SpotBasic

"""This example demonstrates using the WalkToObjectRayInWorld manipulation API request from the spot-sdk through the 
    ros2 wrapper. The robot should walk forward about 0.5 meters.

    This script assumes that the spot is standing and has about 1-2 meters of space in front of it.
"""


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example for running the WalkToObjectRayInWorld command")
    parser.add_argument("robot", type=str, help="name of the robot")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> int:
    spot = SpotBasic(robot_name=args.robot)
    tf = TFListenerWrapper(cache_time_s=30)
    tf.wait_for_a_tform_b(spot.body_frame_name, spot.vision_frame_name)
    spot.set_tf_listener(tf)
    spot.initialize_robot()

    body_t_ground = spot.lookup_a_tform_b(SpotBasic.FrameHint.BODY, SpotBasic.FrameHint.GROUND_PLANE)

    # Values based on example_pick_object_ray_in_world.py
    object_rt_frame = [1.25, 0, body_t_ground.z]
    ray_start_rt_frame = object_rt_frame.copy()
    ray_start_rt_frame[2] += 1
    ray_end_rt_frame = object_rt_frame.copy()
    ray_end_rt_frame[2] -= 1

    try:
        spot.walk_to_object_ray_in_world(
            ray_start_rt_frame=ray_start_rt_frame,
            ray_end_rt_frame=ray_end_rt_frame,
            frame_name=SpotBasic.FrameHint.BODY.value,
            offset_distance=0.25,
        )
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
