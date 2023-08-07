# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import time

import rclpy
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo gazing at a point in space with the body.

Use:
    1) tele op the robot so that the origin of the vision frame is outside the support
        polygon of the feet
    2) start spot driver
    3) execute this script, pass in the robot name
    4) spot will orient itself such that the front of the body is facing towards the origin of the vision frame
"""


def main() -> None:
    """Command-line interface."""
    import argparse

    ###############################################
    # Boiler plate code
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot.")
    parser.add_argument("asynchronous", help="Flag to exercise the async code.", default=False)
    options = parser.parse_args()

    robot_name = options.robot

    rclpy.init()
    node = rclpy.create_node("gaze_at_example")
    spot = SpotBasic(robot_name, parent_node_name=node.get_name())
    spot.initialize_robot()
    ###############################################

    tf = TFListenerWrapper(
        spot.robot_name + "_tf",
        cache_time_s=30,
        wait_for_transform=[spot.body_frame_name, spot.vision_frame_name],
    )
    time.sleep(1.0)  # this sleep addresses a race condition in creating the tf listener and asking it for data
    vision_to_body = tf.lookup_a_tform_b(spot.vision_frame_name, spot.body_frame_name, wait_for_frames=True)

    # gaze at the origin of the vision frame, which should be where the robot started
    if options.asynchronous:
        spot.set_tf_listener(tf)  # you only need the tf listener if you aren't providing vision_to_body
        spot.body_gaze_at_async(
            body_t_target=vision_to_body.inv(), result_callback=(lambda x: print("Body gaze at complete!"))
        )
    else:
        spot.body_gaze_at(vision_to_body.inv(), vision_to_body)
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
