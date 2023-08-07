#!/usr/bin/env python3
#  Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
"""Demonstrate commanding SE2 trajectories via SpotBasic"""

import argparse
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai.trajectory.se2_trajectory_plotting import draw_shape
from bdai.trajectory.spot_se2_properties import BODY_SHAPE, MAX_SPEED_FORWARD, MAX_SPEED_VEC
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from spatialmath import SE2
from spot_utilities.spot_basic import SpotBasic

""" This is a test script to demo having spot walk along a trajectory.

Use:
    1) start spot driver
    2) make sure there isn't anything within a 2 meters of spot
    3) python relative/path/to/this/script/here arguments-here
    4) spot will walk in a pattern based on which example you picked
"""


def plot_trajectory(ax: plt.Axes, trajectory: SE2Trajectory) -> None:
    draw_shape(ax, trajectory, BODY_SHAPE)  # type: ignore
    ax.axis("equal")


def get_vision_t_body(robot_name: str) -> SE2:
    body_frame_name = f"{robot_name}/{BODY_FRAME_NAME}"
    vision_frame_name = f"{robot_name}/{VISION_FRAME_NAME}"
    tf_listener_wrapper = TFListenerWrapper(
        "body_movement_examples", cache_time_s=30, wait_for_transform=[body_frame_name, vision_frame_name]
    )
    return tf_listener_wrapper.lookup_a_tform_b_se2(vision_frame_name, body_frame_name)


def run_example(example_name: str, robot_name: Optional[str] = None) -> None:
    if example_name == "first":
        # Create trajectory to move 2 meters forward and turned 90 degrees ccw
        xf = SE2(1, 2, -np.pi / 2)
        body_t_trajectory = SE2Trajectory.from_se2_pair(target=xf, skewness=0.5, se2_speed_vec=MAX_SPEED_VEC)

    elif example_name == "basic_circle":
        body_t_trajectory = SE2Trajectory.from_circle(radius=1.5, forward_speed=MAX_SPEED_FORWARD)

    elif example_name == "counterspin_circle":
        # NOTE the speed reference is an extremely coarse guideline for setting duration
        # when you have arbitrary spinning going on
        body_t_trajectory = SE2Trajectory.from_circle(radius=1.5, total_body_rotation=-np.pi * 2, forward_speed=0.8)
    else:
        raise ValueError(f"Unknown example name: {example_name}")

    if (robot_name is None) or (robot_name.lower() == "preview"):
        # Plot sampled poses from the trajectory
        _, ax = plt.subplots(1, 1)
        plot_trajectory(ax, body_t_trajectory(n_timesteps=20))  # API / naming could be improved
        plt.show()
        return

    # Initialize
    rclpy.init()
    spot_basic = SpotBasic(robot_name=robot_name, parent_node_name="dummy_node", disable_collisions=False, dry=False)
    spot_basic.initialize_robot()
    spot_basic.stand()
    time.sleep(1.0)

    # Transform and command the trajectory
    vision_t_body = get_vision_t_body(robot_name)
    vision_t_trajectory = vision_t_body * body_t_trajectory
    spot_basic.walk_trajectory(vision_t_trajectory, timestep_sec=0.2)

    # Shut down
    time.sleep(1.0)
    spot_basic.sit()
    spot_basic.shutdown()
    rclpy.shutdown()

    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of robot to use.  Enter 'preview' to simply plot the trajectory")
    parser.add_argument("example_name", help="Name of example to run")
    options = parser.parse_args()

    run_example(options.example_name, options.robot_name)


if __name__ == "__main__":
    main()
