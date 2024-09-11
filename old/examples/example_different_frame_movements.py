#!/usr/bin/env python
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""Example illustrating how to command movements in frames other than the vision frame for some functions"""
import argparse
import time

import numpy as np
from spatialmath import SE2, SE3, SO3
from spot_utilities.spot_basic import SpotBasic

import bdai_ros2_wrappers.process as ros_process
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai.trajectory.spot_se2_properties import MAX_SPEED_FORWARD
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    robot = SpotBasic(args.robot)
    robot.initialize_robot()
    robot.stand()

    tf_listener = TFListenerWrapper()
    tf_listener.wait_for_a_tform_b(robot.body_frame_name, robot.odom_frame_name)
    robot.set_tf_listener(tf_listener)
    robot.stow_arm()

    try:
        # Move the end effector forward and upwards
        end_goal = SE3.Rt(SO3.Rx(np.pi / 6), t=[1.0, 0.0, 0.5])
        robot.move_end_effector_cartesian(frame_t_goal=end_goal, body_follow=True, base_frame=robot.FrameHint.BODY)
        robot.stow_arm()
        time.sleep(1.0)

        # Turn to look at origin of the odometry frame
        odom_t_body = tf_listener.lookup_a_tform_b(robot.odom_frame_name, robot.body_frame_name)
        robot.body_gaze_at(odom_t_body.inv(), frame_t_body=odom_t_body, base_frame=robot.FrameHint.ODOM)
        time.sleep(1.0)

        # Walks 1 meter forward as specified in the odometry frame
        odom_t_body = tf_listener.lookup_a_tform_b_se2(robot.odom_frame_name, robot.body_frame_name)
        body_t_goal = SE2(1.0, 0.0, 0.0)
        odom_t_goal = odom_t_body * body_t_goal
        robot.walk_to(odom_t_goal, base_frame=robot.FrameHint.ODOM)
        time.sleep(1.0)

        # Walks 1.5 meters and then turns around
        odom_t_body = tf_listener.lookup_a_tform_b(robot.odom_frame_name, robot.body_frame_name)
        body_t_goal = SE3.Rt(SO3.Rz(np.pi), t=[1.5, 0.0, 0.0])
        odom_t_goal = odom_t_body * body_t_goal
        robot.walk_and_orient(odom_t_goal, base_frame=robot.FrameHint.ODOM)
        time.sleep(1.0)

        # Walks in a circle with radius 1.25 meters
        odom_t_body = tf_listener.lookup_a_tform_b_se2(robot.odom_frame_name, robot.body_frame_name)
        body_t_traj = SE2Trajectory.from_circle(radius=1.25, forward_speed=MAX_SPEED_FORWARD)
        odom_t_traj = odom_t_body * body_t_traj
        robot.walk_trajectory(odom_t_traj, base_frame=robot.FrameHint.ODOM, timestep_sec=0.2)
        time.sleep(1.0)

    except Exception as e:
        # If something happens, have the robot stop
        main.node.get_logger().error(str(e))
        robot.stop()
    finally:
        robot.stow_arm()


if __name__ == "__main__":
    main()
