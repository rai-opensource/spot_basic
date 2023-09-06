#!/usr/bin/env python
"""Example illustrating how to command movements in frames other than the vision frame for some functions"""
import argparse
import time
from threading import Event

import numpy as np
import rclpy
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai.trajectory.spot_se2_properties import MAX_SPEED_FORWARD
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from spatialmath import SE2, SE3, SO3
from spot_utilities.spot_basic import SpotBasic


def main() -> None:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node("move_spot_ee_body_follow")

    robot = SpotBasic(robot_name=args.robot, parent_node_name=node.get_name())
    robot.initialize_robot()
    robot.stand()
    Event()

    tf_listener = TFListenerWrapper("action_exec_tf", wait_for_transform=[robot.body_frame_name, robot.odom_frame_name])
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
        node.get_logger().error(str(e))
        robot.stop()

    robot.stow_arm()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
