#!/urs/bin/env python
# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#
# To test this program, try the following commands. Feel free
# to set different values for the arguments. Note that five meters
# in front of spot would be required for running these commands as-is.
#
# python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25
# python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 --ac
# python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 0 15 0 --ac
# python example_move_ee_with_body_follow.py Stitch 1.0 0.1 0.25 0 15 0
import argparse

import bdai_ros2_wrappers.process as ros_process
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.utilities import namespace_with
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_msgs.conversions import convert
from spatialmath import SE3, UnitQuaternion
from spot_msgs.action import RobotCommand
from spot_utilities.spot_basic import SpotBasic


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    parser.add_argument(
        "body_t_hand_goal",
        type=float,
        nargs="+",
        help=(
            "target SE3 pose of hand in body frame."
            "Could be either of length 3 (x,y,z), 6 (x,y,z,roll,pitch,yaw), or 7 (x,y,z,qx,qy,qz,qw)."
            "Note that roll, pitch, yaw are in degrees."
        ),
    )
    parser.add_argument("--manual", action="store_true", help="If True, use my own code; Otherwise, use SpotBasic")
    parser.add_argument("--ac", action="store_true", help="If True, call the async method")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    spot_basic = SpotBasic(args.robot)
    spot_basic.initialize_robot()

    robot_command_client = ActionClientWrapper(RobotCommand, namespace_with(args.robot, "robot_command"))
    tf_listener = TFListenerWrapper()
    tf_listener.wait_for_a_tform_b(spot_basic.body_frame_name, spot_basic.vision_frame_name)
    spot_basic.set_tf_listener(tf_listener)

    if len(args.body_t_hand_goal) == 3:
        x, y, z = args.body_t_hand_goal
        quaternion = UnitQuaternion(s=1.0, v=[0.0, 0.0, 0.0])
    elif len(args.body_t_hand_goal) == 6:
        x, y, z, thx, thy, thz = args.body_t_hand_goal
        quaternion = UnitQuaternion.RPY([thx, thy, thz], unit="deg")
    elif len(args.body_t_hand_goal) == 7:
        x, y, z, qx, qy, qz, qw = args.body_t_hand_goal
        quaternion = UnitQuaternion(s=qw, v=[qx, qy, qz])
    else:
        raise ValueError(f"invalid body_t_hand_goal: {args.body_t_hand_goal}")

    # Rotation as a quaternion
    body_t_hand = SE3.Rt(R=quaternion.SO3(), t=[x, y, z])
    odom_t_body = tf_listener.lookup_a_tform_b(spot_basic.vision_frame_name, spot_basic.body_frame_name)
    odom_t_hand = odom_t_body * body_t_hand

    if not args.manual:
        try:
            if not args.ac:
                spot_basic.move_end_effector_cartesian(odom_t_hand, body_follow=True, move_duration_sec=3.0)
            else:
                spot_basic.move_end_effector_cartesian_async(
                    odom_t_hand,
                    body_follow=True,
                    move_duration_sec=3.0,
                    result_callback=lambda feedback: print("MOVE COMPLETE!"),
                )
                print("ASYNC METHOD CALLED! ROBOT MAY BE MOVING NOW.")
                rate = main.node.create_rate(1 / 5.0)
                rate.sleep()
        except KeyboardInterrupt:
            spot_basic.stop()
    else:
        seconds = 3.0
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_t_hand.position.x,
            odom_t_hand.position.y,
            odom_t_hand.position.z,
            odom_t_hand.rotation.w,
            odom_t_hand.rotation.x,
            odom_t_hand.rotation.y,
            odom_t_hand.rotation.z,
            VISION_FRAME_NAME,
            seconds,
        )

        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)
        action_goal = RobotCommand.Goal()
        convert(synchro_command, action_goal.command)
        robot_command_client.send_goal_and_wait(action_goal)
        main.node.get_logger().info("movement done.")

    spot_basic.stow_arm()


if __name__ == "__main__":
    main()
