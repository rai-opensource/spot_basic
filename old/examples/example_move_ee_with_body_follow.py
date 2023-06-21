#!/urs/bin/env python
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
import threading

import rclpy
import spot_driver.conversions as conv
from bdai.utilities.math_helpers import (
    Quaternion,
    SE3Pose,
    euler_zyx_to_quaternion,
)
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from spot_msgs.action import RobotCommand
from spot_utilities.spot_basic import SpotBasic


def main() -> None:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    parser.add_argument(
        "body_t_goal",
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
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node("move_spot_ee_body_follow")

    spot_basic = SpotBasic(robot_name=args.robot, parent_node_name=node.get_name())
    spot_basic.initialize_robot()

    t = threading.Thread(target=rclpy.spin, args=(node,))
    t.start()

    robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", "robot_node", namespace=args.robot)
    tf_listener = TFListenerWrapper(
        "action_exec_tf", wait_for_transform=[spot_basic.body_frame_name, spot_basic.vision_frame_name]
    )

    if len(args.body_t_goal) == 3:
        x, y, z = args.body_t_goal
        quaternion = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    elif len(args.body_t_goal) == 6:
        x, y, z, thx, thy, thz = args.body_t_goal
        quaternion = euler_zyx_to_quaternion(euler=[thz, thy, thx])
    elif len(args.body_t_goal) == 7:
        x, y, z, qx, qy, qz, qw = args.body_t_goal
        quaternion = Quaternion(w=qw, x=qx, y=qy, z=qz)
    else:
        raise ValueError(f"invalid body_t_goal: {args.body_t_goal}")

    # Rotation as a quaternion
    body_T_hand = SE3Pose(x, y, z, rotation=quaternion)
    odom_T_body = tf_listener.lookup_a_tform_b(spot_basic.vision_frame_name, spot_basic.body_frame_name)
    odom_T_hand = odom_T_body * body_T_hand

    if not args.manual:
        try:
            if not args.ac:
                spot_basic.move_end_effector_cartesian(odom_T_hand, body_follow=True)
            else:
                spot_basic.move_end_effector_cartesian_async(
                    odom_T_hand,
                    body_follow=True,
                    result_callback=lambda feedback: print("MOVE COMPLETE!"),
                )
                print("ASYNC METHOD CALLED! ROBOT MAY BE MOVING NOW.")
                rate = node.create_rate(1 / 5.0)
                rate.sleep()
        except KeyboardInterrupt:
            spot_basic.stop()
    else:
        seconds = 3.0
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.position.x,
            odom_T_hand.position.y,
            odom_T_hand.position.z,
            odom_T_hand.rotation.w,
            odom_T_hand.rotation.x,
            odom_T_hand.rotation.y,
            odom_T_hand.rotation.z,
            VISION_FRAME_NAME,
            seconds,
        )

        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(synchro_command, action_goal.command)
        robot_command_client.send_goal_and_wait(action_goal)
        node.get_logger().info("movement done.")

    spot_basic.stow_arm()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
