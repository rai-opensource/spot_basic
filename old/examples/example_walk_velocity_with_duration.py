#!/urs/bin/env python
#
# To test this program
#
# python example_walk_velocity_with_duration.py Gosu 0.75 0.0 0.0
# python example_walk_velocity_with_duration.py Gosu -0.75 0.0 0.0 --duration 1.0
# python example_walk_velocity_with_duration.py Gosu 0.75 0.5 30.0 --duration 2.0
import argparse
import math
import threading

import rclpy
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from spatialmath.twist import Twist2
from spot_msgs.action import RobotCommand
from spot_utilities.spot_basic import SpotBasic


def main() -> None:
    parser = argparse.ArgumentParser(description="Command spot body")
    parser.add_argument("robot", type=str, help="name of the robot")
    parser.add_argument(
        "body_velocity",
        type=float,
        nargs="+",
        help="target SE2 velocity of body. Note that yaw is in degrees",
    )
    parser.add_argument(
        "--duration", type=float, help="duration (s) to walk with given velocity. Default (3.0)", default=3.0
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = rclpy.create_node("test_spot_body")

    spot_basic = SpotBasic(robot_name=args.robot, parent_node_name=node.get_name())
    spot_basic.initialize_robot()

    t = threading.Thread(target=rclpy.spin, args=(node,))
    t.start()

    ActionClientWrapper(RobotCommand, "robot_command", "robot_node", namespace=args.robot)
    TFListenerWrapper("action_exec_tf", wait_for_transform=[spot_basic.body_frame_name, spot_basic.vision_frame_name])

    if len(args.body_velocity) == 3:
        vx, vy, vyaw = args.body_velocity
        vyaw = math.pi * vyaw / 180.0
    else:
        raise ValueError(f"invalid body_velocity: {args.body_velocity}")

    try:
        body_velocity = Twist2([vx, vy], vyaw)
        spot_basic.walk_velocity(body_velocity, duration=args.duration)
    except KeyboardInterrupt:
        spot_basic.stop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
