#!/urs/bin/env python
#
# To test this program
#
# python example_walk_velocity_with_duration.py Gosu 0.75 0.0 0.0
# python example_walk_velocity_with_duration.py Gosu -0.75 0.0 0.0 --duration 1.0
# python example_walk_velocity_with_duration.py Gosu 0.75 0.5 30.0 --duration 2.0
import argparse
import math

import bdai_ros2_wrappers.process as ros_process
from spatialmath.twist import Twist2
from spot_utilities.spot_basic import SpotBasic


def cli() -> argparse.ArgumentParser:
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
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    if len(args.body_velocity) == 3:
        vx, vy, vyaw = args.body_velocity
        vyaw = math.pi * vyaw / 180.0
    else:
        raise ValueError(f"invalid body_velocity: {args.body_velocity}")
    body_velocity = Twist2([vx, vy], vyaw)

    spot_basic = SpotBasic(robot_name=args.robot)
    try:
        if spot_basic.initialize_robot():
            spot_basic.stand()
            spot_basic.walk_velocity(body_velocity, duration=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        spot_basic.stop()


if __name__ == "__main__":
    main()
