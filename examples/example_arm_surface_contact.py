#!/usr/bin/env python
# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import argparse

import numpy as np
import synchros2.process as ros_process
from bosdyn.api.arm_surface_contact_pb2 import ArmSurfaceContact as ArmSurfaceContactProto
from spatialmath import SE3

from spot_basic import FrameHint, SpotBasic
from spot_basic.trajectory.se3_trajectory import SE3Trajectory, SE3TrajectoryPoint, get_trajectory_point


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Use arm surface contact to draw a square")
    parser.add_argument("robot_name", help="Name of robot to use")
    parser.add_argument("--side-length", type=float, default=0.2, help="Length of square side")
    parser.add_argument(
        "--hand_p_tool",
        nargs=3,
        type=float,
        help="Offset from the center of the HAND frame in the hand frame (+x out) to the tool",
        default=[0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--press-force-percentage", type=float, default=0.25, help="Percent of possible force to use pushing down"
    )
    parser.add_argument(
        "--body-follow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="True to have the body move to keep the hand in range",
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="Approximate speed of the hand in m/s",
        default=0.5,
    )

    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> None:
    spot = SpotBasic(args.robot_name)

    wrist_t_tool = spot.lookup_a_tform_b(FrameHint.WRIST, FrameHint.HAND, wait_for_frames=True, timeout=5.0) * SE3(
        x=args.hand_p_tool
    )
    gbody_t_tool = (
        spot.lookup_a_tform_b(FrameHint.GRAV_ALIGNED_BODY, FrameHint.WRIST, wait_for_frames=True, timeout=5.0)
        * wrist_t_tool
    )
    offsets = [
        [0.0, -args.side_length, 0.0],
        [-args.side_length, 0.0, 0.0],
        [0.0, args.side_length, 0.0],
        [args.side_length, 0.0, 0.0],
    ]
    waypoints = [SE3TrajectoryPoint(gbody_t_tool, 0.0)]
    for offset in offsets:
        waypoints.append(
            get_trajectory_point(
                waypoints[-1],
                SE3.Rt(t=waypoints[-1].task_t_waypoint.t + offset, R=waypoints[-1].task_t_waypoint.R),
                args.speed,
            )
        )
    # We don't actually want that first point - the hand is already there
    waypoints.pop(0)
    print("\n".join([str(waypoint.task_t_waypoint.t) for waypoint in waypoints]))
    pos_mode = ArmSurfaceContactProto.Request.AxisMode.AXIS_MODE_POSITION
    force_mode = ArmSurfaceContactProto.Request.AxisMode.AXIS_MODE_FORCE
    spot.arm_surface_contact.synchronously(
        root_frame=FrameHint.VISION,
        task_traj_tool=SE3Trajectory(waypoints),
        x_axis_mode=pos_mode,
        y_axis_mode=pos_mode,
        z_axis_mode=force_mode,
        press_force_percentage=[0.0, 0.0, -1.0 * np.abs(args.press_force_percentage)],
        root_t_task=spot.lookup_a_tform_b(
            FrameHint.VISION, FrameHint.GRAV_ALIGNED_BODY, wait_for_frames=True, timeout=5.0
        ),
        wrist_t_tool=wrist_t_tool,
        xy_to_z_cross_term_admittance=ArmSurfaceContactProto.Request.AdmittanceSetting.ADMITTANCE_SETTING_NORMAL,
        follow_hand=args.body_follow,
    )


if __name__ == "__main__":
    main()
