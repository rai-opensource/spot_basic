#!/usr/bin/env python
# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import contextlib
import logging
import sys
from typing import Optional

import synchros2.process as ros_process
from rclpy.duration import Duration
from rclpy.time import Time
from spatialmath import SE2
from spot_msgs.msg import JointCommand
from synchros2.futures import unwrap_future

from spot_basic import SpotBasic
from spot_basic.control import SpotJointControllerInterface


class ControlledSquatExample:
    """
    Have Spot squat in a controlled manner.

    The Spot driver must be running in full controllable mode before attempting to use this example.
    You can start it in controllable mode by adding controllable:=true to the launch command
    """

    SQUAT_JOINT_POSITIONS = {
        "front_left_hip_x": 0.15,
        "front_left_hip_y": 1.30,
        "front_left_knee": -2.25,
        "front_right_hip_x": -0.15,
        "front_right_hip_y": 1.30,
        "front_right_knee": -2.25,
        "rear_left_hip_x": 0.15,
        "rear_left_hip_y": 1.30,
        "rear_left_knee": -2.25,
        "rear_right_hip_x": -0.15,
        "rear_right_hip_y": 1.30,
        "rear_right_knee": -2.25,
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self._logger = logging.getLogger()
        self._robot = SpotBasic(args.robot_name)
        self._duration = Duration(seconds=args.duration_sec)
        self._throttle_duration = Duration(seconds=0.5)
        self._dry_run = args.dry_run

    def start(self) -> int:
        with self._robot.leash(weak=self._dry_run):
            world_t_start = self._robot.lookup_a_tform_b(
                self._robot.vision_frame_name, self._robot.body_frame_name
            ).yaw_SE2()

            robot_t_goal = SE2(0.5, 0.0, 0.0)
            world_t_goal = world_t_start * robot_t_goal
            self._logger.info("Attempting to walk forward")
            if not self._dry_run:
                self._robot.walk.synchronously(world_t_goal)
            self._logger.info("Robot walked forward")

            clock = self._robot.node.get_clock()
            tf_prefix = self._robot.robot_name + "/"  # NOTE: this is not very practical
            with SpotJointControllerInterface(self._robot) as ifc:
                # Get initial joint positions
                joint_states = unwrap_future(ifc.joint_states_feedback.latest_update, timeout_sec=5.0)
                initial_joint_positions = dict(
                    zip(
                        map(lambda name: name.removeprefix(tf_prefix), joint_states.name),
                        joint_states.position,
                        strict=True,
                    )
                )

                start_time = clock.now()
                last_log_time: Optional[Time] = None
                half_duration_nsec = self._duration.nanoseconds / 2.0
                self._logger.info("Attempting controlled squat")
                with contextlib.closing(ifc.joint_states_feedback.stream()) as joint_states_stream:
                    for joint_states in joint_states_stream:
                        now = clock.now()

                        elapsed_time = now - start_time
                        if elapsed_time >= self._duration:
                            break

                        # Interpolate joint position setpoints along the joint position trajectory
                        alpha = float(elapsed_time.nanoseconds) / half_duration_nsec
                        if elapsed_time.nanoseconds > half_duration_nsec:
                            alpha = 2 - alpha
                        alpha = max(0, min(1, alpha))

                        # Update joint setpoints
                        joint_setpoints = JointCommand()
                        for joint, pos in self.SQUAT_JOINT_POSITIONS.items():
                            joint_setpoints.name.append(tf_prefix + joint)
                            joint_setpoints.position.append(alpha * pos + (1 - alpha) * initial_joint_positions[joint])

                        if last_log_time is None or now - last_log_time > self._throttle_duration:

                            joint_positions = {
                                name: position
                                for name, position in zip(
                                    map(lambda name: name.removeprefix(tf_prefix), joint_states.name),
                                    joint_states.position,
                                    strict=True,
                                )
                            }
                            self._logger.info(f"Currently at {joint_positions}")
                            self._logger.info(f"Commanding {joint_setpoints}")
                            last_log_time = now

                        if not self._dry_run:
                            ifc.update_joints_command(joint_setpoints)

                self._logger.info("Done squatting")

            self._logger.info("Attempting to walk back")
            if not self._dry_run:
                self._robot.walk.synchronously(world_t_start)
            self._logger.info("Robot walked back")

        return 0


def cli() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=ControlledSquatExample.__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="If true, run as if commanding the robot without doing so"
    )
    parser.add_argument(
        "-d", "--duration", dest="duration_sec", type=float, default=8.0, help="Duration of the squat motion"
    )
    parser.add_argument("robot_name", help="Name of the robot")
    return parser


@ros_process.main(cli(), interruptible=True, uses_tf=True)
def main(args: argparse.Namespace) -> int:
    example = ControlledSquatExample(args)
    return example.start()


if __name__ == "__main__":
    sys.exit(main())
