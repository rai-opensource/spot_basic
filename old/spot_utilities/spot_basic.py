#  Copyright [2023] Boston Dynamics AI Institute, Inc.
from enum import IntEnum
import sys
import time
from typing import List, Optional

from geometry_msgs.msg import Twist
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import ColorRGBA as Color
from std_srvs.srv import SetBool, Trigger

from bosdyn.api.spot import robot_command_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder

from bosdyn_msgs.msg import ListWorldObjectRequest, ListWorldObjectResponse
import spot_driver.conversions as conv
from spot_msgs.action import RobotCommand
from spot_msgs.srv import ClearBehaviorFault, ListGraph, ListWorldObjects, SetLocomotion, SetVelocity
from visualization_msgs.msg import Marker

from bdai.utilities.math_helpers import SE2Pose, SE3Pose
from bdai_ros.utilities.manual_conversions import to_ros_pose


class SpotBasic(object):
    """ Base ROS2 Node that provides interfaces to calling various commands """

    TRIGGER_SERVICES = [
        "claim", "release", "stop", "self_right", "sit", "stand", "power_on", "power_off", "estop/hard", "estop/gentle",
        "estop/release"
    ]

    class LocomotionHint(IntEnum):
        """ Enumeration of the locomotion hints

        See
        https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=mobilityparams#locomotionhint
        Note: Comments below for each enumerator come directly from the above link
        """
        UNKNOWN = 0  # Listed for completeness, but the docs say do not use
        AUTO = 1  # No hint, robot chooses an appropriate gait (typically trot.)
        TROT = 2  # Most robust gait which moves diagonal legs together.
        SPEED_SELECT_TROT = 3  # Trot which comes to a stand when not commanded to move.
        CRAWL = 4  # Slow and steady gait which moves only one foot at a time.
        SPEED_SELECT_CRAWL = 10  # Crawl which comes to a stand when not commanded to move.
        AMBLE = 5  # Four beat gait where one foot touches down at a time.
        SPEED_SELECT_AMBLE = 6  # Amble which comes to a stand when not commanded to move.
        JOG = 7  # Demo gait which moves diagonal leg pairs together with an aerial phase.
        HOP = 8  # Demo gait which hops while holding some feet in the air.

    class BatteryChangeDirHint(IntEnum):
        RIGHT = 1
        LEFT = 2

    def __init__(self,
                 robot_name: str,
                 parent_node_name: str,
                 *,
                 disable_collisions: bool = False,
                 dry: bool = False,
                 rviz: bool = False):
        """
        Args:
            robot_name: name of the robot
            parent_node_name: The name of the parent node containing this SpotBasic
            disable_collisions: True if collisions should be ignored, False otherwise
            dry: True if this is a dry run and commands should not actually be sent, False otherwise
            rviz: True if visuals should be published, False otherwise
        """
        self.__node = Node(f"{parent_node_name}_spot_basic", namespace=robot_name)

        self.__robot_name = robot_name
        self.__disable_collisions = disable_collisions
        self.__dry = dry
        self.__rviz = rviz

        # TODO: are there other frame names to grab?
        self.__body_frame_name = f"{self.robot_name}/{BODY_FRAME_NAME}"
        self.__hand_frame_name = f"{self.robot_name}/{HAND_FRAME_NAME}"
        self.__vision_frame_name = f"{self.robot_name}/{VISION_FRAME_NAME}"

        self.__executor = SingleThreadedExecutor()
        self.__executor.add_node(self.__node)
        self.__callback_group = ReentrantCallbackGroup()

        if self.__rviz:
            self.__vis_pub = self.__node.create_publisher(Marker, "/visualization_marker", 10)

        self.__trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self.__trigger_clients[command_name] = self.__node.create_client(Trigger,
                                                                             f"/{self.__robot_name}/{command_name}",
                                                                             callback_group=self.__callback_group)
        self.__stair_mode_client = self.__node.create_client(SetBool,
                                                             f"/{self.__robot_name}/stair_mode",
                                                             callback_group=self.__callback_group)
        self.__locomotion_mode_client = self.__node.create_client(SetLocomotion,
                                                                  f"/{self.__robot_name}/locomotion_mode",
                                                                  callback_group=self.__callback_group)
        self.__max_velocity_client = self.__node.create_client(SetVelocity,
                                                               f"/{self.__robot_name}/max_velocity",
                                                               callback_group=self.__callback_group)
        self.__clear_behavior_fault_client = self.__node.create_client(ClearBehaviorFault,
                                                                       f"/{self.__robot_name}/clear_behavior_fault",
                                                                       callback_group=self.__callback_group)
        self.__list_graph_client = self.__node.create_client(ListGraph,
                                                             f"/{self.__robot_name}/list_graph",
                                                             callback_group=self.__callback_group)
        self.__list_world_objects_client = self.__node.create_client(ListWorldObjects,
                                                                     f"/{self.__robot_name}/list_world_objects",
                                                                     callback_group=self.__callback_group)
        self.__robot_command_action_client = ActionClient(self.__node, RobotCommand, "robot_command")
        self.__node.get_logger().info(f"Waiting for the 'RobotCommand' action server for {self.__robot_name}")
        self.__robot_command_action_client.wait_for_server()

    @property
    def robot_name(self):
        return self.__robot_name

    @property
    def body_frame_name(self):
        return self.__body_frame_name

    @property
    def hand_frame_name(self):
        return self.__hand_frame_name

    @property
    def vision_frame_name(self):
        return self.__vision_frame_name

    def initialize_robot(self) -> bool:
        """ Returns true/false based on success of claiming and powering up Spot. """
        # Claim robot.
        if not self.claim():
            return False

        # Power the robot.
        if not self.power_motors_on():
            return False
        return True

    def trigger_service(self, command_name: str) -> bool:
        """ Triggers a command service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Trigger command '{command_name}' for {self.robot_name}")
        request = Trigger.Request()
        try:
            future = self.__trigger_clients[command_name].call_async(request)
        except KeyError:
            self.__node.get_logger().error(f"No trigger command {command_name}")
            return False

        self.__executor.spin_until_future_complete(future)

        response = future.result()
        if not response.success:
            self.__node.get_logger().error(
                f"Trigger command '{command_name}' for {self.robot_name} failed. Msg: {response.message}")
            return False
        return True

    def claim(self) -> bool:
        """ Triggers the 'claim' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("claim")

    def release(self) -> bool:
        """ Triggers the 'release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("release")

    def stop(self) -> bool:
        """ Triggers the 'stop' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("stop")

    def self_right(self) -> bool:
        """ Triggers the 'self_right' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("self_right")

    def sit(self) -> bool:
        """ Triggers the 'sit' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("sit")

    def stand(self) -> bool:
        """ Triggers the 'stand' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("stand")

    def rollover(self) -> bool:
        """ Triggers the 'rollover' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("rollover")

    def power_on(self) -> bool:
        """ Triggers the 'power_on' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("power_on")

    # Alias for the power_on function
    power_motors_on = power_on

    def power_off(self) -> bool:
        """ Triggers the 'power_off' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("power_off")

    # Alias for the power_off function
    power_motors_off = power_off

    def estop_hard(self) -> bool:
        """ Triggers the 'estop/hard' service on the SpotDriver. The robot will immediately cut power to the motors

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/hard")

    def estop_gentle(self) -> bool:
        """ Triggers the 'estop/gentle' service on the SpotDriver. The robot will try to settle on the ground before
                cutting power to the motors.

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/hard")

    def estop_release(self) -> bool:
        """ Triggers the 'estop/release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self.trigger_service("estop/hard")

    def set_stair_mode(self, value: bool) -> bool:
        """ Sets whether to be in stair_mode on the SpotDriver
        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Setting stair_mode to '{value}' for {self.robot_name}")
        request = SetBool.Request()
        request.data = value
        future = self.__stair_mode_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self.robot_name} failed. Msg: {response.message}.")
            return False
        return True

    def set_locomotion_mode(self, l_hint: LocomotionHint) -> bool:
        """ Set the locomotion mode on the SpotDriver
        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Setting locomotion_mode to '{l_hint}' for {self.robot_name}")
        request = SetLocomotion.Request()
        request.locomotion_mode = int(l_hint)
        future = self.__locomotion_mode_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting locomotion_mode to '{l_hint}' for {self.robot_name} failed. Msg: {response.message}.")
            return False
        return True

    def set_max_velocity(self, velocity_limit: Twist) -> SetVelocity.Response:
        """ Set the maximum velocity on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.__node.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self.robot_name}")
        request = SetVelocity.Request()
        request.velocity = velocity_limit
        future = self.__max_velocity_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self.robot_name} failed. Msg: {response.message}.")
            return False
        return True

    def clear_behavior_fault(self, bid: int) -> ClearBehaviorFault.Response:
        """ Clear a behavior fault on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        # TODO: how to set the bid?
        self.__node.get_logger().info(f"Clearing behavior fault {bid} for {self.robot_name}")
        request = ClearBehaviorFault.Request()
        request.id = bid
        future = self.__clear_behavior_fault_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.__node.get_logger().debug(
                f"Clearing behavior fault {bid} for {self.robot_name} failed. Msg: {response.message}")
            return False
        return True

    def list_graph(self, upload_filepath: str) -> List[str]:
        """ Request a list of waypoint ids """
        self.__node.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self.__list_graph_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        return response.waypoint_ids

    def list_world_objects(self, p_request: ListWorldObjectRequest) -> ListWorldObjectResponse:
        """ Requests a list of world objects

        TODO: not use ROS msgs
        """
        self.__node.get_logger().info(f"List world objects for {self.robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self.__list_world_objects_client.call_async(request)
        self.__executor.spin_until_future_complete(future)
        response = future.result()
        return response.response

    def _send_command_goal_and_wait(self, goal: RobotCommand.Goal, timeout_sec=None):
        future = self.__robot_command_action_client.send_goal_async(goal)
        self.__executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.__node.get_logger().error("Goal was not accepted")
            return None
        future = goal_handle.get_result_async()
        self.__executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        return future.result()

    def command(self, goal: RobotCommand.Goal, timeout_sec=None) -> RobotCommand.Result:
        """ Runs the `RobotCommand` action

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if goal is None:
            # TODO: Is there something better that we can return with information about the failure?
            # maybe a RobotCommand.Result that represents failure?
            return None
        if not self.__dry:
            return self._send_command_goal_and_wait(goal, timeout_sec)
        # TODO: What to return on a dry run? Just success?
        return None

    def command_async(self, goal: RobotCommand.Goal) -> Future:
        """ Runs the `RobotCommand` action asynchronously

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if goal is None:
            # Is there something better that we can return with information about the failure?
            return None
        if not self.__dry:
            goal_future = self.__robot_command_action_client.send_goal_async(goal)
            return goal_future
        # TODO: What to return on a dry run? Just success?
        return None

    def _draw_arrow(self, frame_id: str, ns: str, frame_t_arrow: SE2Pose | SE3Pose, color: Color) -> None:
        """ TODO """
        marker = Marker()
        marker.header.stamp = self.__node.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.id = 0
        marker.ns = ns
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = to_ros_pose(frame_t_arrow)
        marker.color = color
        marker.scale.x = 0.3
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        self.__vis_pub.publish(marker)

    @staticmethod
    def _convert_proto_to_action_goal(proto) -> RobotCommand.Goal:
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(proto, action_goal.command)
        return action_goal

    def _get_battery_change_pose(self, dir_hint: BatteryChangeDirHint):
        proto = RobotCommandBuilder.battery_change_pose_command(int(dir_hint))
        return self._convert_proto_to_action_goal(proto)

    def go_to_battery_change_pose(self,
                                  dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT,
                                  timeout_sec: Optional[float] = None):
        """ Send command to go to the battery change pose asynchronously """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command(action_goal, timeout_sec)

    def go_to_battery_change_pose_async(self, dir_hint: BatteryChangeDirHint = BatteryChangeDirHint.RIGHT):
        """ Send command to go to the battery change pose asynchronously """
        action_goal = self._get_battery_change_pose(dir_hint)
        return self.command_async(action_goal)

    def _get_walk_to_action_goal(self, vision_t_goal: SE2Pose, height: float) -> RobotCommand.Goal:
        """ TODO

        Args:
            vision_t_goal:
            height:

        Returns:

        """
        self.__node.get_logger().info("Walking to goal...")

        # Mobility command
        obstacles = robot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=self.__disable_collisions,
                                                     disable_vision_foot_obstacle_avoidance=self.__disable_collisions,
                                                     disable_vision_foot_constraint_avoidance=self.__disable_collisions,
                                                     obstacle_avoidance_padding=.001)
        mobility_params = robot_command_pb2.MobilityParams(obstacle_params=obstacles)

        if self.__rviz:
            self._draw_arrow(self.vision_frame_name, self.body_frame_name, vision_t_goal, color=Color(g=1.0, a=1.0))

        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=vision_t_goal.x,
                                                                              goal_y=vision_t_goal.y,
                                                                              goal_heading=vision_t_goal.yaw,
                                                                              frame_name=VISION_FRAME_NAME,
                                                                              body_height=height,
                                                                              params=mobility_params)
        return self._convert_proto_to_action_goal(proto_goal)

    def walk_to(self,
                vision_t_goal: SE2Pose,
                height: float = 0,
                timeout_sec: Optional[float] = None) -> Optional[RobotCommand.Result]:
        """ TODO
        
        Args:
            vision_t_goal:
            height:
            timeout_sec:

        Returns:

        """
        action_goal = self._get_walk_to_action_goal(vision_t_goal, height)
        return self.command(action_goal, timeout_sec)

    def walk_to_async(self, vision_t_goal, height: float = 0) -> Optional[Future]:
        """ TODO

        Args:
            vision_t_goal:
            height:

        Returns:

        """
        action_goal = self._get_walk_to_action_goal(vision_t_goal, height)
        return self.command_async(action_goal)

    def _get_arm_stow_action_goal(self):
        """ Create the ROS msg to stow the arm """
        proto = RobotCommandBuilder.arm_stow_command()
        return self._convert_proto_to_action_goal(proto)

    def stow_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommand.Result]:
        """ Send a command to stow the arm synchronously """
        action_goal = self._get_arm_stow_action_goal()
        return self.command(action_goal, timeout_sec)

    def stow_arm_async(self) -> Optional[RobotCommand.Result]:
        """ Send a command to stow the arm asynchronously """
        action_goal = self._get_arm_stow_action_goal()
        return self.command_async(action_goal)

    def _get_arm_ready_action_goal(self):
        """ Create the ROS msg to move the arm to the ready position"""
        proto = RobotCommandBuilder.arm_ready_command()
        return self._convert_proto_to_action_goal(proto)

    def ready_arm(self, timeout_sec: Optional[float] = None) -> Optional[RobotCommand.Result]:
        """ Send a command to move the arm to the ready position synchronously """
        action_goal = self._get_arm_ready_action_goal()
        return self.command(action_goal, timeout_sec)

    # Alias
    unstow_arm = ready_arm

    def ready_arm_async(self):
        """ Send a command to move the arm to the ready position asynchronously"""
        action_goal = self._get_arm_ready_action_goal()
        return self.command_async(action_goal)

    # Alias
    unstow_arm_async = ready_arm_async

    def _get_arm_carry_action_goal(self):
        """ Create the ROS msg to move the arm to the carry position """
        proto = RobotCommandBuilder.arm_carry_command()
        return self._convert_proto_to_action_goal(proto)

    def carry_arm(self):
        """ Move the arm to the carry position synchronously"""
        action_goal = self._get_arm_carry_action_goal()
        return self.command(action_goal)

    def carry_arm_async(self):
        """ Move the arm to the carry position asynchronously"""
        action_goal = self._get_arm_carry_action_goal()
        return self.command_async(action_goal)

    def _get_arm_pose_action_goal(self, goal: SE3Pose, frame_name: str, duration_sec: float) -> RobotCommand.Goal:
        """ Create the ROS Msg for moving the end effector to specific pose """
        proto = RobotCommandBuilder.arm_pose_command(x=goal.x,
                                                     y=goal.y,
                                                     z=goal.z,
                                                     qw=goal.rotation.w,
                                                     qx=goal.rotation.x,
                                                     qy=goal.rotation.y,
                                                     qz=goal.rotation.z,
                                                     frame_name=frame_name,
                                                     seconds=duration_sec)

        return self._convert_proto_to_action_goal(proto)

    def move_end_effector_cartesian(self,
                                    goal: SE3Pose,
                                    frame_name: str = VISION_FRAME_NAME,
                                    move_duration_sec: float = 5.0,
                                    timeout_sec: Optional[float] = None):
        """ Send command to move the end effort to a specific pose synchronously

        Args:
            goal: A pose for the end effector to move to relative to the base frame
            frame_name: Name of the base frame
            move_duration_sec: The duration of the move
            timeout_sec: The timeout on the command

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        action_goal = self._get_arm_pose_action_goal(goal, frame_name, move_duration_sec)
        return self.command(action_goal, timeout_sec)

    def move_end_effector_cartesian_async(self,
                                          goal: SE3Pose,
                                          frame_name: str = VISION_FRAME_NAME,
                                          move_duration_sec: float = 5.0):
        """ Send command to move the end effort to a specific pose asynchronously

        Args:
            goal: A pose for the end effector to move to relative to the base frame
            frame_name: Name of the base frame
            move_duration_sec: The duration of the move

        See:
        https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L1224
        """
        action_goal = self._get_arm_pose_action_goal(goal, frame_name, move_duration_sec)
        return self.command_async(action_goal)

    def move_end_effector_force(self):
        pass

    def move_end_effector_force_async(self):
        pass

    def move_end_effector_impedance(self):
        pass

    def move_end_effector_impedance_async(self):
        pass

    def claw_gripper_open(self):
        action_goal = self._convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_open_command())
        return self.command(action_goal)

    def claw_gripper_open_async(self):
        action_goal = self._convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_open_command())
        return self.command_async(action_goal)

    def claw_gripper_close(self):
        action_goal = self._convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command(action_goal)

    def claw_gripper_close_async(self):
        action_goal = self._convert_proto_to_action_goal(RobotCommandBuilder.claw_gripper_close_command())
        return self.command_async(action_goal)

    def shutdown(self) -> None:
        """ TODO """
        self.__robot_command_action_client.destroy()
        self.__node.destroy_node()
        # TODO: Is there anything else that we should shutdown?


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    parser.add_argument("--disable-collisions", action="store_true", help="Flag for disabling collision avoidance")
    parser.add_argument("--dry", action="store_true", help="Flag for if this is a dry run")
    options = parser.parse_args()

    rclpy.init()
    spot_basic = SpotBasic(robot_name=options.robot_name,
                           parent_node_name="dummy_node",
                           disable_collisions=options.disable_collisions,
                           dry=options.dry)
    spot_basic.initialize_robot()
    time.sleep(1.0)
    spot_basic.sit()
    spot_basic.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
