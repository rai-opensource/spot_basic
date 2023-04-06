#  Copyright [2023] Boston Dynamics AI Institute, Inc.
from enum import IntEnum
import sys
import time
from typing import Callable, Optional

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger, SetBool
from tf2_ros import TransformException

from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME

import spot_driver.conversions as conv
from spot_msgs.action import NavigateTo, RobotCommand, Trajectory
from spot_msgs.srv import ClearBehaviorFault, ListGraph, ListWorldObjects, SetLocomotion, SetVelocity

from bdai.utilities.math_helpers import SE2Pose
from bdai_ros.utilities.manual_conversions import to_se2_pose
from bdai_ros.wrappers.action_client import ActionClientWrapper
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper


class SpotBasic(Node):
    """ Base ROS2 Node that provides interfaces to calling various commands """

    TRIGGER_SERVICES = [
        'claim', 'release', 'stop', 'self_right', 'sit', 'stand', 'power_on', 'power_off', 'estop/hard', 'estop/gentle',
        'estop/release'
    ]

    class LocomotionHint(IntEnum):
        """ Enumeration of the locomotion hints

        See https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html?highlight=mobilityparams#locomotionhint
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

    def __init__(self, options, node_name: Optional[str] = None, num_threads: int = 1):
        if node_name is None:
            node_name = "spot_init"
        self._robot_name = options.robot_name
        super().__init__(node_name, namespace=self._robot_name)

        self._disable_collisions = options.disable_collisions
        self._dry = options.dry

        if num_threads == 1:
            self._executor = SingleThreadedExecutor()
        elif num_threads > 1:
            self._executor = MultiThreadedExecutor(num_threads)
        else:
            raise ValueError("Cannot have a negative number of threads")
        self._executor.add_node(self)
        self._callback_group = ReentrantCallbackGroup()

        self._trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self._trigger_clients[command_name] = self.create_client(Trigger,
                                                                     command_name,
                                                                     callback_group=self._callback_group)
        self._stair_mode_client = self.create_client(SetBool, "stair_mode", callback_group=self._callback_group)
        self._locomotion_mode_client = self.create_client(SetLocomotion,
                                                          "locomotion_mode",
                                                          callback_group=self._callback_group)
        self._max_velocity_client = self.create_client(SetVelocity, "max_velocity", callback_group=self._callback_group)
        self._clear_behavior_fault_client = self.create_client(ClearBehaviorFault,
                                                               "clear_behavior_fault",
                                                               callback_group=self._callback_group)
        self._list_graph_client = self.create_client(ListGraph, "list_graph", callback_group=self._callback_group)
        self._list_world_objects_client = self.create_client(ListWorldObjects,
                                                             "list_world_objects",
                                                             callback_group=self._callback_group)
        self._robot_command_action_client = ActionClientWrapper(RobotCommand,
                                                                'robot_command',
                                                                namespace=self._robot_name)

        self._body_frame_name = f"{self._robot_name}/{BODY_FRAME_NAME}"
        self._vision_frame_name = f"{self._robot_name}/{VISION_FRAME_NAME}"
        self._tf = TFListenerWrapper(self.get_name() + '_tf',
                                     cache_time_s=30,
                                     wait_for_transform=[self._body_frame_name, self._vision_frame_name])

    def initialize_robot(self) -> bool:
        ''' Returns true/false based on success of claiming and powering up Spot.
        '''
        # Claim robot.
        if not self.claim():
            return False

        # Power the robot.
        if not self.power_motors_on():
            return False
        return True

    def _trigger(self, command_name: str) -> bool:
        """ Triggers a command service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Trigger command '{command_name}' for {self._robot_name}")
        request = Trigger.Request()
        try:
            future = self._trigger_clients[command_name].call_async(request)
        except KeyError:
            self.get_logger().error(f"No trigger command {command_name}")
            return False
        # TODO: Are there other exceptions that should be caught?
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().error(
                f"Trigger command '{command_name}' for {self._robot_name} failed. Msg: {response.message}")
            return False
        return True

    def claim(self) -> bool:
        """ Trigger the 'claim' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger('claim')

    def release(self) -> bool:
        """ Trigger the 'release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger('release')

    def stop(self) -> bool:
        """ Trigger the 'stop' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("stop")

    def self_right(self) -> bool:
        """ Trigger the 'self_right' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("self_right")

    def sit(self) -> bool:
        """ Trigger the 'sit' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("sit")

    def stand(self) -> bool:
        """ Trigger the 'stand' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("stand")

    def rollover(self) -> bool:
        """ Trigger the 'rollover' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("rollover")

    def power_on(self) -> bool:
        """ Trigger the 'power_on' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("power_on")

    # Alias for the power_on function
    power_motors_on = power_on

    def power_off(self) -> bool:
        """ Trigger the 'power_off' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("power_off")

    # Alias for the power_off function
    power_motors_off = power_off

    def estop_hard(self) -> bool:
        """ Trigger the 'estop/hard' service on the SpotDriver. The robot will immediately cut power to the motors

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("estop/hard")

    def estop_gentle(self) -> bool:
        """ Trigger the 'estop/gentle' service on the SpotDriver. The robot will try to settle on the ground before
                cutting power to the motors.

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("estop/hard")

    def estop_release(self) -> bool:
        """ Trigger the 'estop/release' service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        return self._trigger("estop/hard")

    def set_stair_mode(self, value: bool) -> bool:
        """ Set whether to be in stair_mode on the SpotDriver
        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Setting stair_mode to '{value}' for {self._robot_name}")
        request = SetBool.Request()
        request.data = value
        future = self._stair_mode_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self._robot_name} failed. Msg: {response.message}")
            return False
        return True

    def set_locomotion_mode(self, l_hint: LocomotionHint) -> bool:
        """ Set the locomotion mode on the SpotDriver
        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Setting locomotion_mode to '{l_hint}' for {self._robot_name}")
        request = SetLocomotion.Request()
        request.locomotion_mode = int(l_hint)
        future = self._locomotion_mode_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"Setting locomotion_mode to '{l_hint}' for {self._robot_name} response {response}")
            return False
        return True

    def set_max_velocity(self, velocity_limit: Twist) -> SetVelocity.Response:
        """ Set the maximum velocity on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self._robot_name}")
        request = SetVelocity.Request()
        request.velocity = velocity_limit
        future = self._max_velocity_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self._robot_name} response {response}")
            return False
        return True

    def clear_behavior_fault(self, bid: int) -> ClearBehaviorFault.Response:
        """ Clear a behavior fault on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        # TODO: how to set the bid?
        self.get_logger().info(f"Clearing behavior fault {bid} for {self._robot_name}")
        request = ClearBehaviorFault.Request()
        request.id = bid
        future = self._clear_behavior_fault_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"Clearing behavior fault {bid} for {self._robot_name} response {response}")
            return False
        return True

    def list_graph(self, upload_filepath: str) -> ListGraph.Response:
        """  """
        self.get_logger().info(f"List graph {upload_filepath} for {self._robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._list_graph_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"List graph {upload_filepath} for {self._robot_name} response {response}")
        return response

    def list_world_objects(self, p_request) -> ListWorldObjects.Response:
        """  """
        self.get_logger().info(f"List world objects for {self._robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self._clear_behavior_fault_client.call_async(request)
        self._executor.spin_until_future_complete(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"List world objects for {self._robot_name} response {response}")
        return response

    def command(self, goal: RobotCommand.Goal, timeout_sec=None) -> RobotCommand.Result:
        """ Runs the `RobotCommand` action

        Args:
            goal:
            timeout_sec:

        Returns:


        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        return self._robot_command_action_client.send_goal_and_wait(goal, timeout_sec)

    def command_async(self, goal: RobotCommand.Goal, response_callback: Callable) -> Future:
        """ Runs the `RobotCommand` action asynchronously

        Args:
            goal:
            response_callback:

        Returns:

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        goal_future = self._robot_command_action_client.send_goal_async(goal)
        goal_future.add_done_callback(response_callback)
        return goal_future

    def _get_walk_to_goal_action(self, goal_se2: SE2Pose, height: float, base_frame: str) -> RobotCommand.Goal:
        self.get_logger().info("Walking to goal...")
        # Visual

        # TODO: mobility parameters

        if base_frame == BODY_FRAME_NAME:
            try:
                vision_t_body = self._tf.lookup_a_tform_b_se2(self._vision_frame_name, self._body_frame_name)
            except TransformException as e:
                self.get_logger().error(f"Cannot get transform from vision to body for {self._robot_name}. Error was "
                                        f"{e}.")
                return None
            goal_se2 = vision_t_body * goal_se2

        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=goal_se2.x,
            goal_y=goal_se2.y,
            goal_heading=goal_se2.yaw,
            frame_name=VISION_FRAME_NAME,
            body_height=height,
        )

        # Convert proto to bosdyn_msgs command and issue command
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(proto_goal, action_goal.command)
        return action_goal

    def walk_to_goal(self, goal_se2: SE2Pose, height: float = 0, base_frame: str = VISION_FRAME_NAME) -> bool:
        """
        
        Args:
            goal_se2:
            height:
            base_frame:
        """
        action_goal = self._get_walk_to_goal_action(goal_se2, height, base_frame)
        if action_goal is None:
            return False
        if not self._dry:
            return self.command(action_goal)

    def walk_to_goal_async(self, goal_se2, height: float = 0, base_frame: str = VISION_FRAME_NAME) -> bool:
        """

        Args:
            goal_se2:
            height:
            base_frame:
        """
        action_goal = self._get_walk_to_goal_action(goal_se2, height, base_frame)
        if action_goal is None:
            return False
        if not self._dry:
            return self.command_async(action_goal)

    def shutdown(self):
        self.get_logger().info('Shutdown tf listener.')
        self._tf.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot_name", help="Name of the robot if the ROS driver is inside that namespace")
    parser.add_argument("--disable-collisions", action="store_true", help="Flag for disabling collision avoidance")
    parser.add_argument("--dry", action="store_true", help="Flag for if this is a dry run")
    parser.add_argument("--playback", action="store_true", help="Flag for if this playing back a rosbag")
    options = parser.parse_args()

    rclpy.init()
    spot_init = SpotBasic(options)
    spot_init.initialize_robot()
    spot_init.stand()
    time.sleep(1.0)
    spot_init.walk_to_goal(SE2Pose(1, 0, 0), base_frame=BODY_FRAME_NAME)
    time.sleep(1.0)
    spot_init.sit()
    spot_init.shutdown()
    spot_init.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
