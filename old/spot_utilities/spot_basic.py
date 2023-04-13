#  Copyright [2023] Boston Dynamics AI Institute, Inc.
from enum import IntEnum
import sys
from threading import Event
import time
from typing import Callable, Optional

from geometry_msgs.msg import PoseStamped, Twist
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import ColorRGBA as Color
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformException

from bosdyn.api.spot.robot_command_pb2 import MobilityParams, ObstacleParams
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder

import spot_driver.conversions as conv
from spot_msgs.action import NavigateTo, RobotCommand, Trajectory
from spot_msgs.srv import ClearBehaviorFault, ListGraph, ListWorldObjects, SetLocomotion, SetVelocity
from visualization_msgs.msg import Marker, MarkerArray

from bdai.utilities.math_helpers import SE2Pose, SE3Pose
from bdai_ros.utilities.manual_conversions import to_ros_pose
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

    def __init__(self, node_name: Optional[str] = None, num_threads: int = 2):
        if node_name is None:
            node_name = "spot_basic"
        super().__init__(node_name)

        self.declare_parameters(namespace='',
                                parameters=[('robot_name', ''), ('disable_collisions', False), ('dry', False),
                                            ('visualize_goals', False)])
        # Get the robot name
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self._namespace = self.robot_name

        self._disable_collisions = self.get_parameter('disable_collisions').get_parameter_value().bool_value
        self._dry = self.get_parameter('dry').get_parameter_value().bool_value
        self._rviz = self.get_parameter("rviz").get_parameter_value().bool_value

        self._callback_group = ReentrantCallbackGroup()

        self._trigger_clients = {}
        for command_name in self.TRIGGER_SERVICES:
            self._trigger_clients[command_name] = self.create_client(Trigger,
                                                                     f"/{self._namespace}/{command_name}",
                                                                     callback_group=self._callback_group)
        self._stair_mode_client = self.create_client(SetBool,
                                                     f"/{self._namespace}/stair_mode",
                                                     callback_group=self._callback_group)
        self._locomotion_mode_client = self.create_client(SetLocomotion,
                                                          f"/{self._namespace}/locomotion_mode",
                                                          callback_group=self._callback_group)
        self._max_velocity_client = self.create_client(SetVelocity,
                                                       f"/{self._namespace}/max_velocity",
                                                       callback_group=self._callback_group)
        self._clear_behavior_fault_client = self.create_client(ClearBehaviorFault,
                                                               f"/{self._namespace}/clear_behavior_fault",
                                                               callback_group=self._callback_group)
        self._list_graph_client = self.create_client(ListGraph,
                                                     f"/{self._namespace}/list_graph",
                                                     callback_group=self._callback_group)
        self._list_world_objects_client = self.create_client(ListWorldObjects,
                                                             f"/{self._namespace}/list_world_objects",
                                                             callback_group=self._callback_group)
        self._robot_command_action_client = ActionClientWrapper(RobotCommand, f'robot_command', self.get_name(),
                                                                self._namespace)

        self.body_frame_name = f"{self.robot_name}/{BODY_FRAME_NAME}"
        self.vision_frame_name = f"{self.robot_name}/{VISION_FRAME_NAME}"
        self._tf = TFListenerWrapper(self.get_name() + '_tf',
                                     cache_time_s=30,
                                     wait_for_transform=[self.body_frame_name, self.vision_frame_name])

        if self._rviz:
            self._vis_pub = self.create_publisher(Marker, '/visualization_marker', 10)

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

    def _wait_for_future(self, future):
        event = Event()

        def done_callback(_):
            nonlocal event
            event.set()

        future.add_done_callback(done_callback)
        event.wait()

    def _trigger(self, command_name: str) -> bool:
        """ Triggers a command service on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Trigger command '{command_name}' for {self.robot_name}")
        request = Trigger.Request()
        try:
            future = self._trigger_clients[command_name].call_async(request)
        except KeyError:
            self.get_logger().error(f"No trigger command {command_name}")
            return False

        self._wait_for_future(future)

        response = future.result()
        if not response.success:
            self.get_logger().error(
                f"Trigger command '{command_name}' for {self.robot_name} failed. Msg: {response.message}")
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
        self.get_logger().info(f"Setting stair_mode to '{value}' for {self.robot_name}")
        request = SetBool.Request()
        request.data = value
        future = self._stair_mode_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(
                f"Setting stair_mode to '{value}' for {self.robot_name} failed. Msg: {response.message}")
            return False
        return True

    def set_locomotion_mode(self, l_hint: LocomotionHint) -> bool:
        """ Set the locomotion mode on the SpotDriver
        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Setting locomotion_mode to '{l_hint}' for {self.robot_name}")
        request = SetLocomotion.Request()
        request.locomotion_mode = int(l_hint)
        future = self._locomotion_mode_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"Setting locomotion_mode to '{l_hint}' for {self.robot_name} response {response}")
            return False
        return True

    def set_max_velocity(self, velocity_limit: Twist) -> SetVelocity.Response:
        """ Set the maximum velocity on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        self.get_logger().info(f"Setting max_velocity to '{velocity_limit}' for {self.robot_name}")
        request = SetVelocity.Request()
        request.velocity = velocity_limit
        future = self._max_velocity_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(
                f"Setting stair_mode to '{velocity_limit}' for {self.robot_name} response {response}")
            return False
        return True

    def clear_behavior_fault(self, bid: int) -> ClearBehaviorFault.Response:
        """ Clear a behavior fault on the SpotDriver

        Returns:
            True if the command succeeded, False otherwise
        """
        # TODO: how to set the bid?
        self.get_logger().info(f"Clearing behavior fault {bid} for {self.robot_name}")
        request = ClearBehaviorFault.Request()
        request.id = bid
        future = self._clear_behavior_fault_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"Clearing behavior fault {bid} for {self.robot_name} response {response}")
            return False
        return True

    def list_graph(self, upload_filepath: str) -> ListGraph.Response:
        """ TODO """
        self.get_logger().info(f"List graph {upload_filepath} for {self.robot_name}")
        request = ListGraph.Request()
        request.upload_filepath = upload_filepath
        future = self._list_graph_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"List graph {upload_filepath} for {self.robot_name} response {response}")
        return response

    def list_world_objects(self, p_request) -> ListWorldObjects.Response:
        """  TODO """
        self.get_logger().info(f"List world objects for {self.robot_name}")
        request = ListWorldObjects.Request()
        request.request = p_request
        future = self._clear_behavior_fault_client.call_async(request)
        self._wait_for_future(future)
        response = future.result()
        if not response.success:
            self.get_logger().debug(f"List world objects for {self.robot_name} response {response}")
        return response

    def command(self, goal: RobotCommand.Goal, timeout_sec=None) -> RobotCommand.Result:
        """ Runs the `RobotCommand` action

        Args:
            goal:
            timeout_sec:

        Returns:


        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if goal is None:
            # TODO: Is there something better that we can return with information about the failure?
            # maybe a RobotCommand.Result that represents failure?
            return None
        if not self._dry:
            return self._robot_command_action_client.send_goal_and_wait(goal, timeout_sec)
        # TODO: What to return on a dry run? Just success?
        return None

    def command_async(self, goal: RobotCommand.Goal, response_callback: Callable) -> Future:
        """ Runs the `RobotCommand` action asynchronously

        Args:
            goal:
            response_callback:

        Returns:

        See https://github.com/bdaiinstitute/spot_ros2/blob/main/spot_msgs/action/RobotCommand.action
        """
        if goal is None:
            # Is there something better that we can return with information about the failure?
            return None
        if not self._dry:
            goal_future = self._robot_command_action_client.send_goal_async(goal)
            goal_future.add_done_callback(response_callback)
            return goal_future
        # TODO: What to return on a dry run? Just success?
        return None

    def _draw_arrow(self, frame_id: str, ns: str, frame_t_arrow: SE2Pose | SE3Pose, color: Color) -> None:
        """ TODO

        Args:
            frame_id:
            ns:
            frame_t_arrow:
            color
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
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
        self._vis_pub.publish(marker)

    @staticmethod
    def _convert_proto_to_action_goal(proto) -> RobotCommand.Goal:
        action_goal = RobotCommand.Goal()
        conv.convert_proto_to_bosdyn_msgs_robot_command(proto, action_goal.command)
        return action_goal

    def _get_walk_to_action_goal(self, goal_se2: SE2Pose, height: float, base_frame: str) -> RobotCommand.Goal:
        """ TODO

        Args:
            goal_se2:
            height:
            base_frame:

        Returns:

        """
        self.get_logger().info("Walking to goal...")

        # Mobility command
        obstacles = ObstacleParams(disable_vision_body_obstacle_avoidance=self._disable_collisions,
                                   disable_vision_foot_obstacle_avoidance=self._disable_collisions,
                                   disable_vision_foot_constraint_avoidance=self._disable_collisions,
                                   obstacle_avoidance_padding=.001)
        mobility_params = MobilityParams(obstacle_params=obstacles)

        if base_frame == BODY_FRAME_NAME:
            try:
                vision_t_body = self._tf.lookup_a_tform_b_se2(self.vision_frame_name, self.body_frame_name)
            except TransformException as e:
                self.get_logger().error(f"Cannot get transform from vision to body for {self.robot_name}. Error was "
                                        f"{e}.")
                return None
            vision_t_goal = vision_t_body * goal_se2
        else:
            vision_t_goal = goal_se2
        # TODO: are there other frames that we should take care of

        if self._rviz:
            self._draw_arrow(self.vision_frame_name, self.body_frame_name, vision_t_goal, color=Color(g=1.0, a=1.0))

        proto_goal = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=vision_t_goal.x,
                                                                              goal_y=vision_t_goal.y,
                                                                              goal_heading=vision_t_goal.yaw,
                                                                              frame_name=VISION_FRAME_NAME,
                                                                              body_height=height,
                                                                              params=mobility_params)
        return self._convert_proto_to_action_goal(proto_goal)

    def walk_to(self,
                goal_se2: SE2Pose,
                height: float = 0,
                base_frame: str = VISION_FRAME_NAME,
                timeout_sec: float = None) -> Optional[RobotCommand.Result]:
        """ TODO
        
        Args:
            goal_se2:
            height:
            base_frame:

        Returns:

        """
        action_goal = self._get_walk_to_action_goal(goal_se2, height, base_frame)
        return self.command(action_goal, timeout_sec)

    def walk_to_async(self, goal_se2, height: float = 0, base_frame: str = VISION_FRAME_NAME) -> Optional[Future]:
        """ TODO

        Args:
            goal_se2:
            height:
            base_frame:

        Returns:

        """
        action_goal = self._get_walk_to_action_goal(goal_se2, height, base_frame)
        return self.command_async(action_goal)

    def _get_arm_stow_action_goal(self):
        proto = RobotCommandBuilder.arm_stow_command()
        return self._convert_proto_to_action_goal(proto)

    def stow_arm(self) -> Optional[RobotCommand.Result]:
        """ TODO

        Returns:

        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command(action_goal)

    def stow_arm_async(self) -> Optional[RobotCommand.Result]:
        """ TODO

        Returns:

        """
        action_goal = self._get_arm_stow_action_goal()
        return self.command_async(action_goal)

    def _get_arm_ready_action_goal(self):
        proto = RobotCommandBuilder.arm_ready_command()
        return self._convert_proto_to_action_goal(proto)

    def ready_arm(self):
        action_goal = self._get_arm_ready_action_goal()
        return self.command(action_goal)

    def ready_arm_async(self):
        action_goal = self._get_arm_ready_action_goal()
        return self.command_async(action_goal)

    def _get_arm_carry_action_goal(self):
        proto = RobotCommandBuilder.arm_carry_command()
        return self._convert_proto_to_action_goal(proto)

    def carry_arm(self):
        action_goal = self._get_arm_carry_action_goal()
        return self.command(action_goal)

    def carry_arm_async(self):
        action_goal = self._get_arm_carry_action_goal()
        return self.command_async(action_goal)

    def _get_arm_pose_action_goal(self, goal: SE3Pose, base_frame: str, time_sec: float) -> RobotCommand.Goal:
        proto = RobotCommandBuilder.arm_pose_command(x=goal.x,
                                                     y=goal.y,
                                                     z=goal.z,
                                                     qw=goal.rotation.w,
                                                     qx=goal.rotation.x,
                                                     qy=goal.rotation.y,
                                                     qz=goal.rotation.z,
                                                     frame_name=base_frame,
                                                     seconds=time_sec)

        return self._convert_proto_to_action_goal(proto)

    def move_end_effector_to(self, goal: SE3Pose, base_frame: str, time_sec: float):
        action_goal = self._get_arm_pose_action_goal(goal, base_frame, time_sec)
        return self.command(action_goal)

    def move_end_effector_to_async(self, goal: SE3Pose, base_frame: str, time_sec: float):
        action_goal = self._get_arm_pose_action_goal(goal, base_frame, time_sec)
        return self.command_async(action_goal)

    def shutdown(self) -> None:
        """ TODO """
        self.get_logger().info('Shutdown tf listener.')
        self._tf.shutdown()
        # TODO: Is there anything else that we should shutdown?


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
    spot_init.walk_to(SE2Pose(1, 0, 0), base_frame=BODY_FRAME_NAME)
    time.sleep(1.0)
    spot_init.sit()
    spot_init.shutdown()
    spot_init.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
