# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math
from unittest.mock import Mock

import numpy as np
import pytest
from bosdyn.api.basic_command_pb2 import (
    BatteryChangePoseCommand,
    RobotCommandFeedbackStatus,
    SE2TrajectoryCommand,
    SelfRightCommand,
    SitCommand,
    StandCommand,
)
from bosdyn.api.robot_command_pb2 import (
    ClearBehaviorFaultResponse,
    RobotCommandFeedbackResponse,
    RobotCommandResponse,
)
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn_api_msgs.msg import (
    SE2TrajectoryCommandFeedbackStatus,
    SitCommandFeedbackStatus,
    StandCommandFeedbackStatus,
)
from geometry_msgs.msg import Twist, Vector3
from spatialmath import SE2, SE3, SO3, Twist2, UnitQuaternion
from spot_basic import SpotBasic
from spot_basic.command import LocomotionHint, SpotCommandProtocol, StairsMode
from spot_basic.command.arm import SpotArmCommandBuilder
from spot_basic.command.full_body import BatteryChangeDirHint, SpotFullBodyCommandBuilder
from spot_basic.command.mobility import SpotMobilityCommandBuilder
from spot_basic.state import FrameHint
from spot_basic.trajectory.se2_trajectory import SE2Trajectory
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_mobility_params(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    assert spot_interface.set_stairs_mode.synchronously(StairsMode.AUTO, timeout_sec=5.0)
    assert spot_interface.set_locomotion_mode.synchronously(LocomotionHint.TROT, timeout_sec=5.0)
    assert spot_interface.set_max_velocity.synchronously(
        Twist(linear=Vector3(x=1.0), angular=Vector3(z=1.0)), timeout_sec=5.0
    )

    spot_interface.walk_velocity(Twist2.Tx(1.0))
    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_velocity_request"
    mobility_params = spot_command_pb2.MobilityParams()
    synchro_command.mobility_command.params.Unpack(mobility_params)
    assert mobility_params.stairs_mode == spot_command_pb2.MobilityParams.StairsMode.STAIRS_MODE_AUTO
    assert mobility_params.locomotion_hint == spot_command_pb2.HINT_TROT
    assert mobility_params.vel_limit.max_vel.linear.x == 1.0
    assert mobility_params.vel_limit.max_vel.linear.y == 0.0
    assert mobility_params.vel_limit.max_vel.angular == 1.0
    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(response)


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_clear_behavior_fault(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    future = spot_interface.clear_behavior_fault.asynchronously(127)

    call = simple_spot.api.ClearBehaviorFault.serve(timeout=2.0)
    assert call is not None
    assert call.request.behavior_fault_id == 127
    response = ClearBehaviorFaultResponse()
    response.status = ClearBehaviorFaultResponse.Status.STATUS_CLEARED
    call.returns(response)

    assert wait_for_future(future, timeout_sec=5.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stop(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    future = spot_interface.stop.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "full_body_command"
    full_body_command = call.request.command.full_body_command
    assert full_body_command.WhichOneof("command") == "stop_request"
    assert full_body_command.HasField("stop_request")
    response = RobotCommandResponse()
    response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(response)

    assert wait_for_future(future, timeout_sec=5.0)
    result = future.result()
    assert result.success, result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_execute(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.execute.asynchronously(RobotCommandBuilder.synchro_sit_command())

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "sit_request"
    assert synchro_command.mobility_command.HasField("sit_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    sit_feedback = mobility_command_feedback.sit_feedback
    sit_feedback.status = SitCommand.Feedback.Status.STATUS_IS_SITTING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_start_execution(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    future = spot_interface.start_executing.asynchronously(RobotCommandBuilder.synchro_sit_command())

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "sit_request"
    assert synchro_command.mobility_command.HasField("sit_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
    result = future.result()
    assert result.success, result.message
    assert result.robot_command_id == command_response.robot_command_id


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_sit_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.sit.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "sit_request"
    assert synchro_command.mobility_command.HasField("sit_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    sit_feedback = mobility_command_feedback.sit_feedback
    sit_feedback.status = SitCommand.Feedback.Status.STATUS_IS_SITTING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    sit_feedback = synchro_feedback.mobility_command_feedback.feedback.sit_feedback
    assert sit_feedback.status.value == SitCommandFeedbackStatus.STATUS_IS_SITTING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stand_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.stand.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "stand_request"
    assert synchro_command.mobility_command.HasField("stand_request")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stand_with_orientation_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.stand_with_orientation.asynchronously(
        gravbody_r_goal=SO3.Rz(45.0, unit="deg"), body_height=0.5
    )

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "stand_request"
    assert synchro_command.mobility_command.HasField("stand_request")
    mobility_params = spot_command_pb2.MobilityParams()
    synchro_command.mobility_command.params.Unpack(mobility_params)
    assert mobility_params.body_control.WhichOneof("param") == "base_offset_rt_footprint"
    assert mobility_params.body_control.HasField("base_offset_rt_footprint")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    goal = SE2(1.0, -1.0, math.pi / 4.0)
    action = spot_interface.walk.asynchronously(goal)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_trajectory_request"
    assert synchro_command.mobility_command.HasField("se2_trajectory_request")
    trajectory = synchro_command.mobility_command.se2_trajectory_request.trajectory
    assert len(trajectory.points) == 1
    assert trajectory.points[0].pose.position.x == 1.0
    assert trajectory.points[0].pose.position.y == -1.0
    assert trajectory.points[0].pose.angle == math.pi / 4.0
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    se2_trajectory_feedback = synchro_feedback.mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_trajectory_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    trajectory = SE2Trajectory(poses=[SE2(), SE2(1.0, -1.0, math.pi / 4.0)], duration_sec=5.0)
    action = spot_interface.walk_trajectory.asynchronously(trajectory, timestep_sec=0.1)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_trajectory_request"
    assert synchro_command.mobility_command.HasField("se2_trajectory_request")
    trajectory = synchro_command.mobility_command.se2_trajectory_request.trajectory
    assert len(trajectory.points) > 2
    assert trajectory.points[0].pose.position.x == 0.0
    assert trajectory.points[0].pose.position.y == 0.0
    assert trajectory.points[0].pose.angle == 0.0
    assert trajectory.points[-1].pose.position.x == 1.0
    assert trajectory.points[-1].pose.position.y == -1.0
    assert trajectory.points[-1].pose.angle == math.pi / 4.0
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    se2_trajectory_feedback = synchro_feedback.mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_and_orient_maneuver(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    goal = SE3.Rt(SO3.Ry(math.pi / 6.0), [1.0, -1.0, 0.0])
    future = spot_interface.walk_and_orient.asynchronously(goal)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_trajectory_request"
    assert synchro_command.mobility_command.HasField("se2_trajectory_request")
    trajectory = synchro_command.mobility_command.se2_trajectory_request.trajectory
    assert len(trajectory.points) == 1
    assert trajectory.points[0].pose.position.x == 1.0
    assert trajectory.points[0].pose.position.y == -1.0
    assert trajectory.points[0].pose.angle == 0.0
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    call.returns(feedback_response)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "stand_request"
    assert synchro_command.mobility_command.HasField("stand_request")
    mobility_params = spot_command_pb2.MobilityParams()
    synchro_command.mobility_command.params.Unpack(mobility_params)
    assert mobility_params.body_control.WhichOneof("param") == "base_offset_rt_footprint"
    assert mobility_params.body_control.HasField("base_offset_rt_footprint")
    base_offset_rt_footprint = mobility_params.body_control.base_offset_rt_footprint
    assert len(base_offset_rt_footprint.points) == 1
    assert base_offset_rt_footprint.points[0].pose.rotation.y == math.sin(math.pi / 12.0)
    assert base_offset_rt_footprint.points[0].pose.rotation.w == math.cos(math.pi / 12.0)
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1235
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    call.returns(feedback_response)

    assert wait_for_future(future, timeout_sec=pytest.DEFAULT_TIMEOUT)
    walk_result, orient_result = future.result()

    mobility_command_feedback = walk_result.result.command.synchronized_feedback.mobility_command_feedback
    se2_trajectory_feedback = mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL

    synchro_feedback = orient_result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_self_right_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.self_right.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "full_body_command"
    full_body_command = call.request.command.full_body_command
    assert full_body_command.WhichOneof("command") == "selfright_request"
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    full_body_feedback = feedback_response.feedback.full_body_feedback
    full_body_feedback.selfright_feedback.status = SelfRightCommand.Feedback.Status.STATUS_COMPLETED
    full_body_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_roll_to_battery_change_pose_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    action = spot_interface.roll_to_battery_change_pose.asynchronously(BatteryChangeDirHint.LEFT)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "full_body_command"
    full_body_command = call.request.command.full_body_command
    assert full_body_command.WhichOneof("command") == "battery_change_pose_request"
    assert full_body_command.battery_change_pose_request.direction_hint == (
        BatteryChangePoseCommand.Request.DirectionHint.HINT_LEFT
    )
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 4321
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    full_body_feedback = feedback_response.feedback.full_body_feedback
    full_body_feedback.battery_change_pose_feedback.status = BatteryChangePoseCommand.Feedback.Status.STATUS_COMPLETED
    full_body_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_body_gaze_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    x_axis = np.array([1.0, 0.0, 0.0])
    expected_orientation = SO3.Ry(-30.0, unit="deg")

    action = spot_interface.body_gaze.asynchronously(
        np.squeeze(expected_orientation * x_axis), FrameHint.GRAV_ALIGNED_BODY
    )

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "stand_request"
    assert synchro_command.mobility_command.HasField("stand_request")
    mobility_params = spot_command_pb2.MobilityParams()
    synchro_command.mobility_command.params.Unpack(mobility_params)
    assert mobility_params.body_control.WhichOneof("param") == "base_offset_rt_footprint"
    base_offset_rt_footprint = mobility_params.body_control.base_offset_rt_footprint
    assert len(base_offset_rt_footprint.points) == 1
    point = base_offset_rt_footprint.points[0]
    q = UnitQuaternion(expected_orientation)
    qx, qy, qz, qw = q.vec_xyzs
    orientation = point.pose.rotation
    assert math.isclose(orientation.x, qx)
    assert math.isclose(orientation.y, qy)
    assert math.isclose(orientation.z, qz)
    assert math.isclose(orientation.w, qw)
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 4321
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_with_velocity_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.wait_for_servers()
    assert spot_interface.claim.synchronously().success
    assert spot_interface.power_on.synchronously().success

    expected_body_velocity = Twist2([1.0, 0.0, 1.0])
    spot_interface.walk_with_velocity(expected_body_velocity)
    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    synchro_command = call.request.command.synchronized_command
    assert synchro_command.mobility_command.WhichOneof("command") == "se2_velocity_request"
    velocity = synchro_command.mobility_command.se2_velocity_request.velocity
    assert velocity.linear.x == expected_body_velocity.v[0]
    assert velocity.linear.y == expected_body_velocity.v[1]
    assert velocity.angular == expected_body_velocity.w
    call.returns(RobotCommandResponse(status=RobotCommandResponse.Status.STATUS_OK))


def test_whole_body_command_builder() -> None:
    mock_target = Mock(spec=SpotCommandProtocol)
    fb_command_builder = SpotFullBodyCommandBuilder(mock_target)
    # create a torso pose and arm joint angles filled with some dummy values
    quaternion = UnitQuaternion(s=0, v=[0, 0, 1])
    pose = SE3.Rt(R=quaternion.SO3(), t=[1.0, 2.0, 3.0])
    arm_joint_angles = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    command = fb_command_builder.whole_body_command(pose, arm_joint_angles)
    arm_trajectory = command.synchronized_command.arm_command.arm_joint_move_command.trajectory
    assert len(arm_trajectory.points) == 1
    arm_trajectory_point = arm_trajectory.points[0]
    # ensure arm joint angles match the desired input
    assert arm_trajectory_point.position.sh0.value == arm_joint_angles[0]
    assert arm_trajectory_point.position.sh1.value == arm_joint_angles[1]
    assert arm_trajectory_point.position.el0.value == arm_joint_angles[2]
    assert arm_trajectory_point.position.el1.value == arm_joint_angles[3]
    assert arm_trajectory_point.position.wr0.value == arm_joint_angles[4]
    assert arm_trajectory_point.position.wr1.value == arm_joint_angles[5]
    mobility_params = spot_command_pb2.MobilityParams()
    command.synchronized_command.mobility_command.params.Unpack(mobility_params)
    assert len(mobility_params.body_control.base_offset_rt_footprint.points) == 1
    base_offset_point = mobility_params.body_control.base_offset_rt_footprint.points[0]
    # ensure body pose matches the desired input
    assert base_offset_point.pose.position.x == pose.x
    assert base_offset_point.pose.position.y == pose.y
    assert base_offset_point.pose.position.z == pose.z
    assert base_offset_point.pose.rotation.w == quaternion.s
    assert base_offset_point.pose.rotation.x == quaternion.v[0]
    assert base_offset_point.pose.rotation.y == quaternion.v[1]
    assert base_offset_point.pose.rotation.z == quaternion.v[2]


def test_torso_pose_command_builder() -> None:
    mock_target = Mock(spec=SpotCommandProtocol)
    mobility_command_builder = SpotMobilityCommandBuilder(mock_target)
    # create a dummy torso pose command with the optional duration parameter set
    quaternion = UnitQuaternion(s=0, v=[0, 0, 1])
    pose = SE3.Rt(R=quaternion.SO3(), t=[1.0, 2.0, 3.0])
    duration = 1.0
    command = mobility_command_builder.torso_pose_command(pose, duration=duration)
    mobility_params = spot_command_pb2.MobilityParams()
    command.synchronized_command.mobility_command.params.Unpack(mobility_params)
    assert len(mobility_params.body_control.base_offset_rt_footprint.points) == 1
    base_offset_point = mobility_params.body_control.base_offset_rt_footprint.points[0]
    # ensure torso pose and duration are forwarded correctly to the command
    assert base_offset_point.pose.position.x == pose.x
    assert base_offset_point.pose.position.y == pose.y
    assert base_offset_point.pose.position.z == pose.z
    assert base_offset_point.pose.rotation.w == quaternion.s
    assert base_offset_point.pose.rotation.x == quaternion.v[0]
    assert base_offset_point.pose.rotation.y == quaternion.v[1]
    assert base_offset_point.pose.rotation.z == quaternion.v[2]
    assert base_offset_point.time_since_reference.seconds == duration


def test_arm_joint_command_builder() -> None:
    mock_target = Mock(spec=SpotCommandProtocol)
    arm_command_builder = SpotArmCommandBuilder(mock_target)
    # create a dummy arm joint command with duration, max velocity, and max acceleration
    duration_sec = 1.0
    max_velocity = 2.0
    max_acceleration = 3.0
    arm_joint_angles = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    command = arm_command_builder.arm_joint_command(arm_joint_angles, duration_sec, max_velocity, max_acceleration)
    arm_trajectory = command.synchronized_command.arm_command.arm_joint_move_command.trajectory
    # ensure duration, velocity, acceleration, and joint angles are forwarded correctly to the command.
    assert arm_trajectory.maximum_velocity.value == max_velocity
    assert arm_trajectory.maximum_acceleration.value == max_acceleration
    assert len(arm_trajectory.points) == 1
    arm_trajectory_point = arm_trajectory.points[0]
    assert arm_trajectory_point.position.sh0.value == arm_joint_angles[0]
    assert arm_trajectory_point.position.sh1.value == arm_joint_angles[1]
    assert arm_trajectory_point.position.el0.value == arm_joint_angles[2]
    assert arm_trajectory_point.position.el1.value == arm_joint_angles[3]
    assert arm_trajectory_point.position.wr0.value == arm_joint_angles[4]
    assert arm_trajectory_point.position.wr1.value == arm_joint_angles[5]
    assert arm_trajectory_point.time_since_reference.seconds == duration_sec


def test_walk_velocity_command_builder() -> None:
    mock_target = Mock(spec=SpotCommandProtocol)
    # put a locomotion hint into the target's mobility params to test that this is forwarded to the command correctly
    mock_target.mobility_params = spot_command_pb2.MobilityParams(locomotion_hint=LocomotionHint.CRAWL)
    mobility_command_builder = SpotMobilityCommandBuilder(mock_target)
    v_x = 1.0
    v_y = 2.0
    v_rot = 3.0
    # by not specifying mobility params here, we are using the ones from target
    command = mobility_command_builder.walk_velocity_command(v_x, v_y, v_rot)
    velocity_request = command.synchronized_command.mobility_command.se2_velocity_request
    # check that the velocity matches the desired input
    assert velocity_request.velocity.linear.x == v_x
    assert velocity_request.velocity.linear.y == v_y
    assert velocity_request.velocity.angular == v_rot
    mobility_params = spot_command_pb2.MobilityParams()
    command.synchronized_command.mobility_command.params.Unpack(mobility_params)
    # check that the mobility params match that of the target
    assert mobility_params == mock_target.mobility_params


def test_whole_body_including_gripper_command_builder() -> None:
    mock_target = Mock(spec=SpotCommandProtocol)
    fb_command_builder = SpotFullBodyCommandBuilder(mock_target)
    # Uses the same arm joint angles and pose as previous tests. Here we are testing the gripper command portion.
    quaternion = UnitQuaternion(s=0, v=[0, 0, 1])
    pose = SE3.Rt(R=quaternion.SO3(), t=[1.0, 2.0, 3.0])
    arm_joint_angles = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    gripper_angle = -0.5
    command = fb_command_builder.whole_body_command(pose, arm_joint_angles, gripper_angle=gripper_angle)
    arm_trajectory = command.synchronized_command.arm_command.arm_joint_move_command.trajectory
    assert len(arm_trajectory.points) == 1
    arm_trajectory_point = arm_trajectory.points[0]
    # ensure arm joint angles match the desired input
    assert arm_trajectory_point.position.sh0.value == arm_joint_angles[0]
    assert arm_trajectory_point.position.sh1.value == arm_joint_angles[1]
    assert arm_trajectory_point.position.el0.value == arm_joint_angles[2]
    assert arm_trajectory_point.position.el1.value == arm_joint_angles[3]
    assert arm_trajectory_point.position.wr0.value == arm_joint_angles[4]
    assert arm_trajectory_point.position.wr1.value == arm_joint_angles[5]
    mobility_params = spot_command_pb2.MobilityParams()
    command.synchronized_command.mobility_command.params.Unpack(mobility_params)
    assert len(mobility_params.body_control.base_offset_rt_footprint.points) == 1
    base_offset_point = mobility_params.body_control.base_offset_rt_footprint.points[0]
    # ensure body pose matches the desired input
    assert base_offset_point.pose.position.x == pose.x
    assert base_offset_point.pose.position.y == pose.y
    assert base_offset_point.pose.position.z == pose.z
    assert base_offset_point.pose.rotation.w == quaternion.s
    assert base_offset_point.pose.rotation.x == quaternion.v[0]
    assert base_offset_point.pose.rotation.y == quaternion.v[1]
    assert base_offset_point.pose.rotation.z == quaternion.v[2]
    # Ensure gripper trajectory matches desired input
    gripper_trajectory = command.synchronized_command.gripper_command.claw_gripper_command.trajectory
    assert len(gripper_trajectory.points) == 1
    assert gripper_trajectory.points[0].point == gripper_angle
