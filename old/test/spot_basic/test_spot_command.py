# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math

import numpy as np
import pytest
from bdai.trajectory.se2_trajectory import SE2Trajectory
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
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
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.command import LocomotionHint
from spot_utilities.spot_basic.command.full_body import BatteryChangeDirHint
from spot_utilities.spot_basic.state import FrameHint
from spot_wrapper.testing.fixtures import SpotFixture


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_mobility_params(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    assert spot_interface.set_stair_mode.synchronously(True, timeout_sec=5.0)
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
    assert mobility_params.stair_hint is True
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
    assert spot_interface.initialize_robot()

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
    assert spot_interface.initialize_robot()

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
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_start_execution(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()
    assert result.success, result.message
    assert result.robot_command_id == command_response.robot_command_id


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_sit_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    sit_feedback = synchro_feedback.mobility_command_feedback.feedback.sit_feedback
    assert sit_feedback.status.value == SitCommandFeedbackStatus.STATUS_IS_SITTING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_sit_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    sit_feedback = mobility_command_feedback.sit_feedback
    sit_feedback.status = SitCommand.Feedback.Status.STATUS_IS_SITTING
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    result = spot_interface.sit(timeout_sec=2.0)
    assert result is not None

    synchro_feedback = result.command.synchronized_feedback
    sit_feedback = synchro_feedback.mobility_command_feedback.feedback.sit_feedback
    assert sit_feedback.status.value == SitCommandFeedbackStatus.STATUS_IS_SITTING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stand_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_stand_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    result = spot_interface.stand(timeout_sec=2.0)
    assert result is not None

    synchro_feedback = result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_stand_with_orientation_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_stand_with_orientation_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    result = spot_interface.stand_with_orientation(
        gravbody_r_goal=SO3.Rz(45, unit="deg"), body_height=0.5, timeout_sec=2.0
    )
    assert result is not None

    synchro_feedback = result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    se2_trajectory_feedback = synchro_feedback.mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_walk_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    goal = SE2(1.0, -1.0, math.pi / 4.0)
    result = spot_interface.walk(goal, timeout_sec=2.0)
    assert result is not None

    mobility_command_feedback = result.command.synchronized_feedback.mobility_command_feedback
    se2_trajectory_feedback = mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_trajectory_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    se2_trajectory_feedback = synchro_feedback.mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_walk_trajectory_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    trajectory = SE2Trajectory(poses=[SE2(), SE2(1.0, -1.0, math.pi / 4.0)], duration_sec=1.0)
    result = spot_interface.walk_trajectory(trajectory, timeout_sec=2.0)
    assert result is not None

    mobility_command_feedback = result.command.synchronized_feedback.mobility_command_feedback
    se2_trajectory_feedback = mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_and_orient_maneuver(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(future, timeout_sec=2.0)
    walk_result, orient_result = future.result()

    mobility_command_feedback = walk_result.result.command.synchronized_feedback.mobility_command_feedback
    se2_trajectory_feedback = mobility_command_feedback.feedback.se2_trajectory_feedback
    assert se2_trajectory_feedback.status.value == SE2TrajectoryCommandFeedbackStatus.STATUS_AT_GOAL

    synchro_feedback = orient_result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_walk_and_orient_maneuver(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1235
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    se2_trajectory_feedback = mobility_command_feedback.se2_trajectory_feedback
    se2_trajectory_feedback.status = SE2TrajectoryCommand.Feedback.Status.STATUS_AT_GOAL
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    goal = SE3.Rt(SO3.Ry(math.pi / 6.0), [1.0, -1.0, 0.0])
    result = spot_interface.walk_and_orient(goal, timeout_sec=2.0)
    assert result is not None

    synchro_feedback = result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_self_right_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_roll_to_battery_change_pose_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_body_gaze_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded
    assert action.result.success, action.result.message

    synchro_feedback = action.result.result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_legacy_body_gaze_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    simple_spot.api.RobotCommand.future.returns(command_response)

    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    mobility_command_feedback = synchro_feedback.mobility_command_feedback
    mobility_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    stand_feedback = mobility_command_feedback.stand_feedback
    stand_feedback.status = StandCommand.Feedback.Status.STATUS_IS_STANDING
    simple_spot.api.RobotCommandFeedback.future.returns(feedback_response)

    result = spot_interface.body_gaze_at(SE3.Trans([1.0, 1.0, 1.0]), timeout_sec=5.0)
    assert result is not None

    synchro_feedback = result.command.synchronized_feedback
    stand_feedback = synchro_feedback.mobility_command_feedback.feedback.stand_feedback
    assert stand_feedback.status.value == StandCommandFeedbackStatus.STATUS_IS_STANDING


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_spot_walk_with_velocity_command(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

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
