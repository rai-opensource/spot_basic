# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math

import pytest
from bosdyn.api.arm_command_pb2 import (
    ArmCartesianCommand,
    ArmImpedanceCommand,
    ArmJointMoveCommand,
    GazeCommand,
    NamedArmPositionsCommand,
)
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.robot_command_pb2 import RobotCommandFeedbackResponse, RobotCommandResponse
from bosdyn.api.spot.robot_command_pb2 import MobilityParams
from bosdyn_api_msgs.msg import GazeCommandFeedbackStatus
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.spatialvector import SpatialVector
from spot_basic import SpotBasic
from spot_basic.command.arm import AxisModeMap
from spot_basic.state import ArmJointMap, FrameHint
from spot_wrapper.testing.fixtures import SpotFixture
from synchros2.futures import wait_for_future
from synchros2.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_deployment(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    for robot_command_id, (actuate_arm, named_position) in enumerate(
        (
            (spot_interface.ready_arm, NamedArmPositionsCommand.Positions.POSITIONS_READY),
            (spot_interface.carry_arm, NamedArmPositionsCommand.Positions.POSITIONS_CARRY),
            (spot_interface.stow_arm, NamedArmPositionsCommand.Positions.POSITIONS_STOW),
        )
    ):
        action = actuate_arm.asynchronously()

        call = simple_spot.api.RobotCommand.serve(timeout=2.0)
        assert call is not None
        assert call.request.command.WhichOneof("command") == "synchronized_command"
        arm_command = call.request.command.synchronized_command.arm_command
        assert arm_command.WhichOneof("command") == "named_arm_position_command"
        assert arm_command.named_arm_position_command.position == named_position
        command_response = RobotCommandResponse()
        command_response.robot_command_id = robot_command_id
        command_response.status = RobotCommandResponse.Status.STATUS_OK
        call.returns(command_response)

        call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
        assert call is not None
        assert call.request.robot_command_id == robot_command_id
        feedback_response = RobotCommandFeedbackResponse()
        synchro_feedback = feedback_response.feedback.synchronized_feedback
        arm_command_feedback = synchro_feedback.arm_command_feedback
        arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
        arm_command_feedback.named_arm_position_feedback.status = (
            NamedArmPositionsCommand.Feedback.Status.STATUS_COMPLETE
        )
        call.returns(feedback_response)

        assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
        assert action.succeeded

        assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_locking(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    action = spot_interface.lock_arm.asynchronously()

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_joint_move_command"
    assert len(arm_command.arm_joint_move_command.trajectory.points) == 1
    assert not arm_command.arm_joint_move_command.trajectory.points[0].HasField("position")
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_joint_move_feedback.status = ArmJointMoveCommand.Feedback.Status.STATUS_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_gazing(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    target = x, y, z = (2.0, 3.0, 1.0)

    action = spot_interface.arm_gaze.asynchronously(target, FrameHint.VISION)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_gaze_command"
    assert arm_command.arm_gaze_command.frame1_name == FrameHint.VISION.value
    target_trajectory = arm_command.arm_gaze_command.target_trajectory_in_frame1
    assert len(target_trajectory.points) == 1
    assert target_trajectory.points[0].point.x == x
    assert target_trajectory.points[0].point.y == y
    assert target_trajectory.points[0].point.z == z
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 4321
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_gaze_feedback.status = GazeCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message
    synchro_feedback = action.result.result.command.synchronized_feedback
    arm_gaze_feedback = synchro_feedback.arm_command_feedback.feedback.arm_gaze_feedback
    assert arm_gaze_feedback.status.value == GazeCommandFeedbackStatus.STATUS_TRAJECTORY_COMPLETE


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_motion(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    target_configuration = ArmJointMap(sh0=0.5, sh1=0.0, el0=0.0, el1=0.0, wr0=-0.5, wr1=0.0)
    action = spot_interface.move_arm.asynchronously(target_configuration, duration_sec=5.0)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_joint_move_command"
    assert len(arm_command.arm_joint_move_command.trajectory.points) == 1
    trajectory_point = arm_command.arm_joint_move_command.trajectory.points[0]
    assert trajectory_point.position.sh0.value == target_configuration.sh0
    assert trajectory_point.position.sh1.value == target_configuration.sh1
    assert trajectory_point.position.el0.value == target_configuration.el0
    assert trajectory_point.position.el1.value == target_configuration.el1
    assert trajectory_point.position.wr0.value == target_configuration.wr0
    assert trajectory_point.position.wr1.value == target_configuration.wr1
    assert trajectory_point.time_since_reference.seconds == 5
    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_joint_move_feedback.status = ArmJointMoveCommand.Feedback.Status.STATUS_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_trajectory(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    times = [0.0, 1.0, 2.0]
    positions = [
        ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.0, wr1=0.0),
        ArmJointMap(sh0=0.5, sh1=0.0, el0=0.0, el1=0.0, wr0=-0.5, wr1=0.0),
        ArmJointMap(sh0=1.0, sh1=0.0, el0=0.0, el1=0.0, wr0=-1.0, wr1=0.0),
    ]
    velocities = [
        ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.0, wr1=0.0),
        ArmJointMap(sh0=0.25, sh1=0.0, el0=0.0, el1=0.0, wr0=-0.25, wr1=0.0),
        ArmJointMap(sh0=0.0, sh1=0.0, el0=0.0, el1=0.0, wr0=0.0, wr1=0.0),
    ]

    action = spot_interface.move_arm_along_trajectory.asynchronously(
        joint_positions=positions, joint_velocities=velocities, times=times
    )

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_joint_move_command"
    trajectory_points = arm_command.arm_joint_move_command.trajectory.points
    assert len(trajectory_points) == 3

    for point, position, velocity, time in zip(trajectory_points, positions, velocities, times, strict=False):
        assert point.position.sh0.value == position.sh0
        assert point.position.sh1.value == position.sh1
        assert point.position.el0.value == position.el0
        assert point.position.el1.value == position.el1
        assert point.position.wr0.value == position.wr0
        assert point.position.wr1.value == position.wr1

        assert point.velocity.sh0.value == velocity.sh0
        assert point.velocity.sh1.value == velocity.sh1
        assert point.velocity.el0.value == velocity.el0
        assert point.velocity.el1.value == velocity.el1
        assert point.velocity.wr0.value == velocity.wr0
        assert point.velocity.wr1.value == velocity.wr1

        assert point.time_since_reference.seconds == time

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_joint_move_feedback.status = ArmJointMoveCommand.Feedback.Status.STATUS_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_end_effector_motion(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    duration_sec = 5.0
    vision_t_goal = SE3.Rt(SO3.Rz(45.0, unit="deg"), [0.5, 0.5, 1.0])
    action = spot_interface.move_end_effector.asynchronously(vision_t_goal, duration_sec)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_cartesian_command"
    assert arm_command.arm_cartesian_command.root_frame_name == FrameHint.VISION.value
    assert len(arm_command.arm_cartesian_command.pose_trajectory_in_task.points) == 1
    trajectory_point = arm_command.arm_cartesian_command.pose_trajectory_in_task.points[0]
    vision_t_goal_q = UnitQuaternion(vision_t_goal.R)
    qx, qy, qz, qw = vision_t_goal_q.vec_xyzs
    x, y, z = vision_t_goal.t
    assert trajectory_point.pose.position.x == x
    assert trajectory_point.pose.position.y == y
    assert trajectory_point.pose.position.z == z
    assert trajectory_point.pose.rotation.w == qw
    assert trajectory_point.pose.rotation.x == qx
    assert trajectory_point.pose.rotation.y == qy
    assert trajectory_point.pose.rotation.z == qz
    assert trajectory_point.time_since_reference.seconds == int(duration_sec)
    mobility_command = call.request.command.synchronized_command.mobility_command
    assert mobility_command.WhichOneof("command") == "stand_request"

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1234
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_cartesian_feedback.status = ArmCartesianCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_end_effector_and_body_motion(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    duration_sec = 10.0
    vision_t_goal = SE3.Rt(SO3(), [1.0, 1.0, 1.0])
    action = spot_interface.move_end_effector.asynchronously(vision_t_goal, duration_sec, body_follow=True)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_cartesian_command"
    assert arm_command.arm_cartesian_command.root_frame_name == FrameHint.VISION.value
    assert len(arm_command.arm_cartesian_command.pose_trajectory_in_task.points) == 1
    trajectory_point = arm_command.arm_cartesian_command.pose_trajectory_in_task.points[0]
    assert trajectory_point.pose.position.x == vision_t_goal.x
    assert trajectory_point.pose.position.y == vision_t_goal.y
    assert trajectory_point.pose.position.z == vision_t_goal.z
    assert trajectory_point.pose.rotation.w == 1.0
    assert trajectory_point.time_since_reference.seconds == duration_sec
    mobility_command = call.request.command.synchronized_command.mobility_command
    assert mobility_command.WhichOneof("command") == "follow_arm_request"

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 4321
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_cartesian_feedback.status = ArmCartesianCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_end_effector_and_body_hip_motion(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    duration_sec = 7.5
    vision_t_goal = SE3.Rt(SO3.Rz(45.0, unit="deg"), [1.0, 1.0, 1.0])
    action = spot_interface.move_end_effector.asynchronously(vision_t_goal, duration_sec, body_hip_assist=True)

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_cartesian_command"
    assert arm_command.arm_cartesian_command.root_frame_name == FrameHint.VISION.value
    assert len(arm_command.arm_cartesian_command.pose_trajectory_in_task.points) == 1
    trajectory_point = arm_command.arm_cartesian_command.pose_trajectory_in_task.points[0]
    vision_t_goal_q = UnitQuaternion(vision_t_goal.R)
    qx, qy, qz, qw = vision_t_goal_q.vec_xyzs
    x, y, z = vision_t_goal.t
    assert trajectory_point.pose.position.x == x
    assert trajectory_point.pose.position.y == y
    assert trajectory_point.pose.position.z == z
    assert trajectory_point.pose.rotation.w == qw
    assert trajectory_point.pose.rotation.x == qx
    assert trajectory_point.pose.rotation.y == qy
    assert trajectory_point.pose.rotation.z == qz
    fraction, seconds = math.modf(duration_sec)
    nanos = int(fraction * 1e9)
    assert trajectory_point.time_since_reference.seconds == seconds
    assert trajectory_point.time_since_reference.nanos == nanos
    mobility_command = call.request.command.synchronized_command.mobility_command
    assert mobility_command.WhichOneof("command") == "stand_request"
    mobility_params = MobilityParams()
    mobility_command.params.Unpack(mobility_params)
    assert mobility_params.body_control.body_assist_for_manipulation.enable_hip_height_assist

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 1324
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_cartesian_feedback.status = ArmCartesianCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_end_effector_trajectory(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    timestamps = [0.0, 10.0]
    poses = [SE3(), SE3.Rt(SO3.Rz(45.0, unit="deg"), [2.0, 1.0, 1.0])]
    action = spot_interface.move_end_effector_along_trajectory.asynchronously(
        pose_vecs=poses,
        timestamps_between_poses=timestamps,
        axis_mode_map=AxisModeMap.position_mode(),
        root_frame=FrameHint.VISION,
        body_follow_arm=True,
    )

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_cartesian_command"
    arm_cartesian_command = arm_command.arm_cartesian_command
    assert arm_cartesian_command.root_frame_name == FrameHint.VISION.value

    assert len(arm_cartesian_command.pose_trajectory_in_task.points) == 2
    trajectory_points = arm_cartesian_command.pose_trajectory_in_task.points
    for point, pose, timestamp in zip(trajectory_points, poses, timestamps, strict=False):
        x, y, z = pose.t
        q = UnitQuaternion(pose.R)
        qx, qy, qz, qw = q.vec_xyzs

        assert point.pose.position.x == x
        assert point.pose.position.y == y
        assert point.pose.position.z == z
        assert point.pose.rotation.w == qw
        assert point.pose.rotation.x == qx
        assert point.pose.rotation.y == qy
        assert point.pose.rotation.z == qz

        assert point.time_since_reference.seconds == timestamp

    assert arm_cartesian_command.x_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION
    assert arm_cartesian_command.y_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION
    assert arm_cartesian_command.z_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION
    assert arm_cartesian_command.rx_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION
    assert arm_cartesian_command.ry_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION
    assert arm_cartesian_command.rz_axis == ArmCartesianCommand.Request.AxisMode.AXIS_MODE_POSITION

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 9876
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_cartesian_feedback.status = ArmCartesianCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_arm_end_effector_impedance_controlled_trajectory(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    trajectory = [(SE3(), 0.0), (SE3.Rt(SO3.Rz(45.0, unit="deg"), [0.5, 0.5, 1.0]), 10.0)]
    stiffness = SpatialVector([100, 100, 100, 10, 10, 10])
    damping = SpatialVector([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    action = spot_interface.move_end_effector_with_controlled_impedance.asynchronously(
        trajectory, root_frame=FrameHint.ODOM, stiffness=stiffness, damping=damping
    )

    call = simple_spot.api.RobotCommand.serve(timeout=2.0)
    assert call is not None
    assert call.request.command.WhichOneof("command") == "synchronized_command"
    arm_command = call.request.command.synchronized_command.arm_command
    assert arm_command.WhichOneof("command") == "arm_impedance_command"
    assert arm_command.arm_impedance_command.root_frame_name == FrameHint.ODOM.value
    assert arm_command.arm_impedance_command.diagonal_stiffness_matrix.values == list(stiffness.A)
    assert arm_command.arm_impedance_command.diagonal_damping_matrix.values == list(damping.A)
    trajectory_points = arm_command.arm_impedance_command.task_tform_desired_tool.points
    assert len(trajectory_points) == 2

    for (pose, timestamp), point in zip(trajectory, trajectory_points, strict=False):
        x, y, z = pose.t
        q = UnitQuaternion(pose.R)
        qx, qy, qz, qw = q.vec_xyzs

        assert point.pose.position.x == x
        assert point.pose.position.y == y
        assert point.pose.position.z == z
        assert point.pose.rotation.w == qw
        assert point.pose.rotation.x == qx
        assert point.pose.rotation.y == qy
        assert point.pose.rotation.z == qz

        assert point.time_since_reference.seconds == timestamp

    command_response = RobotCommandResponse()
    command_response.robot_command_id = 6789
    command_response.status = RobotCommandResponse.Status.STATUS_OK
    call.returns(command_response)

    call = simple_spot.api.RobotCommandFeedback.serve(timeout=2.0)
    assert call is not None
    assert call.request.robot_command_id == command_response.robot_command_id
    feedback_response = RobotCommandFeedbackResponse()
    synchro_feedback = feedback_response.feedback.synchronized_feedback
    arm_command_feedback = synchro_feedback.arm_command_feedback
    arm_command_feedback.status = RobotCommandFeedbackStatus.Status.STATUS_PROCESSING
    arm_command_feedback.arm_impedance_feedback.status = ArmImpedanceCommand.Feedback.Status.STATUS_TRAJECTORY_COMPLETE
    call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=pytest.DEFAULT_TIMEOUT)
    assert action.succeeded

    assert action.result.success, action.result.message
