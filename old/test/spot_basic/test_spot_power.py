# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope
from bosdyn.api.robot_command_pb2 import RobotCommandResponse
from bosdyn.api.robot_state_pb2 import PowerState
from spot_utilities.spot_basic import SpotBasic
from spot_wrapper.testing.fixtures import SpotFixture

MotorPowerState = PowerState.MotorPowerState


@pytest.mark.usefixtures("simple_spot_node")
def test_spot_power_cycle(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    power_state = simple_spot.api.robot_state.power_state

    with spot_interface.lease():
        assert power_state.motor_power_state == MotorPowerState.MOTOR_POWER_STATE_OFF

        assert spot_interface.power_on()
        assert power_state.motor_power_state == MotorPowerState.MOTOR_POWER_STATE_ON

        assert ros.executor is not None
        future = ros.executor.create_task(lambda: spot_interface.power_off())
        call = simple_spot.api.RobotCommand.serve(timeout=5.0)
        assert call is not None
        full_body_command = call.request.command.full_body_command
        assert full_body_command.WhichOneof("command") == "safe_power_off_request"
        power_state.motor_power_state = MotorPowerState.MOTOR_POWER_STATE_OFF
        response = RobotCommandResponse()
        response.status = RobotCommandResponse.Status.STATUS_OK
        call.returns(response)

        assert wait_for_future(future, timeout_sec=5.0)
        assert future.result() is True
        assert power_state.motor_power_state == MotorPowerState.MOTOR_POWER_STATE_OFF

        assert spot_interface.power_on()
        assert power_state.motor_power_state == MotorPowerState.MOTOR_POWER_STATE_ON
