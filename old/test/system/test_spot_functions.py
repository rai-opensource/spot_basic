# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pytest
from spot_msgs.msg import PowerState
from spot_utilities.spot_basic import SpotBasic
from synchros2.futures import wait_for_future


@pytest.mark.manual
@pytest.mark.launch(fixture=pytest.hardware_spot_graph_description)
def test_power_control(hardware_spot_interface: SpotBasic) -> None:
    assert hardware_spot_interface.power_motors_on.synchronously(timeout_sec=10.0).success
    assert wait_for_future(
        hardware_spot_interface.power_states.matching_update(
            lambda message: message.motor_power_state == PowerState.STATE_ON
        ),
        timeout_sec=5.0,
    )
    assert hardware_spot_interface.power_motors_off.synchronously(timeout_sec=10.0).success
    assert wait_for_future(
        hardware_spot_interface.power_states.matching_update(
            lambda message: message.motor_power_state == PowerState.STATE_OFF
        ),
        timeout_sec=5.0,
    )
    assert hardware_spot_interface.power_motors_on.synchronously(timeout_sec=10.0).success
    assert wait_for_future(
        hardware_spot_interface.power_states.matching_update(
            lambda message: message.motor_power_state == PowerState.STATE_ON
        ),
        timeout_sec=5.0,
    )
