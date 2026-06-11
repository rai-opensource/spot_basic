# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Module containing test fixtures."""

# We disable Pylint warnings for all Protobuf files which contain objects with
# dynamically added member attributes.
# pylint: disable=no-member

# When a test needs a fixture, it must specify the fixture name as a parameter.
# In doings so, Pylint raises an incorrect warning about a name being redefined,
# warning that we want disabled.
# pylint: disable=redefined-outer-name

import os
import pathlib
from typing import Callable, Iterator

import domain_coordinator
import launch_pytest
import pytest
import spot_wrapper.testing
import synchros2.scope as ros_scope
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_pytest.actions import ReadyToTest
from launch_ros.substitutions import FindPackageShare
from spot_wrapper.testing.fixtures import SpotFixture
from spot_wrapper.testing.mocks import MockSpot
from synchros2.launch.actions import update_sigterm_sigkill_timeout
from synchros2.scope import ROSAwareScope

from bdai_core.os.path import get_bdai_data_path

TEST_PATH = pathlib.Path(__file__).parent
os.environ.update(
    {
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "FASTRTPS_DEFAULT_PROFILES_FILE": str(TEST_PATH / "profile.xml"),
    }
)


@spot_wrapper.testing.fixture
class simple_spot(MockSpot):
    """
    This is a factory that returns an instance of the class MockSpot, served by a local GRPC server.
    The MockSpot and the GRPC server are used to handle calls to a simulated Spot robot.
    """

    def __init__(self) -> None:
        super().__init__()
        manipulator_state = self.robot_state.manipulator_state
        manipulator_state.is_gripper_holding_item = False


# pylint: disable=invalid-name,unused-argument
@spot_wrapper.testing.fixture
class simple_spot_without_arm(MockSpot):
    """Same as `simple_spot` with this one does not have an arm."""

    def __init__(self) -> None:
        super().__init__()
        self.robot_state.ClearField("manipulator_state")
        assert not self.robot_state.HasField("manipulator_state")


@pytest.fixture
def domain_id() -> Iterator[int]:
    with domain_coordinator.domain_id() as domain_id:  # to ensure node isolation
        yield domain_id


@pytest.fixture
def ros(domain_id: int) -> Iterator[ROSAwareScope]:
    """
    This method is a generator function that returns a different ROS2 scope
    each time it is invoked, to handle a ROS2 context lifecycle.
    """
    with ros_scope.top(global_=True, domain_id=domain_id, uses_tf=True) as top:
        yield top


@pytest.fixture
def spot_graph_description_factory() -> Callable:
    def make(spot: SpotFixture, domain_id: int, controllable: bool = False) -> LaunchDescription:
        config_file = [PathJoinSubstitution([FindPackageShare("spot_basic"), "config/spot_ros.testing.yaml"])]
        ld = LaunchDescription(
            [
                SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=str(domain_id)),
                SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value=os.environ["RMW_IMPLEMENTATION"]),
                SetEnvironmentVariable(
                    name="FASTRTPS_DEFAULT_PROFILES_FILE", value=os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"]
                ),
                SetEnvironmentVariable(name="BOSDYN_CLIENT_USERNAME", value="spot"),
                SetEnvironmentVariable(name="BOSDYN_CLIENT_PASSWORD", value="spot"),
                SetEnvironmentVariable(name="SPOT_NAME", value=spot.api.name),
                SetEnvironmentVariable(name="SPOT_IP", value=spot.address),
                SetEnvironmentVariable(name="SPOT_PORT", value=str(spot.port)),
                SetEnvironmentVariable(name="SPOT_CERTIFICATE", value=str(spot.certificate_path)),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        [PathJoinSubstitution([FindPackageShare("spot_driver"), "launch/spot_driver.launch.py"])]
                    ),
                    launch_arguments={
                        "spot_name": spot.api.name,
                        "config_file": config_file,
                        "depth_registered_mode": "from_spot",
                        "controllable": str(controllable),
                    }.items(),
                ),
                ReadyToTest(),
            ],
        )
        update_sigterm_sigkill_timeout(ld, sigterm_timeout_s=0.0, sigkill_timeout_s=0.0)
        return ld

    return make


@launch_pytest.fixture
def simple_spot_graph_description(
    spot_graph_description_factory: Callable, simple_spot: SpotFixture, domain_id: int
) -> LaunchDescription:
    return spot_graph_description_factory(simple_spot, domain_id)


@launch_pytest.fixture
def simple_spot_without_arm_graph_description(
    spot_graph_description_factory: Callable, simple_spot_without_arm: SpotFixture, domain_id: int
) -> LaunchDescription:
    return spot_graph_description_factory(simple_spot_without_arm, domain_id)


@launch_pytest.fixture
def simple_controllable_spot_graph_description(
    spot_graph_description_factory: Callable, simple_spot: SpotFixture, domain_id: int
) -> LaunchDescription:
    return spot_graph_description_factory(simple_spot, domain_id, controllable=True)


@pytest.fixture(scope="session")
def test_data_path() -> pathlib.Path:
    return pathlib.Path(get_bdai_data_path()) / "bdai" / "spot_utilities" / "test_data"


def pytest_configure() -> None:
    pytest.simple_spot_graph_description = simple_spot_graph_description
    pytest.simple_spot_without_arm_graph_description = simple_spot_without_arm_graph_description
    pytest.simple_controllable_spot_graph_description = simple_controllable_spot_graph_description
    pytest.DEFAULT_TIMEOUT = 30.0  # CI occassionally just requires super long timeouts
