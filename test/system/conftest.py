# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import pathlib
from typing import Iterator

import domain_coordinator
import launch_pytest
import pytest
import synchros2.scope as ros_scope
import yaml
from ament_index_python import get_package_share_directory
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_pytest.actions import ReadyToTest
from launch_ros.substitutions import FindPackageShare
from synchros2.scope import ROSAwareScope

from spot_basic import SpotBasic
from spot_basic.testing_prompt_session import TestingPromptSession


@pytest.fixture(scope="session")
def hardware_robot_login(prompt_session: TestingPromptSession) -> tuple[str, str, str]:
    ip = prompt_session.ask("What is the robot's IP?", who="pytest")
    username = prompt_session.ask("What is the robot's username?", who="pytest")
    password = prompt_session.ask("What is the robot's password?", who="pytest")
    return (ip, username, password)


@pytest.fixture(scope="session")
def hardware_robot_name(prompt_session: TestingPromptSession) -> str:
    return prompt_session.ask("What is the robot's name (used as a namespace)?", who="pytest")


@pytest.fixture(scope="session")
def hardware_bound_domain_id() -> Iterator[int]:
    with domain_coordinator.domain_id() as domain_id:
        yield domain_id


@pytest.fixture(scope="session")
def hardware_bound_ros(hardware_bound_domain_id: int) -> Iterator[ROSAwareScope]:
    with ros_scope.top(global_=True, domain_id=hardware_bound_domain_id, uses_tf=True) as top:
        yield top


@pytest.fixture(scope="session")
def hardware_spot_interface(hardware_robot_name: str, hardware_bound_ros: ROSAwareScope) -> Iterator[SpotBasic]:
    robot_interface = SpotBasic(hardware_robot_name, node=hardware_bound_ros.node)
    with robot_interface.leash():
        try:
            yield robot_interface
        finally:
            if robot_interface.has_arm():
                robot_interface.claw_gripper_open.synchronously(timeout_sec=10.0)
                robot_interface.claw_gripper_close.synchronously(timeout_sec=10.0)


@pytest.fixture(scope="session")
def hardware_spot_parameters(hardware_robot_name: str, prompt_session: TestingPromptSession) -> dict:
    share_directory = pathlib.Path(get_package_share_directory("spot_basic"))
    config_file = share_directory / "config" / "spot_ros.testing.yaml"
    config = yaml.safe_load(config_file.read_text())
    parameters = config["/**"]["ros__parameters"]
    parameters["initialize_spot_cam"] = prompt_session.boolean_ask(
        f"does {hardware_robot_name} have a Spot CAM payload?", who="pytest"
    )
    parameters["publish_graph_nav_pose"] = prompt_session.boolean_ask(
        f"was the Graph Nav service initialized on {hardware_robot_name}?", who="pytest"
    )
    return parameters


@launch_pytest.fixture(scope="session")
def hardware_spot_graph_description(
    tmp_path_factory: pytest.TempPathFactory,
    hardware_robot_name: str,
    hardware_robot_login: tuple[str, str, str],
    hardware_bound_domain_id: int,
    hardware_spot_parameters: dict,
) -> LaunchDescription:
    config_file = tmp_path_factory.mktemp("hardware_spot_graph_description") / "spot_ros.yaml"
    config_file.write_text(yaml.dump({"/**": {"ros__parameters": hardware_spot_parameters}}))
    ip, username, password = hardware_robot_login

    return LaunchDescription(
        [
            SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=str(hardware_bound_domain_id)),
            SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value=os.environ["RMW_IMPLEMENTATION"]),
            SetEnvironmentVariable(
                name="FASTRTPS_DEFAULT_PROFILES_FILE", value=os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"]
            ),
            SetEnvironmentVariable(name="BOSDYN_CLIENT_USERNAME", value=username),
            SetEnvironmentVariable(name="BOSDYN_CLIENT_PASSWORD", value=password),
            SetEnvironmentVariable(name="SPOT_IP", value=ip),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [PathJoinSubstitution([FindPackageShare("spot_driver"), "launch/spot_driver.launch.py"])]
                ),
                launch_arguments={
                    "spot_name": hardware_robot_name,
                    "config_file": str(config_file),
                    "depth_registered_mode": "from_spot",
                }.items(),
            ),
            ReadyToTest(),
        ],
    )


def pytest_configure() -> None:
    pytest.hardware_spot_graph_description = hardware_spot_graph_description


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Exempt the system tests from the global pytest-timeout limit.

    These tests drive real hardware and block on interactive prompts, so wall
    clock time is not a meaningful signal of a hang the way it is for the unit
    tests.
    """
    system_test_dir = str(pathlib.Path(__file__).parent)
    for item in items:
        if str(item.fspath).startswith(system_test_dir):
            item.add_marker(pytest.mark.timeout(0))
