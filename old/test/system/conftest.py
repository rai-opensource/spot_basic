# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import pathlib
from typing import Iterator

import bdai_ros2_wrappers.scope as ros_scope
import domain_coordinator
import launch_pytest
import pytest
import yaml
from ament_index_python import get_package_share_directory
from bdai.test_utilities.testing_prompt_session import TestingPromptSession
from bdai_core.robots.load_robot_configs import (
    get_default_spot_password,
    get_default_spot_username,
    get_spot_config,
    get_spot_ip,
)
from bdai_ros2_wrappers.scope import ROSAwareScope
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_pytest.actions import ReadyToTest
from launch_ros.substitutions import FindPackageShare
from spot_utilities.spot_basic import SpotBasic


@pytest.fixture(scope="session")
def hardware_robot_name(prompt_session: TestingPromptSession) -> str:
    spot_names = list(get_spot_config()["spots"].keys())
    return prompt_session.choose("which robot shall we use?", spot_names, who="pytest")


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
    assert robot_interface.initialize_robot()
    try:
        yield robot_interface
    finally:
        if robot_interface.has_arm():
            robot_interface.stow_arm.synchronously(timeout_sec=10.0)
        robot_interface.sit.synchronously(timeout_sec=10.0)


@pytest.fixture(scope="session")
def hardware_spot_parameters(hardware_robot_name: str, prompt_session: TestingPromptSession) -> dict:
    share_directory = pathlib.Path(get_package_share_directory("spot_utilities"))
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
    hardware_bound_domain_id: int,
    hardware_spot_parameters: dict,
) -> LaunchDescription:
    config_file = tmp_path_factory.mktemp("hardware_spot_graph_description") / "spot_ros.yaml"
    config_file.write_text(yaml.dump({"/**": {"ros__parameters": hardware_spot_parameters}}))

    return LaunchDescription(
        [
            SetEnvironmentVariable(name="ROS_DOMAIN_ID", value=str(hardware_bound_domain_id)),
            SetEnvironmentVariable(name="BOSDYN_CLIENT_USERNAME", value=get_default_spot_username()),
            SetEnvironmentVariable(name="BOSDYN_CLIENT_PASSWORD", value=get_default_spot_password()),
            SetEnvironmentVariable(name="SPOT_NAME", value=hardware_robot_name),
            SetEnvironmentVariable(name="SPOT_IP", value=get_spot_ip(hardware_robot_name)),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [PathJoinSubstitution([FindPackageShare("spot_utilities"), "launch/spot_driver.launch.py"])]
                ),
                launch_arguments={"spot_name": hardware_robot_name, "config_file": str(config_file)}.items(),
            ),
            ReadyToTest(),
        ],
    )


def pytest_configure() -> None:
    pytest.hardware_spot_graph_description = hardware_spot_graph_description
