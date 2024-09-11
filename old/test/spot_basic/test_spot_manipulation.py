# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import pytest
from bosdyn.api import manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import ApiGraspOverride
from bosdyn.api.robot_state_pb2 import ManipulatorState
from bosdyn_api_msgs.msg import ManipulationFeedbackState
from sensor_msgs.msg import CameraInfo
from spot_utilities.spot_basic import SpotBasic
from spot_utilities.spot_basic.state import FrameHint
from spot_wrapper.testing.fixtures import SpotFixture

from bdai_ros2_wrappers.futures import wait_for_future
from bdai_ros2_wrappers.scope import ROSAwareScope


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_generic_manipulation(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    tag_frame_name = "tag36h11:530"
    manipulation_request = manipulation_api_pb2.ManipulationApiRequest()
    manipulation_request.pick_object.frame_name = tag_frame_name
    action = spot_interface.perform.asynchronous(manipulation_request)

    request_call = simple_spot.api.ManipulationApi.serve(timeout=2.0)
    assert request_call is not None
    assert request_call.request.WhichOneof("manipulation_cmd") == "pick_object"
    assert request_call.request.pick_object.frame_name == tag_frame_name
    request_response = manipulation_api_pb2.ManipulationApiResponse()
    request_response.manipulation_cmd_id = 1234
    request_call.returns(request_response)

    feedback_call = simple_spot.api.ManipulationApiFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.manipulation_cmd_id == request_response.manipulation_cmd_id
    feedback_response = manipulation_api_pb2.ManipulationApiFeedbackResponse()
    feedback_response.manipulation_cmd_id = feedback_call.request.manipulation_cmd_id
    feedback_response.current_state = manipulation_api_pb2.ManipulationFeedbackState.MANIP_STATE_DONE
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    assert action.result.result.current_state.value == ManipulationFeedbackState.MANIP_STATE_DONE


@pytest.mark.launch(fixture=pytest.simple_spot_without_arm_graph_description)
def test_generic_manipulation_with_no_arm(simple_spot_without_arm: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot_without_arm.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    dummy_manipulation_command = manipulation_api_pb2.ManipulationApiRequest()
    with pytest.raises(RuntimeError):
        spot_interface.perform.synchronous(dummy_manipulation_command)


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_pick_object_in_image(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    image_pixel = [10, 10]
    camera_info = CameraInfo()
    camera_info.width = 640
    camera_info.height = 480
    camera_frame_id = FrameHint.FRONT_RIGHT_CAMERA.value
    camera_info.header.frame_id = camera_frame_id
    camera_info.k = [150.0, 0.0, 200.0, 0.0, 180.0, 300.0, 0.0, 0.0, 1.0]

    action = spot_interface.pick_object_in_image.asynchronously(image_pixel, camera_info)

    request_call = simple_spot.api.ManipulationApi.serve(timeout=2.0)
    assert request_call is not None
    assert request_call.request.WhichOneof("manipulation_cmd") == "pick_object_in_image"
    pick_object_in_image = request_call.request.pick_object_in_image
    assert pick_object_in_image.frame_name_image_sensor == camera_frame_id
    px, py = image_pixel
    assert pick_object_in_image.pixel_xy.x == px
    assert pick_object_in_image.pixel_xy.y == py
    fx, _, cx, _, fy, cy, *_ = camera_info.k
    assert pick_object_in_image.camera_model.intrinsics.focal_length.x == fx
    assert pick_object_in_image.camera_model.intrinsics.focal_length.y == fy
    assert pick_object_in_image.camera_model.intrinsics.principal_point.x == cx
    assert pick_object_in_image.camera_model.intrinsics.principal_point.y == cy
    request_response = manipulation_api_pb2.ManipulationApiResponse()
    request_response.manipulation_cmd_id = 4321
    request_call.returns(request_response)

    feedback_call = simple_spot.api.ManipulationApiFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.manipulation_cmd_id == request_response.manipulation_cmd_id
    feedback_response = manipulation_api_pb2.ManipulationApiFeedbackResponse()
    feedback_response.manipulation_cmd_id = feedback_call.request.manipulation_cmd_id
    feedback_response.current_state = manipulation_api_pb2.ManipulationFeedbackState.MANIP_STATE_GRASP_SUCCEEDED
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    assert action.result.result.current_state.value == ManipulationFeedbackState.MANIP_STATE_GRASP_SUCCEEDED


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_walk_to_object_in_image(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    image_pixel = [40, 20]
    camera_info = CameraInfo()
    camera_info.width = 640
    camera_info.height = 480
    camera_frame_id = FrameHint.FRONT_RIGHT_CAMERA.value
    camera_info.header.frame_id = camera_frame_id
    camera_info.k = [150.0, 0.0, 200.0, 0.0, 180.0, 300.0, 0.0, 0.0, 1.0]
    offset_distance = 1.0

    action = spot_interface.walk_to_object_in_image.asynchronously(
        image_pixel, camera_info, offset_distance=offset_distance
    )

    request_call = simple_spot.api.ManipulationApi.serve(timeout=2.0)
    assert request_call is not None
    assert request_call.request.WhichOneof("manipulation_cmd") == "walk_to_object_in_image"
    walk_to_object_in_image = request_call.request.walk_to_object_in_image
    assert walk_to_object_in_image.frame_name_image_sensor == camera_frame_id
    px, py = image_pixel
    assert walk_to_object_in_image.pixel_xy.x == px
    assert walk_to_object_in_image.pixel_xy.y == py
    fx, _, cx, _, fy, cy, *_ = camera_info.k
    assert walk_to_object_in_image.camera_model.intrinsics.focal_length.x == fx
    assert walk_to_object_in_image.camera_model.intrinsics.focal_length.y == fy
    assert walk_to_object_in_image.camera_model.intrinsics.principal_point.x == cx
    assert walk_to_object_in_image.camera_model.intrinsics.principal_point.y == cy
    assert walk_to_object_in_image.offset_distance.value == 1.0
    request_response = manipulation_api_pb2.ManipulationApiResponse()
    request_response.manipulation_cmd_id = 1324
    request_call.returns(request_response)

    feedback_call = simple_spot.api.ManipulationApiFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.manipulation_cmd_id == request_response.manipulation_cmd_id
    feedback_response = manipulation_api_pb2.ManipulationApiFeedbackResponse()
    feedback_response.manipulation_cmd_id = feedback_call.request.manipulation_cmd_id
    feedback_response.current_state = manipulation_api_pb2.ManipulationFeedbackState.MANIP_STATE_DONE
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    assert action.result.result.current_state.value == ManipulationFeedbackState.MANIP_STATE_DONE


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_pick_object_ray_in_world(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    ray_start = np.array([0.0, 0.0, 0.0])
    ray_end = np.array([-1.0, -1.0, -1.0])
    frame = FrameHint.FRONT_LEFT_CAMERA

    grasp_params = manipulation_api_pb2.GraspParams(grasp_palm_to_fingertip=0.5)

    action = spot_interface.pick_object_ray_in_world.asynchronously(
        ray_start, ray_end, frame, grasp_params=grasp_params
    )

    request_call = simple_spot.api.ManipulationApi.serve(timeout=2.0)
    assert request_call is not None
    assert request_call.request.WhichOneof("manipulation_cmd") == "pick_object_ray_in_world"
    pick_object_ray_in_world = request_call.request.pick_object_ray_in_world
    assert pick_object_ray_in_world.frame_name == frame.value
    ray_start_x, ray_start_y, ray_start_z = ray_start
    assert pick_object_ray_in_world.ray_start_rt_frame.x == ray_start_x
    assert pick_object_ray_in_world.ray_start_rt_frame.y == ray_start_y
    assert pick_object_ray_in_world.ray_start_rt_frame.z == ray_start_z
    ray_end_x, ray_end_y, ray_end_z = ray_end
    assert pick_object_ray_in_world.ray_end_rt_frame.x == ray_end_x
    assert pick_object_ray_in_world.ray_end_rt_frame.y == ray_end_y
    assert pick_object_ray_in_world.ray_end_rt_frame.z == ray_end_z
    grasp_palm_to_fingertip = pick_object_ray_in_world.grasp_params.grasp_palm_to_fingertip
    assert grasp_palm_to_fingertip == grasp_params.grasp_palm_to_fingertip
    request_response = manipulation_api_pb2.ManipulationApiResponse()
    request_response.manipulation_cmd_id = 4231
    request_call.returns(request_response)

    feedback_call = simple_spot.api.ManipulationApiFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.manipulation_cmd_id == request_response.manipulation_cmd_id
    feedback_response = manipulation_api_pb2.ManipulationApiFeedbackResponse()
    feedback_response.manipulation_cmd_id = feedback_call.request.manipulation_cmd_id
    feedback_response.current_state = manipulation_api_pb2.ManipulationFeedbackState.MANIP_STATE_GRASP_SUCCEEDED
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    assert action.result.result.current_state.value == ManipulationFeedbackState.MANIP_STATE_GRASP_SUCCEEDED


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_walk_to_object_ray_in_world(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    ray_start = np.array([0.0, 0.0, 0.0])
    ray_end = np.array([1.0, 1.0, -1.0])
    frame = FrameHint.FRONT_RIGHT_CAMERA
    offset_distance = 1.0

    action = spot_interface.walk_to_object_ray_in_world.asynchronously(ray_start, ray_end, frame, offset_distance)

    request_call = simple_spot.api.ManipulationApi.serve(timeout=2.0)
    assert request_call is not None
    assert request_call.request.WhichOneof("manipulation_cmd") == "walk_to_object_ray_in_world"
    walk_to_object_ray_in_world = request_call.request.walk_to_object_ray_in_world
    assert walk_to_object_ray_in_world.frame_name == frame.value
    ray_start_x, ray_start_y, ray_start_z = ray_start
    assert walk_to_object_ray_in_world.ray_start_rt_frame.x == ray_start_x
    assert walk_to_object_ray_in_world.ray_start_rt_frame.y == ray_start_y
    assert walk_to_object_ray_in_world.ray_start_rt_frame.z == ray_start_z
    ray_end_x, ray_end_y, ray_end_z = ray_end
    assert walk_to_object_ray_in_world.ray_end_rt_frame.x == ray_end_x
    assert walk_to_object_ray_in_world.ray_end_rt_frame.y == ray_end_y
    assert walk_to_object_ray_in_world.ray_end_rt_frame.z == ray_end_z
    assert walk_to_object_ray_in_world.offset_distance.value == offset_distance
    request_response = manipulation_api_pb2.ManipulationApiResponse()
    request_response.manipulation_cmd_id = 4231
    request_call.returns(request_response)

    feedback_call = simple_spot.api.ManipulationApiFeedback.serve(timeout=2.0)
    assert feedback_call is not None
    assert feedback_call.request.manipulation_cmd_id == request_response.manipulation_cmd_id
    feedback_response = manipulation_api_pb2.ManipulationApiFeedbackResponse()
    feedback_response.manipulation_cmd_id = feedback_call.request.manipulation_cmd_id
    feedback_response.current_state = manipulation_api_pb2.ManipulationFeedbackState.MANIP_STATE_DONE
    feedback_call.returns(feedback_response)

    assert wait_for_future(action.finalization, timeout_sec=2.0)
    assert action.succeeded

    assert action.result.success, action.result.message
    assert action.result.result.current_state.value == ManipulationFeedbackState.MANIP_STATE_DONE


@pytest.mark.launch(fixture=pytest.simple_spot_graph_description)
def test_override_grasp_or_carry(simple_spot: SpotFixture, ros: ROSAwareScope) -> None:
    spot_interface = SpotBasic(simple_spot.api.name, node=ros.node)
    assert spot_interface.initialize_robot()

    future = spot_interface.override_grasp_or_carry.asynchronously(
        ApiGraspOverride.Override.OVERRIDE_HOLDING, ManipulatorState.CarryState.CARRY_STATE_CARRIABLE
    )

    call = simple_spot.api.OverrideGrasp.serve(timeout=2.0)
    assert call is not None
    assert call.request.api_grasp_override.override_request == (ApiGraspOverride.Override.OVERRIDE_HOLDING)
    assert call.request.carry_state_override.override_request == (ManipulatorState.CarryState.CARRY_STATE_CARRIABLE)
    call.returns(manipulation_api_pb2.ApiGraspOverrideResponse())

    assert wait_for_future(future, timeout_sec=2.0)
    result = future.result()

    assert result.success, result.message
