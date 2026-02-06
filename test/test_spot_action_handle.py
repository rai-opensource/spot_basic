#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import contextlib
import inspect
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type
from unittest.mock import Mock

import pytest
import synchros2.scope as ros_scope
from domain_coordinator import domain_id
from rclpy.action import ActionClient, ActionServer
from rclpy.action.server import CancelResponse, GoalResponse, ServerGoalHandle
from spot_basic import SpotActionHandle
from spot_msgs.action import Manipulation, RobotCommand
from synchros2.action_handle import ActionHandle

from bdai_ros.utilities.custom_types import Action

GOAL_ORDER = 5


@dataclass
class ActionHandleMockCallbacks:
    result_callback: Mock = field(default_factory=Mock)
    failure_callback: Mock = field(default_factory=Mock)
    feedback_callback: Mock = field(default_factory=Mock)
    cancel_success_callback: Mock = field(default_factory=Mock)
    cancel_failure_callback: Mock = field(default_factory=Mock)


class SpotActionHandleTest(unittest.TestCase):
    action_name: str = ""
    action_type: Type[Action] = type(Action)

    def execute_and_succeed_callback(self, goal_handle: ServerGoalHandle) -> Action.Result:
        goal_handle.succeed()
        result = self.action_type.Result()
        result.success = True
        result.message = "Succeeded"
        return result

    def execute_slowly_and_succeed_callback(self, goal_handle: ServerGoalHandle) -> Action.Result:
        time.sleep(0.5)
        return self.execute_and_succeed_callback(goal_handle)

    def execute_but_abort_callback(self, goal_handle: ServerGoalHandle) -> Action.Result:
        goal_handle.abort()
        result = self.action_type.Result()
        result.success = False
        result.message = "Aborted"
        return result

    def execute_with_feedback_callback(self, goal_handle: ServerGoalHandle) -> Action.Result:
        feedback = self.action_type.Feedback()

        for _ in range(GOAL_ORDER):
            goal_handle.publish_feedback(feedback)
            time.sleep(0.01)
        time.sleep(0.05)

        goal_handle.succeed()
        result = self.action_type.Result()
        result.success = True
        result.message = "Succeeded"
        return result

    def execute_until_canceled_callback(self, goal_handle: ServerGoalHandle) -> Action.Result:
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = self.action_type.Result()
                result.success = False
                result.message = "Canceled"
                return result
            time.sleep(0.001)

    def goal_rejection_callback(self, goal: Action.Goal) -> GoalResponse:
        return GoalResponse.REJECT

    def cancel_accept_callback(self, cancel_request: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def setUp(self) -> None:
        self.fixture = contextlib.ExitStack()
        self.domain_id = self.fixture.enter_context(domain_id())
        self.ros = self.fixture.enter_context(ros_scope.top(namespace="fixture", domain_id=self.domain_id))
        self.client = ActionClient(self.ros.node, self.action_type, self.action_name)

    def setUpServer(self, execute_callback: Optional[Callable] = None, **kwargs: Any) -> None:
        if execute_callback is None:
            execute_callback = self.execute_and_succeed_callback
        self.server = ActionServer(self.ros.node, self.action_type, self.action_name, execute_callback, **kwargs)

    def doSendGoal(
        self, goal: Optional[Action.Goal] = None, label: Optional[str] = None
    ) -> Tuple[ActionHandle, ActionHandleMockCallbacks]:
        if goal is None:
            goal = self.action_type.Goal()
        if label is None:
            label = inspect.stack()[1].function
        self.assertTrue(self.client.wait_for_server(1.0))

        mocks = ActionHandleMockCallbacks()

        handle = SpotActionHandle("spot", label, context=self.ros.context)
        handle.set_result_callback(mocks.result_callback)
        handle.set_on_failure_callback(mocks.failure_callback)
        handle.set_feedback_callback(mocks.feedback_callback)
        handle.set_on_cancel_success_callback(mocks.cancel_success_callback)
        handle.set_on_cancel_failure_callback(mocks.cancel_failure_callback)

        send_goal_future = self.client.send_goal_async(goal, feedback_callback=handle.get_feedback_callback)
        handle.set_send_goal_future(send_goal_future)
        return handle, mocks

    def tearDown(self) -> None:
        self.fixture.close()


class SpotActionHandleRobotCommandTest(SpotActionHandleTest):
    action_name = "robot_command"
    action_type = RobotCommand

    def test_result(self) -> None:
        """Tests that the result callback is correctly called when the ActionGoal completes successfully"""
        self.setUpServer()
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_reject(self) -> None:
        """Tests the case where the action server rejects the goal"""
        self.setUpServer(goal_callback=self.goal_rejection_callback)
        handle, mocks = self.doSendGoal()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_abort(self) -> None:
        """Tests the case where the action server aborts the goal"""
        self.setUpServer(execute_callback=self.execute_but_abort_callback)
        handle, mocks = self.doSendGoal()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_feedback(self) -> None:
        self.setUpServer(execute_callback=self.execute_with_feedback_callback)
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertGreater(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_cancel_success(self) -> None:
        self.setUpServer(
            execute_callback=self.execute_until_canceled_callback, cancel_callback=self.cancel_accept_callback
        )
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_acceptance(timeout_sec=pytest.DEFAULT_TIMEOUT))
        handle.cancel()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertTrue(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_cancel_failure(self) -> None:
        self.setUpServer(execute_callback=self.execute_slowly_and_succeed_callback)
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_acceptance(timeout_sec=pytest.DEFAULT_TIMEOUT))
        handle.cancel()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertTrue(mocks.cancel_failure_callback.called)


class SpotActionHandleManipulationTest(SpotActionHandleTest):
    action_name = "manipulation"
    action_type = Manipulation

    def test_result(self) -> None:
        """Tests that the result callback is correctly called when the ActionGoal completes successfully"""
        self.setUpServer()
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_reject(self) -> None:
        """Tests the case where the action server rejects the goal"""
        self.setUpServer(goal_callback=self.goal_rejection_callback)
        handle, mocks = self.doSendGoal()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_abort(self) -> None:
        """Tests the case where the action server aborts the goal"""
        self.setUpServer(execute_callback=self.execute_but_abort_callback)
        handle, mocks = self.doSendGoal()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_feedback(self) -> None:
        self.setUpServer(execute_callback=self.execute_with_feedback_callback)
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertGreater(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_cancel_success(self) -> None:
        self.setUpServer(
            execute_callback=self.execute_until_canceled_callback, cancel_callback=self.cancel_accept_callback
        )
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_acceptance(timeout_sec=pytest.DEFAULT_TIMEOUT))
        handle.cancel()
        self.assertFalse(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNone(handle.result)
        self.assertFalse(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertTrue(mocks.cancel_success_callback.called)
        self.assertFalse(mocks.cancel_failure_callback.called)

    def test_cancel_failure(self) -> None:
        self.setUpServer(execute_callback=self.execute_slowly_and_succeed_callback)
        handle, mocks = self.doSendGoal()
        self.assertTrue(handle.wait_for_acceptance(timeout_sec=pytest.DEFAULT_TIMEOUT))
        handle.cancel()
        self.assertTrue(handle.wait_for_result(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)
        self.assertEqual(mocks.feedback_callback.call_count, 0)
        self.assertFalse(mocks.cancel_success_callback.called)
        self.assertTrue(mocks.cancel_failure_callback.called)


if __name__ == "__main__":
    unittest.main()

# FIXME(WUD-2731)
pytestmark = pytest.mark.skip(reason="Temporarily disabled")
