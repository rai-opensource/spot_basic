#  Copyright (c) 2023-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import contextlib
import inspect
import unittest
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type
from unittest.mock import Mock

import pytest
import rclpy.node
import synchros2.scope as ros_scope
from domain_coordinator import domain_id
from spot_basic import SpotServiceHandle
from spot_msgs.srv import Dock
from std_srvs.srv import Empty, Trigger
from synchros2.utilities import ensure


@dataclass
class ServiceHandleMockCallbacks:
    result_callback: Mock = field(default_factory=Mock)
    failure_callback: Mock = field(default_factory=Mock)


class SpotServiceHandleTest(unittest.TestCase):
    service_name: str
    service_type: Type

    def setUp(self) -> None:
        self.fixture = contextlib.ExitStack()
        self.domain_id = self.fixture.enter_context(domain_id())
        self.ros = self.fixture.enter_context(ros_scope.top(namespace="fixture", domain_id=self.domain_id))
        self.node: rclpy.node.Node = ensure(self.ros.node)
        self.client = self.node.create_client(self.service_type, self.service_name)

    def setUpServer(self, callback: Callable) -> None:
        self.server = self.node.create_service(self.service_type, self.service_name, callback=callback)

    def doSendRequest(
        self, request: Optional[Type] = None, label: Optional[str] = None
    ) -> Tuple[SpotServiceHandle, ServiceHandleMockCallbacks]:
        if request is None:
            request = self.service_type.Request()
        if label is None:
            label = inspect.stack()[1].function
        self.assertTrue(self.client.wait_for_service(1.0))

        mocks = ServiceHandleMockCallbacks()
        handle = SpotServiceHandle("spot", label, context=self.ros.context)
        handle.set_result_callback(mocks.result_callback)
        handle.set_on_failure_callback(mocks.failure_callback)

        service_future = self.client.call_async(request)
        handle.set_send_service_future(service_future)
        return handle, mocks

    def tearDown(self) -> None:
        self.fixture.close()


class SpotServiceHandleRobotCommandTest(SpotServiceHandleTest):
    service_name = "trigger_service"
    service_type = Trigger

    def test_result(self) -> None:
        """Tests that the result callback iDocks correctly called when the ActionGoal completes successfully"""

        def callback(req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
            result = Trigger.Response()
            result.success = True
            result.message = "Succeeded"
            return result

        self.setUpServer(callback)
        handle, mocks = self.doSendRequest()
        self.assertTrue(handle.wait(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(handle.result.success)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)

    def test_reject(self) -> None:
        """Tests the case where the action server rejects the goal"""

        def callback(req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
            result = Trigger.Response()
            result.success = False
            result.message = "Failed"
            return result

        self.setUpServer(callback)
        handle, mocks = self.doSendRequest()
        self.assertTrue(handle.wait(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertFalse(handle.result.success)
        self.assertTrue(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)


class SpotServiceHandleDockTest(SpotServiceHandleTest):
    service_name = "dock_service"
    service_type = Dock

    def test_success(self) -> None:
        """Tests that the result callback should give a warning"""

        def callback(req: Dock.Request, resp: Dock.Response) -> Dock.Response:
            result = Dock.Response()
            result.success = True
            return result

        self.setUpServer(callback)
        handle, mocks = self.doSendRequest()
        self.assertTrue(handle.wait(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(handle.result.success)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)

    def test_reject(self) -> None:
        """Tests the case where the action server rejects the goal"""

        def callback(req: Dock.Request, resp: Dock.Response) -> Dock.Response:
            result = Dock.Response()
            result.success = False
            return result

        self.setUpServer(callback)
        handle, mocks = self.doSendRequest()
        self.assertTrue(handle.wait(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertFalse(handle.result.success)
        self.assertTrue(mocks.result_callback.called)
        self.assertTrue(mocks.failure_callback.called)


class SpotServiceHandleEmptyTest(SpotServiceHandleTest):
    service_name = "empty_service"
    service_type = Empty

    def test_warning(self) -> None:
        """Tests that the result callback should give a warning"""

        def callback(req: Empty.Request, resp: Empty.Response) -> Empty.Response:
            result = Empty.Response()
            return result

        self.setUpServer(callback)
        handle, mocks = self.doSendRequest()
        self.assertTrue(handle.wait(timeout_sec=pytest.DEFAULT_TIMEOUT))
        self.assertIsNotNone(handle.result)
        self.assertTrue(mocks.result_callback.called)
        self.assertFalse(mocks.failure_callback.called)


if __name__ == "__main__":
    unittest.main()
