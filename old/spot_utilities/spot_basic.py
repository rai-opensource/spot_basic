import sys
import time

import rclpy
import spot_driver.conversions as conv
from bdai_ros.wrappers.action_client import ActionClientWrapper
from bdai_ros.wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME
from rclpy.node import Node
from spot_msgs.action import RobotCommand
from spot_utilities.spot_commander import SpotCommander


class SpotBasic(Node):

    def __init__(self, options, process_name=None):
        if process_name is None:
            process_name = "spot_init"
        super().__init__(process_name, namespace=options.robot)

        self._body_frame_name = options.robot + '/' + BODY_FRAME_NAME
        self._vision_frame_name = options.robot + '/' + VISION_FRAME_NAME
        self._tf_listener = TFListenerWrapper("{}_{}".format(process_name, "tf"),
                                              wait_for_transform=[self._body_frame_name, self._vision_frame_name])
        self._robot = SpotCommander(options.robot)
        self._robot_command_client = ActionClientWrapper(RobotCommand, 'robot_command', namespace=options.robot)

    def initialize_robot(self):
        '''Returns true/false based on success of claiming and powering up Spot.
        '''
        # Claim robot.
        self.get_logger().info('Claiming robot ...')
        result = self._robot.command('claim')
        if not result.success:
            self.get_logger().error('Unable to claim robot, message was ' + result.message)
            return False
        self.get_logger().info('Claimed robot.')

        # Power the robot.
        self.get_logger().info('Powering robot on ...')
        result = self._robot.command('power_on')
        if not result.success:
            self.get_logger().error('Unable to power on, robot message was ' + result.message)
            return False
        self.get_logger().info('Robot powered.')

    def stand(self) -> bool:
        '''Returns true/false based on success of standing up Spot.
        '''
        self.get_logger().info('Standing robot up ...')
        result = self._robot.command('stand')
        if not result.success:
            self.get_logger().error('Robot did not stand, message was ' + result.message)
            return False
        self.get_logger().info('Successfully stood up.')

        return True

    def sit(self) -> bool:
        '''Returns true/false based on success sitting down Spot.
        '''
        self.get_logger().info('Sitting robot down ...')
        result = self._robot.command('sit')
        if not result.success:
            self.get_logger().error('Robot did not stand, message was ' + result.message)
            return False
        self.get_logger().info('Successfully sat down.')

        return True

    def shutdown(self):
        self.get_logger().info('Shutdown tf listener.')
        self._tf_listener.shutdown()


def main():
    rclpy.init()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="Name of the robot if the ROS driver is inside that namespace")
    options = parser.parse_args()

    spot_init = SpotBasic(options)
    spot_init.initialize_robot()
    spot_init.stand()
    time.sleep(3.0)
    spot_init.sit()
    spot_init.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
