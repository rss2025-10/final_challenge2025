#!/usr/bin/env python3
from ast import Tuple
from math import pi
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Header
from safe_drive_msgs.msg import SafeDriveMsg


class SafetyController(Node):

    def __init__(self):
        super().__init__("safety_controller")

        self.declare_parameter("safety_topic", "/vesc/low_level/input/safety")
        self.declare_parameter("scan_topic", "/scan")

        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.SAFETY_TOPIC = self.get_parameter("safety_topic").get_parameter_value().string_value

        self.scan_subscriber = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.laser_scan_callback, 10)
        self.safety_publisher = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)

        self.scan_length = None
        self.max_speed = 1.5
        self.speed = 1.5

    def laser_scan_callback(self, msg):
        # self.get_logger().info("Scanning")
        if not self.scan_length:
            self.scan_length = len(msg.ranges)
            self.left_bound = self.scan_length // 6
            self.right_bound = self.scan_length - self.scan_length // 6

        scan = msg
        ranges = scan.ranges

        # calculated range values
        use_range = range(self.left_bound, self.right_bound, 10)

        HARD_STOP_BOUND = 0.7
        # SLOW_BOUND = 0.8

        for k in use_range:
            chunk_end = min(k+10, len(ranges))
            chunk_values = [ranges[i] for i in range(k, chunk_end) if not np.isnan(ranges[i]) and not np.isinf(ranges[i])]

            if len(chunk_values) > 0:
                average = sum(chunk_values) / len(chunk_values)
                if average < HARD_STOP_BOUND:
                    new_msg = AckermannDriveStamped()
                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = "racecar"
                    new_msg.header = header
                    new_msg.drive = AckermannDrive()
                    new_msg.drive.speed = 0.0
                    self.safety_publisher.publish(new_msg)

    def drive_msg_callback(self, msg):
        """Processes the drive message."""
        # get range slicing boundaries

        # new_msg = AckermannDriveStamped()
        # new_msg.header = msg.header
        # new_msg.drive = msg.drive
        # new_msg.drive.speed = self.speed
        msg.drive.speed = self.speed

        # Send drive
        # self.get_logger().info("self.speed: " + str(self.speed))
        self.get_logger().info("self.angle: " + str(msg.drive.steering_angle))
        self.safety_publisher.publish(msg)


def main():
    rclpy.init()
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
