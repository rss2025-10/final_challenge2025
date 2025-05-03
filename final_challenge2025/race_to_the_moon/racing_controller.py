#!/usr/bin/env python3
"""
racing_controller.py

A controller for racing (lane following) using a pure pursuit controller.
This node subscribes to a camera image, calls the lane_detector module to
compute the centerline of the lane, and then produces drive commands using a 
pure pursuit formulation.

It follows a structure similar to parking_controller but with the backup
capability removed.
"""

import rclpy
from rclpy.node import Node
import math
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Import the lane detector module.
import lane_detector


class RacingController(Node):
    def __init__(self):
        super().__init__('racing_controller')

        # Declare parameters.
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("camera_topic", "/camera/rgb/image_raw")
        self.declare_parameter("car_length", 0.325)          # meters
        self.declare_parameter("lookahead_distance", 1.0)      # meters, pure pursuit lookahead
        self.declare_parameter("max_speed", 2.0)               # maximum forward speed
        self.declare_parameter("road_width", 3.0)              # estimated road width in meters

        self.drive_topic = self.get_parameter("drive_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.car_length = self.get_parameter("car_length").value
        self.lookahead_distance = self.get_parameter("lookahead_distance").value
        self.max_speed = self.get_parameter("max_speed").value
        self.road_width = self.get_parameter("road_width").value

        # Publisher for drive commands.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        
        # Subscription for camera images.
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)
        
        # CvBridge for converting ROS image msg to OpenCV image.
        self.bridge = CvBridge()

        self.get_logger().info("Racing Controller Initialized")

    def image_callback(self, msg):
        # Convert the incoming ROS Image message to an OpenCV BGR image.
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Convert BGR to RGB as the lane_detector expects an RGB image.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Process the image to detect lanes.
        output_img, x_position, theta = lane_detector.detect_lanes(rgb_image)
        
        # If lane detection fails (e.g. one or both lanes not detected), 
        # then we command a zero speed.
        drive_cmd = AckermannDriveStamped()
        if x_position is None:
            self.get_logger().debug("Lane detection failed; sending zero command.")
            drive_cmd.drive.speed = 0.0
            drive_cmd.drive.steering_angle = 0.0
            self.drive_pub.publish(drive_cmd)
            return

        # Get image dimensions.
        img_height, img_width, _ = rgb_image.shape
        image_center = img_width / 2.0

        # Use the lane detector’s bottom center x-coordinate.
        # Convert pixel error into an estimated real-world lateral offset.
        # (Assume the camera sees about self.road_width meters across the width.)
        pixel_to_meter = self.road_width / img_width
        lateral_offset = (x_position - image_center) * pixel_to_meter

        # Compute the angle to the target in the vehicle coordinate system.
        # The target is assumed to be at a lookahead distance directly ahead with a lateral offset.
        alpha = math.atan2(lateral_offset, self.lookahead_distance)

        # Pure pursuit steering law.
        # Steering angle = arctan(2 * L * sin(alpha) / L_d)
        steering_angle = math.atan2(2.0 * self.car_length * math.sin(alpha), self.lookahead_distance)

        # Optionally reduce speed when steering sharply.
        # For example, linearly reduce speed when the absolute angle exceeds 0.
        speed_reduction_factor = max(0.3, 1 - abs(alpha) / (math.pi / 4))
        speed = self.max_speed * speed_reduction_factor

        # Populate and publish the drive command.
        drive_cmd.drive.speed = float(speed)
        drive_cmd.drive.steering_angle = float(steering_angle)

        # Optionally, log the values.
        self.get_logger().debug(
            f"Lateral offset: {lateral_offset:.3f} m, alpha: {alpha:.3f} rad, "
            f"steering_angle: {steering_angle:.3f} rad, speed: {speed:.3f} m/s"
        )

        self.drive_pub.publish(drive_cmd)

def main(args=None):
    rclpy.init(args=args)
    racing_controller = RacingController()
    rclpy.spin(racing_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
