#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from vs_msgs.msg import ConeLocationPixel, ConeLocation  # Using the same message type for now

# Import the YOLO detector
from shrinkray_heist.model.detector import Detector

# POINTS WERE GATHERED IN A CLOCKWISE MANNER, STARTING FROM THE TOP LEFT CORNER OF THE PAPER
######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_IMAGE_PLANE = [[270.0, 255.0],
                   [436.0, 254.0],
                   [509.0, 325.0],
                   [244.0, 328.0]] # dummy points
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[22, 3.25],
                    [22, -7.75],
                    [13.5, -7.75],
                    [13.5, 3.25]] # dummy points
######################################################

METERS_PER_INCH = 0.0254


class BananaDetector(Node):
    """
    A class for applying banana detection using YOLO model to the robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_banana_px (ConeLocationPixel) : the coordinates of the banana in the image frame (units are pixels).
    Also publishes a debug image with detection visualization.
    """
    def __init__(self):
        super().__init__("banana_detector")

        # Initialize the YOLO detector
        self.detector = Detector(threshold=0.3)

        # Publishers
        self.banana_pub = self.create_publisher(ConeLocationPixel, "/banana_px", 10)
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)

        # Subscribe to ZED camera RGB frames
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()  # Converts between ROS images and OpenCV Images

        # Add a counter to track frames processed
        self.frame_count = 0
        self.detection_count = 0

        self.get_logger().info("Banana Detector Initialized")
        self.get_logger().info(f"Available classes in YOLO model: {self.detector.classes}")

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info("Homography Transformer Initialized")

    def image_callback(self, image_msg):
        """
        Process incoming images to detect bananas using YOLO.
        Publishes the coordinates of detected bananas and a debug image.
        """
        try:
            # Count frames
            self.frame_count += 1

            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Log frame processing
            if self.frame_count % 10 == 0:
                self.get_logger().info(f"Processing frame #{self.frame_count}, image shape: {image.shape}")

            # Run YOLO detection
            results = self.detector.predict(image)
            predictions = results["predictions"]

            # If no predictions at all, return early
            if not predictions:
                if self.frame_count % 30 == 0:
                    self.get_logger().info("No objects detected in this frame")
                return

            # Log all detected objects for debugging
            detected_classes = [pred[1] for pred in predictions]
            if detected_classes:
                self.get_logger().info(f"Detected objects: {detected_classes}")

            # Filter predictions to only include bananas
            banana_predictions = [pred for pred in predictions if pred[1] == "banana"]

            if not banana_predictions:
                if self.frame_count % 30 == 0:
                    self.get_logger().info("No bananas detected in this frame")
                return

            # Increment detection counter and log
            self.detection_count += 1

            # Get the first banana detection
            (xmin, ymin, xmax, ymax), _ = banana_predictions[0]

            # Calculate the center pixel at the bottom of the bounding box
            # This is the point that corresponds to the ground plane
            u = (xmin + xmax) / 2  # center x-coordinate
            v = float(ymax)  # bottom y-coordinate

            # Log banana detection with coordinates
            self.get_logger().info(f"BANANA DETECTED #{self.detection_count}")
            self.get_logger().info(f"Bounding box: ({int(xmin)}, {int(ymin)}) to ({int(xmax)}, {int(ymax)})")
            self.get_logger().info(f"Ground point: u={int(u)}, v={int(v)}")

            # ######## HOMOGRAPHY TRANSFORM HERE

            # Create and publish the banana location message
            banana_msg = ConeLocationPixel()
            banana_msg.u = u
            banana_msg.v = v
            self.banana_pub.publish(banana_msg)

            # Create debug image with bounding box and detection point
            debug_img = image.copy()

            # Draw the bounding box and bottom center point on the debug image
            cv2.rectangle(debug_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(u), int(v)), 5, (0, 0, 255), -1)

            # Add text label
            cv2.putText(debug_img, "banana", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Publish the debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")





def main(args=None):
    rclpy.init(args=args)
    banana_detector = BananaDetector()
    rclpy.spin(banana_detector)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
