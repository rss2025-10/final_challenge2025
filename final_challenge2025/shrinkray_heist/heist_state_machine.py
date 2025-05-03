import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from enum import Enum, auto
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image
from shrinkray_heist.red_light_detector import cd_color_segmentation
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


# TODO: Import banana detection message type
# from shrinkray_heist_msgs.msg import BananaDetection

class HeistState(Enum):
    IDLE = auto()
    PLANNING = auto()
    FOLLOWING = auto()
    BANANA_DETECTED = auto()
    WAITING = auto()
    FINISHED = auto()

class HeistStateMachine(Node):
    """
    High-level state machine for the Shrink Ray Heist (Part A).
    Orchestrates planning, following, and banana detection as per the whiteboard plan.
    """
    def __init__(self):
        super().__init__('heist_state_machine')

        # State
        self.state = HeistState.IDLE
        self.goal_points = []  # List of Pose
        self.current_goal_idx = 0
        self.current_pose = None
        self.banana_detected = False
        self.at_end = False

        self.declare_parameter("safety_topic", "/safety_topic")

        self.SAFETY_TOPIC = self.get_parameter("safety_topic").get_parameter_value().string_value

        # Subscribers
        self.create_subscription(PoseArray, '/shrinkray_part', self.goal_points_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.stoplight_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.stoplight_publisher = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        # TODO: Subscribe to banana detection topic
        # self.create_subscription(BananaDetection, '/banana_detection', self.banana_callback, 10)

        # Publishers
        self.trajectory_pub = self.create_publisher(PoseArray, '/trajectory/current', 10)
        # TODO: Optionally, publish drive commands directly if not using follower node

        # Timers (for WAITING state)
        self.wait_timer = None
        self.wait_duration = 5.0  # seconds to wait at banana

        self.get_logger().info('Heist State Machine Initialized')

    def goal_points_callback(self, msg: PoseArray):
        self.goal_points = list(msg.poses)
        self.current_goal_idx = 0
        self.state = HeistState.PLANNING
        self.get_logger().info(f'Received {len(self.goal_points)} goal points. Starting plan to first point.')
        self.plan_to_next_point()

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        # TODO: Check if at goal/end, update state if needed
        if self.state == HeistState.FOLLOWING and self.is_at_goal():
            if self.current_goal_idx >= len(self.goal_points) - 1:
                self.state = HeistState.FINISHED
                self.get_logger().info('Reached final goal. Stopping and disabling banana search.')
                # TODO: Stop the car, disable banana search
            else:
                self.current_goal_idx += 1
                self.state = HeistState.PLANNING
                self.plan_to_next_point()

    def banana_callback(self, msg):
        # TODO: Update this for actual banana detection message
        if msg.detected:
            self.banana_detected = True
            if self.state == HeistState.FOLLOWING:
                self.state = HeistState.BANANA_DETECTED
                self.get_logger().info('Banana detected! Driving to banana.')
                # TODO: Drive to banana, then wait
                self.enter_wait_state()

    def plan_to_next_point(self):
        if self.current_goal_idx < len(self.goal_points):
            goal = self.goal_points[self.current_goal_idx]
            self.get_logger().info(f'Planning to point {self.current_goal_idx + 1}/{len(self.goal_points)}')
            # TODO: Call planner node/service to plan from current_pose to goal
            # Publish planned trajectory to /trajectory/current
            self.state = HeistState.FOLLOWING
            # TODO: Start follower node if needed
        else:
            self.get_logger().warn('No more goal points to plan to.')
            self.state = HeistState.FINISHED

    def is_at_goal(self):
        # TODO: Implement logic to check if current_pose is close to current goal
        return False

    def enter_wait_state(self):
        self.state = HeistState.WAITING
        self.get_logger().info(f'Waiting {self.wait_duration} seconds at banana.')
        if self.wait_timer:
            self.wait_timer.cancel()
        self.wait_timer = self.create_timer(self.wait_duration, self.wait_done)

    def wait_done(self):
        self.get_logger().info('Done waiting at banana. Setting new goal point (start = current loc).')
        # TODO: Set new goal point, set start as current location
        self.state = HeistState.PLANNING
        self.plan_to_next_point()
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None

    def image_callback(self, image_msg):
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Apply color segmentation to detect the cone
            stop_detected = cd_color_segmentation(image)
            if stop_detected:
                new_drive_msg = AckermannDrive()
                new_msg = AckermannDriveStamped()
                new_drive_msg.speed = 0.0
                new_msg.drive = new_drive_msg
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "racecar"
                new_msg.header = header
                self.stoplight_publisher.publish(new_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"No cone detected: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HeistStateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
