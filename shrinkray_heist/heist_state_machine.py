import rclpy
import math
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from enum import Enum, auto
from geometry_msgs.msg import PoseArray,PoseStamped, PoseWithCovarianceStamped, Quaternion, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool, Header, Float64, String
from sensor_msgs.msg import Image
from shrinkray_heist.red_light_detector import cd_color_segmentation
from vs_msgs.msg import ConeLocation, ConeLocationPixel
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from path_planning.hybrid_a_star import HybridAStarPlanner
import os


# TODO: Import banana detection message type
# from shrinkray_heist_msgs.msg import BananaDetection

class HeistState(Enum):
    IDLE = auto()
    PLANNING = auto()
    FOLLOWING = auto()
    BANANA_DETECTED = auto()
    WAITING = auto()
    FINISHED = auto()

FINISH_RADIUS = 1.0       # meters; when within this radius of the final waypoint, finish (stop)
LOOKAHEAD_DISTANCE = 0.75   # meters; the lookahead distance for pure pursuit
CAR_LENGTH = 0.325         # meters; your vehicle's wheelbase length
SPEED = 1.0

def quaternion_to_yaw(q: Quaternion):
    # Convert quaternion to yaw angle.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class HeistStateMachine(Node):
    """
    High-level state machine for the Shrink Ray Heist (Part A).
    Orchestrates planning, following, and banana detection as per the whiteboard plan.
    """
    def __init__(self):
        super().__init__('heist_state_machine')

        self.lookahead = LOOKAHEAD_DISTANCE     # meters (from constant)
        self.speed = SPEED                      # meters/second (constant speed)
        self.wheelbase_length = CAR_LENGTH

        self.a_star_planner = HybridAStarPlanner(
            xy_resolution=0.5,
            yaw_resolution=math.radians(30)
        )
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, 1)

        self.state = HeistState.IDLE
        self.goal_points = []
        self.current_goal_idx = 0

        self.current_pose = None
        self.at_end = False
        self.current_traj = None

        self.declare_parameter("safety_topic", "/vesc/low_level/input/safety")
        self.declare_parameter('odom_topic', '/vesc/odom')
        self.declare_parameter("drive_topic", "/vesc/input/navigation")

        self.SAFETY_TOPIC = self.get_parameter("safety_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        # Subscribers
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_points_callback, 10)
        self.create_subscription(PoseArray, '/shrinkray_part', self.shrinkray_points_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initial_pose_callback, 10)
        self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.stoplight_publisher = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)
        self.create_subscription(ConeLocationPixel, '/banana_px', self.banana_callback, 10)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.error_pub = self.create_publisher(Float64, "/error", 1)
        self.create_timer(0.1, self.state_monitor)


        # Subscribe to odometry updates.
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pure_pursuit,
            1)

        # Create a publisher for drive commands.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)

        # Timers (for WAITING state)
        self.wait_timer = None
        self.wait_duration = 5.0  # seconds to wait at banana
        
        # Homography transform
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
        
        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info('Heist State Machine Initialized')

    def initial_pose_callback(self, msg):
        """Stores initial pose."""
        self.get_logger().info("Initial pose stored")
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, quaternion_to_yaw(msg.pose.pose.orientation)]

    def state_monitor(self):
        #self.get_logger().info(f"{self.state.name}")
        pass

    def map_cb(self, msg):
        self.get_logger().info("Map recieved. Setting up occupancy grid for planner")
        self.a_star_planner.set_occupancy_grid(msg)
        self.resolution = msg.info.resolution

    def shrinkray_points_callback(self, msg):
        """Process points from BasementPointPublisher for shrinkray part locations"""
        self.get_logger().info("Received shrinkray part locations")

        # Reset current state before planning to new points
        self.goal_points = []
        self.current_goal_idx = 0

        # Extract the points and add them to the goal points
        for pose in msg.poses:
            position = [pose.position.x, pose.position.y, 0.0]  # Setting yaw to 0 since orientation doesn't matter for these waypoints
            self.goal_points.append(position)
            self.get_logger().info(f"Added goal point: {position[0]:.2f}, {position[1]:.2f}")

        # Add the starting position as the final goal to return to
        if self.current_pose is not None:
            self.goal_points.append([self.current_pose[0], self.current_pose[1], self.current_pose[2]])
            self.get_logger().info(f"Added return point: {self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}")

        # If we have goals and a valid pose, start planning
        if self.goal_points and self.current_pose is not None:
            self.state = HeistState.PLANNING
            self.plan_to_next_point()
        else:
            self.get_logger().warn("Waiting for initial pose before planning can begin")

    def goal_points_callback(self, msg):
        """Stores goal points as they come. If a 4th point comes in, resets all points collected.
        Stores whether or not the point is an end point as well."""

        position = [msg.pose.position.x, msg.pose.position.y, quaternion_to_yaw(msg.pose.orientation)]

        if len(self.goal_points) >= 3:
            self.goal_points = []
            self.current_goal_idx = 0

        self.goal_points.append(position)

        if len(self.goal_points) == 3:
            self.state = HeistState.PLANNING
            self.plan_to_next_point()


    def banana_callback(self, msg):
        """If a banana location message is published,
        adjusts behavior appropriately."""

        if self.state == HeistState.FOLLOWING:
            self.state = HeistState.BANANA_DETECTED
            self.get_logger().info('Banana detected! Driving to banana.')


        if self.state == HeistState.BANANA_DETECTED:
            u = msg.u
            v = msg.v

                #Call to main function
            dx, dy = self.transformUvToXy(u, v)

            if np.sqrt(dx**2 + dy**2) < FINISH_RADIUS:
                self.enter_wait_state()

            else:

                alpha = math.atan2(dy, dx)

                if self.lookahead == 0:
                            self.get_logger().error("Lookahead is 0; cannot compute steering.")
                            return

                        # Pure pursuit curvature formula.
                steering_angle = math.atan2(self.wheelbase_length * math.sin(alpha), self.lookahead)

                        #self.get_logger().info(f"Target in vehicle frame: ({local_x:.2f}, {local_y:.2f}), "
                        #                       f"alpha: {alpha:.2f}, steering: {steering_angle:.2f}")

                self.publish_drive_command(self.speed, steering_angle)

    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

    def plan_to_next_point(self):
        """Just calls A Star planner or decides to be finished.
        Should only be called when the car is ready to switch
        to following mode or if it's about to finish."""

        if self.current_goal_idx < len(self.goal_points):
            goal = self.goal_points[self.current_goal_idx]
            self.get_logger().info(f'Planning to point {self.current_goal_idx + 1}/{len(self.goal_points)}')
            self.current_traj = self.a_star_planner.plan_path(self.current_pose, goal)
            if self.current_traj:
                self.state = HeistState.FOLLOWING

        else:

            self.get_logger().warn('No more goal points to plan to.')
            self.state = HeistState.FINISHED

    def pure_pursuit(self, msg):
        """Trajectory follower code. Makes sure car is in following mode."""

        if self.state is not HeistState.FOLLOWING:
            return

        #self.get_logger().info("Pursuing")

        # Get the vehicle pose from odometry.
        self.current_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, quaternion_to_yaw(msg.pose.pose.orientation)]
        vehicle_position = msg.pose.pose.position
        vehicle_orientation = msg.pose.pose.orientation
        vehicle_yaw = quaternion_to_yaw(vehicle_orientation)

        # --- Check if we are close enough to the end of the trajectory ---
        if self.current_traj:
                end_point = self.current_traj[-1]  # (x,y) tuple of the final waypoint
                distance_to_end = math.hypot(end_point[0] - vehicle_position.x,
                                            end_point[1] - vehicle_position.y)

                if distance_to_end < FINISH_RADIUS:
                    self.get_logger().info(f"Reached end of trajectory (within {FINISH_RADIUS} m). Stopping.")
                    self.publish_drive_command(0.0, 0.0)
                    self.stopped = True
                    self.current_goal_idx += 1
                    self.plan_to_next_point()

        # Get the target point (as a Pose) that is at least lookahead distance ahead.
        target_point = self.find_path_target_point(vehicle_position)
        if target_point is None:
                    self.get_logger().warn("No valid target point found. Stopping.")
                    self.publish_drive_command(0.0, 0.0)
                    return

        # Compute the vector from vehicle to target point in world frame.
        dx = target_point.position.x - vehicle_position.x
        dy = target_point.position.y - vehicle_position.y

        # Transform the target point into the vehicle coordinate frame.
        local_x = math.cos(-vehicle_yaw) * dx - math.sin(-vehicle_yaw) * dy
        local_y = math.sin(-vehicle_yaw) * dx + math.cos(-vehicle_yaw) * dy
        error = np.abs(local_y)

        # Compute the angle to the target point relative to the vehicle's x-axis.
        alpha = math.atan2(local_y, local_x)

        if self.lookahead == 0:
            self.get_logger().error("Lookahead is 0; cannot compute steering.")
            return

        # Pure pursuit curvature formula.
        steering_angle = math.atan2(self.wheelbase_length * math.sin(alpha), self.lookahead)

        #self.get_logger().info(f"Target in vehicle frame: ({local_x:.2f}, {local_y:.2f}), "
        #                       f"alpha: {alpha:.2f}, steering: {steering_angle:.2f}")

        #self.get_logger().info("publishing drive command")
        self.publish_drive_command(self.speed, steering_angle)
        error_msg = Float64()
        error_msg.data = error
        self.error_pub.publish(error_msg)


    def enter_wait_state(self):
        """Waiting when banana is detected."""

        self.state = HeistState.WAITING
        self.get_logger().info(f'Waiting {self.wait_duration} seconds at banana.')
        if self.wait_timer:
            self.wait_timer.cancel()
        self.wait_timer = self.create_timer(self.wait_duration, self.wait_done)

    def wait_done(self):
        self.get_logger().info('Done waiting at banana. Setting new goal point (start = current loc).')
        self.current_goal_idx += 1
        self.state = HeistState.PLANNING
        self.plan_to_next_point()
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None


    def find_path_target_point(self, current_position):
        """
        Use the stored trajectory points (which are 2-tuples), and return
        the first point (converted to a Pose) at least lookahead distance ahead.
        If all points are closer, return the last point (as a Pose).
        """
        if not self.current_traj:
            return None

        # Find the index of the trajectory point closest to the current position.
        closest_index = 0
        min_dist = float("inf")
        for i, pt in enumerate(self.current_traj):
            distance = math.hypot(pt[0] - current_position.x, pt[1] - current_position.y)
            if distance < min_dist:
                min_dist = distance
                closest_index = i

        selected = None
        for pt in self.current_traj[closest_index:]:
            distance = math.hypot(pt[0] - current_position.x, pt[1] - current_position.y)
            if distance >= self.lookahead:
                selected = pt
                break

        # If no point meets lookahead, select the last one.
        if selected is None:
            selected = self.current_traj[-1]

        # Convert the (x,y) tuple into a Pose message.
        pose = Pose()
        pose.position.x = selected[0]
        pose.position.y = selected[1]
        return pose

    def image_callback(self, image_msg):
        try:
            # Set OpenCV to not use GUI elements
            cv2.setUseOptimized(True)
            cv2.ocl.setUseOpenCL(False)
            
            # Disable any GUI functionality
            os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "0"
            os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
            
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

    def publish_drive_command(self, speed, steering_angle):
        """
        Helper function that publishes an AckermannDriveStamped command.
        """
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = HeistStateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
