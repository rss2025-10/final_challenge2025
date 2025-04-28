import rclpy
from rclpy.node import Node
from enum import Enum, auto
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

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
        
        # Subscribers
        self.create_subscription(PoseArray, '/shrinkray_part', self.goal_points_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
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


def main(args=None):
    rclpy.init(args=args)
    node = HeistStateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 