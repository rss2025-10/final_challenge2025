#!/usr/bin/env python3
import rclpy
import time
import random
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped

BANANA_LOCS = (
    {
        "position": {"x": -5.537989139556885, "y": 25.822851181030273, "z": 0.0},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.7053219448334878,
            "w": 0.7088871236919926,
        },
    },
    {
        "position": {"x": -20.475706100463867, "y": 32.773101806640625, "z": 0.0},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.000904241558244784,
            "w": 0.9999995911735186,
        },
    },
    {
        "position": {"x": -20.35629653930664, "y": 25.77571678161621, "z": 0.0},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": -0.999970764838653,
            "w": 0.007646533070562698,
        },
    },
)

POSITION_NOISE = 0.07
THETA_NOISE = 5

class BasementPointPublisher(Node):
    def __init__(self):
        super().__init__("basement_point_publisher")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.banana_pub = self.create_publisher(
            PoseStamped, "/goal_pose", qos_profile
        )

        selected_locs = random.sample(BANANA_LOCS, 2)

        while self.banana_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for at least 1 matching subscription(s)...")
            time.sleep(1)

        selected_locs = random.sample(BANANA_LOCS, 2)

        for loc in selected_locs:
            time.sleep(1)
            msg = PoseStamped()

            now = self.get_clock().now().to_msg()
            msg.header.stamp = now
            msg.header.frame_id = "map"

            msg.pose.position.x = loc["position"]["x"] + np.random.normal(0, POSITION_NOISE)
            msg.pose.position.y = loc["position"]["y"] + np.random.normal(0, POSITION_NOISE)
            msg.pose.position.z = loc["position"]["z"]

            theta = 2 * np.arctan2(loc["orientation"]["z"], loc["orientation"]["w"])
            theta += np.random.normal(0, np.deg2rad(THETA_NOISE))
            msg.pose.orientation.z = np.sin(theta / 2.0)
            msg.pose.orientation.w = np.cos(theta / 2.0)

            self.get_logger().info(f"Publishing {msg}\n")
            self.banana_pub.publish(msg)


def main():
    rclpy.init()
    node = BasementPointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
