#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped

class GTPoseToTF(Node):
    def __init__(self):
        super().__init__('gt_pose_to_tf_broadcaster')

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribe to the /gt_pose topic
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/gt_pose',
            self.pose_callback,
            10  # QoS profile depth
        )

        # Frame names
        self.parent_frame = "map"  # Change this if needed
        self.child_frame = "new_frame"

    def pose_callback(self, msg):
        # Create the TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # Copy pose data
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        # Publish the transform
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = GTPoseToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

