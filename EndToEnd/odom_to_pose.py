import rclpy
from rclpy.node import Node
import tf_transformations as tft
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np

class OdomToPoseTransformer(Node):
    def __init__(self):
        super().__init__('odom_to_pose_transformer')
        
        # Parameters
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')
        self.declare_parameter('pose_topic', '/gt_pose')
        self.declare_parameter('target_frame', 'map')  # The desired output frame
        
        # Get parameters
        self.odom_topic = self.get_parameter('odom_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        
        # Subscriber to odometry
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        
        # Publisher for transformed pose
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        
        # Define a fixed transformation (translation + rotation)
        self.fixed_transform = TransformStamped()
        self.fixed_transform.transform.translation.x = 151.108 # Example: shift x by 1m
        self.fixed_transform.transform.translation.y = -535.164  # Example: shift y by -0.5m
        self.fixed_transform.transform.translation.z = 5.869 
        
        # Example rotation (convert Euler angles to quaternion)
        #roll, pitch, yaw = (0.0, 0.0, np.pi/4)  # Example: rotate 45 degrees (pi/4 rad)
        roll, pitch, yaw = (0.0, -0.01082, -1.0)  # Example: rotate 45 degrees (pi/4 rad)
        q = tft.quaternion_from_euler(roll, pitch, yaw)
        self.fixed_transform.transform.rotation.x = q[0]
        self.fixed_transform.transform.rotation.y = q[1]
        self.fixed_transform.transform.rotation.z = q[2]
        self.fixed_transform.transform.rotation.w = q[3]

    def odom_callback(self, msg):
        # Extract pose from odometry
        pose = msg.pose.pose

        # Convert pose quaternion to transformation matrix
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        T_odom = tft.quaternion_matrix(q)
        T_odom[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        # Convert fixed transformation quaternion to matrix
        q_fixed = [
            self.fixed_transform.transform.rotation.x,
            self.fixed_transform.transform.rotation.y,
            self.fixed_transform.transform.rotation.z,
            self.fixed_transform.transform.rotation.w
        ]
        T_fixed = tft.quaternion_matrix(q_fixed)
        T_fixed[:3, 3] = [
            self.fixed_transform.transform.translation.x,
            self.fixed_transform.transform.translation.y,
            self.fixed_transform.transform.translation.z
        ]

        # Apply transformation: T_final = T_fixed * T_odom
        T_final = np.dot(T_fixed, T_odom)
        
        # Extract new pose
        new_pos = T_final[:3, 3]
        new_quat = tft.quaternion_from_matrix(T_final)

        # Publish transformed pose
        transformed_pose = PoseStamped()
        transformed_pose.header.stamp = msg.header.stamp
        transformed_pose.header.frame_id = self.target_frame
        transformed_pose.pose.position.x = new_pos[0]
        transformed_pose.pose.position.y = new_pos[1]
        transformed_pose.pose.position.z = new_pos[2]
        transformed_pose.pose.orientation.x = new_quat[0]
        transformed_pose.pose.orientation.y = new_quat[1]
        transformed_pose.pose.orientation.z = new_quat[2]
        transformed_pose.pose.orientation.w = new_quat[3]

        self.pose_pub.publish(transformed_pose)
        self.get_logger().info(f'Published transformed pose: {transformed_pose}')

def main(args=None):
    rclpy.init(args=args)
    node = OdomToPoseTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

