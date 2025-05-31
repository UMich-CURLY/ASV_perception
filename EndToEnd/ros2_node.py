import yaml
import os
import torch
from Segmentation.utils import *
from utils import *
import time
import rclpy
from rclpy.node import Node
import ros2_numpy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from tf_transformations import quaternion_matrix
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
from cv_bridge import CvBridge
from PIL import Image as PILImage  # Rename PIL Image
import cv2


class LidarPosesSubscriber(Node):

    def __init__(self, pc_topic, pose_topic, img_topic, res, e2e_net, dev, dtype, voxel_sizes, color, publish=False):
        super().__init__('lidar_poses_image_subscriber')

        self.get_logger().info("Initializing the node!")
        self.publish = publish
        self.bridge = CvBridge()

        # Publishers
        self.map_pub = self.create_publisher(MarkerArray, 'SemMap_global', 10)
        self.var_pub = self.create_publisher(MarkerArray, 'VarMap_global', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, 'filtered_lidar', 10)  # New LiDAR Publisher
        self.next_map = MarkerArray()
        self.var_map = MarkerArray()

        # Message Filters
        self.pc_sub = Subscriber(self, PointCloud2, pc_topic)
        self.pose_sub = Subscriber(self, PoseStamped, pose_topic)
        self.img_sub = Subscriber(self, Image, img_topic)  # New Image Subscriber
        
        self.ts = TimeSynchronizer([self.pc_sub, self.pose_sub, self.img_sub], 5)
        self.ts.registerCallback(self.callback)

        # Other initialization
        self.lidar = None
        self.res = res
        self.seg_input = None
        self.inv = None
        self.lidar_pose = None
        self.e2e_net = e2e_net
        self.dev = dev
        self.dtype = dtype
        self.voxel_sizes = voxel_sizes
        self.color = color
        self.get_logger().info(f"Subscribed to pointcloud topic: {pc_topic}, pose topic: {pose_topic}, image topic: {img_topic}")

    def callback(self, pc_msg, pose_msg, img_msg):
        # Convert PointCloud2 msg to numpy array
        lidar_pc = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)

        # Extract 'xyz' and 'intensity' fields
        xyz = lidar_pc['xyz']  # (N, 3) array
        intensity = lidar_pc['intensity']  # (N, 1) array

        # Initialize a new array to hold XYZ + intensity data
        lidar_raw = np.zeros((xyz.shape[0], 4))
        lidar_raw[:, :3] = xyz  # Populate XYZ
        lidar_raw[:, 3] = intensity[:, 0]  # Populate intensity
        print("lidar_raw.shape: ", lidar_raw.shape)
        # Convert Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.get_logger().info("Received synchronized Image, LiDAR, and Pose data.")

        # Filter LiDAR points based on the camera’s field of view
        projected_pixels, depths, valid_camera_indices = project_lidar_to_image(lidar_raw[:, :3])

        # Check if projected pixels are empty
        if projected_pixels.shape[0] == 0:
            self.get_logger().warn("No valid LiDAR points in camera FOV. Skipping this callback.")
            return  # Exit early and wait for the next callback
        
        print("projected_pixels.shape: ", projected_pixels.shape)
        print("projected_pixels[0]: ", projected_pixels[0])
        # projected_pixels = projected_pixels[valid_camera_indices]

        # Step 3️⃣: Assign class labels based on projection
        img_h, img_w, _ = cv_image.shape
        valid_indices = (
            (0 <= projected_pixels[:, 0]) & (projected_pixels[:, 0] < img_w) &
            (0 <= projected_pixels[:, 1]) & (projected_pixels[:, 1] < img_h)
        )
        print("valid_indices.shape: ", valid_indices.shape)

        # Apply the same filtering to lidar
        self.lidar = lidar_raw[valid_camera_indices]  # Only keep LiDAR points in the image frame
        print("self.lidar.shape: ", self.lidar.shape)

        pose_t = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        pose_quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                              pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        self.lidar_pose = quaternion_matrix(pose_quat)
        self.lidar_pose[:3, 3] = pose_t
        
        # Convert Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.get_logger().info("Received synchronized Image, LiDAR, and Pose data.")

        # Convert OpenCV image to PIL Image
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            # self.seg_input, self.inv = generate_seg_in(self.lidar, self.res)
            # _, _, point_labels = generate_point_labels(self.lidar, self.res, pil_image)
            _, _, point_labels = generate_point_labels(self.lidar, self.res, pil_image, img_msg)

            input_data = [
                torch.tensor(self.lidar_pose).to(self.dev).type(self.dtype),
                torch.tensor(self.lidar).to(self.dev).type(self.dtype),
                point_labels,
                None
            ]
            
            start_t = time.time()
            self.e2e_net(input_data)
            end_t = time.time()
            self.get_logger().info(f"Inference completed in {end_t - start_t:.2f} seconds wall time.")

            if self.publish:
                self.next_map = publish_local_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
                                                  self.voxel_sizes, self.color, self.next_map, 
                                                  self.e2e_net.propagation_net.translation)
                self.map_pub.publish(self.next_map)
                self.var_map = publish_var_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
                                               self.voxel_sizes, self.color, self.var_map, 
                                               self.e2e_net.propagation_net.translation)
                self.var_pub.publish(self.var_map)


def main():
    MODEL_CONFIG = "KITTI"

    # Load model parameters
    model_params_file = os.path.join(os.getcwd(), "Configs", MODEL_CONFIG + ".yaml")
    with open(model_params_file, "r") as stream:
        try:
            model_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    e2e_net = load_model(model_params, dev)
    e2e_net.eval()

    # Initialize ROS2 node
    rclpy.init()
    node = LidarPosesSubscriber(
        pc_topic=model_params["pc_topic"],
        pose_topic=model_params["pose_topic"],
        img_topic=model_params["img_topic"],  # Added Image Topic
        res=model_params["res"],
        e2e_net=e2e_net,
        dev=dev,
        dtype=dtype,
        voxel_sizes=model_params["voxel_sizes"],
        color=model_params["colors"],
        publish=model_params["publish"]
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
