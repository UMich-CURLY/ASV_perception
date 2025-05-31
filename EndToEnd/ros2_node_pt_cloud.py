import yaml
import os
import torch
from Segmentation.utils import *
from utils import *
import time
import rclpy
from rclpy.node import Node
import ros2_numpy
from sensor_msgs.msg import PointCloud2, Image, PointField
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from tf_transformations import quaternion_matrix
from message_filters import Subscriber, TimeSynchronizer
import numpy as np
from cv_bridge import CvBridge
from PIL import Image as PILImage  # Rename PIL Image
import cv2
import struct  # Needed for proper color packing

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
        self.pc_pub = self.create_publisher(PointCloud2, 'semantic_map', 10)  # Publisher for PointCloud2

        self.global_cloud = None  # To store the accumulated point cloud
        self.initialize_global_grid()  # Initialize world with default points


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
    
    def initialize_global_grid(self, grid_size=50, step=1.0, default_color=(128, 128, 128)):
        """ Initializes the global map with a uniform grid of points. """
        x_range = np.arange(-grid_size, grid_size, step)
        y_range = np.arange(-grid_size, grid_size, step)
        z_range = np.arange(-5, 5, step)  # Example: 10m vertical height

        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

        # Assign initial color (e.g., gray)
        r, g, b = default_color
        rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        colors = np.full((points.shape[0], 1), rgb_packed, dtype=np.float32)

        # Store in global cloud
        self.global_cloud = np.hstack((points, colors))
        print(f"Initialized global grid with {self.global_cloud.shape[0]} points.")

        # Publish once
        self.publish_global_pointcloud()

    def publish_global_pointcloud(self):
        """ Publishes the global point cloud. """
        if self.global_cloud is None:
            return

        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header.stamp = self.get_clock().now().to_msg()
        pc2_msg.header.frame_id = "map"

        # Define PointCloud2 fields
        pc2_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # Set cloud size
        pc2_msg.width = self.global_cloud.shape[0]
        pc2_msg.height = 1
        pc2_msg.is_dense = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.data = self.global_cloud.tobytes()

        # Publish global point cloud
        self.pc_pub.publish(pc2_msg)

    # def callback(self, pc_msg, pose_msg, img_msg):
    #     # Convert PointCloud2 msg to numpy array
    #     lidar_pc = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)

    #     # Extract 'xyz' and 'intensity' fields
    #     xyz = lidar_pc['xyz']  # (N, 3) array
    #     intensity = lidar_pc['intensity']  # (N, 1) array

    #     # Initialize a new array to hold XYZ + intensity data
    #     lidar_raw = np.zeros((xyz.shape[0], 4))
    #     lidar_raw[:, :3] = xyz  # Populate XYZ
    #     lidar_raw[:, 3] = intensity[:, 0]  # Populate intensity
    #     print("lidar_raw.shape: ", lidar_raw.shape)

    #     # Convert Image message to OpenCV format
    #     cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    #     self.get_logger().info("Received synchronized Image, LiDAR, and Pose data.")

    #     # Filter LiDAR points based on the camera’s field of view
    #     projected_pixels, depths, valid_camera_indices = project_lidar_to_image(lidar_raw[:, :3])

    #     # Check if projected pixels are empty
    #     if projected_pixels.shape[0] == 0:
    #         self.get_logger().warn("No valid LiDAR points in camera FOV. Skipping this callback.")
    #         return  # Exit early and wait for the next callback

    #     # Apply the same filtering to lidar
    #     self.lidar = lidar_raw[valid_camera_indices]  # Only keep LiDAR points in the image frame

    #     pose_t = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
    #     pose_quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
    #                         pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
    #     self.lidar_pose = quaternion_matrix(pose_quat)
    #     self.lidar_pose[:3, 3] = pose_t

    #     # Convert Image message to OpenCV format
    #     cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

    #     # Convert OpenCV image to PIL Image
    #     pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
    #     with torch.no_grad():
    #         _, _, point_labels = generate_point_labels(self.lidar, self.res, pil_image, img_msg)

    #         input_data = [
    #             torch.tensor(self.lidar_pose).to(self.dev).type(self.dtype),
    #             torch.tensor(self.lidar).to(self.dev).type(self.dtype),
    #             point_labels,
    #             None
    #         ]
            
    #         start_t = time.time()
    #         self.e2e_net(input_data)
    #         end_t = time.time()
    #         self.get_logger().info(f"Inference completed in {end_t - start_t:.2f} seconds wall time.")

    #         if self.publish:
    #             # Generate and publish local map
    #             self.next_map = publish_local_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
    #                                             self.voxel_sizes, self.color, self.next_map, 
    #                                             self.e2e_net.propagation_net.translation)
    #             self.map_pub.publish(self.next_map)

    #             # Generate and publish variance map
    #             self.var_map = publish_var_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
    #                                         self.voxel_sizes, self.color, self.var_map, 
    #                                         self.e2e_net.propagation_net.translation)
    #             self.var_pub.publish(self.var_map)

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

        # Apply the same filtering to lidar
        self.lidar = lidar_raw[valid_camera_indices]  # Only keep LiDAR points in the image frame

        # Extract pose from PoseStamped message
        pose_t = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        pose_quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                            pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        self.lidar_pose = quaternion_matrix(pose_quat)
        self.lidar_pose[:3, 3] = pose_t

        # Convert OpenCV image to PIL Image
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            # Generate point labels
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


            # Optionally, continue using the semantic map and variance map publishing
            if self.publish:
                # Generate the local semantic map (this will return a single marker)
                marker = publish_local_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
                                            self.voxel_sizes, self.color, None, self.e2e_net.propagation_net.translation)

                # Extract points and colors from the marker
                points = []
                rgb_colors = []

                for point, color in zip(marker.points, marker.colors):
                    points.append([point.x, point.y, point.z])

                    # Convert normalized (0.0-1.0) RGB values to 0-255 scale
                    r = int(color.r * 255)
                    g = int(color.g * 255)
                    b = int(color.b * 255)

                    # Pack into a float field (compatible with PointCloud2)
                    rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
                    rgb_colors.append(rgb_packed)

                # Convert points and colors into numpy arrays
                cloud_points = np.array(points, dtype=np.float32)
                cloud_colors = np.array(rgb_colors, dtype=np.float32).reshape(-1, 1)  # Ensure shape is correct

                # Combine XYZ and RGB fields
                pc2_data = np.hstack((cloud_points, cloud_colors))

                # Create PointCloud2 message
                pc2_msg = PointCloud2()
                pc2_msg.header.stamp = pc_msg.header.stamp
                pc2_msg.header.frame_id = "map"  

                # Define PointCloud2 fields
                pc2_msg.fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)  # Stored as float for visualization
                ]

                # Set width and height for unordered cloud
                pc2_msg.width = cloud_points.shape[0]
                pc2_msg.height = 1  

                # Set point step and row step
                pc2_msg.point_step = 16  # Each point has 4 floats (4 bytes each)
                pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width  

                # Convert data to byte format
                pc2_msg.data = pc2_data.tobytes()

                # Publish the PointCloud2 message
                self.pc_pub.publish(pc2_msg)

                # Optionally, continue using the semantic map and variance map publishing
                if self.publish:
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
