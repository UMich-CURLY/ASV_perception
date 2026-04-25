import yaml
import os
import torch
from utils import *
import time
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from tf_transformations import quaternion_matrix
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
import numpy as np
from cv_bridge import CvBridge
import struct  # Needed for proper color packing
import cv2


class ConvBKIMap(Node):

    def __init__(self, model_params, res, e2e_net, dev, dtype, voxel_sizes, color, publish=False):
        super().__init__('convBKI_map_node')

        self.get_logger().info("Initializing the node!")
        self.publish = publish
        self.bridge = CvBridge()

        # self.fixed_class_id_mapping = {
        #     "tree": 0,
        #     "grass - shore": 1,
        #     "concreteplatform": 2,
        #     "water": 3,
        #     "redbuoy": 4,
        #     "blackbuoy": 5,
        #     "greenbuoy": 6,
        #     "orangebuoy": 7,
        #     "whitebuoy": 8,
        # }
        self.num_classes = model_params["num_classes"]
        self.ros_topic = model_params["ros_parameters"]

        # Publishers
        self.overlay_pub = self.create_publisher(Image, "/debug/lidar_camera_overlay", 10)
        # self.map_pub = self.create_publisher(MarkerArray, self.ros_topic["map_topic"], 10)
        self.var_pub = self.create_publisher(MarkerArray, self.ros_topic["var_topic"], 10)
        self.next_map = MarkerArray()
        self.var_map = MarkerArray()
        self.pc_pub = self.create_publisher(PointCloud2, self.ros_topic["semantic_pcd_topic"], 10)  # Publisher for PointCloud2

        self.global_cloud = None  # To store the accumulated point cloud
        self.initialize_global_grid()  # Initialize world with default points

        # Subscriber Message Filters
        self.filt_pcd_sub = Subscriber(self, PointCloud2, self.ros_topic["filt_pcd_topic"])  # Filtered LiDAR pcd
        self.pose_sub = Subscriber(self, PoseStamped, self.ros_topic["pose_topic"])          # Vehicle Pose
        self.mask_sub = Subscriber(self, Image, self.ros_topic["det_mask_topic"])            # Segmentation Mask
        self.annot_sub = Subscriber(self, Image, self.ros_topic["annot_topic"])              # annotated img from cam_det 

        # self.ts = TimeSynchronizer([self.filt_pcd_sub, self.pose_sub, self.mask_sub], 10)
        self.ts = ApproximateTimeSynchronizer(
            [self.filt_pcd_sub, self.pose_sub, self.mask_sub, self.annot_sub],
            queue_size=10,
            slop=0.05,
        )
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

        self.get_logger().info(
            "Map Publisher Node up. Subscribed to:\n"
            f"  filtered pcd topic: {self.ros_topic['filt_pcd_topic']}\n"
            f"  pose topic: {self.ros_topic['pose_topic']}\n"
            f"  mask topic: {self.ros_topic['det_mask_topic']}"
        )
    
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

    def create_point_labels(self, mask_msg):        
        """
        Assign per-point semantic labels to LiDAR points using 2D detections
        and segmentation masks in the camera frame.

        Args:
            mask_msg (sensor_msgs.msg.Image):            // segmentation mask
                mask pixel 0:       background;
                          (k+1):    class index k

        Returns:
            point_labels (np.ndarray):
                One-hot encoded labels of shape (N, num_classes) for projected LiDAR points.
        """
        
        # Convert Image message to OpenCV format
        label_mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        combined_mask = np.asarray(label_mask, dtype=np.uint8)

        # Initialize point labels (N, num_classes) → One-hot encoding
        point_labels = np.zeros((self.proj_pix.shape[0], self.num_classes), dtype=np.float32)

        for i, (u, v) in enumerate(self.proj_pix):
            if combined_mask[int(v), int(u)] > 0:  # Check if point falls inside any mask
                class_idx = combined_mask[int(v), int(u)] - 1  # Subtract 1 to get correct class index
                point_labels[i, class_idx] = 1.0
        
        return point_labels

    def publish_overlay_image(self, annot_msg):
        annot_img = self.bridge.imgmsg_to_cv2(annot_msg, desired_encoding="bgr8")

        u = self.proj_pix[:, 0]
        v = self.proj_pix[:, 1]
        depth = self.filtered_lidar[:, 2]

        h, w = annot_img.shape[:2]

        for i in range(len(u)):
            ui = int(round(float(u[i])))
            vi = int(round(float(v[i])))

            if 0 <= ui < w and 0 <= vi < h:
                d = float(depth[i])

                # simple depth-based color
                d_clip = max(0.0, min(d, 30.0))
                val = int(255 * (d_clip / 30.0))
                color = (255 - val, 0, val)   # near red, far blue

                cv2.circle(annot_img, (ui, vi), 2, color, -1)

        overlay_msg = self.bridge.cv2_to_imgmsg(annot_img, encoding="bgr8")
        overlay_msg.header = annot_msg.header
        self.overlay_pub.publish(overlay_msg)

    def callback(self, filt_pcd_msg, pose_msg, mask_msg, annot_msg):

        # Convert filtered pcd2 msg to NumPy array
        # https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
        points = point_cloud2.read_points_numpy(filt_pcd_msg, field_names=['x', 'y', 'z', 'intensity', 'u', 'v'])
        
        self.filtered_lidar = points[:, :4]                  # N x 4, (x, y, z, intensity)
        self.proj_pix = points[:, 4:]                        # N x 2, (u, v)
        self.publish_overlay_image(annot_msg)

        # Generate one-hot encoded labels for projected LiDAR points.
        start_time = time.time()
        self.point_labels = self.create_point_labels(mask_msg)
        self.get_logger().info(f"mask processed in {time.time()-start_time:.2f} seconds.")

        # Extract pose from PoseStamped message
        pose_t = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
        pose_quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                            pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        self.lidar_pose = quaternion_matrix(pose_quat)
        self.lidar_pose[:3, 3] = pose_t

        with torch.no_grad():
            input_data = [
                torch.tensor(self.lidar_pose, device=self.dev, dtype=self.dtype),
                torch.tensor(self.filtered_lidar, device=self.dev, dtype=self.dtype),
                torch.tensor(self.point_labels, device=self.dev, dtype=self.dtype),
                None
            ]

            start_t = time.time()
            self.e2e_net(input_data)
            self.get_logger().info(f"Inference completed in {time.time() - start_t:.2f} seconds wall time.")


            # Optionally, continue using the semantic map and variance map publishing
            if self.publish:
                # Generate the local semantic map (this will return a single marker)
                marker = publish_local_map(self.e2e_net.grid, self.e2e_net.convbki_net.centroids, 
                                            self.voxel_sizes, self.color, None, self.e2e_net.propagation_net.translation)

                # Extract points and colors from the marker
                points = []
                rgb_colors = []

                if marker is None or len(marker.points) == 0:       ## added by yui
                    self.get_logger().info("Local map empty; skipping pointcloud publish.")
                    return

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
                pc2_msg.header.stamp = filt_pcd_msg.header.stamp
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

    pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
    config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
    
    # Load model parameters
    with open(config_path, "r") as stream:
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
    node = ConvBKIMap(
        model_params=model_params,
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