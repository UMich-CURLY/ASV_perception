import yaml
import os
import torch
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import ros2_numpy
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from sensor_msgs_py import point_cloud2
from tf_transformations import quaternion_matrix
import transforms3d

import numpy as np
from std_msgs.msg import Header

## TODO: move these to params.yaml --------------------------##
# Camera to LiDAR Transformation
ROTATION_CAMERA_TO_LIDAR = np.array([0.0, 0.0, 0.0])  
TRANSLATION_CAMERA_TO_LIDAR = np.array([0.0, 0.0, 0.15])  

# Compute rotation matrix
R_matrix = transforms3d.euler.euler2mat(
    np.radians(ROTATION_CAMERA_TO_LIDAR[0]),
    np.radians(ROTATION_CAMERA_TO_LIDAR[1]),
    np.radians(ROTATION_CAMERA_TO_LIDAR[2])
)

CAMERA_TO_LIDAR_TRANSFORM = np.eye(4)
CAMERA_TO_LIDAR_TRANSFORM[:3, :3] = R_matrix
CAMERA_TO_LIDAR_TRANSFORM[:3, 3] = TRANSLATION_CAMERA_TO_LIDAR

Cf_TO_Cw_TRANSFORM = np.eye(4)
Cf_TO_Cw_TRANSFORM[:3, :3] = np.array([[0, -1, 0],
                                       [0, 0, -1],
                                       [1, 0, 0]])

##--------------------------------------------------------------##

class PcdFilterNode(Node):
    def __init__(self):
        super().__init__('pcd_filter_node')

        # Load parameters from YAML file
        config_path = os.path.join(os.getcwd(), 'configs/params.yaml')

        with open(config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            
        self.num_classes = self.config["num_classes"]
        self.ros_topic = self.config["ros_parameters"]
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Subscriber
        self.pcd_sub = self.create_subscription(
            PointCloud2,
            self.ros_topic["lidar_topic"],
            self.pcd_callback,
            qos_profile_sensor_data
        )
        # Camera intrinsics / Subscriber for Camera Info
        self.caminfo_ready = False
        self.caminfo_sub = self.create_subscription(
            CameraInfo, 
            self.ros_topic["caminfo_topic"], 
            self.caminfo_callback, 
            qos_profile_sensor_data
        )
        
        # Publisher
        self.filt_pcd_pub = self.create_publisher(PointCloud2, self.ros_topic["filt_pcd_topic"], 10)  # Filtered pcd Publisher

        # Worker thread
        self.running = True
        self.latest_lidar_raw = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        self.cb_count = 0

        self.get_logger().info('PcdFilterNode initialized and ready.')

    # Camera Intrinsics Callback
    def caminfo_callback(self, msg: CameraInfo):
        if self.caminfo_ready:                  ## only run once
            return
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]
        self.img_w, self.img_h = msg.width, msg.height
        self.camera_header = msg.header
        self.caminfo_ready = True
        self.CAMERA_INTRINSICS = np.array([[self.fx, 0.0, self.cx],
                                           [0.0, self.fy, self.cy],
                                           [0.0, 0.0, 1.0]])
        # self.get_logger().info(f"Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, w={self.img_w}, h={self.img_h}")
        # x=762.7223205566406, fy=762.7223110198975, cx=640.0, cy=360.0, w=1280, h=720

    # LiDAR PointCloud Callback
    def pcd_callback(self, msg: PointCloud2):
        lastest_pcd = ros2_numpy.point_cloud2.point_cloud2_to_array(msg)

        # Header info
        self.pcd_header = msg.header
        
        # Extract 'xyz' and 'intensity' fields
        xyz = lastest_pcd['xyz']
        intensity = lastest_pcd['intensity']
        lidar_raw = np.empty((xyz.shape[0], 4), dtype=np.float32)
        lidar_raw[:, :3] = xyz.astype(np.float32, copy=False)                  # populate xyz
        lidar_raw[:, 3] = intensity.reshape(-1).astype(np.float32, copy=False) # populate intensity

        with self.lock:
            self.latest_lidar_raw = lidar_raw
        # print(f"Received LiDAR pcd with {self.latest_lidar_raw.shape[0]} points.")

    
    def worker_loop(self):
        self.get_logger().info("PCD Filter Worker Thread Started")
        while rclpy.ok() and self.running:
            if not self.caminfo_ready:
                time.sleep(0.1)
                continue

            want_filt_pcd = self.filt_pcd_pub.get_subscription_count() > 0

            if not want_filt_pcd:
                time.sleep(0.01)
                print("No subscribers for filter pcd, skipping...")
                continue
    
            self.cb_count += 1
            t0 = time.time()
            with self.lock:
                if self.latest_lidar_raw is None:
                    lidar_raw = None
                else:
                    lidar_raw = self.latest_lidar_raw.copy()  # snapshot
                    self.latest_lidar_raw = None

            if lidar_raw is None:
                time.sleep(0.005)
                continue

            try:

                projected_pixels, cam_pts, valid_lidar_indices = self.project_lidar_to_image(lidar_raw[:, :3], self.img_w, self.img_h)

                # Check if projected pixels are empty
                if projected_pixels.shape[0] == 0:
                    self.get_logger().warn("No valid LiDAR points in camera FOV. Skipping this callback.")
                    continue

                # Apply the same filtering to lidar
                self.lidar = lidar_raw[valid_lidar_indices]  # Only keep LiDAR points in the image frame
                self.proj_pix = projected_pixels

                if want_filt_pcd:
                    # self.lidar contains [x, y, z, intensity] in the Camera Frame
                    # self.proj_pix contains [u, v]
                    # Stack them: (N, 6) -> [x, y, z, intensity, u, v]
                    combined_data = np.hstack((
                        self.lidar,                       # x, y, z, intensity
                        self.proj_pix.astype(np.float32)  # u, v
                    ))

                    # PCD Fields
                    # https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
                    pcd_fields = [
                        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                        PointField(name="u", offset=16, datatype=PointField.FLOAT32, count=1),
                        PointField(name="v", offset=20, datatype=PointField.FLOAT32, count=1)
                    ]

                    header = Header()
                    header.stamp = self.pcd_header.stamp
                    header.frame_id = "map"
                    filtered_pcd = point_cloud2.create_cloud(header, pcd_fields, combined_data)
                    self.filt_pcd_pub.publish(filtered_pcd)
                
                t1 = time.time()
                infer_ms = (t1 - t0) * 1000.0
                if self.cb_count % 10 == 0:
                    self.get_logger().info(
                        f"PCD process time: {infer_ms:.1f} ms | "
                        f"filt_pcd_pub={want_filt_pcd}"
                    )
            
            except Exception as e:
                self.get_logger().error(f"Error in worker_loop: {e}")
                time.sleep(0.01)

        
    def destroy_node(self):
        self.running = False
        super().destroy_node()
    
    def project_lidar_to_image(self, points_3d, image_width=1280, image_height=720):
        """Project 3D LiDAR points into 2D camera image plane and filter points outside the FOV."""
        
        points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # [x, y, z, 1]

        # Transform LiDAR points to camera frame: (n, 3)
        points_camera = (Cf_TO_Cw_TRANSFORM @ CAMERA_TO_LIDAR_TRANSFORM @ points_3d_h.T).T[:, :3]

        # Keep only points in front of the camera
        valid_camera_indices = points_camera[:, 2] > 0  
        points_camera = points_camera[valid_camera_indices]

        # Project onto image plane
        pixels = (self.CAMERA_INTRINSICS @ points_camera.T).T                # (n, 3)
        pixels = pixels[:, :2] / pixels[:, 2:]  # Normalize by depth    # (n, 2)

        # Filter points within image bounds
        valid_fov_indices = (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width) & \
                            (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)

        # Apply final filtering
        valid_indices = np.where(valid_camera_indices)[0][valid_fov_indices]  # Indices w.r.t original input
        pixels = pixels[valid_fov_indices].astype(int)
        
        return pixels, points_camera[valid_fov_indices, 2], valid_indices


def main(args=None):
    rclpy.init(args=args)
    node = PcdFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()