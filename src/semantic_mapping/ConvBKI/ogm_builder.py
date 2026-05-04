#!/usr/bin/env python3

"""
Creates a global occupancy grid from semantic point clouds or operates in 
free-space mode for testing. The grid origin is fixed at the boat's starting 
position to provide a consistent global reference frame.

Features:
- Fixed global coordinate system (doesn't follow the boat)
- Semantic point cloud processing with color-based obstacle detection
- Free-space mode for testing without perception data
"""
import os
import yaml
import pathlib
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf_transformations as tft

# File paths for persistent storage
SAVE_DIR = pathlib.Path.home() / ".ros"
GRID_FILE = SAVE_DIR / "ogm.npy"
META_FILE = SAVE_DIR / "ogm_meta.json"

class OccupancyGridBuilder(Node):
    def __init__(self):
        super().__init__('occupancy_grid_builder')

        pkg_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
        config_path = os.path.join(pkg_path, 'configs', 'params.yaml')
        
        # Load model parameters
        with open(config_path, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.ros_topic = self.configs["ros_parameters"]
        self.ogm_params = self.configs["OGM_builder"]
        self.rviz_colors = self.configs["rviz_colors"]

        self._setup_parameters()
        self._setup_tf()
        self._setup_ros_communication()
        self._initialize_state()
        self._configure_free_space_mode()

    def _setup_parameters(self):
        """Initialize all OGM parameters with defaults."""
        self.resolution = self.ogm_params["grid_resolution"]
        self.grid_width = self.ogm_params["grid_width"]
        self.grid_height = self.ogm_params["grid_height"]
        self.free_space_mode = self.ogm_params["no_perception"]

    def _setup_tf(self):
        """Initialize transform buffer, listener, and broadcaster."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def _setup_ros_communication(self):
        """Setup publishers and subscribers."""

        self.create_subscription(PoseStamped, self.ros_topic["pose_topic"], self._pose_callback, 10)
        self.create_subscription(PointCloud2, self.ros_topic["global_pcd_topic"], self._pointcloud_callback, 10)
        self.grid_publisher = self.create_publisher(OccupancyGrid, self.ros_topic["ogm_topic"], 10)
        self.create_timer(0.2, self._publish_grid)

    def _initialize_state(self):
        """Initialize internal state variables."""
        self.origin_x = None
        self.origin_y = None
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.current_pose = None
        self.origin_initialized = False


    # no perception mode
    def _configure_free_space_mode(self):       
        """Configure settings for free-space testing mode."""
        if self.free_space_mode:
            # Create a 100m x 100m map for testing
            map_size_meters = 100
            self.grid_width = int(map_size_meters / self.resolution)
            self.grid_height = int(map_size_meters / self.resolution)
            self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
            
            self.get_logger().info(
                f"Free-space mode: Creating {map_size_meters}m x {map_size_meters}m global map")

    def _broadcast_map_transform(self):
        """Broadcast the static transform from world to map frame."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'map'
        
        # Set translation (map origin in world coordinates)
        transform.transform.translation.x = float(self.origin_x)
        transform.transform.translation.y = float(self.origin_y)
        transform.transform.translation.z = 0.0
        
        # No rotation needed
        transform.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(transform)

    def _pose_callback(self, msg: PoseStamped):
        """Handle incoming pose messages and initialize map origin if needed."""
        self.current_pose = msg
        
        if not self.origin_initialized:
            self._initialize_map_origin(msg)

    def _initialize_map_origin(self, pose_msg: PoseStamped):
        """Set the map origin based on the boat's starting position."""
        boat_x = pose_msg.pose.position.x
        boat_y = pose_msg.pose.position.y
        
        # Center the map around the starting position
        self.origin_x = boat_x - 0.5 * self.grid_width * self.resolution
        self.origin_y = boat_y - 0.5 * self.grid_height * self.resolution
        self.origin_initialized = True
        
        # Broadcast the map transform
        self._broadcast_map_transform()
        
        # Log the map configuration
        map_size_x = self.grid_width * self.resolution
        map_size_y = self.grid_height * self.resolution
        
        self.get_logger().info(f"Map origin set at ({self.origin_x:.2f}, {self.origin_y:.2f})")
        self.get_logger().info(
            f"Map coverage: X[{self.origin_x:.1f} to {self.origin_x + map_size_x:.1f}], "
            f"Y[{self.origin_y:.1f} to {self.origin_y + map_size_y:.1f}]")

    def _pointcloud_callback(self, msg: PointCloud2):
        """Process incoming point cloud data to update the occupancy grid."""
        # Skip point cloud processing in free-space mode
        if self.free_space_mode:
            return
            
        # Wait for map origin to be initialized
        if not self.origin_initialized:
            return

        # Parse point cloud data
        points, labels = self._parse_pointcloud(msg)
        if points is None:
            return
            
        # Transform points to map frame if necessary
        points = self._transform_to_map_frame(points, msg)
        if points is None:
            return
            
        # Update occupancy grid with new observations
        self._update_occupancy_grid(points, labels)

    def _parse_pointcloud(self, msg: PointCloud2):
        """Extract XYZ coordinates and semantic labels from point cloud."""
        try:
            # Unpack binary data as Nx4 float32 array (x, y, z, rgb)
            point_array = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        except ValueError:
            self.get_logger().warning("Invalid point cloud format")
            return None, None
            
        xyz_points = point_array[:, :3]
        rgb_values = point_array[:, 3].view(np.uint32)
        
        # Convert RGB to semantic labels
        labels = self._rgb_to_semantic_labels(rgb_values)
        
        # Filter out unknown labels
        valid_mask = labels != -1
        return xyz_points[valid_mask], labels[valid_mask]

    def _rgb_to_semantic_labels(self, rgb_values):
        """Convert RGB color values to semantic labels."""
        # Extract RGB components
        red = (rgb_values >> 16) & 0xFF
        green = (rgb_values >> 8) & 0xFF
        blue = rgb_values & 0xFF
        colors = np.stack([red, green, blue], axis=1)
        
        # Color to semantic mapping
        # color_map = {
        #     (200, 275, 200): 100,  # Vegetation -> Obstacle
        #     (0, 102, 0): 100,      # Dark vegetation -> Obstacle
        #     (160, 160, 160): 100,  # Structures -> Obstacle
        #     (30, 60, 150): 0,      # Water -> Free space
        #     (255, 30, 30): 100,    # Red objects -> Obstacle
        #     (0, 0, 0): 100,        # Black objects -> Obstacle
        #     (150, 240, 80): 100,   # Light vegetation -> Obstacle
        #     (255, 128, 0): 100,    # Orange objects -> Obstacle
        #     (255, 255, 255): 100   # White objects -> Obstacle
        # }
        color_map = {}

        for class_id, rgb in self.rviz_colors.items():
            rgb_tuple = tuple(rgb)

            if class_id == self.ogm_params["free_space_id"]:  # water
                color_map[rgb_tuple] = 0
            elif class_id == self.ogm_params["unknown_id"]:   # background
                color_map[rgb_tuple] = -1  # keep as -1
            else:
                color_map[rgb_tuple] = 100
        labels = np.full(colors.shape[0], -1, dtype=np.int8)
        
        # Assign labels based on color matching
        for color, label in color_map.items():
            color_match = np.all(colors == color, axis=1)
            labels[color_match] = label
            
        return labels

    def _transform_to_map_frame(self, points, msg):
        """Transform points to map coordinate frame if needed."""
        if msg.header.frame_id == 'map':
            return points
            
        try:
            # Look up transform from point cloud frame to map frame
            transform = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.2))
                
        except Exception as e:
            self.get_logger().warning(f"Transform lookup failed: {e}")
            return None
            
        # Apply transformation
        rotation_matrix = tft.quaternion_matrix([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w])[:3, :3]
            
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z])
            
        transformed_points = (rotation_matrix @ points.T).T + translation
        return transformed_points

    def _update_occupancy_grid(self, points, labels):
        """Update the occupancy grid with new point observations."""
        # Convert world coordinates to grid indices
        grid_x = np.floor((points[:, 0] - self.origin_x) / self.resolution).astype(int)
        grid_y = np.floor((points[:, 1] - self.origin_y) / self.resolution).astype(int)
        
        # Filter points that fall within the grid bounds
        valid_points = ((grid_x >= 0) & (grid_x < self.grid_width) & 
                       (grid_y >= 0) & (grid_y < self.grid_height))
        
        grid_x = grid_x[valid_points]
        grid_y = grid_y[valid_points]
        labels = labels[valid_points]
        
        # Update grid cells based on observations
        for x, y, label in zip(grid_x, grid_y, labels):
            if label == 100:  # Obstacle
                self.occupancy_grid[y, x] = 100
            elif label == 0:  # Free space (water)
                self.occupancy_grid[y, x] = 0

    def _publish_grid(self):
        """Publish the current occupancy grid."""
        if not self.origin_initialized:
            return
            
        # Create and populate OccupancyGrid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        
        # Set grid metadata
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # Flatten grid data for message
        grid_msg.data = self.occupancy_grid.flatten().tolist()
        
        self.grid_publisher.publish(grid_msg)

def main(args=None):
    """Main entry point for the occupancy grid builder node."""
    rclpy.init(args=args)
    
    try:
        node = OccupancyGridBuilder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()