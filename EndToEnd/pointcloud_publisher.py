import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudMerger(Node):
    def __init__(self):
        super().__init__('pointcloud_merger')

        # Initialize global point cloud storage
        self.global_point_cloud = None
        self.latest_header = None  # Store the latest header
        
        # Subscribe to local point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/semantic_map',  # Replace with your topic name
            self.listener_callback,
            10
        )

        # Publisher for the merged global point cloud
        self.publisher = self.create_publisher(PointCloud2, '/global_pointcloud', 10)

    def listener_callback(self, msg):
        # Extract all fields: x, y, z, and rgb
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        pc_array = np.stack([pc_data['x'], pc_data['y'], pc_data['z'], pc_data['rgb']], axis=-1)

        # Ensure point cloud has shape (N, 4) for (x, y, z, rgb)
        if pc_array.shape[1] != 4:
            self.get_logger().error(f"Incoming point cloud has unexpected shape: {pc_array.shape}")
            return

        # Merge with existing global point cloud
        if self.global_point_cloud is None:
            self.global_point_cloud = pc_array
        else:
            try:
                self.global_point_cloud = np.vstack((self.global_point_cloud, pc_array))
            except ValueError as e:
                self.get_logger().error(f"Error merging point clouds: {e}")
                return

        # Store latest header for correct frame and timestamp
        self.latest_header = msg.header

        # Create a new PointCloud2 message
        global_pc = self.create_pointcloud2(self.latest_header, self.global_point_cloud)

        # Publish the merged global point cloud
        self.publisher.publish(global_pc)

    def create_pointcloud2(self, header, points):
        """ Convert numpy array to PointCloud2 message """
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)  # Keep RGB as float
        ]

        return pc2.create_cloud(header, fields, points)

def main(args=None):
    rclpy.init(args=args)
    point_cloud_merger = PointCloudMerger()
    rclpy.spin(point_cloud_merger)
    point_cloud_merger.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

