import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudMerger(Node):
    def __init__(self):
        super().__init__('pointcloud_merger')

        # Dictionary to store unique (x, y, z) -> rgb mapping
        self.global_point_cloud = {}
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

        # Update or add new points
        for point in pc_data:
            xyz = (point[0], point[1], point[2])  # (x, y, z)
            self.global_point_cloud[xyz] = point[3]  # Update RGB value

        # Store latest header for correct frame and timestamp
        self.latest_header = msg.header

        # Convert dictionary back to array
        global_pc_array = np.array([[x, y, z, rgb] for (x, y, z), rgb in self.global_point_cloud.items()])

        # Create a new PointCloud2 message
        global_pc = self.create_pointcloud2(self.latest_header, global_pc_array)

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

