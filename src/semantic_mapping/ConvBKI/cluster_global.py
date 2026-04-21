import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
import struct

class PointCloudCluster(Node):
    def __init__(self):
        super().__init__('pointcloud_cluster')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/global_pointcloud',  # Change this to match your topic name
            self.pointcloud_callback,
            10)
        self.cluster_centroids = []  # List to store cluster centroids
        self.eps = 1.15  # Distance threshold for clustering
        self.merge_threshold = 1.5  # Distance threshold to merge clusters
        self.min_samples = 2  # Minimum points to form a cluster
        self.tracked_colors = {(255, 30, 30), (0, 0, 0), (150, 240, 80), (255, 128, 0), (255, 255, 255)}  # Tracked colors
        self.z_threshold = -1.4  # Ignore points below this z-value
        
    def pointcloud_callback(self, msg):
        points = []
        colors = []

        self.get_logger().info("Received PointCloud2 message")

        # Clear past cluster information before processing the new frame
        self.cluster_centroids = []  # Reset to track only current frame

        for p in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = p
            if z < self.z_threshold:
                continue  # Ignore points below the threshold
            # Convert float32 RGB to integer representation
            rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            r = (rgb_int >> 16) & 255
            g = (rgb_int >> 8) & 255
            b = rgb_int & 255
            color_tuple = (r, g, b)

            if color_tuple in self.tracked_colors:
                points.append([x, y, z])
                colors.append(color_tuple)

        self.get_logger().info(f"Total extracted points: {len(points)}")
        self.get_logger().info(f"Filtered colors: {set(colors)}")

        if len(points) == 0:
            self.get_logger().warn("No valid points found with the tracked colors!")
            return

        points = np.array(points)

        # Run DBSCAN on all points together
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        clusters = []
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise

            cluster_points = points[labels == label]
            mean_position = np.mean(cluster_points, axis=0)
            clusters.append(mean_position)

        # Merge close clusters
        merged_clusters = self.merge_clusters(clusters)

        # Update stored clusters with the merged ones
        self.cluster_centroids = merged_clusters
        self.get_logger().info(f'Tracked clusters: {len(self.cluster_centroids)}')
        self.get_logger().info(f'Clustering centroids: {self.cluster_centroids}')

    def merge_clusters(self, clusters):
        """ Merge clusters that are within the merge_threshold distance """
        merged = []
        for cluster in clusters:
            merged = self._merge_with_existing(merged, cluster)
        return merged

    def _merge_with_existing(self, merged, cluster):
        """ Helper function to merge a cluster if it's close to any existing one """
        for i, existing in enumerate(merged):
            if np.linalg.norm(cluster - existing) < self.merge_threshold:
                merged[i] = (existing + cluster) / 2  # Average positions
                return merged
        merged.append(cluster)
        return merged

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudCluster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

