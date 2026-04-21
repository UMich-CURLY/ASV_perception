import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = 'asv_object_slam_ros2'
    params_file = os.path.join(get_package_share_directory(pkg), 'config', 'params.yaml')

    return LaunchDescription([
        Node(package=pkg, executable='convBKI_map_node', name='convBKI_map',
             parameters=[params_file]
        ),
        Node(
            package=pkg, executable='objdet_node', name='objdet',
             parameters=[params_file],
        )
    ])