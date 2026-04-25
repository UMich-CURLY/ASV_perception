import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.abspath(os.path.join(launch_dir, '..'))

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'python3',
                'sensor_frontend/objseg_yolo.py',
            ],
            cwd=src_dir,
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                'sensor_frontend/pcdFilter_node.py',
            ],
            cwd=src_dir,
            output='screen'
        ),
    ])