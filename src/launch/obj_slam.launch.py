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
                'object_slam/drift_bridge.py',
            ],
            cwd=src_dir,
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'python3',
                'object_slam/landmark_tracker.py',
            ],
            cwd=src_dir,
            output='screen'
        ),
        # ExecuteProcess(
        #     cmd=[
        #         'python3',
        #         'object_slam/gtsam_optimizer.py',
        #     ],
        #     cwd=src_dir,
        #     output='screen'
        # ),
    ])