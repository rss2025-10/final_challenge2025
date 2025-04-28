from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch the banana detector node."""
    return LaunchDescription([
        Node(
            package='final_challenge2025',
            executable='banana_detector',
            name='banana_detector',
            output='screen',
            emulate_tty=True,
        )
    ]) 