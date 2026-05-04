import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_parol6_gazebo = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_parol6_gazebo, 'worlds', 'jezero.world')
    gazebo_models_path = os.path.join(pkg_parol6_gazebo, 'models')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items(),
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=gazebo_models_path + ':' + os.environ.get('GAZEBO_MODEL_PATH', ''),
        ),
        gazebo,
    ])
