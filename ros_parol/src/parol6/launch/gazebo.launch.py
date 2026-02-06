import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_parol6 = get_package_share_directory('parol6')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # # Ensure gazebo can find models in the share directory
    # # This is needed because sometimes package:// paths are not resolved properly without it
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] =  os.environ['GAZEBO_MODEL_PATH'] + ':' + os.path.join(pkg_parol6, '..')
    else:
        os.environ['GAZEBO_MODEL_PATH'] =  os.path.join(pkg_parol6, '..')

    # Ruta al archivo URDF
    urdf_file = os.path.join(pkg_parol6, 'urdf', 'parol6.urdf')

    # Incluir el lanzamiento de Gazebo vacío
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    # Nodo para spawnear el robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'parol6', '-file', urdf_file],
        output='screen'
    )

    # Transformación estática (ROS 2 usa tf2_ros)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'base_link', '--child-frame-id', 'base_footprint'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity,
        static_tf
    ])
