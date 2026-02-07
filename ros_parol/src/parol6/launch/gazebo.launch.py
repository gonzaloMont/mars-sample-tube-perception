import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_parol6 = get_package_share_directory('parol6')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Ensure gazebo can find models in the share directory
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = os.environ['GAZEBO_MODEL_PATH'] + ':' + os.path.join(pkg_parol6, '..')
    else:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(pkg_parol6, '..')

    # Procesar el xacro que incluye el URDF + ros2_control + gazebo plugin
    xacro_file = os.path.join(pkg_parol6, 'urdf', 'parol6_gazebo.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    # Robot State Publisher (publica /robot_description y TF)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # Incluir el lanzamiento de Gazebo vacío
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    # Spawnear el robot desde el topic /robot_description
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'parol6', '-topic', 'robot_description'],
        output='screen'
    )

    # Spawner de controladores (se lanzan secuencialmente tras el spawn del robot)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_group_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Transformación estática (ROS 2 usa tf2_ros)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--yaw', '0', '--pitch', '0', '--roll', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'base_footprint'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        static_tf,
        # Lanzar joint_state_broadcaster después de que el robot esté spawneado
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        # Lanzar arm_group_controller después de joint_state_broadcaster
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[arm_controller_spawner],
            )
        ),
    ])
