import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_parol6 = get_package_share_directory('parol6')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_parol6_moveit = get_package_share_directory('parol6_moveit')

    # GAZEBO_MODEL_PATH for mesh resolution
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ':' + os.path.join(pkg_parol6, '..')
    else:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(pkg_parol6, '..')

    # Process Gazebo xacro (URDF + GazeboSystem + gazebo_ros2_control plugin)
    xacro_file = os.path.join(pkg_parol6, 'urdf', 'parol6_gazebo.xacro')
    robot_description_content = xacro.process_file(xacro_file).toxml()

    # Overhead camera (modelo independiente del robot)
    camera_xacro = os.path.join(pkg_parol6, 'urdf', 'overhead_camera.urdf.xacro')
    camera_description = xacro.process_file(camera_xacro).toxml()

    # Escribir a archivo temporal (evita problemas de shell con XML largo)
    camera_urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', prefix='overhead_camera_', delete=False
    )
    camera_urdf_tmp.write(camera_description)
    camera_urdf_tmp.close()
    camera_urdf_path = camera_urdf_tmp.name

    # use_sim_time for all nodes (Gazebo publishes sim clock)
    use_sim_time = {'use_sim_time': True}

    # --- Gazebo simulation nodes ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}, use_sim_time],
        output='screen',
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'parol6', '-topic', 'robot_description', '-timeout', '120'],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_group_controller', '--controller-manager', '/controller_manager'],
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
    )

    # --- Overhead camera ---
    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'overhead_camera',
            '-file', camera_urdf_path,
            '-x', '0.0', '-y', '0.0', '-z', '1.0',
            '-R', '0.0', '-P', '1.5707963267949', '-Y', '0.0',
            '-timeout', '120',
        ],
        output='screen',
    )

    # TF estática world → overhead_camera_link (misma pose que en spawn)
    # En el robot real se publica con la pose calibrada
    camera_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '1.0',
            '--roll', '0.0', '--pitch', '1.5707963267949', '--yaw', '0.0',
            '--frame-id', 'world', '--child-frame-id', 'overhead_camera_link',
        ],
        output='screen',
    )

    # --- MoveIt 2 nodes ---
    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    # Override robot_description to use Gazebo version (GazeboSystem instead of FakeSystem)
    moveit_config.robot_description = {"robot_description": robot_description_content}

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), use_sim_time],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_parol6_moveit, 'config', 'moveit.rviz')],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            use_sim_time,
        ],
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        spawn_camera,
        camera_static_tf,
        # Sequential controller startup after entity spawn
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )),
        # Start MoveIt after all controllers are active
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[move_group_node, rviz_node],  # añadir rviz_node para abrir tmbn rviz: on_exit=[move_group_node, rviz_node]
        )),
    ])
