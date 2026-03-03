import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_parol6_desc = get_package_share_directory('parol6_description')
    pkg_parol6_gazebo = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_parol6_moveit = get_package_share_directory('parol6_moveit')

    world_file = os.path.join(pkg_parol6_gazebo, 'worlds', 'parol6.world')


    # ── Argumentos configurables en tiempo de lanzamiento ──────────────────
    # Uso:
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py            # sin RViz, sin gripper
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py rviz:=true
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py gripper:=true
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py rviz:=true gripper:=true
    rviz_arg    = DeclareLaunchArgument('rviz',    default_value='false', description='Lanzar RViz con la configuración de MoveIt')
    gripper_arg = DeclareLaunchArgument('gripper', default_value='false', description='Lanzar el nodo de gripper virtual')

    use_rviz    = LaunchConfiguration('rviz')
    use_gripper = LaunchConfiguration('gripper')


    # GAZEBO_MODEL_PATH for mesh resolution
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ':' + os.path.join(pkg_parol6_desc, '..')
    else:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(pkg_parol6_desc, '..')

    # Process Gazebo xacro (URDF + GazeboSystem + gazebo_ros2_control plugin)
    xacro_file = os.path.join(pkg_parol6_desc, 'urdf', 'parol6_gazebo.xacro')
    robot_description_content = xacro.process_file(xacro_file).toxml()

    # Overhead camera (modelo independiente del robot)
    camera_xacro = os.path.join(pkg_parol6_desc, 'urdf', 'overhead_camera.urdf.xacro')
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

    # --- Nodos de simulación ---
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
        launch_arguments={'world': world_file}.items(),
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

    # --- Cámara cenital ---
    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'overhead_camera',
            '-file', camera_urdf_path,
            '-x', '0.2', '-y', '0.0', '-z', '0.75',
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
            '--x', '0.2', '--y', '0.0', '--z', '0.75',
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

    # ── RViz (opcional) ─────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
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

    # ── Gripper virtual (opcional) ──────────────────────────────────────────
    gripper_node = Node(
        package='parol6_controllers',
        executable='virtual_gripper_node.py',
        output='screen',
        condition=IfCondition(use_gripper),
    )

    return LaunchDescription([
        rviz_arg,
        gripper_arg,

        gazebo,
        robot_state_publisher,
        spawn_entity,
        spawn_camera,
        camera_static_tf,

        # Arranque secuencial de controladores tras el spawn del robot
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
        # MoveIt + RViz + gripper arrancan cuando todos los controladores están activos
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[move_group_node, rviz_node, gripper_node],
        )),
    ])
