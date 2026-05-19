import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
from moveit_configs_utils import MoveItConfigsBuilder


def _write_urdf_tmp(xacro_path, mappings=None):
    """Procesa un xacro y escribe el URDF en un fichero temporal; devuelve la ruta."""
    xml = xacro.process_file(xacro_path, mappings=mappings or {}).toxml()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(xml)
    tmp.close()
    return tmp.name, xml


def launch_setup(context, *args, **kwargs):
    pkg_parol6_desc   = get_package_share_directory('parol6_description')
    pkg_parol6_gazebo = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros    = get_package_share_directory('gazebo_ros')
    pkg_parol6_moveit = get_package_share_directory('parol6_moveit')

    camera_type = LaunchConfiguration('camera').perform(context)
    use_rviz    = LaunchConfiguration('rviz').perform(context)
    use_gripper = LaunchConfiguration('gripper').perform(context)
    use_ik      = LaunchConfiguration('ik').perform(context)

    world_file = os.path.join(pkg_parol6_gazebo, 'worlds', 'parol6.world')

    # URDF del robot (sin navcam: gazebo_ros2_control no admite URDFs largos en --param)
    xacro_file = os.path.join(pkg_parol6_desc, 'urdf', 'parol6_gazebo.xacro')
    _, robot_description_content = _write_urdf_tmp(xacro_file)

    use_sim_time = {'use_sim_time': True}

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

    moveit_config = (
        MoveItConfigsBuilder("parol6", package_name="parol6_moveit")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    moveit_config.robot_description = {"robot_description": robot_description_content}

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), use_sim_time],
    )

    after_controllers = [move_group_node]

    if use_rviz == 'true':
        after_controllers.append(Node(
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
        ))

    if use_gripper == 'true':
        after_controllers.append(Node(
            package='parol6_controllers',
            executable='virtual_gripper_node.py',
            output='screen',
        ))

    if use_ik == 'true':
        after_controllers.append(Node(
            package='parol6_apps',
            executable='pose_ik_subscriber.py',
            output='screen',
            parameters=[use_sim_time],
        ))

    actions = [
        gazebo,
        robot_state_publisher,
        spawn_entity,
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
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=after_controllers,
        )),
    ]

    if camera_type == 'overhead':
        # Cámara cenital: entidad independiente con pose fija en el mundo
        cam_urdf_path, _ = _write_urdf_tmp(
            os.path.join(pkg_parol6_desc, 'urdf', 'overhead_camera.urdf.xacro')
        )
        actions += [
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'overhead_camera',
                    '-file', cam_urdf_path,
                    '-x', '0.2', '-y', '0.0', '-z', '0.75',
                    '-R', '0.0', '-P', '1.5707963267949', '-Y', '0.0',
                    '-timeout', '120',
                ],
                output='screen',
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=[
                    '--x', '0.2', '--y', '0.0', '--z', '0.75',
                    '--roll', '0.0', '--pitch', '1.5707963267949', '--yaw', '0.0',
                    '--frame-id', 'world', '--child-frame-id', 'overhead_camera_link',
                ],
                output='screen',
            ),
        ]

    else:  # camera_type == 'navcam'
        # NavCam estéreo: misma posición y orientación que la cámara cenital
        cam_urdf_path, _ = _write_urdf_tmp(
            os.path.join(pkg_parol6_desc, 'urdf', 'navcam_stereo.urdf.xacro')
        )
        actions += [
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'navcam_stereo',
                    '-file', cam_urdf_path,
                    '-x', '0.2', '-y', '0.0', '-z', '0.75',
                    '-R', '0.0', '-P', '1.5707963267949', '-Y', '0.0',
                    '-timeout', '120',
                ],
                output='screen',
            ),
            # base_link → navcam_rig_link  (misma pose que la cámara cenital en world)
            Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=['--x', '0.2', '--y', '0.0', '--z', '0.75',
                            '--roll', '0.0', '--pitch', '1.5707963267949', '--yaw', '0.0',
                            '--frame-id', 'base_link',
                            '--child-frame-id', 'navcam_rig_link']),
            # navcam_rig_link → navcam_left_link  (navcam_left_joint origin)
            Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=['--x', '0', '--y', '0.075', '--z', '0',
                            '--roll', '0', '--pitch', '0', '--yaw', '0',
                            '--frame-id', 'navcam_rig_link',
                            '--child-frame-id', 'navcam_left_link']),
            # navcam_left_link → navcam_left_optical_frame  (navcam_left_optical_joint origin)
            Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=['--x', '0', '--y', '0', '--z', '0',
                            '--roll', '-1.5707963267949', '--pitch', '0',
                            '--yaw', '-1.5707963267949',
                            '--frame-id', 'navcam_left_link',
                            '--child-frame-id', 'navcam_left_optical_frame']),
            # navcam_rig_link → navcam_right_link  (navcam_right_joint origin)
            Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=['--x', '0', '--y', '-0.075', '--z', '0',
                            '--roll', '0', '--pitch', '0', '--yaw', '0',
                            '--frame-id', 'navcam_rig_link',
                            '--child-frame-id', 'navcam_right_link']),
            # navcam_right_link → navcam_right_optical_frame  (navcam_right_optical_joint origin)
            Node(package='tf2_ros', executable='static_transform_publisher',
                 arguments=['--x', '0', '--y', '0', '--z', '0',
                            '--roll', '-1.5707963267949', '--pitch', '0',
                            '--yaw', '-1.5707963267949',
                            '--frame-id', 'navcam_right_link',
                            '--child-frame-id', 'navcam_right_optical_frame']),
        ]

    return actions


def generate_launch_description():
    # Uso:
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py camera:=navcam
    #   ros2 launch parol6_gazebo gazebo_moveit.launch.py camera:=overhead rviz:=true gripper:=true ik:=true

    pkg_parol6_desc = get_package_share_directory('parol6_description')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += ':' + os.path.join(pkg_parol6_desc, '..')
    else:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(pkg_parol6_desc, '..')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera', default_value='overhead',
            description="Tipo de cámara: 'overhead' (cenital independiente) o 'navcam' (estéreo en base_link)",
        ),
        DeclareLaunchArgument('rviz',    default_value='false', description='Lanzar RViz con la configuración de MoveIt'),
        DeclareLaunchArgument('gripper', default_value='false', description='Lanzar el nodo de gripper virtual'),
        DeclareLaunchArgument('ik',      default_value='false', description='Lanzar el nodo de cinemática inversa'),
        OpaqueFunction(function=launch_setup),
    ])
