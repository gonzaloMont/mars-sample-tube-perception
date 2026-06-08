import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_parol6_desc = get_package_share_directory('parol6_description')
    pkg_parol6_gazebo = get_package_share_directory('parol6_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    world_file = os.path.join(pkg_parol6_gazebo, 'worlds', 'parol6.world')

    # model://<pkg>/... necesita el 'share' de <pkg> en GAZEBO_MODEL_PATH. Con
    # instalación aislada añadimos el de parol6_description Y el de parol6_gazebo
    # (este provee meshes/mars2020tube.stl); sin él el visual del tubo no carga.
    extra_model_paths = [
        os.path.join(pkg_parol6_desc, '..'),
        os.path.join(pkg_parol6_gazebo, '..'),
    ]
    prev = os.environ.get('GAZEBO_MODEL_PATH', '')
    os.environ['GAZEBO_MODEL_PATH'] = ':'.join(p for p in [prev, *extra_model_paths] if p)

    # Procesar el xacro que incluye el URDF + ros2_control + gazebo plugin
    xacro_file = os.path.join(pkg_parol6_desc, 'urdf', 'parol6_gazebo.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    # Overhead camera (modelo independiente del robot)
    camera_xacro = os.path.join(pkg_parol6_desc, 'urdf', 'overhead_camera.urdf.xacro')
    camera_description = xacro.process_file(camera_xacro).toxml()

    # Escribir el URDF de la cámara a un archivo temporal para evitar
    # problemas de shell al pasar XML largo como argumento
    camera_urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', prefix='overhead_camera_', delete=False
    )
    camera_urdf_tmp.write(camera_description)
    camera_urdf_tmp.close()
    camera_urdf_path = camera_urdf_tmp.name

    use_sim_time = {'use_sim_time': True}

    # Robot State Publisher (publica /robot_description y TF)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, use_sim_time],
        output='screen'
    )

    # Incluir el lanzamiento de Gazebo vacío
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items(),
    )

    # Spawnear el robot desde el topic /robot_description
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'parol6', '-topic', 'robot_description', '-timeout', '120'],
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

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
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

    # --- Overhead camera ---
    # Spawnear la cámara como entidad separada en Gazebo
    # (la pose se define aquí, igual que en el robot real con calibración extrínseca)
    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'overhead_camera',
            '-file', camera_urdf_path,
            '-x', '0.0', '-y', '0.0', '-z', '0.75',
            '-R', '0.0', '-P', '1.5707963267949', '-Y', '0.0',
            '-timeout', '120',
        ],
        output='screen',
    )

    # TF estática world → overhead_camera_link
    # En el robot real esta misma TF se publica con la pose calibrada
    camera_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.75',
            '--roll', '0.0', '--pitch', '1.5707963267949', '--yaw', '0.0',
            '--frame-id', 'world', '--child-frame-id', 'overhead_camera_link',
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        static_tf,
        spawn_camera,
        camera_static_tf,
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
        # Lanzar gripper_controller después de arm_group_controller
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=arm_controller_spawner,
                on_exit=[gripper_controller_spawner],
            )
        ),
    ])
