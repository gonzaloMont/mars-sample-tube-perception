#!/usr/bin/env python3
"""
synthetic_dataset_generator.py

Nodo ROS 2 (rclpy) para generar dataset sintético Sim2Real en Gazebo:
  1) Randomiza escena (tubo, robot, iluminación) [placeholders/skeleton]
  2) Sincroniza física y captura imagen RGB
  3) Consulta TF exacta camera_optical_frame <- tube_link en el instante de captura
  4) Proyecta keypoints 3D conocidos del tubo a píxel 2D con modelo pinhole
  5) Calcula bounding box 2D
  6) Guarda imagen + JSON por muestra

Uso típico:
  ros2 run parol6_vision synthetic_dataset_generator.py

Parámetros relevantes:
    - image_topic:            topic RGB (default: /overhead_camera/overhead_cam/image_raw)
    - camera_info_topic:      topic CameraInfo (default: /overhead_camera/overhead_cam/camera_info)
    - camera_frame:           frame óptico de cámara (default: overhead_camera_optical_frame)
  - tube_frame:             frame del tubo (default: tube_link)
  - output_dir:             carpeta salida dataset
  - num_samples:            número de muestras a guardar
  - settle_time_sec:        espera tras reanudar física antes de capturar
  - capture_timeout_sec:    timeout para esperar imagen nueva
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Empty
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from gazebo_msgs.msg import EntityState
    from gazebo_msgs.msg import LinkStates
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetEntityState
    from gazebo_msgs.srv import SetModelState
    GAZEBO_STATE_AVAILABLE = True
except ImportError:
    GAZEBO_STATE_AVAILABLE = False


@dataclass
class ProjectedSample:
    image: np.ndarray
    image_stamp: Time
    bbox_xyxy: Tuple[float, float, float, float]


class SyntheticDatasetGenerator(Node):
    def __init__(self) -> None:
        super().__init__('synthetic_dataset_generator')

        # ── Parámetros ROS ──────────────────────────────────────────────────
        self.declare_parameter('image_topic', '/overhead_camera/overhead_cam/image_raw')
        self.declare_parameter('camera_info_topic', '/overhead_camera/overhead_cam/camera_info')
        self.declare_parameter('camera_frame', 'overhead_camera_optical_frame')
        self.declare_parameter('camera_link_frame', 'overhead_camera_link')
        self.declare_parameter('tube_frame', 'tube_link')
        self.declare_parameter('tube_entity_name', 'mars2020tube')
        self.declare_parameter('link_states_topic', '/link_states')
        self.declare_parameter('camera_link_name_in_gazebo', 'overhead_camera::overhead_camera_link')
        self.declare_parameter('tube_link_name_in_gazebo', 'mars2020tube::tube_link')
        self.declare_parameter('output_dir', os.path.expanduser('~/parol6_synth_dataset'))
        self.declare_parameter('num_samples', 1000)
        self.declare_parameter('class_id', 0)
        self.declare_parameter('class_names', ['tube'])
        self.declare_parameter('settle_time_sec', 0.15)
        self.declare_parameter('capture_timeout_sec', 3.5)
        self.declare_parameter('settle_max_wait_sec', 0.0)
        self.declare_parameter('settle_min_wait_sec', 0.2)
        self.declare_parameter('tube_linear_vel_threshold', 0.02)
        self.declare_parameter('tube_angular_vel_threshold', 0.15)
        self.declare_parameter('tube_position_delta_threshold', 0.0015)
        self.declare_parameter('tube_orientation_delta_threshold', 0.01)
        self.declare_parameter('settle_stable_samples', 4)
        self.declare_parameter('joint_trajectory_topic', '/arm_group_controller/joint_trajectory')
        self.declare_parameter('parol6_joint_names', ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
        self.declare_parameter('parol6_joint_min_limits', [-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159])
        self.declare_parameter('parol6_joint_max_limits', [3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159])
        self.declare_parameter('joint_trajectory_time_from_start_sec', 0.8)

        # Espacio de trabajo aleatorio del tubo (world)
        self.declare_parameter('tube_x_min', 0.15)
        self.declare_parameter('tube_x_max', 0.45)
        self.declare_parameter('tube_y_min', -0.20)
        self.declare_parameter('tube_y_max', 0.20)
        self.declare_parameter('tube_z_min', 0.75)
        self.declare_parameter('tube_z_max', 0.95)
        self.declare_parameter('tube_z_ground', 0.0)

        # Geometría conocida del tubo (aprox. del SDF adjunto)
        self.declare_parameter('tube_length_m', 0.186)
        self.declare_parameter('tube_radius_m', 0.01016)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.camera_link_frame = str(self.get_parameter('camera_link_frame').value)
        self.tube_frame = str(self.get_parameter('tube_frame').value)
        self.tube_entity_name = str(self.get_parameter('tube_entity_name').value)
        self.link_states_topic = str(self.get_parameter('link_states_topic').value)
        self.camera_link_name_in_gazebo = str(self.get_parameter('camera_link_name_in_gazebo').value)
        self.tube_link_name_in_gazebo = str(self.get_parameter('tube_link_name_in_gazebo').value)
        self.output_dir = str(self.get_parameter('output_dir').value)
        self.num_samples = int(self.get_parameter('num_samples').value)
        self.class_id = int(self.get_parameter('class_id').value)
        self.class_names = [str(name) for name in self.get_parameter('class_names').value]
        self.settle_time_sec = float(self.get_parameter('settle_time_sec').value)
        self.capture_timeout_sec = float(self.get_parameter('capture_timeout_sec').value)
        self.settle_max_wait_sec = float(self.get_parameter('settle_max_wait_sec').value)
        self.settle_min_wait_sec = float(self.get_parameter('settle_min_wait_sec').value)
        self.tube_linear_vel_threshold = float(self.get_parameter('tube_linear_vel_threshold').value)
        self.tube_angular_vel_threshold = float(self.get_parameter('tube_angular_vel_threshold').value)
        self.tube_position_delta_threshold = float(self.get_parameter('tube_position_delta_threshold').value)
        self.tube_orientation_delta_threshold = float(self.get_parameter('tube_orientation_delta_threshold').value)
        self.settle_stable_samples = int(self.get_parameter('settle_stable_samples').value)
        self.joint_trajectory_topic = str(self.get_parameter('joint_trajectory_topic').value)
        self.parol6_joint_names = [str(name) for name in self.get_parameter('parol6_joint_names').value]
        self.parol6_joint_min_limits = [float(limit) for limit in self.get_parameter('parol6_joint_min_limits').value]
        self.parol6_joint_max_limits = [float(limit) for limit in self.get_parameter('parol6_joint_max_limits').value]
        self.joint_trajectory_time_from_start_sec = float(self.get_parameter('joint_trajectory_time_from_start_sec').value)

        self.tube_x_min = float(self.get_parameter('tube_x_min').value)
        self.tube_x_max = float(self.get_parameter('tube_x_max').value)
        self.tube_y_min = float(self.get_parameter('tube_y_min').value)
        self.tube_y_max = float(self.get_parameter('tube_y_max').value)
        self.tube_z_min = float(self.get_parameter('tube_z_min').value)
        self.tube_z_max = float(self.get_parameter('tube_z_max').value)
        self.tube_z_ground = float(self.get_parameter('tube_z_ground').value)

        self.tube_length = float(self.get_parameter('tube_length_m').value)
        self.tube_radius = float(self.get_parameter('tube_radius_m').value)

        # ── Estado interno (imagen + cámara) ───────────────────────────────
        self.bridge = CvBridge()
        self.latest_image_msg: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_link_states = None
        self.latest_image_counter = 0
        self.use_tf_lookup = True
        self.last_warn_times: Dict[str, float] = {}

        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 10)
        if GAZEBO_STATE_AVAILABLE:
            self.create_subscription(LinkStates, self.link_states_topic, self._link_states_cb, 10)
        self.parol6_joint_pub = self.create_publisher(
            JointTrajectory,
            self.joint_trajectory_topic,
            10,
        )

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Servicios Gazebo (pause/unpause + set_entity_state opcional) ──
        self.pause_clients = [
            self.create_client(Empty, '/gazebo/pause_physics'),
            self.create_client(Empty, '/pause_physics'),
        ]
        self.unpause_clients = [
            self.create_client(Empty, '/gazebo/unpause_physics'),
            self.create_client(Empty, '/unpause_physics'),
        ]

        self.set_entity_state_client = None
        self.set_model_state_clients = []
        if GAZEBO_STATE_AVAILABLE:
            self.set_entity_state_clients = [
                self.create_client(SetEntityState, '/gazebo/set_entity_state'),
                self.create_client(SetEntityState, '/set_entity_state'),
            ]
            self.set_model_state_clients = [
                self.create_client(SetModelState, '/gazebo/set_model_state'),
                self.create_client(SetModelState, '/set_model_state'),
            ]

        # ── Estructura de salida con split train/val/test ───────────────────
        self.split_plan = self._build_split_plan(self.num_samples)
        self.images_dirs = {
            'train': os.path.join(self.output_dir, 'images', 'train'),
            'val': os.path.join(self.output_dir, 'images', 'val'),
            'test': os.path.join(self.output_dir, 'images', 'test'),
        }
        self.labels_dirs = {
            'train': os.path.join(self.output_dir, 'labels', 'train'),
            'val': os.path.join(self.output_dir, 'labels', 'val'),
            'test': os.path.join(self.output_dir, 'labels', 'test'),
        }
        for split_name in ('train', 'val', 'test'):
            os.makedirs(self.images_dirs[split_name], exist_ok=True)
            os.makedirs(self.labels_dirs[split_name], exist_ok=True)

        self._write_yolo_data_yaml()

        self.get_logger().info('SyntheticDatasetGenerator inicializado.')
        self.get_logger().info(f'Image topic: {self.image_topic}')
        self.get_logger().info(f'CameraInfo topic: {self.camera_info_topic}')
        self.get_logger().info(f'Output dir: {self.output_dir}')
        self.get_logger().info(f'YOLO data.yaml: {os.path.join(self.output_dir, "data.yaml")}')
        self.get_logger().info(
            f'Split dataset -> train: {self.split_plan.count("train")}, '
            f'val: {self.split_plan.count("val")}, '
            f'test: {self.split_plan.count("test")}'
        )

    def _write_yolo_data_yaml(self) -> None:
        data_yaml_path = os.path.join(self.output_dir, 'data.yaml')

        # Garantiza que class_id está en rango para evitar datasets inválidos.
        if not self.class_names:
            self.class_names = ['tube']
        if self.class_id < 0 or self.class_id >= len(self.class_names):
            self._warn_throttled(
                'class_id_out_of_range',
                f'class_id={self.class_id} fuera de rango para class_names (n={len(self.class_names)}). '
                'Se usará class_id=0.',
                period_sec=60.0,
            )
            self.class_id = 0

        yaml_lines = [
            'train: images/train',
            'val: images/val',
            'test: images/test',
            f'nc: {len(self.class_names)}',
            f'names: {self.class_names}',
        ]
    

        with open(data_yaml_path, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(yaml_lines) + '\n')

    def _warn_throttled(self, key: str, text: str, period_sec: float = 5.0) -> None:
        now = time.monotonic()
        last = self.last_warn_times.get(key, 0.0)
        if (now - last) >= period_sec:
            self.get_logger().warn(text)
            self.last_warn_times[key] = now

    # ───────────────────────────────────────────────────────────────────────
    # Callbacks
    # ───────────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        self.latest_image_msg = msg
        self.latest_image_counter += 1

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def _link_states_cb(self, msg) -> None:
        self.latest_link_states = msg

    # ───────────────────────────────────────────────────────────────────────
    # Flujo principal
    # ───────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        if not self._wait_for_initial_data(timeout_sec=10.0):
            self.get_logger().error('No se recibió imagen/camera_info inicial. Abortando.')
            return

        self.get_logger().info(f'Comenzando generación de {self.num_samples} muestras...')

        saved = 0
        attempts = 0
        max_attempts = max(5 * self.num_samples, 50)

        while rclpy.ok() and saved < self.num_samples and attempts < max_attempts:
            attempts += 1

            # 1) Pausar física
            self._pause_physics()

            # 2) Domain randomization (skeleton + ejemplo básico para tubo)
            self.randomize_tube_pose()
            self.randomize_parol6_joints()
            self.randomize_lighting()

            # 3) Reanudar física y dejar estabilizar sensores
            self._unpause_physics()
            self._spin_sleep(self.settle_time_sec)

            # Si el tubo no está quieto, no se captura nada en este ciclo.
            if not self._wait_for_tube_settle():
                self._pause_physics()
                continue

            # 4) Capturar imagen fresca + proyectar GT
            sample = self._capture_and_project_sample()
            if sample is None:
                continue

            # 5) Guardado de imagen + JSON
            self._save_sample(sample, saved)
            saved += 1

            if saved % 10 == 0 or saved == self.num_samples:
                self.get_logger().info(f'Progreso dataset: {saved}/{self.num_samples}')

        self.get_logger().info(f'Generación finalizada. Guardadas {saved} muestras en {self.output_dir}')

    # ───────────────────────────────────────────────────────────────────────
    # Esperas / sincronización
    # ───────────────────────────────────────────────────────────────────────

    def _wait_for_initial_data(self, timeout_sec: float) -> bool:
        t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_image_msg is not None and self.latest_camera_info is not None:
                return True
        return False

    def _spin_sleep(self, duration_sec: float) -> None:
        t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t0) < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def _wait_for_fresh_image(self, timeout_sec: float) -> Optional[Image]:
        start_counter = self.latest_image_counter
        t0 = time.monotonic()

        while rclpy.ok() and (time.monotonic() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.latest_image_counter > start_counter and self.latest_image_msg is not None:
                return self.latest_image_msg
        return None

    def _wait_for_tube_settle(self) -> bool:
        """
        Espera hasta que el tubo esté prácticamente en reposo según /link_states.
        Devuelve True si detecta reposo.

        Si settle_max_wait_sec <= 0, espera indefinidamente (modo estricto).
        Si settle_max_wait_sec > 0, devuelve False al agotar timeout.
        """
        # Ventana mínima para que la física responda tras el teletransporte.
        self._spin_sleep(self.settle_min_wait_sec)

        t0 = time.monotonic()
        stable_count = 0
        strict_wait = self.settle_max_wait_sec <= 0.0
        prev_pose = None

        while rclpy.ok():
            if not strict_wait and (time.monotonic() - t0) >= self.settle_max_wait_sec:
                self._warn_throttled(
                    'tube_not_settled_timeout',
                    'Tiempo de asentamiento agotado; no se capturará en este ciclo y se reintentará.',
                    period_sec=5.0,
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.01)

            tube_twist = self._find_link_twist(self.latest_link_states, self.tube_link_name_in_gazebo)
            tube_pose = self._find_link_pose(self.latest_link_states, self.tube_link_name_in_gazebo)
            if tube_twist is None:
                self._warn_throttled(
                    'tube_twist_missing',
                    f'No se encontró twist para {self.tube_link_name_in_gazebo} en /link_states; '
                    'esperando datos de velocidad para continuar.',
                    period_sec=5.0,
                )
                continue

            if tube_pose is None:
                self._warn_throttled(
                    'tube_pose_missing',
                    f'No se encontró pose para {self.tube_link_name_in_gazebo} en /link_states; '
                    'esperando datos de pose para continuar.',
                    period_sec=5.0,
                )
                continue

            lin = tube_twist.linear
            ang = tube_twist.angular
            lin_norm = math.sqrt(lin.x * lin.x + lin.y * lin.y + lin.z * lin.z)
            ang_norm = math.sqrt(ang.x * ang.x + ang.y * ang.y + ang.z * ang.z)

            velocity_stable = (
                lin_norm < self.tube_linear_vel_threshold
                and ang_norm < self.tube_angular_vel_threshold
            )

            pose_stable = False
            if prev_pose is not None:
                dx = float(tube_pose.position.x - prev_pose.position.x)
                dy = float(tube_pose.position.y - prev_pose.position.y)
                dz = float(tube_pose.position.z - prev_pose.position.z)
                pos_delta = math.sqrt(dx * dx + dy * dy + dz * dz)

                q_curr = tube_pose.orientation
                q_prev = prev_pose.orientation
                ori_delta = self._quaternion_angular_distance(
                    q_curr.x, q_curr.y, q_curr.z, q_curr.w,
                    q_prev.x, q_prev.y, q_prev.z, q_prev.w,
                )

                pose_stable = (
                    pos_delta < self.tube_position_delta_threshold
                    and ori_delta < self.tube_orientation_delta_threshold
                )

            prev_pose = tube_pose

            if velocity_stable or pose_stable:
                stable_count += 1
                if stable_count >= self.settle_stable_samples:
                    return True
            else:
                stable_count = 0

        return False

    # ───────────────────────────────────────────────────────────────────────
    # Servicios Gazebo
    # ───────────────────────────────────────────────────────────────────────

    def _call_empty_service(self, clients: List) -> bool:
        req = Empty.Request()
        for client in clients:
            if client.wait_for_service(timeout_sec=0.2):
                future = client.call_async(req)
                end = time.monotonic() + 1.0
                while rclpy.ok() and not future.done() and time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.02)
                return bool(future.done())
        return False

    def _pause_physics(self) -> None:
        ok = self._call_empty_service(self.pause_clients)
        if not ok:
            self.get_logger().warn('No fue posible pausar física (servicio no disponible).')

    def _unpause_physics(self) -> None:
        ok = self._call_empty_service(self.unpause_clients)
        if not ok:
            self.get_logger().warn('No fue posible reanudar física (servicio no disponible).')

    # ───────────────────────────────────────────────────────────────────────
    # Domain Randomization (skeleton solicitada)
    # ───────────────────────────────────────────────────────────────────────

    def randomize_tube_pose(self) -> None:
        """
        Placeholder funcional: randomiza pose del tubo en workspace.

        Implementación base:
          - Si /gazebo/set_entity_state está disponible, aplica una pose aleatoria
            al modelo del tubo.
          - Si no, deja warning y continúa.
        """
        if not GAZEBO_STATE_AVAILABLE:
            self.get_logger().debug('gazebo_msgs/SetEntityState no disponible; se omite randomize_tube_pose().')
            return

        model_name = self._resolve_tube_model_name_from_link_states() or self.tube_entity_name

        x = random.uniform(self.tube_x_min, self.tube_x_max)
        y = random.uniform(self.tube_y_min, self.tube_y_max)
        # z fija para que el tubo aparezca en el suelo (sin altura aleatoria).
        z = self.tube_z_ground

        # Sin inclinación aleatoria: roll y pitch fijos.
        # Se mantiene yaw aleatoria para variar orientación horizontal.
        roll = 1.57
        pitch = 0.0
        yaw = random.uniform(-math.pi, math.pi)
        qx, qy, qz, qw = self._quaternion_from_euler(roll, pitch, yaw)

        state = EntityState()
        state.name = model_name
        state.reference_frame = 'world'
        state.pose = Pose()
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        set_ok = self._try_set_entity_state(state)
        if set_ok:
            return

        model_state = ModelState()
        model_state.model_name = model_name
        model_state.reference_frame = 'world'
        model_state.pose = state.pose

        set_ok = self._try_set_model_state(model_state)
        if not set_ok:
            self._warn_throttled(
                'tube_randomization_failed',
                'No fue posible randomizar la pose del tubo: fallaron SetEntityState y SetModelState. '
                'Revisa servicios Gazebo y nombre de entidad/modelo del tubo.',
                period_sec=5.0,
            )

    def _resolve_tube_model_name_from_link_states(self) -> Optional[str]:
        if self.latest_link_states is None:
            return None

        suffix = self.tube_link_name_in_gazebo.split('::')[-1]
        for link_name in self.latest_link_states.name:
            if link_name.endswith('::' + suffix):
                return link_name.split('::')[0]
        return None

    def _try_set_entity_state(self, state: EntityState) -> bool:
        for client in getattr(self, 'set_entity_state_clients', []):
            if not client.wait_for_service(timeout_sec=0.1):
                continue

            req = SetEntityState.Request()
            req.state = state
            future = client.call_async(req)
            end = time.monotonic() + 1.0
            while rclpy.ok() and not future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.01)

            if not future.done():
                continue

            try:
                response = future.result()
            except Exception:
                continue

            if response is None:
                continue

            success = getattr(response, 'success', True)
            if success:
                return True

        return False

    def _try_set_model_state(self, model_state: ModelState) -> bool:
        for client in self.set_model_state_clients:
            if not client.wait_for_service(timeout_sec=0.1):
                continue

            req = SetModelState.Request()
            req.model_state = model_state
            future = client.call_async(req)
            end = time.monotonic() + 1.0
            while rclpy.ok() and not future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.01)

            if not future.done():
                continue

            try:
                response = future.result()
            except Exception:
                continue

            if response is None:
                continue

            success = getattr(response, 'success', True)
            if success:
                return True

        return False

    def randomize_parol6_joints(self) -> None:
        """
        Publica un JointTrajectory (1 punto) con posiciones aleatorias para
        las 6 articulaciones del PAROL6.

        El mensaje se publica en el tópico del controlador ROS 2:
          /arm_group_controller/joint_trajectory
        """
        if len(self.parol6_joint_names) != 6:
            self.get_logger().warn(
                f'Se esperaban 6 joint names para PAROL6 y se recibieron {len(self.parol6_joint_names)}. '
                'Se omite randomización articular en este ciclo.'
            )
            return

        if len(self.parol6_joint_min_limits) != 6 or len(self.parol6_joint_max_limits) != 6:
            self.get_logger().warn(
                'Los límites articulares deben tener longitud 6. '
                'Se omite randomización articular en este ciclo.'
            )
            return

        random_positions = []
        for min_limit, max_limit in zip(self.parol6_joint_min_limits, self.parol6_joint_max_limits):
            if max_limit < min_limit:
                min_limit, max_limit = max_limit, min_limit
            random_positions.append(random.uniform(min_limit, max_limit))

        joint_trajectory_msg = JointTrajectory()
        joint_trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        joint_trajectory_msg.joint_names = self.parol6_joint_names

        point = JointTrajectoryPoint()
        point.positions = random_positions
        point.time_from_start = Duration(
            seconds=self.joint_trajectory_time_from_start_sec
        ).to_msg()

        joint_trajectory_msg.points = [point]
        self.parol6_joint_pub.publish(joint_trajectory_msg)

        self.get_logger().debug(
            f'JointTrajectory publicado en {self.joint_trajectory_topic} '
            f'con posiciones aleatorias: {random_positions}'
        )

    def randomize_lighting(self) -> None:
        """
        Placeholder (skeleton): randomización de iluminación en Gazebo.

        Posibles implementaciones:
          - Servicio /gazebo/set_light_properties (si está disponible).
          - Plugin world para cambiar intensidad/color/luz direccional.
        """
        # TODO: Implementar según configuración del world/plugin.
        self.get_logger().debug('randomize_lighting(): placeholder (sin implementación activa).')

    # ───────────────────────────────────────────────────────────────────────
    # Captura + TF + proyección
    # ───────────────────────────────────────────────────────────────────────

    def _capture_and_project_sample(self) -> Optional[ProjectedSample]:
        image_msg = self._wait_for_fresh_image(timeout_sec=self.capture_timeout_sec)
        if image_msg is None:
            self.get_logger().warn('Timeout esperando imagen fresca; se omite intento.')
            return None

        camera_info = self.latest_camera_info
        if camera_info is None:
            self.get_logger().warn('CameraInfo no disponible en captura; se omite intento.')
            return None

        try:
            image_cv = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Error CvBridge en captura: {exc}')
            return None

        stamp = Time.from_msg(image_msg.header.stamp)

        # Lookup TF exacta camera_frame <- tube_frame al timestamp de imagen.
        # Si no existe en TF (caso típico con modelos SDF), desactivamos el
        # lookup TF y usamos fallback /link_states para iteraciones siguientes.
        if self.use_tf_lookup:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    self.tube_frame,
                    stamp,
                    timeout=Duration(seconds=0.2),
                )
                R_ct, t_ct = self._tf_to_rt(tf_msg)
            except LookupException as exc:
                self.use_tf_lookup = False
                self.get_logger().warn(
                    f'tf2 LookupException ({self.camera_frame} <- {self.tube_frame}): {exc}. '
                    'Se desactiva lookup TF y se usará fallback /link_states.'
                )
                rt = self._compute_rt_from_link_states()
                if rt is None:
                    return None
                R_ct, t_ct = rt
            except ConnectivityException as exc:
                self._warn_throttled('tf_connectivity', f'tf2 ConnectivityException: {exc}')
                return None
            except ExtrapolationException as exc:
                self._warn_throttled('tf_extrapolation', f'tf2 ExtrapolationException: {exc}')
                return None
        else:
            rt = self._compute_rt_from_link_states()
            if rt is None:
                return None
            R_ct, t_ct = rt

        K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)

        # 1) Keypoints 3D (desactivado temporalmente para entrenamiento bbox-only)
        #    Se conserva comentado para usarlo más adelante en YOLO-Pose / PnP.
        # keypoints_3d = {
        #     'center': np.array([0.0, 0.0, 0.0], dtype=np.float64),
        #     'tip_plus': np.array([0.0, 0.0, +0.5 * self.tube_length], dtype=np.float64),
        #     'tip_minus': np.array([0.0, 0.0, -0.5 * self.tube_length], dtype=np.float64),
        #     'edge_top': np.array([0.0, +self.tube_radius, 0.0], dtype=np.float64),
        #     'edge_bottom': np.array([0.0, -self.tube_radius, 0.0], dtype=np.float64),
        # }
        # keypoints_uv: Dict[str, Tuple[float, float]] = {}
        # for name, p_tube in keypoints_3d.items():
        #     p_cam = R_ct @ p_tube + t_ct
        #     uv = self._project_point_pinhole(K, p_cam)
        #     if uv is None:
        #         self.get_logger().debug(f'Keypoint {name} proyectó detrás de cámara; se omite intento.')
        #         return None
        #     keypoints_uv[name] = uv

        # 2) Bounding box 2D a partir de puntos de envolvente del cilindro
        bbox_points_3d = self._tube_envelope_points(n_angles=18)
        uv_list = []
        for p_tube in bbox_points_3d:
            p_cam = R_ct @ p_tube + t_ct
            uv = self._project_point_pinhole(K, p_cam)
            if uv is not None:
                uv_list.append(uv)

        if not uv_list:
            self.get_logger().debug('No hay puntos proyectables para bbox; se omite intento.')
            return None

        h, w = image_cv.shape[:2]
        bbox = self._bbox_from_points(uv_list, w, h)
        if bbox is None:
            self.get_logger().debug('BBox inválida (fuera de imagen); se omite intento.')
            return None

        return ProjectedSample(
            image=image_cv,
            image_stamp=stamp,
            bbox_xyxy=bbox,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Guardado
    # ───────────────────────────────────────────────────────────────────────

    def _save_sample(self, sample: ProjectedSample, sample_index: int) -> None:
        split_name = self.split_plan[sample_index]
        sample_id = f'{sample_index:06d}'

        image_name = f'img_{sample_id}.jpg'
        label_name = f'img_{sample_id}.txt'

        image_path = os.path.join(self.images_dirs[split_name], image_name)
        label_path = os.path.join(self.labels_dirs[split_name], label_name)

        ok = cv2.imwrite(image_path, sample.image)
        if not ok:
            self.get_logger().error(f'No se pudo guardar imagen: {image_path}')
            return

        img_h, img_w = sample.image.shape[:2]
        yolo_bbox = self._xyxy_to_yolo(sample.bbox_xyxy, img_w, img_h)
        if yolo_bbox is None:
            self.get_logger().warn('BBox inválida al normalizar formato YOLO; se omite muestra.')
            return

        cx, cy, bw, bh = yolo_bbox
        with open(label_path, 'w', encoding='utf-8') as fp:
            fp.write(f'{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

    @staticmethod
    def _xyxy_to_yolo(
        bbox_xyxy: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        xmin, ymin, xmax, ymax = bbox_xyxy
        if img_w <= 0 or img_h <= 0:
            return None

        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0.0 or box_h <= 0.0:
            return None

        cx = ((xmin + xmax) * 0.5) / float(img_w)
        cy = ((ymin + ymax) * 0.5) / float(img_h)
        bw = box_w / float(img_w)
        bh = box_h / float(img_h)

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        return cx, cy, bw, bh

    @staticmethod
    def _build_split_plan(total_samples: int) -> List[str]:
        train_count = int(total_samples * 0.70)
        val_count = int(total_samples * 0.20)
        test_count = total_samples - train_count - val_count

        split_plan = ['train'] * train_count + ['val'] * val_count + ['test'] * test_count
        random.shuffle(split_plan)
        return split_plan

    # ───────────────────────────────────────────────────────────────────────
    # Geometría / proyección
    # ───────────────────────────────────────────────────────────────────────

    def _tf_to_rt(self, tf_msg) -> Tuple[np.ndarray, np.ndarray]:
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = self._rotation_matrix_from_quaternion(q.x, q.y, q.z, q.w)
        t_vec = np.array([t.x, t.y, t.z], dtype=np.float64)
        return R, t_vec

    def _compute_rt_from_link_states(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fallback para obtener camera_frame <- tube_frame usando /link_states.

        Asume que /link_states contiene, al menos, las poses de:
          - overhead_camera::overhead_camera_link
          - mars2020tube::tube_link
        """
        if self.latest_link_states is None:
            self._warn_throttled(
                'no_link_states',
                'No hay datos en /link_states para fallback de transformaciones.',
            )
            return None

        camera_pose = self._find_link_pose(self.latest_link_states, self.camera_link_name_in_gazebo)
        tube_pose = self._find_link_pose(self.latest_link_states, self.tube_link_name_in_gazebo)

        if camera_pose is None:
            self._warn_throttled(
                'camera_link_not_found',
                f'No se encontró {self.camera_link_name_in_gazebo} en /link_states. '
                'Ajusta camera_link_name_in_gazebo si el nombre real difiere.'
            )
            return None
        if tube_pose is None:
            self._warn_throttled(
                'tube_link_not_found',
                f'No se encontró {self.tube_link_name_in_gazebo} en /link_states. '
                'Ajusta tube_link_name_in_gazebo si el nombre real difiere.'
            )
            return None

        T_w_cam_link = self._pose_to_matrix(camera_pose)
        T_w_tube = self._pose_to_matrix(tube_pose)

        # Transformación camera_link <- tube
        T_cam_link_tube = np.linalg.inv(T_w_cam_link) @ T_w_tube

        # Si el frame de cámara objetivo es óptico, convertimos link->optical.
        if self.camera_frame == 'overhead_camera_optical_frame':
            # En overhead_camera.urdf.xacro:
            # origin rpy="-1.5707963267949 0 -1.5707963267949" parent=link child=optical
            R_link_opt = self._rotation_matrix_from_euler(-1.5707963267949, 0.0, -1.5707963267949)
            T_link_opt = np.eye(4, dtype=np.float64)
            T_link_opt[:3, :3] = R_link_opt

            # Queremos optical <- tube:
            T_opt_tube = np.linalg.inv(T_link_opt) @ T_cam_link_tube
            R = T_opt_tube[:3, :3]
            t = T_opt_tube[:3, 3]
            return R, t

        R = T_cam_link_tube[:3, :3]
        t = T_cam_link_tube[:3, 3]
        return R, t

    @staticmethod
    def _find_link_pose(link_states_msg, desired_name: str):
        idx = SyntheticDatasetGenerator._find_link_index(link_states_msg, desired_name)
        if idx is None:
            return None
        return link_states_msg.pose[idx]

    @staticmethod
    def _find_link_twist(link_states_msg, desired_name: str):
        idx = SyntheticDatasetGenerator._find_link_index(link_states_msg, desired_name)
        if idx is None:
            return None
        return link_states_msg.twist[idx]

    @staticmethod
    def _find_link_index(link_states_msg, desired_name: str) -> Optional[int]:
        if link_states_msg is None:
            return None

        names = list(link_states_msg.name)

        # 1) Coincidencia exacta
        if desired_name in names:
            return names.index(desired_name)

        # 2) Coincidencia por sufijo (útil si cambia el prefijo de modelo)
        for idx, name in enumerate(names):
            if name.endswith(desired_name) or name.endswith('::' + desired_name.split('::')[-1]):
                return idx

        return None

    def _pose_to_matrix(self, pose: Pose) -> np.ndarray:
        R = self._rotation_matrix_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[0, 3] = float(pose.position.x)
        T[1, 3] = float(pose.position.y)
        T[2, 3] = float(pose.position.z)
        return T

    @staticmethod
    def _project_point_pinhole(K: np.ndarray, p_cam: np.ndarray) -> Optional[Tuple[float, float]]:
        z = float(p_cam[2])
        if z <= 1e-6:
            return None

        x = float(p_cam[0]) / z
        y = float(p_cam[1]) / z

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        u = fx * x + cx
        v = fy * y + cy
        return float(u), float(v)

    def _tube_envelope_points(self, n_angles: int = 18) -> List[np.ndarray]:
        """Puntos de la envolvente del cilindro para estimar bbox 2D."""
        points = []
        z_values = [-0.5 * self.tube_length, +0.5 * self.tube_length]

        for z in z_values:
            for i in range(n_angles):
                a = 2.0 * math.pi * i / n_angles
                x = self.tube_radius * math.cos(a)
                y = self.tube_radius * math.sin(a)
                points.append(np.array([x, y, z], dtype=np.float64))

        # Añadimos también centro y extremos de eje para robustez
        points.extend(
            [
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, +0.5 * self.tube_length], dtype=np.float64),
                np.array([0.0, 0.0, -0.5 * self.tube_length], dtype=np.float64),
            ]
        )
        return points

    @staticmethod
    def _bbox_from_points(
        uv_list: List[Tuple[float, float]],
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        xs = [p[0] for p in uv_list]
        ys = [p[1] for p in uv_list]

        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(float(img_w - 1), max(xs))
        ymax = min(float(img_h - 1), max(ys))

        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax

    # ───────────────────────────────────────────────────────────────────────
    # Utilidades matemáticas
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rotation_matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convierte cuaternión (x,y,z,w) a matriz R 3x3."""
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convierte RPY a cuaternión (x,y,z,w)."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return x, y, z, w

    @staticmethod
    def _rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        R_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float64,
        )
        R_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )
        R_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return R_z @ R_y @ R_x

    @staticmethod
    def _quaternion_angular_distance(
        x1: float, y1: float, z1: float, w1: float,
        x2: float, y2: float, z2: float, w2: float,
    ) -> float:
        dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2
        dot = min(max(abs(dot), 0.0), 1.0)
        return 2.0 * math.acos(dot)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticDatasetGenerator()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido por usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
