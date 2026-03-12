#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetEntityState, GetEntityState, DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, Point, Quaternion
import tf2_ros
import numpy as np
import time


# ── Utilidades de quaternion ───────────────────────────────────────────────────

def quat_to_np(q):
    """geometry_msgs/Quaternion → numpy [x, y, z, w]"""
    return np.array([q.x, q.y, q.z, q.w])

def np_to_quat(q):
    """numpy [x, y, z, w] → geometry_msgs/Quaternion"""
    out = Quaternion()
    out.x, out.y, out.z, out.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return out

def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, ambos [x,y,z,w]"""
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

def quat_conjugate(q):
    """Inversa de un quaternion unitario [x,y,z,w]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def rotate_vector(v, q):
    """Rota el vector v (array 3) por el quaternion q [x,y,z,w]"""
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[:3]

# ──────────────────────────────────────────────────────────────────────────────


class VirtualGripperNode(Node):
    def __init__(self):
        super().__init__('virtual_gripper_node')
        
        # Parámetros configurables
        self.declare_parameter('gripper_closed_threshold', 0.011)  # Posición considerada "cerrado"
        self.declare_parameter('gripper_open_threshold', 0.012)    # Posición considerada "abierto"
        self.declare_parameter('min_contact_time', 0.2)            # Tiempo mínimo de contacto
        
        self.gripper_closed_threshold = self.get_parameter('gripper_closed_threshold').value
        self.gripper_open_threshold = self.get_parameter('gripper_open_threshold').value
        self.min_contact_time = self.get_parameter('min_contact_time').value
        
        # Estado del sistema
        self.left_contact = False
        self.right_contact = False
        self.last_contact_time = 0.0
        self.gripper_position = 0.025  # Abierto por defecto
        self.attached_object = None
        self.object_initial_pose = None
        
        # Offset en frame LOCAL del gripper (se calcula al agarrar)  # CAMBIO
        self.grasp_offset_local = None      # np.array [x,y,z]       # CAMBIO
        self.grasp_orientation_rel = None   # np.array [x,y,z,w]     # CAMBIO

        # Flags para evitar llamadas paralelas a los servicios
        self._attaching = False
        self._updating = False

        # Buffer TF para leer la pose del gripper
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Suscriptores
        self.left_contact_sub = self.create_subscription(
            ContactsState,
            '/left_finger_contact',
            self.left_contact_callback,
            qos_profile_sensor_data
        )
        
        self.right_contact_sub = self.create_subscription(
            ContactsState,
            '/right_finger_contact',
            self.right_contact_callback,
            qos_profile_sensor_data
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Clientes de servicios de Gazebo
        self.get_entity_client = self.create_client(GetEntityState, '/get_entity_state')
        self.set_entity_client = self.create_client(SetEntityState, '/set_entity_state')

        # Esperar a que los servicios estén disponibles
        self.get_logger().info('Esperando servicios de Gazebo...')
        self.get_entity_client.wait_for_service(timeout_sec=10.0)
        self.set_entity_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('Servicios de Gazebo disponibles.')
        
        # Timer para actualizar la posición del objeto agarrado
        self.update_timer = self.create_timer(0.01, self.update_attached_object)  # 100 Hz
        
        self.get_logger().info('Virtual Gripper Node iniciado')
        self.get_logger().info(f'  - Umbral cerrado: {self.gripper_closed_threshold}m')
        self.get_logger().info(f'  - Umbral abierto: {self.gripper_open_threshold}m')
        self.get_logger().info(f'  - Tiempo mínimo contacto: {self.min_contact_time}s')

    def _get_gripper_pose(self):
        """Obtiene la pose actual de gripper_base_link en world via TF.
        Devuelve (pos np.array[3], quat np.array[x,y,z,w]) o (None, None)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', 'gripper_base_link',
                rclpy.time.Time())
            pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ])
            q = tf.transform.rotation
            quat = np.array([q.x, q.y, q.z, q.w])
            return pos, quat
        except Exception as e:
            self.get_logger().warn(f'TF no disponible: {e}')
            return None, None
    
    def _extract_object_name(self, state):
        """Extrae el nombre del objeto de un estado de contacto.
        Prueba collision1_name y collision2_name, devuelve el que no sea 'parol6'."""
        for cname in (state.collision1_name, state.collision2_name):
            if '::' in cname:
                name = cname.split('::')[0]
                if name != 'parol6':
                    return name
        return None

    def left_contact_callback(self, msg):
        """Callback para contactos del dedo izquierdo"""
        if len(msg.states) > 0:
            self.left_contact = True
            self.last_contact_time = time.time()

            if not self.attached_object:
                object_name = self._extract_object_name(msg.states[0])
                if object_name:
                    self.check_and_attach(object_name)
        else:
            self.left_contact = False
    
    def right_contact_callback(self, msg):
        """Callback para contactos del dedo derecho"""
        if len(msg.states) > 0:
            self.right_contact = True
            self.last_contact_time = time.time()

            if not self.attached_object:
                object_name = self._extract_object_name(msg.states[0])
                if object_name:
                    self.check_and_attach(object_name)
        else:
            self.right_contact = False
    
    def joint_state_callback(self, msg):
        """Callback para estados de las articulaciones"""
        if 'gripper_joint' in msg.name:
            idx = msg.name.index('gripper_joint')
            self.gripper_position = msg.position[idx]
            
            if self.attached_object and self.gripper_position > self.gripper_open_threshold:
                self.detach_object()
    
    def check_and_attach(self, object_name):
        """Verifica condiciones y agarra el objeto"""
        time_since_contact = time.time() - self.last_contact_time
        if (self.left_contact and 
            self.right_contact and 
            self.gripper_position < self.gripper_closed_threshold and
            time_since_contact < self.min_contact_time and
            not self.attached_object and
            not self._attaching):
            self._attaching = True
            self.attach_object(object_name)
    
    def attach_object(self, object_name):
        """Crea attachment virtual del objeto al gripper"""
        self.get_logger().info(f'Intentando agarrar objeto: {object_name}')
        
        req = GetEntityState.Request()
        req.name = object_name
        req.reference_frame = 'world'
        
        future = self.get_entity_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_object_pose(f, object_name))

    def _on_object_pose(self, future, object_name):
        """Callback: recibida la pose del objeto, calcular offset en frame local del gripper."""
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Excepción al obtener pose del objeto: {e}')
            self._attaching = False
            return

        if result is None or not result.success:
            self.get_logger().warn(f'No se pudo obtener pose del objeto {object_name}')
            self._attaching = False
            return

        obj_pose = result.state.pose
        self.get_logger().info(f'Pose objeto obtenida: x={obj_pose.position.x:.3f}')

        # Leer pose del gripper via TF
        grip_pos, grip_quat = self._get_gripper_pose()
        if grip_pos is None:
            self.get_logger().warn('No se pudo obtener pose del gripper via TF')
            self._attaching = False
            return

        # Vector offset en world frame
        obj_pos = np.array([obj_pose.position.x,
                            obj_pose.position.y,
                            obj_pose.position.z])
        delta_world = obj_pos - grip_pos

        # Rotar el offset al frame LOCAL del gripper:               # CAMBIO
        # offset_local = q_grip_inv * delta_world                   # CAMBIO
        grip_quat_inv = quat_conjugate(grip_quat)                   # CAMBIO
        self.grasp_offset_local = rotate_vector(delta_world,        # CAMBIO
                                                grip_quat_inv)      # CAMBIO

        # Orientación relativa del objeto respecto al gripper:      # CAMBIO
        # q_rel = q_grip_inv * q_obj                                # CAMBIO
        obj_quat = quat_to_np(obj_pose.orientation)                 # CAMBIO
        self.grasp_orientation_rel = quat_multiply(                 # CAMBIO
            grip_quat_inv, obj_quat)                                # CAMBIO

        self.attached_object = object_name
        self._attaching = False
        self.get_logger().info(f'✅ Objeto {object_name} agarrado!')
        self.get_logger().info(
            f'   Offset local: x={self.grasp_offset_local[0]:.3f}, '
            f'y={self.grasp_offset_local[1]:.3f}, '
            f'z={self.grasp_offset_local[2]:.3f}')
    
    def update_attached_object(self):
        """Actualiza la posición del objeto agarrado para que siga al gripper"""
        if not self.attached_object or self.grasp_offset_local is None:
            return

        if self._updating:
            return
        self._updating = True

        # Leer pose actual del gripper via TF
        grip_pos, grip_quat = self._get_gripper_pose()
        self._updating = False
        if grip_pos is None:
            return

        # Offset local → world: delta_world = q_grip * offset_local # CAMBIO
        delta_world = rotate_vector(self.grasp_offset_local,        # CAMBIO
                                    grip_quat)                      # CAMBIO

        # Orientación del objeto: q_obj = q_grip * q_rel            # CAMBIO
        obj_quat = quat_multiply(grip_quat,                         # CAMBIO
                                 self.grasp_orientation_rel)        # CAMBIO

        new_pose = Pose()
        new_pose.position.x = grip_pos[0] + delta_world[0]
        new_pose.position.y = grip_pos[1] + delta_world[1]
        new_pose.position.z = grip_pos[2] + delta_world[2]
        new_pose.orientation = np_to_quat(obj_quat)                 # CAMBIO: orientación dinámica

        set_req = SetEntityState.Request()
        set_req.state.name = self.attached_object
        set_req.state.pose = new_pose
        set_req.state.reference_frame = 'world'
        
        self.set_entity_client.call_async(set_req)
    
    def detach_object(self):
        """Suelta el objeto"""
        if self.attached_object:
            self.get_logger().info(f'🔓 Soltando objeto: {self.attached_object}')
            self.attached_object = None
            self.grasp_offset_local = None      # CAMBIO
            self.grasp_orientation_rel = None   # CAMBIO


def main(args=None):
    rclpy.init(args=args)
    node = VirtualGripperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()