#!/usr/bin/env python3
"""
pick_and_place_orchestrator.py  —  Orquestador del ciclo pick-and-place del PAROL6.

Controla directamente brazo y gripper via MoveIt sin nodos intermediarios.

Ciclo:
    INIT → SCAN → OPEN_GRIPPER → APPROACH → PICK → CLOSE_GRIPPER
         → LIFT → DEPOSIT → OPEN_GRIPPER → HOME → SCAN

Interfaces:
    Suscribe:
        /vision/detections                        (vision_msgs/Detection2DArray)
        /vision/mask_image                        (sensor_msgs/Image)
        /overhead_camera/.../camera_info          (sensor_msgs/CameraInfo)
        /joint_states                             (sensor_msgs/JointState)

    Publica:
        /pick_and_place_state                            (std_msgs/String)  estado actual

Parámetros configurables:
    cam_x/y/z_world     posición de la cámara en world
    z_approach          Z al acercarse sobre el tubo (seguro)
    z_pick              Z al bajar a coger el tubo
    z_lift              Z al levantar tras coger
    deposit_x/y/z       posición de depósito
    orient_x/y/z/w      quaternion del gripper apuntando hacia abajo
    min_score           confianza mínima YOLO
    confirm_detections  detecciones consecutivas para confirmar objetivo
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    PositionIKRequest,
)
from moveit_msgs.srv import GetPositionIK


# ── Constantes del robot ──────────────────────────────────────────────────────
ARM_JOINT_NAMES     = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
GRIPPER_JOINT_NAMES = ['gripper_joint', 'gripper_right_joint']
ARM_GROUP           = 'arm_group'
GRIPPER_GROUP       = 'gripper'
TIP_LINK            = 'L6_link'
BASE_FRAME          = 'base_link'
GRIPPER_OPEN        = 0.025   # metros (estado 'open' del SRDF)
GRIPPER_CLOSED      = 0.008     # metros (para anchura del tubo)

# ── Estados de la misión ──────────────────────────────────────────────────────
INIT          = 'INIT'
SCAN          = 'SCAN'
OPEN_GRIPPER  = 'OPEN_GRIPPER'
APPROACH      = 'APPROACH'
PICK          = 'PICK'
CLOSE_GRIPPER = 'CLOSE_GRIPPER'
LIFT          = 'LIFT'
DEPOSIT       = 'DEPOSIT'
RELEASE       = 'RELEASE'   # bajar en depósito + abrir gripper
HOME          = 'HOME'
ERROR         = 'ERROR'


class PickAndPlaceOrchestrator(Node):
    def __init__(self):
        super().__init__('pick_and_place_orchestrator')

        # ── Parámetros ───────────────────────────────────────────────────────
        self.declare_parameter('cam_x_world',        0.2)
        self.declare_parameter('cam_y_world',        0.0)
        self.declare_parameter('cam_z_world',        0.75)
        self.declare_parameter('z_approach',         0.15)
        self.declare_parameter('z_pick',             0.085)
        self.declare_parameter('z_lift',             0.20)
        self.declare_parameter('deposit_x',          0.10)
        self.declare_parameter('deposit_y',          0.25)
        self.declare_parameter('deposit_z',          0.15)
        self.declare_parameter('orient_x',           0.707)
        self.declare_parameter('orient_y',           0.0)
        self.declare_parameter('orient_z',          -0.707)
        self.declare_parameter('orient_w',           0.0)
        self.declare_parameter('min_score',          0.2)
        self.declare_parameter('confirm_detections', 3)
        self.declare_parameter('arm_velocity_scaling',     0.3)
        self.declare_parameter('gripper_velocity_scaling', 0.5)
        # Retardo (s) tras completar cada estado antes de pasar al siguiente
        self.declare_parameter('delay_after_open_gripper',  0.0)
        self.declare_parameter('delay_after_approach',      0.0)
        self.declare_parameter('delay_after_pick',          0.0)
        self.declare_parameter('delay_after_close_gripper', 0.0)
        self.declare_parameter('delay_after_lift',          0.0)
        self.declare_parameter('delay_after_deposit',       0.0)
        self.declare_parameter('delay_after_release',       0.0)
        self.declare_parameter('delay_after_home',          0.0)
        self._load_params()

        # ── Estado ───────────────────────────────────────────────────────────
        self.state            = INIT
        self.confirm_count    = 0
        self.target_x         = 0.0
        self.target_y         = 0.0
        self._pending_action  = None   # 'arm' | 'gripper' — qué acción espera respuesta

        # ── Intrínseca ───────────────────────────────────────────────────────
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_ready = False

        # ── CV Bridge ────────────────────────────────────────────────────────
        self.bridge    = CvBridge()
        self.last_mask = None

        # ── Joint state (para leer posición actual del gripper) ──────────────
        self._current_joint_state = None

        # ── Clientes MoveIt ──────────────────────────────────────────────────
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._ik_client     = self.create_client(GetPositionIK, '/compute_ik')

        # ── Suscriptores ─────────────────────────────────────────────────────
        self.create_subscription(CameraInfo,
            '/overhead_camera/overhead_cam/camera_info', self._camera_info_cb, 10)
        self.create_subscription(Image,
            '/vision/mask_image', self._mask_cb, 10)
        self.create_subscription(Detection2DArray,
            '/vision/detections', self._detections_cb, 10)
        self.create_subscription(JointState,
            '/joint_states', self._joint_state_cb, 10)

        # ── Publicadores ─────────────────────────────────────────────────────
        self.pub_state = self.create_publisher(String, '/pick_and_place_state', 10)
        self.pub_vision_need = self.create_publisher(String, '/vision/activate', 10)

        self.get_logger().info('PickAndPlaceOrchestrator iniciado. Esperando MoveIt y cámara...')

        # Esperar servicios antes de arrancar
        self.get_logger().info('Esperando /compute_ik...')
        self._ik_client.wait_for_service()
        self.get_logger().info('Esperando /move_action...')
        self._action_client.wait_for_server()
        self.get_logger().info('MoveIt listo.')

    # ── Parámetros ───────────────────────────────────────────────────────────

    def _load_params(self):
        self.cam_x  = self.get_parameter('cam_x_world').value
        self.cam_y  = self.get_parameter('cam_y_world').value
        self.cam_z  = self.get_parameter('cam_z_world').value
        self.z_approach  = self.get_parameter('z_approach').value
        self.z_pick      = self.get_parameter('z_pick').value
        self.z_lift      = self.get_parameter('z_lift').value
        self.deposit_x   = self.get_parameter('deposit_x').value
        self.deposit_y   = self.get_parameter('deposit_y').value
        self.deposit_z   = self.get_parameter('deposit_z').value
        self.orient_x    = self.get_parameter('orient_x').value
        self.orient_y    = self.get_parameter('orient_y').value
        self.orient_z    = self.get_parameter('orient_z').value
        self.orient_w    = self.get_parameter('orient_w').value
        self.min_score                = self.get_parameter('min_score').value
        self.confirm_detections       = int(self.get_parameter('confirm_detections').value)
        self.arm_velocity_scaling     = self.get_parameter('arm_velocity_scaling').value
        self.gripper_velocity_scaling = self.get_parameter('gripper_velocity_scaling').value
        self.state_delays = {
            OPEN_GRIPPER:  float(self.get_parameter('delay_after_open_gripper').value),
            APPROACH:      float(self.get_parameter('delay_after_approach').value),
            PICK:          float(self.get_parameter('delay_after_pick').value),
            CLOSE_GRIPPER: float(self.get_parameter('delay_after_close_gripper').value),
            LIFT:          float(self.get_parameter('delay_after_lift').value),
            DEPOSIT:       float(self.get_parameter('delay_after_deposit').value),
            RELEASE:       float(self.get_parameter('delay_after_release').value),
            HOME:          float(self.get_parameter('delay_after_home').value),
        }

    # ── Callbacks de sensores ─────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_ready:
            return
        k = msg.k
        self.fx, self.fy = k[0], k[4]
        self.cx, self.cy = k[2], k[5]
        self.camera_ready = True
        self.get_logger().info(
            f'Intrínseca: fx={self.fx:.1f}, cx={self.cx:.1f}')
        self._transition(SCAN)

    def _mask_cb(self, msg: Image):
        try:
            self.last_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error máscara: {e}')

    def _joint_state_cb(self, msg: JointState):
        self._current_joint_state = msg

    def _detections_cb(self, msg: Detection2DArray):
        """Solo actúa en SCAN. Requiere N detecciones consecutivas antes de actuar."""
        if self.state != SCAN or not self.camera_ready:
            return
        if not msg.detections:
            self.confirm_count = 0
            return

        best = max(msg.detections,
                   key=lambda d: d.results[0].hypothesis.score if d.results else 0.0)
        if not best.results or best.results[0].hypothesis.score < self.min_score:
            self.confirm_count = 0
            return

        if self.last_mask is not None:
            u, v = self._centroid_from_mask(self.last_mask)
            print(f'COMPARACIÓN DE CENTROS: bbox=({best.bbox.center.position.x:.1f}, {best.bbox.center.position.y:.1f}), mask=({u:.1f}, {v:.1f})')
        else:
            u = best.bbox.center.position.x
            v = best.bbox.center.position.y

        x_world, y_world = self._pixel_to_world(u, v)
        self.confirm_count += 1
        self.get_logger().info(
            f'Detección {self.confirm_count}/{self.confirm_detections} '
            f'— tubo en x={x_world:.3f}, y={y_world:.3f}')

        if self.confirm_count >= self.confirm_detections:
            self.confirm_count = 0
            self.target_x = x_world
            self.target_y = y_world
            self.get_logger().info(
                f'Tubo confirmado → x={self.target_x:.3f}, y={self.target_y:.3f}')
            self._transition(OPEN_GRIPPER)

    # ── Máquina de estados ────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        self.get_logger().info(f'[{self.state}] → [{new_state}]')
        self.state = new_state
        self._publish_state()

        if new_state == SCAN:
            msg = String()
            msg.data = 'activate'
            self.pub_vision_need.publish(msg)
            self.get_logger().info('Escaneando... esperando detección de tubo.')

        elif new_state == OPEN_GRIPPER:
            msg = String()
            msg.data = 'deactivate'
            self.pub_vision_need.publish(msg)
            self._move_gripper_async(GRIPPER_OPEN)

        elif new_state == APPROACH:
            self._move_arm_to_pose_async(self.target_x, self.target_y, self.z_approach)

        elif new_state == PICK:
            self._move_arm_to_pose_async(self.target_x, self.target_y, self.z_pick)

        elif new_state == CLOSE_GRIPPER:
            self._move_gripper_async(GRIPPER_CLOSED)

        elif new_state == LIFT:
            self._move_arm_to_pose_async(self.target_x, self.target_y, self.z_lift)

        elif new_state == DEPOSIT:
            self._move_arm_to_pose_async(self.deposit_x, self.deposit_y, self.z_lift)

        elif new_state == RELEASE:
            # Bajar en el depósito y luego abrir el gripper
            self._move_arm_to_pose_async(self.deposit_x, self.deposit_y, self.deposit_z)

        elif new_state == HOME:
            self._move_arm_to_pose_async(0.2, 0.0, 0.33)

        elif new_state == ERROR:
            self.get_logger().error('Estado ERROR — misión detenida.')

    def _delayed_transition(self, next_state: str, delay: float = 0.0):
        """Transiciona a next_state tras un retardo opcional sin bloquear el executor."""
        if delay <= 0.0:
            self._transition(next_state)
            return
        self.get_logger().info(f'Esperando {delay:.1f}s antes de → {next_state}')
        # El timer se crea repetitivo pero se cancela tras el primer disparo.
        def _fire():
            _timer.cancel()
            self._transition(next_state)
        _timer = self.create_timer(delay, _fire)

    def _on_motion_done(self, success: bool):
        """Callback unificado cuando cualquier movimiento (brazo o gripper) termina."""
        if not success:
            self.get_logger().error(f'Movimiento fallido en estado {self.state} → SCAN')
            self._transition(SCAN)
            return

        # Avanzar al siguiente estado
        next_state = {
            OPEN_GRIPPER:  APPROACH,
            APPROACH:      PICK,
            PICK:          CLOSE_GRIPPER,
            CLOSE_GRIPPER: LIFT,
            LIFT:          DEPOSIT,
            DEPOSIT:       RELEASE,
            RELEASE:       HOME,
            HOME:          SCAN,
        }.get(self.state)

        if next_state is None:
            self.get_logger().warn(f'Estado inesperado al terminar movimiento: {self.state}')
            return

        # RELEASE tiene dos pasos: bajar brazo → abrir gripper
        # Cuando el brazo termina de bajar, todavía estamos en RELEASE,
        # así que abrimos el gripper antes de pasar a HOME.
        if self.state == RELEASE and self._pending_action == 'arm':
            self._pending_action = 'gripper'
            self._move_gripper_async(GRIPPER_OPEN)
            return

        self._delayed_transition(next_state, self.state_delays.get(self.state, 0.0))

    # ── Control del brazo ─────────────────────────────────────────────────────

    def _move_arm_to_pose_async(self, x: float, y: float, z: float):
        """Resuelve IK para la pose dada y envía el movimiento al brazo."""
        self._pending_action = 'arm'

        pose = Pose()
        pose.position.x    = x - 0.06 # offset de L6 al centro del gripper
        pose.position.y    = y
        pose.position.z    = z
        pose.orientation.x = self.orient_x
        pose.orientation.y = self.orient_y
        pose.orientation.z = self.orient_z
        pose.orientation.w = self.orient_w

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = BASE_FRAME
        pose_stamped.header.stamp    = self.get_clock().now().to_msg()
        pose_stamped.pose            = pose

        ik_request = PositionIKRequest()
        ik_request.group_name       = ARM_GROUP
        ik_request.ik_link_name     = TIP_LINK
        ik_request.avoid_collisions = True
        ik_request.timeout.sec      = 5
        ik_request.pose_stamped     = pose_stamped

        request = GetPositionIK.Request()
        request.ik_request = ik_request

        self.get_logger().info(
            f'Resolviendo IK para x={x:.3f}, y={y:.3f}, z={z:.3f} '
            f'(vel={self.arm_velocity_scaling})')
        future = self._ik_client.call_async(request)
        future.add_done_callback(self._ik_response_cb)

    def _ik_response_cb(self, future):
        response   = future.result()
        error_code = response.error_code.val
        if error_code != 1:
            self.get_logger().error(f'IK fallida (código: {error_code})')
            self._on_motion_done(False)
            return

        joint_state = response.solution.joint_state
        positions   = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        target      = [positions.get(j, 0.0) for j in ARM_JOINT_NAMES]
        self.get_logger().info(f'IK resuelta: {[f"{v:.3f}" for v in target]}')
        self._send_joint_goal(ARM_GROUP, ARM_JOINT_NAMES, target,
                              velocity_scaling=self.arm_velocity_scaling)

    # ── Control del gripper ───────────────────────────────────────────────────

    def _move_gripper_async(self, position: float):
        """Abre (0.025) o cierra (0.008) el gripper."""
        self._pending_action = 'gripper'
        action = 'Abriendo' if position >= GRIPPER_OPEN else 'Cerrando'
        self.get_logger().info(f'{action} gripper (posición: {position})')
        self._send_joint_goal(
            GRIPPER_GROUP,
            GRIPPER_JOINT_NAMES,
            [position, position],
            velocity_scaling=self.gripper_velocity_scaling,
        )

    # ── Envío genérico de goal de joints a MoveIt ─────────────────────────────

    def _send_joint_goal(self, group: str, joint_names: list,
                         positions: list, velocity_scaling: float = 0.3):
        goal = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name                      = group
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 5.0
        req.max_velocity_scaling_factor     = velocity_scaling
        req.max_acceleration_scaling_factor = velocity_scaling

        constraints = Constraints()
        for name, pos in zip(joint_names, positions):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = pos
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request                          = req
        goal.planning_options                 = PlanningOptions()
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rechazado por move_group.')
            self._on_motion_done(False)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        ok = result.error_code.val == 1
        if ok:
            self.get_logger().info('Movimiento completado OK.')
        else:
            self.get_logger().error(f'Movimiento fallido (código: {result.error_code.val}).')
        self._on_motion_done(ok)

    # ── Utilidades de estado ──────────────────────────────────────────────────

    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.pub_state.publish(msg)

    # ── Utilidades de visión ──────────────────────────────────────────────────

    def _centroid_from_mask(self, mask):
        M = cv2.moments(mask)
        if M['m00'] < 1.0:
            return self.cx, self.cy
        return M['m10'] / M['m00'], M['m01'] / M['m00']

    def _pixel_to_world(self, u: float, v: float):
        depth = self.cam_z - 0.03
        x_opt = (u - self.cx) / self.fx * depth
        y_opt = (v - self.cy) / self.fy * depth
        return self.cam_x - y_opt, self.cam_y - x_opt


# ── main ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()