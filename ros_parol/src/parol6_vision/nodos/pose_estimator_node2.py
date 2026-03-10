#!/usr/bin/env python3

import numpy as np
import rclpy
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Pose
from std_msgs.msg import String


# Estados del nodo
IDLE    = 'idle'     # esperando detección, listo para publicar
MOVING  = 'moving'  # movimiento en curso, ignorar detecciones

class PoseEstimatorNode2(Node):
    def __init__(self):
        super().__init__('pose_estimator_node2')

        # ── Parámetros ──────────────────────────────────────────────────────
        
        # Posición de la cámara en world 
        self.declare_parameter('cam_x_world', 0.2)   # m
        self.declare_parameter('cam_y_world', 0.0)   # m
        self.declare_parameter('cam_z_world', 0.75)  # m


        # Confianza mínima para procesar una detección
        self.declare_parameter('min_score', 0.3)

        self.cam_x   = self.get_parameter('cam_x_world').value
        self.cam_y   = self.get_parameter('cam_y_world').value
        self.cam_z   = self.get_parameter('cam_z_world').value

        self.min_score = self.get_parameter('min_score').value

        # ── Intrínseca (se actualiza al recibir camera_info) ────────────────
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_ready = False

        # ── Estado ───────────────────────────────────────────────────────────
        self.state = IDLE

        # ── CV Bridge ───────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Última máscara recibida ──────────────────────────────────────────
        self.last_mask = None

        # ── Suscriptores ────────────────────────────────────────────────────
        cam_info_topic = '/overhead_camera/overhead_cam/camera_info'
        mask_topic     = '/vision/mask_image'
        det_topic      = '/vision/detections'

        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._camera_info_cb, 10)
        self.create_subscription(Image, mask_topic,
                                 self._mask_cb, 10)
        self.create_subscription(Detection2DArray, det_topic,
                                 self._detections_cb, 10)

        # Escucha el estado que publica pose_ik_subscriber
        # 'done'  → movimiento completado, volver a IDLE
        # 'error' → IK fallida o movimiento rechazado, volver a IDLE
        self.create_subscription(String, '/robot_motion_status',
                                 self._motion_status_cb, 10)

        # ── Publicador ───────────────────────────────────────────────────────
        output_topic = '/pose_target_cuaterniones'
        self.pub_pose = self.create_publisher(Pose, output_topic, 10)

        self.get_logger().info(
            f'PoseEstimatorNode iniciado.\n'
            f'  Cámara world: x={self.cam_x}, y={self.cam_y}, z={self.cam_z}\n'
        )

    
    # ── Callbacks ────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        """Recibe la intrínseca una sola vez y cancela la suscripción."""
        if self.camera_ready:
            return
        k = msg.k  # matriz 3x3 en row-major
        self.fx = k[0]
        self.fy = k[4]
        self.cx = k[2]
        self.cy = k[5]
        self.camera_ready = True
        self.get_logger().info(
            f'Intrínseca recibida: fx={self.fx:.2f}, fy={self.fy:.2f}, '
            f'cx={self.cx:.2f}, cy={self.cy:.2f}'
        )

    
    def _mask_cb(self, msg: Image):
        """Guarda la última máscara para usarla en el callback de detecciones."""
        try:
            self.last_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error convirtiendo máscara: {e}')

    def _motion_status_cb(self, msg: String):
        """Recibe 'done' o 'error' de pose_ik_subscriber cuando termina el movimiento."""
        status = msg.data
        if status in ('done', 'error'):
            self.get_logger().info(f'Movimiento terminado ({status}) → volviendo a IDLE')
            self.state = IDLE

    def _detections_cb(self, msg: Detection2DArray):
        """Procesa las detecciones y publica la pose del tubo."""
        if not self.camera_ready:
            self.get_logger().warn('Intrínseca aún no recibida, ignorando detección.')
            return
        if self.state == MOVING:
            return
        if not msg.detections:
            return

        # Seleccionar la detección con mayor confianza
        best = max(msg.detections,
                   key=lambda d: d.results[0].hypothesis.score if d.results else 0.0)

        if not best.results or best.results[0].hypothesis.score < self.min_score:
            self.get_logger().warn('Detección con confianza insuficiente, ignorando.')
            return

        score = best.results[0].hypothesis.score
        self.get_logger().info(f'Procesando detección (score={score:.2f})')

        # ── Posición: centroide de la máscara ────────────────────────────────
        if self.last_mask is not None:
            u, v = self._centroid_from_mask(self.last_mask)
        else:
            # Fallback: centro del bounding box
            u = best.bbox.center.position.x
            v = best.bbox.center.position.y
            self.get_logger().warn('Sin máscara, usando centro del bbox.')

        # ── Proyección inversa píxel → world ────────────────────────────────
        x_world, y_world = self._pixel_to_world(u, v)

        # Z del gripper = altura del tubo + offset para no chocar
        z_gripper = 0.085

        # ── Orientación: yaw alineado con el eje mayor del tubo ──────────────
        qx, qy, qz, qw = 0.707, 0.0, -0.707, 0.0  # default: gripper hacia abajo

        # ── Construir y publicar Pose ─────────────────────────────────────────
        pose = Pose()
        pose.position.x = x_world - 0.06 # offset de L6 al centro del gripper
        pose.position.y = y_world
        pose.position.z = z_gripper
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        # Pasar a MOVING ANTES de publicar para evitar que otro callback
        # llegue entre la publicación y el cambio de estado
        self.state = MOVING
        self.pub_pose.publish(pose)
        self.get_logger().info(
            f'Pose publicada → '
            f'x={x_world:.3f}, y={y_world:.3f}, z={z_gripper:.3f}, '
            f'qx={qx:.3f}, qy={qy:.3f}, qz={qz:.3f}, qw={qw:.3f}'
        )

    def _centroid_from_mask(self, mask: np.ndarray):
        """Calcula el centroide (u, v) en píxeles a partir de la máscara binaria."""
        M = cv2.moments(mask)
        if M['m00'] < 1.0:
            # Sin área: usar centro de imagen
            self.get_logger().warn('Máscara vacía, usando centro de imagen.')
            return self.cx, self.cy
        u = M['m10'] / M['m00']
        v = M['m01'] / M['m00']
        return u, v


    def _pixel_to_world(self, u: float, v: float):
        """
        Proyección inversa: píxel (u,v) → coordenadas world (x, y).

        La cámara está a (cam_x, cam_y, cam_z) en world, mirando hacia abajo.
        El frame óptico ROS (Z fwd, X right, Y down) con pitch=90° mapea así:
            depth = cam_z - z_tube          (distancia al plano del tubo)
            X_optical =  (u - cx) / fx * depth
            Y_optical =  (v - cy) / fy * depth

        Con la rotación world→óptico (pitch=90°):
            x_world =  cam_x - Y_optical    (Y_optical apunta en -X_world)
            y_world =  cam_y - X_optical    (X_optical apunta en -Y_world)
        """
        depth = self.cam_z - 0.03 # distancia al plano del tubo (asumiendo tubo a 3cm del suelo)
        x_opt = (u - self.cx) / self.fx * depth
        y_opt = (v - self.cy) / self.fy * depth

        x_world = self.cam_x - y_opt
        y_world = self.cam_y - x_opt
        return x_world, y_world


# ── main ─────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()