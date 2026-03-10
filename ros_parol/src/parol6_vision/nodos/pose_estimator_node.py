#!/usr/bin/env python3
"""
pose_estimator_node.py

Suscribe a:
  - /vision/detections     (vision_msgs/Detection2DArray)  bboxes del tubo
  - /vision/mask_image     (sensor_msgs/Image)             máscara binaria del tubo
  - /overhead_camera/overhead_cam/camera_info  (sensor_msgs/CameraInfo)

Publica en:
  - /pose_target_cuaterniones  (geometry_msgs/Pose)  pose del tubo en frame base_link,
                                                      lista para pose_ik_subscriber.py

──────────────────────────────────────────────────────────────────────────────
GEOMETRÍA
──────────────────────────────────────────────────────────────────────────────
La cámara es cenital perfecta (pitch=90°) montada en:
    world → overhead_camera_link: x=0.2, y=0.0, z=0.75

Proyección inversa píxel → world (plano suelo Z=Z_TUBE_WORLD):
    X_cam = (u - cx) / fx * depth          donde depth = Z_cam - Z_tube_world
    Y_cam = (v - cy) / fy * depth

Como la cámara apunta hacia abajo con pitch=90°, los ejes del frame óptico
(convención ROS: Z fwd, X right, Y down) se mapean a world así:
    X_world =  X_cam_offset + Y_cam   (eje Y_cam apunta hacia world -X)
    Y_world = -X_cam                   (eje X_cam apunta hacia world -Y)
    Z_world =  Z_TUBE_WORLD            (constante)

ORIENTACIÓN del tubo tumbado:
    cv2.minAreaRect sobre la máscara devuelve el eje mayor del rectángulo mínimo.
    Ese ángulo (en imagen) se convierte a yaw en world.
    El gripper debe alinearse con ese eje → orientación como quaternion.

NOTA: Z_TUBE_WORLD y la altura del gripper sobre el tubo son parámetros
      configurables para ajustar sin tocar el código.
"""

import math
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray
import cv2


def yaw_to_quaternion(yaw: float):
    """Convierte un ángulo yaw (radianes) a quaternion (x, y, z, w).
    Roll y pitch son 0 — el gripper apunta hacia abajo."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    # roll=0, pitch=0, yaw → quaternion
    # Para el gripper apuntando hacia abajo se añade pitch=-pi/2
    pitch = -math.pi / 2.0
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return (
        sp * cy,   # qx
        sp * sy,   # qy
        cp * sy,   # qz
        cp * cy,   # qw
    )


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('camera_info_topic',
                               '/overhead_camera/overhead_cam/camera_info')
        self.declare_parameter('mask_topic',       '/vision/mask_image')
        self.declare_parameter('detections_topic', '/vision/detections')
        self.declare_parameter('output_topic',     '/pose_target_cuaterniones')

        # Posición de la cámara en world 
        self.declare_parameter('cam_x_world', 0.2)   # m
        self.declare_parameter('cam_y_world', 0.0)   # m
        self.declare_parameter('cam_z_world', 0.75)  # m

        # Altura del tubo sobre el suelo (world Z) y del gripper sobre el tubo
        self.declare_parameter('z_tube_world',    0.03)  # m  (radio del tubo)
        self.declare_parameter('z_gripper_offset', 0.05) # m  separación gripper-tubo

        # Confianza mínima para procesar una detección
        self.declare_parameter('min_score', 0.3)

        self.cam_x   = self.get_parameter('cam_x_world').value
        self.cam_y   = self.get_parameter('cam_y_world').value
        self.cam_z   = self.get_parameter('cam_z_world').value
        self.z_tube  = self.get_parameter('z_tube_world').value
        self.z_grip  = self.get_parameter('z_gripper_offset').value
        self.min_score = self.get_parameter('min_score').value

        # ── Intrínseca (se actualiza al recibir camera_info) ────────────────
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_ready = False

        # ── CV Bridge ───────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Última máscara recibida ──────────────────────────────────────────
        self.last_mask = None

        # ── Suscriptores ────────────────────────────────────────────────────
        cam_info_topic = self.get_parameter('camera_info_topic').value
        mask_topic     = self.get_parameter('mask_topic').value
        det_topic      = self.get_parameter('detections_topic').value

        self.create_subscription(CameraInfo, cam_info_topic,
                                 self._camera_info_cb, 10)
        self.create_subscription(Image, mask_topic,
                                 self._mask_cb, 10)
        self.create_subscription(Detection2DArray, det_topic,
                                 self._detections_cb, 10)

        # ── Publicador ───────────────────────────────────────────────────────
        output_topic = self.get_parameter('output_topic').value
        self.pub_pose = self.create_publisher(Pose, output_topic, 10)

        self.get_logger().info(
            f'PoseEstimatorNode iniciado.\n'
            f'  Cámara world: x={self.cam_x}, y={self.cam_y}, z={self.cam_z}\n'
            f'  Z tubo: {self.z_tube} m   Z gripper offset: {self.z_grip} m'
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

    def _detections_cb(self, msg: Detection2DArray):
        """Procesa las detecciones y publica la pose del tubo."""
        if not self.camera_ready:
            self.get_logger().warn('Intrínseca aún no recibida, ignorando detección.')
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
            orientation_angle = self._orientation_from_mask(self.last_mask)
        else:
            # Fallback: centro del bounding box
            u = best.bbox.center.position.x
            v = best.bbox.center.position.y
            orientation_angle = 0.0
            self.get_logger().warn('Sin máscara, usando centro del bbox.')

        # ── Proyección inversa píxel → world ────────────────────────────────
        x_world, y_world = self._pixel_to_world(u, v)

        # Z del gripper = altura del tubo + offset para no chocar
        z_gripper = self.z_tube + self.z_grip

        # ── Orientación: yaw alineado con el eje mayor del tubo ──────────────
        qx, qy, qz, qw = yaw_to_quaternion(orientation_angle)

        # ── Construir y publicar Pose ─────────────────────────────────────────
        pose = Pose()
        pose.position.x = x_world
        pose.position.y = y_world
        pose.position.z = z_gripper
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        self.pub_pose.publish(pose)
        self.get_logger().info(
            f'Pose publicada → '
            f'x={x_world:.3f}, y={y_world:.3f}, z={z_gripper:.3f}, '
            f'yaw={math.degrees(orientation_angle):.1f}°'
        )

    # ── Utilidades ───────────────────────────────────────────────────────────

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

    def _orientation_from_mask(self, mask: np.ndarray) -> float:
        """
        Calcula el ángulo del eje mayor del tubo (en radianes) a partir de la máscara.

        Usa cv2.minAreaRect sobre los contornos de la máscara.
        El ángulo devuelto es el yaw en world que debe tener el gripper para
        alinearse con el eje longitudinal del tubo.

        cv2.minAreaRect devuelve angle en [-90, 0). Lo convertimos al eje
        mayor (el lado largo del rectángulo) y lo transformamos a world.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        # Contorno más grande (el tubo)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return 0.0

        rect = cv2.minAreaRect(largest)
        # rect = ((cx, cy), (width, height), angle_deg)
        # angle está en [-90, 0)
        # Si width < height el eje mayor es vertical → angle + 90
        # Si width >= height el eje mayor es horizontal → angle
        (_, (w, h), angle_deg) = rect
        if w < h:
            angle_deg += 90.0

        # Convertir a radianes y ajustar al frame world
        # La cámara cenital con pitch=90° y el frame óptico ROS hace que
        # el eje X de imagen apunte a -Y_world y el eje Y apunte a -X_world.
        # Por tanto el ángulo en imagen se mapea directamente como yaw negado.
        angle_rad = math.radians(-angle_deg)
        return angle_rad

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
        depth = self.cam_z - self.z_tube
        x_opt = (u - self.cx) / self.fx * depth
        y_opt = (v - self.cy) / self.fy * depth

        x_world = self.cam_x - y_opt
        y_world = self.cam_y - x_opt
        return x_world, y_world


# ── main ─────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()