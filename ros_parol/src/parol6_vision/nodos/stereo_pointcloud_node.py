#!/usr/bin/env python3
"""
stereo_pointcloud_node.py — Nodo ROS 2 que genera una nube de puntos 3D densa
a partir del par estéreo NavCam usando Semi-Global Block Matching (SGBM).

──────────────────────────────────────────────────────────────────────────────
ENTRADAS (sincronizadas con ApproximateTimeSynchronizer)
  /navcam/left/navcam_left/image_raw          sensor_msgs/Image
  /navcam/left/navcam_left/camera_info        sensor_msgs/CameraInfo
  /navcam/right/navcam_right/image_raw        sensor_msgs/Image
  /navcam/right/navcam_right/camera_info      sensor_msgs/CameraInfo

SALIDA
  /navcam/pointcloud                          sensor_msgs/PointCloud2
      Frame: navcam_left_optical_frame (XYZ float32 unordered)

DISEÑO
  • Primera llamada válida: inicializa la calibración estéreo desde los
    mensajes CameraInfo y el TF2 entre los marcos ópticos (lookup estático).
  • cv2.stereoRectify genera los mapas de rectificación y la matriz Q de
    reproyección, válidos para cualquier orientación de baseline.
  • cv2.StereoSGBM (modo SGBM_3WAY) calcula la disparidad en las imágenes
    rectificadas.
  • Filtro WLS opcional (opencv-contrib, ya instalado para LINEMOD) suaviza
    artefactos en bordes sin degradar la precisión métrica.
  • cv2.reprojectImageTo3D + masking NumPy vectorizado convierte la disparidad
    en puntos 3D sin bucles Python; los datos se pasan directamente a
    PointCloud2 mediante .tobytes() (zero-copy desde NumPy).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
import rclpy.time
import rclpy.duration

from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
import tf2_ros
from geometry_msgs.msg import TransformStamped


class StereoPointCloudNode(Node):
    """Genera una nube de puntos 3D densa desde la NavCam estéreo con SGBM."""

    def __init__(self) -> None:
        super().__init__('stereo_pointcloud')

        # ── Parámetros ───────────────────────────────────────────────────────
        self.declare_parameter('left_image_topic',
                               '/navcam/left/navcam_left/image_raw')
        self.declare_parameter('right_image_topic',
                               '/navcam/right/navcam_right/image_raw')
        self.declare_parameter('left_info_topic',
                               '/navcam/left/navcam_left/camera_info')
        self.declare_parameter('right_info_topic',
                               '/navcam/right/navcam_right/camera_info')
        self.declare_parameter('output_topic',   '/navcam/pointcloud')
        self.declare_parameter('output_frame',   'navcam_left_optical_frame')
        self.declare_parameter('left_frame_id',  'navcam_left_optical_frame')
        self.declare_parameter('right_frame_id', 'navcam_right_optical_frame')

        # Parámetros SGBM
        self.declare_parameter('min_disparity',    0)
        self.declare_parameter('num_disparities',  96)   # múltiplo de 16
        self.declare_parameter('block_size',       7)     # impar, 3–11
        self.declare_parameter('uniqueness_ratio', 5)
        self.declare_parameter('speckle_window',   100)
        self.declare_parameter('speckle_range',    2)
        self.declare_parameter('disp12_max_diff',  1)

        # Filtros de profundidad
        self.declare_parameter('min_depth',    0.30)   # metros
        self.declare_parameter('max_depth',    1.0)   # metros
        self.declare_parameter('scale_factor', 0.5)    # factor de reducción

        # Filtro WLS post-procesado (requiere opencv-contrib)
        self.declare_parameter('use_wls_filter', True)
        self.declare_parameter('wls_lambda',     8000.0)
        self.declare_parameter('wls_sigma',      1.5)

        # Tolerancia de sincronización temporal
        self.declare_parameter('sync_slop', 0.15)

        # ── QoS ─────────────────────────────────────────────────────────────
        # Gazebo libgazebo_ros_camera.so publica con RELIABLE. Usamos RELIABLE
        # en los subscribers para garantizar la entrega y que ApproximateTime
        # Synchronizer reciba todos los mensajes sin pérdidas silenciosas.
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Suscriptores sincronizados ───────────────────────────────────────
        left_img_t   = self.get_parameter('left_image_topic').value
        right_img_t  = self.get_parameter('right_image_topic').value
        left_info_t  = self.get_parameter('left_info_topic').value
        right_info_t = self.get_parameter('right_info_topic').value

        self._sub_li = Subscriber(self, Image,      left_img_t,   qos_profile=reliable_qos)
        self._sub_ri = Subscriber(self, Image,      right_img_t,  qos_profile=reliable_qos)
        self._sub_lc = Subscriber(self, CameraInfo, left_info_t,  qos_profile=reliable_qos)
        self._sub_rc = Subscriber(self, CameraInfo, right_info_t, qos_profile=reliable_qos)

        slop = self.get_parameter('sync_slop').value
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_li, self._sub_ri, self._sub_lc, self._sub_rc],
            queue_size=10, slop=slop,
        )
        self._sync.registerCallback(self._stereo_cb)

        # ── Publishers ───────────────────────────────────────────────────────
        # RELIABLE para que RViz2 (que suscribe con RELIABLE por defecto)
        # pueda conectarse sin incompatibilidad de QoS.
        self._pc_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('output_topic').value,
            reliable_qos,
        )
        # Topic de debug: imagen de disparidad normalizada (visible en RViz/rqt)
        self._disp_pub = self.create_publisher(Image, '/navcam/disparity', reliable_qos)

        # ── TF2 ─────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lst = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Utilidades ───────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Estado de calibración (inicializado una sola vez) ────────────────
        self._calib_ok   = False
        self._map1l      = None   # mapa rectificación izquierda
        self._map2l      = None
        self._map1r      = None   # mapa rectificación derecha
        self._map2r      = None
        self._Q          = None   # matriz de reproyección
        self._scale      = self.get_parameter('scale_factor').value
        self._img_size   = None

        # ── Matcher SGBM ─────────────────────────────────────────────────────
        blk  = self.get_parameter('block_size').value
        P1   = 8  * blk * blk    # penalización suave (1 canal grayscale)
        P2   = 32 * blk * blk    # penalización fuerte

        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=self.get_parameter('min_disparity').value,
            numDisparities=self.get_parameter('num_disparities').value,
            blockSize=blk,
            P1=P1,
            P2=P2,
            disp12MaxDiff=self.get_parameter('disp12_max_diff').value,
            uniquenessRatio=self.get_parameter('uniqueness_ratio').value,
            speckleWindowSize=self.get_parameter('speckle_window').value,
            speckleRange=self.get_parameter('speckle_range').value,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # ── Filtro WLS (opcional, opencv-contrib) ─────────────────────────────
        self._wls     = None
        self._sgbm_r  = None

        if self.get_parameter('use_wls_filter').value:
            try:
                self._sgbm_r = cv2.ximgproc.createRightMatcher(self._sgbm)
                self._wls    = cv2.ximgproc.createDisparityWLSFilter(self._sgbm)
                self._wls.setLambda(self.get_parameter('wls_lambda').value)
                self._wls.setSigmaColor(self.get_parameter('wls_sigma').value)
                self.get_logger().info('WLS disparity filter habilitado.')
            except AttributeError:
                self._wls = None
                self.get_logger().warn(
                    'cv2.ximgproc no disponible; WLS deshabilitado. '
                    'Instala opencv-contrib-python para activarlo.'
                )

        self.get_logger().info(
            f'StereoPointCloudNode iniciado.\n'
            f'  Izquierda : {left_img_t}\n'
            f'  Derecha   : {right_img_t}\n'
            f'  Salida    : {self.get_parameter("output_topic").value}\n'
            f'  Scale     : {self._scale}  NumDisp: {self.get_parameter("num_disparities").value}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Inicialización de calibración estéreo (se ejecuta una sola vez)
    # ──────────────────────────────────────────────────────────────────────────

    def _init_calib(self, li: CameraInfo, ri: CameraInfo) -> bool:
        """
        Calcula los mapas de rectificación a partir de CameraInfo + TF2.
        Devuelve True si la inicialización fue exitosa.
        """
        scale    = self._scale
        img_size = (int(li.width * scale), int(li.height * scale))

        # Matrices intrínsecas
        K1 = np.array(li.k, dtype=np.float64).reshape(3, 3)
        K2 = np.array(ri.k, dtype=np.float64).reshape(3, 3)
        D1 = np.array(li.d, dtype=np.float64)
        D2 = np.array(ri.d, dtype=np.float64)

        # Escalar K si reducimos resolución
        if scale != 1.0:
            K1[:2] *= scale
            K2[:2] *= scale

        # Gazebo ideal → sin distorsión, pero nos aseguramos de que el vector
        # tenga al menos 4 coeficientes para que OpenCV no proteste.
        if D1.size < 4:
            D1 = np.zeros(5, dtype=np.float64)
        if D2.size < 4:
            D2 = np.zeros(5, dtype=np.float64)

        # Extrínseca estéreo vía TF2.
        # lookup(target=left, source=right) devuelve la transformada que lleva
        # un punto del frame RIGHT al frame LEFT:
        #   P_left = R_lr · P_right + T_lr   →  T_lr = [+0.15, 0, 0]
        #
        # OpenCV stereoRectify espera R, T con la convención OPUESTA:
        #   P_right = R · P_left + T
        # Es decir, necesita la inversa de la transformada:
        #   R = R_lr^T
        #   T = −R_lr^T · T_lr  = [−0.15, 0, 0]
        # Esto da Q[3,2] = −1/Tx = +6.667 y Z > 0 para disparidad > 0.
        left_fr  = self.get_parameter('left_frame_id').value
        right_fr = self.get_parameter('right_frame_id').value
        try:
            tf_msg: TransformStamped = self._tf_buf.lookup_transform(
                target_frame=left_fr,
                source_frame=right_fr,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF2 lookup {right_fr}→{left_fr} fallido: {exc}. Reintentando…'
            )
            return False

        t    = tf_msg.transform.translation
        q    = tf_msg.transform.rotation
        T_lr = np.array([[t.x], [t.y], [t.z]], dtype=np.float64)
        R_lr = self._quat_to_mat(q.x, q.y, q.z, q.w)

        # Invertir para obtener la convención de OpenCV (left→right)
        R = R_lr.T
        T = -R_lr.T @ T_lr

        self.get_logger().info(
            f'Extrínseca estéreo  T={T.ravel().round(4)}  '
            f'R_diag={np.diag(R).round(4)}'
        )

        # Rectificación: alpha=0.0 recorta solo a píxeles válidos, evita bordes
        R1, R2, P1_m, P2_m, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2,
            img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        self._Q = Q

        self._map1l, self._map2l = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1_m, img_size, cv2.CV_32FC1)
        self._map1r, self._map2r = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2_m, img_size, cv2.CV_32FC1)

        self._img_size = img_size
        self._calib_ok = True
        self.get_logger().info(
            f'Calibración estéreo inicializada. '
            f'Imagen rectificada: {img_size}  Q[3,2]={Q[3,2]:.6f}'
        )
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Callback principal
    # ──────────────────────────────────────────────────────────────────────────

    def _stereo_cb(
        self,
        left_img:  Image,
        right_img: Image,
        left_info: CameraInfo,
        right_info: CameraInfo,
    ) -> None:

        if not self._calib_ok:
            if not self._init_calib(left_info, right_info):
                return

        # ── Conversión ROS → OpenCV ──────────────────────────────────────────
        try:
            left_cv  = self._bridge.imgmsg_to_cv2(left_img,  'mono8')
            right_cv = self._bridge.imgmsg_to_cv2(right_img, 'mono8')
        except CvBridgeError as exc:
            self.get_logger().warn(f'CvBridge error: {exc}')
            return

        # ── Reducción de escala (mejora velocidad ~4x con scale=0.5) ─────────
        if self._scale != 1.0:
            left_cv  = cv2.resize(left_cv,  self._img_size, interpolation=cv2.INTER_AREA)
            right_cv = cv2.resize(right_cv, self._img_size, interpolation=cv2.INTER_AREA)

        # ── Rectificación ────────────────────────────────────────────────────
        left_rect  = cv2.remap(left_cv,  self._map1l, self._map2l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_cv, self._map1r, self._map2r, cv2.INTER_LINEAR)

        # ── Cómputo de disparidad ────────────────────────────────────────────
        if self._wls is not None:
            disp_l = self._sgbm.compute(left_rect,  right_rect)
            disp_r = self._sgbm_r.compute(right_rect, left_rect)
            disp_f = self._wls.filter(disp_l, left_rect, None, disp_r)
            disp   = disp_f.astype(np.float32) / 16.0
        else:
            disp = self._sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0

        # ── Publicar mapa de disparidad normalizado (debug) ──────────────────
        if self._disp_pub.get_subscription_count() > 0:
            min_d_val = float(self.get_parameter('min_disparity').value)
            valid_disp = disp[disp > min_d_val + 1.0]
            if valid_disp.size > 0:
                d_min, d_max = float(valid_disp.min()), float(valid_disp.max())
                norm = np.zeros_like(disp, dtype=np.uint8)
                if d_max > d_min:
                    mask = disp > min_d_val + 1.0
                    norm[mask] = (
                        (disp[mask] - d_min) / (d_max - d_min) * 255
                    ).astype(np.uint8)
                disp_msg = self._bridge.cv2_to_imgmsg(norm, encoding='mono8')
                disp_msg.header = Header(
                    stamp=left_img.header.stamp,
                    frame_id=self.get_parameter('output_frame').value,
                )
                self._disp_pub.publish(disp_msg)

        # ── Reproyección 3D (operación vectorizada) ──────────────────────────
        pts3d = cv2.reprojectImageTo3D(disp, self._Q)    # (H, W, 3)

        min_d = float(self.get_parameter('min_disparity').value)
        z     = pts3d[:, :, 2]

        valid = (
            (disp > min_d + 1.0) &
            np.isfinite(z) &
            (z > self.get_parameter('min_depth').value) &
            (z < self.get_parameter('max_depth').value)
        )

        n_valid = int(valid.sum())
        self.get_logger().debug(
            f'Puntos válidos: {n_valid} / {disp.size}  '
            f'disp_rng=[{disp.min():.1f}, {disp.max():.1f}]'
        )

        pts = pts3d[valid].astype(np.float32)   # (N, 3)
        if pts.size == 0:
            self.get_logger().warn(
                'SGBM no generó puntos válidos. '
                'Verifica que la escena tenga textura suficiente, '
                'o reduce num_disparities / ajusta min_depth.',
                throttle_duration_sec=5.0,
            )
            return

        pc_msg = self._make_pointcloud2(
            pts,
            left_img.header.stamp,
            self.get_parameter('output_frame').value,
        )
        self._pc_pub.publish(pc_msg)
        self.get_logger().debug(f'PointCloud2 publicado: {len(pts)} pts')

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convierte cuaternión unitario a matriz de rotación 3×3."""
        return np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

    @staticmethod
    def _make_pointcloud2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        """
        Convierte array NumPy (N×3, float32) a sensor_msgs/PointCloud2.
        Usa .tobytes() para evitar copias: zero-copy desde el buffer NumPy.
        """
        msg              = PointCloud2()
        msg.header       = Header(stamp=stamp, frame_id=frame_id)
        msg.height       = 1
        msg.width        = len(pts)
        msg.fields       = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12            # 3 × 4 bytes
        msg.row_step     = 12 * len(pts)
        msg.data         = pts.tobytes()
        msg.is_dense     = True
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StereoPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
