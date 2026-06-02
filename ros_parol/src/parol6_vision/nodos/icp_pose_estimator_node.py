#!/usr/bin/env python3
"""
icp_pose_estimator_node.py — Nodo ROS 2 de estimación de pose 6D del tubo
mediante una arquitectura híbrida YOLOv8 + ICP (Open3D).

╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE                                                                  ║
║                                                                            ║
║   1. YOLO detecta el tubo en 2D            → vision_msgs/Detection2DArray   ║
║   2. La caja 2D recorta la nube 3D escena  → solo puntos dentro del bbox    ║
║   3. Limpieza (voxel + outliers + plano + cluster) → puntos del tubo        ║
║   4. ICP alinea la escena con la nube de referencia del .stl ESCALADO       ║
║   5. Se publica la pose 6D como geometry_msgs/PoseStamped                   ║
╚══════════════════════════════════════════════════════════════════════════╝

ENTRADAS
  /vision/detections                       vision_msgs/Detection2DArray
                                            (bbox del tubo — de YOLO)
  <pointcloud_topic>                        sensor_msgs/PointCloud2
        — o, si cloud_source == 'depth' —
  <depth_topic>                             sensor_msgs/Image  (32FC1 / 16UC1)
  <camera_info_topic>                       sensor_msgs/CameraInfo
                                            (intrínseca de ESA cámara)

SALIDAS
  /vision/tube_pose                         geometry_msgs/PoseStamped
                                            (pose 6D del tubo — salida principal)
  /pose_target_cuaterniones                 geometry_msgs/Pose
                                            (compat. con pose_ik_subscriber.py)
  /vision/icp_scene_cloud                   sensor_msgs/PointCloud2  (debug RViz)
  /vision/icp_aligned_model                 sensor_msgs/PointCloud2  (debug RViz)
  TF: <output_frame> -> tube_link_estimated

SERVICIO
  ~/estimate_now   std_srvs/Trigger
      Fuerza una estimación con la última detección + nube recibidas.

──────────────────────────────────────────────────────────────────────────────
COHERENCIA DE FRAMES (importante)
  La nube de puntos, la matriz intrínseca (camera_info) y la caja de YOLO
  deben venir TODAS de la misma cámara. En este repositorio la nube la
  genera `stereo_pointcloud_node.py` en `navcam_left_optical_frame`; por
  tanto YOLO debe ejecutarse sobre la imagen IZQUIERDA de la NavCam
  (/navcam/left/navcam_left/image_raw) y `camera_info_topic` debe apuntar a
  /navcam/left/navcam_left/camera_info.

  En Gazebo las cámaras no tienen distorsión y el par estéreo está alineado,
  así que la matriz K cruda del camera_info coincide con la de la imagen
  rectificada usada para generar la nube. Con una cámara real habría que
  proyectar con la matriz P rectificada.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray

import tf2_ros

from parol6_vision import icp_utils


# QoS para tópicos de sensor de Gazebo (BEST_EFFORT).
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=5,
)


class IcpPoseEstimatorNode(Node):
    """Estima la pose 6D del tubo con YOLO (bbox) + ICP (Open3D)."""

    def __init__(self) -> None:
        super().__init__("icp_pose_estimator_node")

        # ═══════════════ PARÁMETROS ═══════════════════════════════════════
        # ── Tópicos ──────────────────────────────────────────────────────
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("pointcloud_topic", "/navcam/pointcloud")
        self.declare_parameter("depth_topic",
                               "/navcam/left/navcam_left/depth/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/navcam/left/navcam_left/camera_info")
        self.declare_parameter("image_topic",
                               "/navcam/left/navcam_left/image_raw")
        self.declare_parameter("pose_topic",        "/vision/tube_pose")
        self.declare_parameter("legacy_pose_topic", "/pose_target_cuaterniones")
        self.declare_parameter("scene_debug_topic", "/vision/icp_scene_cloud")
        self.declare_parameter("model_debug_topic", "/vision/icp_aligned_model")
        self.declare_parameter("mesh_debug_image_topic", "/vision/icp_mesh_overlay")

        # ── Fuente de la nube: 'pointcloud' (PointCloud2) o 'depth' (Image) ─
        self.declare_parameter("cloud_source", "pointcloud")
        self.declare_parameter("depth_scale", 1.0)   # 1.0=metros, 0.001=mm

        # ── Frame de salida (vacío => frame de la nube; si no, se hace TF) ─
        self.declare_parameter("output_frame", "")
        self.declare_parameter("tf_child_frame", "tube_link_estimated")
        self.declare_parameter("broadcast_tf", True)
        self.declare_parameter("publish_legacy_pose", True)
        self.declare_parameter("publish_debug_clouds", True)

        # ── Modelo .stl ──────────────────────────────────────────────────
        # Vacío => share/parol6_gazebo/meshes/mars2020tube.stl
        self.declare_parameter("stl_path", "")
        self.declare_parameter("model_points", 5000)

        # ── YOLO / recorte ───────────────────────────────────────────────
        self.declare_parameter("min_score", 0.3)
        self.declare_parameter("target_class", "")     # "" => cualquier clase
        self.declare_parameter("bbox_margin_px", 8.0)

        # ── Filtrado de la escena ────────────────────────────────────────
        self.declare_parameter("voxel_size", 0.003)
        self.declare_parameter("sor_neighbors", 20)
        self.declare_parameter("sor_std_ratio", 2.0)
        self.declare_parameter("remove_plane", True)
        self.declare_parameter("plane_dist_thresh", 0.004)
        self.declare_parameter("cluster_eps", 0.012)
        self.declare_parameter("cluster_min_points", 30)

        # ── ICP ──────────────────────────────────────────────────────────
        self.declare_parameter("icp_max_corr_dist", 0.0)   # 0 => 3*voxel
        self.declare_parameter("icp_max_iter", 80)
        self.declare_parameter("icp_point_to_plane", True)
        self.declare_parameter("use_global_registration", False)
        self.declare_parameter("min_fitness", 0.45)        # rechazo de la pose
        self.declare_parameter("max_rmse", 0.006)          # m
        self.declare_parameter("min_cloud_points", 60)

        # ── Frecuencia de procesado ──────────────────────────────────────
        self.declare_parameter("process_rate", 2.0)        # Hz


        # ── Edad máxima de la detección YOLO ─────────────────────────────
        # Si la última detección recibida tiene más de esta edad (segundos),
        # se considera obsoleta y NO se procesa. Esto evita que el ICP
        # publique poses basadas en detecciones antiguas (e.g., del objeto
        # del frame anterior) cuando YOLO ha dejado de publicar.
        self.declare_parameter("detection_max_age_s", 1.0)

        # ── Lectura de parámetros ────────────────────────────────────────
        gp = self.get_parameter
        self.cloud_source   = gp("cloud_source").value
        self.depth_scale    = float(gp("depth_scale").value)
        self.output_frame   = gp("output_frame").value
        self.tf_child       = gp("tf_child_frame").value
        self.broadcast_tf   = bool(gp("broadcast_tf").value)
        self.pub_legacy     = bool(gp("publish_legacy_pose").value)
        self.pub_debug      = bool(gp("publish_debug_clouds").value)
        self.model_points   = int(gp("model_points").value)
        self.min_score      = float(gp("min_score").value)
        self.target_class   = gp("target_class").value
        self.bbox_margin    = float(gp("bbox_margin_px").value)
        self.voxel_size     = float(gp("voxel_size").value)
        self.sor_neighbors  = int(gp("sor_neighbors").value)
        self.sor_std_ratio  = float(gp("sor_std_ratio").value)
        self.remove_plane   = bool(gp("remove_plane").value)
        self.plane_dist     = float(gp("plane_dist_thresh").value)
        self.cluster_eps    = float(gp("cluster_eps").value)
        self.cluster_minpts = int(gp("cluster_min_points").value)
        self.icp_corr       = float(gp("icp_max_corr_dist").value)
        self.icp_iter       = int(gp("icp_max_iter").value)
        self.icp_p2p        = bool(gp("icp_point_to_plane").value)
        self.use_global     = bool(gp("use_global_registration").value)
        self.min_fitness    = float(gp("min_fitness").value)
        self.max_rmse       = float(gp("max_rmse").value)
        self.min_cloud_pts  = int(gp("min_cloud_points").value)
        process_rate        = float(gp("process_rate").value)
        self.det_max_age_s  = float(gp("detection_max_age_s").value)

        # ═══════════════ NUBE DE REFERENCIA (del .stl ESCALADO) ════════════
        stl_path = gp("stl_path").value
        if not stl_path:
            stl_path = os.path.join(
                get_package_share_directory("parol6_gazebo"),
                "meshes", "mars2020tube.stl")
        if not os.path.isfile(stl_path):
            raise FileNotFoundError(
                f"No se encuentra el .stl del tubo en: {stl_path}\n"
                f"Indica la ruta correcta con el parametro 'stl_path'.")

        self.get_logger().info(f"Cargando modelo de referencia: {stl_path}")
        # icp_utils aplica la escala NO UNIFORME obligatoria (0.022939,
        # 0.022939, 0.033212) y recentra la malla en el origen.
        self.model_cloud = icp_utils.load_reference_cloud(
            stl_path, n_points=self.model_points)
        dims = icp_utils.reference_cloud_dimensions(self.model_cloud)
        self.get_logger().info(
            f"Modelo listo: {len(self.model_cloud.points)} puntos, "
            f"extents (X,Y,Z) = {dims.round(4)} m  "
            f"(longitud ~{dims[2]:.3f} m, diametro ~{dims[0]:.4f} m).")

        # ═══════════════ ESTADO ═══════════════════════════════════════════
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._busy = False

        self._K: np.ndarray | None = None
        self._cloud_xyz: np.ndarray | None = None  # nube de escena (N,3)
        self._cloud_frame: str = ""
        self._cloud_stamp = None
        self._last_detection = None
        self._last_detection_time: float | None = None  # monotonic time
        self._last_image: np.ndarray | None = None
        self._last_image_stamp = None

        # ═══════════════ SUSCRIPTORES ═════════════════════════════════════
        self.create_subscription(
            Detection2DArray, gp("detections_topic").value,
            self._detections_cb, 10)
        self.create_subscription(
            CameraInfo, gp("camera_info_topic").value,
            self._camera_info_cb, SENSOR_QOS)
        self.create_subscription(
            Image, gp("image_topic").value,
            self._image_cb, SENSOR_QOS)

        if self.cloud_source == "depth":
            self.create_subscription(
                Image, gp("depth_topic").value, self._depth_cb, SENSOR_QOS)
            self.get_logger().info(
                f"Fuente de nube: DEPTH IMAGE ({gp('depth_topic').value})")
        else:
            self.create_subscription(
                PointCloud2, gp("pointcloud_topic").value,
                self._pointcloud_cb, SENSOR_QOS)
            self.get_logger().info(
                f"Fuente de nube: POINTCLOUD2 ({gp('pointcloud_topic').value})")

        # ═══════════════ PUBLICADORES ═════════════════════════════════════
        self.pub_pose = self.create_publisher(
            PoseStamped, gp("pose_topic").value, 10)
        self.pub_legacy_pose = self.create_publisher(
            Pose, gp("legacy_pose_topic").value, 10)
        self.pub_scene = self.create_publisher(
            PointCloud2, gp("scene_debug_topic").value, 1)
        self.pub_model = self.create_publisher(
            PointCloud2, gp("model_debug_topic").value, 1)
        self.pub_mesh_image = self.create_publisher(
            Image, gp("mesh_debug_image_topic").value, 1)

        # ═══════════════ TF ═══════════════════════════════════════════════
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ═══════════════ SERVICIO + TIMER ═════════════════════════════════
        self.create_service(Trigger, "~/estimate_now", self._estimate_now_cb)

        period = 1.0 / max(process_rate, 0.1)
        self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f"IcpPoseEstimatorNode iniciado. Procesando a {process_rate} Hz.")

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS DE SUSCRIPCIÓN
    # ══════════════════════════════════════════════════════════════════════

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._K is None:
            self.get_logger().info(
                f"Intrinseca recibida: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}")
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        try:
            xyz = icp_utils.pointcloud2_to_xyz(msg)
        except (ValueError, AttributeError) as exc:
            self.get_logger().warn(f"No se pudo parsear PointCloud2: {exc}")
            return
        with self._lock:
            self._cloud_xyz = xyz
            self._cloud_frame = msg.header.frame_id
            self._cloud_stamp = msg.header.stamp

    def _depth_cb(self, msg: Image) -> None:
        if self._K is None:
            return
        try:
            # 32FC1 (m) o 16UC1 (mm) — passthrough conserva el dtype original.
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"CvBridge (depth): {exc}")
            return
        xyz = icp_utils.depth_to_xyz(np.asarray(depth), self._K,
                                     depth_scale=self.depth_scale)
        with self._lock:
            self._cloud_xyz = xyz
            self._cloud_frame = msg.header.frame_id
            self._cloud_stamp = msg.header.stamp

    def _image_cb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"CvBridge (image): {exc}")
            return
        with self._lock:
            self._last_image = img
            self._last_image_stamp = msg.header.stamp

    def _detections_cb(self, msg: Detection2DArray) -> None:
        """Solo CACHEA la detección; el procesado lo dispara el timer."""
        best = self._select_detection(msg)
        if best is not None:
            import time as _time
            with self._lock:
                self._last_detection = best
                self._last_detection_time = _time.monotonic()

    # ══════════════════════════════════════════════════════════════════════
    # DISPARADORES DEL PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    def _timer_cb(self) -> None:
        self._run_pipeline(log_prefix="[timer]")

    def _estimate_now_cb(self, request, response):
        ok, info = self._run_pipeline(log_prefix="[service]")
        response.success = ok
        response.message = info
        return response

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE 
    # ══════════════════════════════════════════════════════════════════════

    def _run_pipeline(self, log_prefix: str = "") -> tuple[bool, str]:
        """Ejecuta el pipeline completo YOLO-bbox → recorte → ICP → pose."""
        # Evita ejecuciones solapadas (el ICP puede tardar > periodo del timer).
        if self._busy:
            return False, "ocupado: estimacion anterior en curso"
        self._busy = True
        try:
            return self._pipeline_impl(log_prefix)
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().error(f"{log_prefix} fallo inesperado: {exc}")
            return False, f"excepcion: {exc}"
        finally:
            self._busy = False

    def _pipeline_impl(self, log_prefix: str) -> tuple[bool, str]:
        # ── 0. Comprobaciones de disponibilidad ─────────────────────────────
        import time as _time
        with self._lock:
            K = None if self._K is None else self._K.copy()
            cloud_frame = self._cloud_frame
            cloud_stamp = self._cloud_stamp
            detection = self._last_detection
            detection_time = self._last_detection_time

        if K is None:
            return False, "sin camera_info todavia"
        if detection is None:
            return False, "sin deteccion YOLO todavia"

        # Rechazar detecciones obsoletas: si YOLO ha dejado de publicar (p.ej.
        # porque el objeto ha desaparecido o se ha respawneado en otra zona),
        # no debemos seguir publicando poses basadas en la última detección
        # válida, que ahora apunta a una región sin el objeto.
        if detection_time is not None:
            age = _time.monotonic() - detection_time
            if age > self.det_max_age_s:
                return False, (f"deteccion YOLO obsoleta ({age:.2f}s > "
                               f"{self.det_max_age_s:.2f}s)")

        bbox_xyxy, score = detection

        # ── Lectura de la nube cacheada ─────────────────────────────────────
        with self._lock:
            xyz = None if self._cloud_xyz is None else self._cloud_xyz.copy()
            cloud_frame = self._cloud_frame
            cloud_stamp = self._cloud_stamp

        if xyz is None or xyz.shape[0] == 0:
            return False, "sin nube de puntos todavia"

        # ── 1. Recorte de la nube con la caja 2D de YOLO ────────────────────
        cropped = icp_utils.crop_cloud_by_bbox(
            xyz, K, bbox_xyxy, margin_px=self.bbox_margin)
        self.get_logger().info(
            f"{log_prefix} bbox YOLO (score={score:.2f}): "
            f"{xyz.shape[0]} pts escena -> {cropped.shape[0]} pts en la caja")

        if cropped.shape[0] < self.min_cloud_pts:
            return False, (f"pocos puntos tras recorte "
                            f"({cropped.shape[0]} < {self.min_cloud_pts})")

        # ── 2. Limpieza / segmentación de la escena ─────────────────────────
        scene_pcd = icp_utils.make_o3d_cloud(cropped)
        scene_pcd = icp_utils.preprocess_scene_cloud(
            scene_pcd,
            voxel_size=self.voxel_size,
            sor_neighbors=self.sor_neighbors,
            sor_std_ratio=self.sor_std_ratio,
            remove_plane=self.remove_plane,
            plane_dist_thresh=self.plane_dist,
            cluster_eps=self.cluster_eps,
            cluster_min_points=self.cluster_minpts,
        )
        n_scene = len(scene_pcd.points)
        if n_scene < self.min_cloud_pts:
            return False, (f"pocos puntos tras limpieza "
                            f"({n_scene} < {self.min_cloud_pts})")

        # ── 3. Registro ICP contra el modelo del .stl escalado ──────────────
        result = icp_utils.register_tube_icp(
            scene_pcd, self.model_cloud,
            voxel_size=self.voxel_size,
            max_corr_dist=(self.icp_corr if self.icp_corr > 0 else None),
            use_point_to_plane=self.icp_p2p,
            max_iter=self.icp_iter,
            use_global_registration=self.use_global,
        )

        T = result["T_cam_obj"]
        fitness = result["fitness"]
        rmse = result["rmse"]

        if T is None:
            return False, "ICP no convergio"

        self.get_logger().info(
            f"{log_prefix} ICP: fitness={fitness:.3f}  "
            f"rmse={rmse * 1000:.2f} mm  (scene={n_scene} pts)")

        # ── 4. Rechazo de estimaciones de baja calidad ──────────────────────
        if fitness < self.min_fitness or rmse > self.max_rmse:
            return False, (f"pose rechazada (fitness={fitness:.3f} "
                            f"< {self.min_fitness} o rmse={rmse*1000:.2f}mm "
                            f"> {self.max_rmse*1000:.1f}mm)")

        # ── 5. Construcción y publicación de la pose ────────────────────────
        self._publish_pose(T, cloud_frame, cloud_stamp)

        # ── 6. Nubes de debug para RViz ─────────────────────────────────────
        if self.pub_debug:
            self._publish_debug_clouds(scene_pcd, T, cloud_frame, cloud_stamp)
            self._publish_mesh_overlay_image(T, bbox_xyxy, cloud_stamp)

        return True, (f"pose publicada (fitness={fitness:.3f}, "
                      f"rmse={rmse*1000:.2f} mm)")

    # ══════════════════════════════════════════════════════════════════════
    # PUBLICACIÓN
    # ══════════════════════════════════════════════════════════════════════

    def _publish_pose(self, T_cam_obj: np.ndarray, cloud_frame: str,
                      stamp) -> None:
        """Construye PoseStamped a partir de la matriz 4x4 y la publica.

        `T_cam_obj` es la pose del tubo en el frame ÓPTICO de la cámara.
        Si `output_frame` está definido, se transforma a ese frame con TF2.
        """
        pos = T_cam_obj[:3, 3]
        qx, qy, qz, qw = icp_utils.matrix_to_quaternion(T_cam_obj[:3, :3])

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp if stamp is not None \
            else self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = cloud_frame
        pose_stamped.pose.position.x = float(pos[0])
        pose_stamped.pose.position.y = float(pos[1])
        pose_stamped.pose.position.z = float(pos[2])
        pose_stamped.pose.orientation.x = float(qx)
        pose_stamped.pose.orientation.y = float(qy)
        pose_stamped.pose.orientation.z = float(qz)
        pose_stamped.pose.orientation.w = float(qw)

        # ── Transformación opcional a otro frame (p.ej. base_link/world) ────
        if self.output_frame and self.output_frame != cloud_frame:
            transformed = self._transform_pose(pose_stamped, self.output_frame)
            if transformed is not None:
                pose_stamped = transformed
            else:
                self.get_logger().warn(
                    f"TF {cloud_frame}->{self.output_frame} no disponible; "
                    f"se publica en {cloud_frame}.")

        # ── Publicar PoseStamped (salida principal) ─────────────────────────
        self.pub_pose.publish(pose_stamped)

        # ── Pose plana para compatibilidad con pose_ik_subscriber.py ────────
        if self.pub_legacy:
            self.pub_legacy_pose.publish(pose_stamped.pose)

        # ── TF para visualizar en RViz ──────────────────────────────────────
        if self.broadcast_tf:
            tf_msg = TransformStamped()
            tf_msg.header = pose_stamped.header
            tf_msg.child_frame_id = self.tf_child
            tf_msg.transform.translation.x = pose_stamped.pose.position.x
            tf_msg.transform.translation.y = pose_stamped.pose.position.y
            tf_msg.transform.translation.z = pose_stamped.pose.position.z
            tf_msg.transform.rotation = pose_stamped.pose.orientation
            self._tf_broadcaster.sendTransform(tf_msg)

        p = pose_stamped.pose.position
        self.get_logger().info(
            f"Pose 6D publicada [{pose_stamped.header.frame_id}]: "
            f"x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}")

    def _transform_pose(self, pose: PoseStamped, target_frame: str):
        """Transforma una PoseStamped a `target_frame` usando TF2.

        Aplica la transformación con álgebra propia (`_manual_transform_pose`)
        en lugar de depender de `tf2_geometry_msgs.do_transform_pose_stamped`,
        cuya firma cambia entre distros de ROS 2 (Humble/Iron/Jazzy).
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, pose.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

        return self._manual_transform_pose(pose, tf, target_frame)

    @staticmethod
    def _manual_transform_pose(pose: PoseStamped, tf: TransformStamped,
                               target_frame: str) -> PoseStamped:
        """Aplica una TransformStamped a una PoseStamped a mano (4x4)."""
        def quat_to_mat(x, y, z, w):
            return np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
        t = tf.transform.translation
        r = tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(r.x, r.y, r.z, r.w)
        T[:3, 3] = [t.x, t.y, t.z]

        P = np.eye(4)
        P[:3, :3] = quat_to_mat(pose.pose.orientation.x, pose.pose.orientation.y,
                                pose.pose.orientation.z, pose.pose.orientation.w)
        P[:3, 3] = [pose.pose.position.x, pose.pose.position.y,
                    pose.pose.position.z]
        M = T @ P

        out = PoseStamped()
        out.header.stamp = pose.header.stamp
        out.header.frame_id = target_frame
        out.pose.position.x, out.pose.position.y, out.pose.position.z = M[:3, 3]
        qx, qy, qz, qw = icp_utils.matrix_to_quaternion(M[:3, :3])
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def _publish_debug_clouds(self, scene_pcd, T_cam_obj, frame, stamp) -> None:
        """Publica la nube de escena y el modelo alineado para inspección RViz."""
        scene_xyz = np.asarray(scene_pcd.points, dtype=np.float32)
        self.pub_scene.publish(self._xyz_to_pc2(scene_xyz, frame, stamp))

        # Modelo transformado a la pose estimada (debe solaparse con la escena).
        model_xyz = np.asarray(self.model_cloud.points, dtype=np.float64)
        model_h = np.column_stack([model_xyz, np.ones(len(model_xyz))])
        model_cam = (T_cam_obj @ model_h.T).T[:, :3].astype(np.float32)
        self.pub_model.publish(self._xyz_to_pc2(model_cam, frame, stamp))

    def _publish_mesh_overlay_image(self, T_cam_obj, bbox_xyxy, stamp) -> None:
        """Proyecta el mesh en la imagen y lo dibuja como wireframe semi-transparente."""
        with self._lock:
            img = None if self._last_image is None else self._last_image.copy()

        if img is None or self._K is None:
            return

        # Proyectar puntos del modelo 3D
        model_xyz = np.asarray(self.model_cloud.points, dtype=np.float64)
        model_h = np.column_stack([model_xyz, np.ones(len(model_xyz))])
        model_cam = (T_cam_obj @ model_h.T).T[:, :3]

        # Proyectar a imagen con la matriz intrínseca K
        model_h_img = (self._K @ model_cam.T).T
        model_2d = (model_h_img[:, :2] / model_h_img[:, 2:3]).astype(np.int32)

        # Filtrar puntos dentro de la imagen
        h, w = img.shape[:2]
        valid = (model_2d[:, 0] >= 0) & (model_2d[:, 0] < w) & \
                (model_2d[:, 1] >= 0) & (model_2d[:, 1] < h)
        model_2d_valid = model_2d[valid]

        # Crear overlay semi-transparente
        overlay = img.copy()

        # Dibujar puntos proyectados con bajo alpha
        for pt in model_2d_valid:
            cv2.circle(overlay, tuple(pt), 1, (0, 255, 255), -1)

        # Dibujar bbox de YOLO en rojo
        x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Blender overlay con imagen original (transparencia)
        alpha = 0.4
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        # Convertir a mensaje ROS y publicar
        try:
            msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
            self.pub_mesh_image.publish(msg)
        except CvBridgeError as exc:
            self.get_logger().warn(f"No se pudo publicar imagen overlay: {exc}")

    @staticmethod
    def _xyz_to_pc2(xyz: np.ndarray, frame: str, stamp) -> PointCloud2:
        """Convierte un array (N,3) float32 a sensor_msgs/PointCloud2."""
        msg = PointCloud2()
        msg.header = Header(frame_id=frame)
        if stamp is not None:
            msg.header.stamp = stamp
        msg.height = 1
        msg.width = int(xyz.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
        msg.is_dense = True
        return msg

    # ══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ══════════════════════════════════════════════════════════════════════

    def _select_detection(self, msg: Detection2DArray):
        """Elige la mejor detección y devuelve (bbox_xyxy, score) o None."""
        if not msg.detections:
            return None

        candidates = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            # Filtro por clase (si se ha configurado una concreta).
            if self.target_class and hyp.class_id != self.target_class:
                continue
            if hyp.score < self.min_score:
                continue
            candidates.append((det, hyp.score))

        if not candidates:
            return None

        det, score = max(candidates, key=lambda c: c[1])

        # vision_msgs/Detection2D: bbox.center (x,y) + size_x/size_y → xyxy.
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        sx = det.bbox.size_x
        sy = det.bbox.size_y
        bbox_xyxy = (cx - sx / 2.0, cy - sy / 2.0,
                     cx + sx / 2.0, cy + sy / 2.0)
        return bbox_xyxy, score


# ════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)
    node = IcpPoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
