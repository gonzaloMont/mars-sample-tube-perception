#!/usr/bin/env python3
"""
linemod_pose_estimator_node.py — Nodo ROS 2 de estimación de pose 6D (LINEMOD).

Este es el nodo de INFERENCIA. Sustituye/complementa a pose_estimator_node.py:
en lugar de una estimación 2.5D (centroide + minAreaRect), recupera la pose
6D completa del tubo mediante template matching LINEMOD.

──────────────────────────────────────────────────────────────────────────────
ENTRADAS
  /overhead_camera/overhead_cam/image_raw      sensor_msgs/Image
  /overhead_camera/overhead_cam/camera_info    sensor_msgs/CameraInfo
  /vision/detections                           vision_msgs/Detection2DArray
                                               (bbox del tubo, de YOLO)

SALIDAS
  /vision/pose_estimates        vision_msgs/Detection3DArray   (pose 6D + score)
  /pose_target_cuaterniones     geometry_msgs/Pose             (compatibilidad
                                con el pipeline existente: pose_ik_subscriber)
  /vision/linemod_debug_image   sensor_msgs/Image              (anotada, debug)
  TF: <base_frame> -> tube_link_estimated                      (para RViz)

SERVICIO
  ~/estimate_now   std_srvs/Trigger
      Fuerza una estimación con la última imagen+detección recibidas y
      publica el resultado. Pensado para el estado SCAN del orquestador:
      el orquestador pide la pose CUANDO la necesita en vez de depender
      del último mensaje publicado.

──────────────────────────────────────────────────────────────────────────────
DISEÑO
  • La base de datos LINEMOD (templates + poses) se genera OFFLINE con
    scripts/train_linemod_templates.py. Este nodo solo la CARGA.
  • LINEMOD se ejecuta sobre el ROI recortado del bounding box de YOLO, no
    sobre la imagen completa: más rápido y menos falsos positivos.
  • La pose se recupera en el frame óptico de la cámara y se transforma a
    `base_link` con tf2. Si tf2 no tiene la transformada, se usa el montaje
    cenital conocido como fallback.
"""

from __future__ import annotations

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import cv2

from parol6_vision.linemod_db import LinemodDatabase
from parol6_vision import pose_solver


class LinemodPoseEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("linemod_pose_estimator_node")

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter("image_topic",
                               "/overhead_camera/overhead_cam/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/overhead_camera/overhead_cam/camera_info")
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("pose_array_topic", "/vision/pose_estimates")
        self.declare_parameter("legacy_pose_topic", "/pose_target_cuaterniones")
        self.declare_parameter("debug_image_topic", "/vision/linemod_debug_image")

        # Base de datos LINEMOD generada offline.
        self.declare_parameter("linemod_db_dir", "")  # vacío -> share/.../models/linemod
        self.declare_parameter("match_threshold", 80.0)   # similitud LINEMOD [0..100]
        self.declare_parameter("min_score", 0.3)          # confianza mínima YOLO
        self.declare_parameter("bbox_padding", 12)        # px de margen al recortar ROI
        self.declare_parameter("min_roi_size", 160)       # px mínimos para evitar segfaults en cv2.linemod

        # Frames y geometría de fallback (montaje cenital conocido).
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_optical_frame", "overhead_camera_optical_frame")
        self.declare_parameter("object_frame", "tube_link_estimated")
        self.declare_parameter("ground_plane_z", 0.03)    # altura del tubo sobre el suelo
        self.declare_parameter("cam_x_world", 0.2)
        self.declare_parameter("cam_y_world", 0.0)
        self.declare_parameter("cam_z_world", 0.75)
        self.declare_parameter("publish_on_detection", True)

        self.match_threshold = float(self.get_parameter("match_threshold").value)
        self.min_score = float(self.get_parameter("min_score").value)
        self.bbox_padding = int(self.get_parameter("bbox_padding").value)
        self.min_roi_size = int(self.get_parameter("min_roi_size").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_optical_frame = str(self.get_parameter("camera_optical_frame").value)
        self.object_frame = str(self.get_parameter("object_frame").value)
        self.ground_plane_z = float(self.get_parameter("ground_plane_z").value)
        self.publish_on_detection = bool(self.get_parameter("publish_on_detection").value)

        # ── Carga de la base de datos LINEMOD ───────────────────────────────
        db_dir = str(self.get_parameter("linemod_db_dir").value)
        if not db_dir:
            from ament_index_python.packages import get_package_share_directory
            db_dir = os.path.join(
                get_package_share_directory("parol6_vision"), "models", "linemod")

        if not os.path.isdir(db_dir):
            self.get_logger().error(
                f"No se encontró la base de datos LINEMOD en '{db_dir}'.\n"
                f"Genérala primero con:\n"
                f"  ros2 run parol6_vision train_linemod_templates.py "
                f"--stl <mars2020tube.stl> --out {db_dir}")
            raise SystemExit(1)

        self.get_logger().info(f"Cargando base de datos LINEMOD: {db_dir}")
        self.db = LinemodDatabase.load(db_dir)
        self.get_logger().info(
            f"  {len(self.db.entries)} templates cargados "
            f"(render_distance={self.db.render_distance:.3f} m).")

        # ── Estado interno ──────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self.latest_detections: Detection2DArray | None = None
        self.camera_K: np.ndarray | None = None
        self.camera_ready = False

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Suscriptores ────────────────────────────────────────────────────
        self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self._image_cb, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._camera_info_cb, 10)
        self.create_subscription(
            Detection2DArray, self.get_parameter("detections_topic").value,
            self._detections_cb, 10)

        # ── Publicadores ────────────────────────────────────────────────────
        # QoS transient_local en la pose: un suscriptor tardío (el orquestador)
        # recibe la última pose estimada sin tener que esperar a una nueva.
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_pose_array = self.create_publisher(
            Detection3DArray, self.get_parameter("pose_array_topic").value, 10)
        self.pub_legacy_pose = self.create_publisher(
            Pose, self.get_parameter("legacy_pose_topic").value, latched)
        self.pub_debug = self.create_publisher(
            Image, self.get_parameter("debug_image_topic").value, 10)

        # ── Servicio de estimación bajo demanda ─────────────────────────────
        self.srv_estimate = self.create_service(
            Trigger, "~/estimate_now", self._estimate_now_cb)

        self.get_logger().info("LinemodPoseEstimatorNode listo.")

    # ════════════════════════════════════════════════════════════════════════
    # Callbacks de suscripción
    # ════════════════════════════════════════════════════════════════════════

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_ready:
            return
        self.camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # La inferencia debe usar las MISMAS intrínsecas con las que se
        # entrenaron los templates: avisamos si difieren notablemente.
        if not np.allclose(self.camera_K, self.db.K, rtol=0.05, atol=2.0):
            self.get_logger().warn(
                "Las intrínsecas de camera_info difieren de las usadas al "
                "generar los templates. El matching puede degradarse; "
                "regenera la base de datos con --camera-info.")
        self.camera_ready = True
        self.get_logger().info(
            f"Intrínsecas recibidas: fx={self.camera_K[0,0]:.1f}, "
            f"fy={self.camera_K[1,1]:.1f}.")

    def _image_cb(self, msg: Image) -> None:
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._latest_image_header = msg.header
        except CvBridgeError as exc:
            self.get_logger().error(f"CvBridge: {exc}")

    def _detections_cb(self, msg: Detection2DArray) -> None:
        self.latest_detections = msg
        if self.publish_on_detection:
            self._run_estimation(trigger_source="detection")

    # ════════════════════════════════════════════════════════════════════════
    # Servicio
    # ════════════════════════════════════════════════════════════════════════

    def _estimate_now_cb(self, request, response):
        """Servicio Trigger: fuerza una estimación con los datos más recientes."""
        ok, info = self._run_estimation(trigger_source="service")
        response.success = ok
        response.message = info
        return response

    # ════════════════════════════════════════════════════════════════════════
    # Núcleo de estimación
    # ════════════════════════════════════════════════════════════════════════

    def _run_estimation(self, trigger_source: str) -> tuple[bool, str]:
        """Ejecuta el pipeline LINEMOD completo y publica la pose 6D."""
        if not self.camera_ready or self.camera_K is None:
            return False, "camera_info aún no recibido"
        if self.latest_image is None:
            return False, "imagen aún no recibida"
        if self.latest_detections is None or not self.latest_detections.detections:
            return False, "sin detecciones YOLO disponibles"

        # ── Seleccionar la mejor detección ──────────────────────────────────
        best_det = max(
            self.latest_detections.detections,
            key=lambda d: d.results[0].hypothesis.score if d.results else 0.0)
        if not best_det.results or best_det.results[0].hypothesis.score < self.min_score:
            return False, "detección por debajo de min_score"

        # ── Recortar ROI a partir del bbox YOLO ─────────────────────────────
        roi, roi_offset = self._crop_roi(self.latest_image, best_det)
        if roi is None:
            return False, "bounding box fuera de la imagen"

        # ── Matching LINEMOD sobre el ROI ───────────────────────────────────
        matches = self.db.match(roi, threshold=self.match_threshold)
        if not matches:
            return False, (f"LINEMOD sin match (umbral={self.match_threshold}); "
                           f"baja match_threshold o densifica los templates")
        best_match = matches[0]

        # ── Transformada base_link <- cámara óptica (tf2 o fallback) ────────
        R_cam_plane, t_cam_plane = self._get_camera_to_base()

        # ── Recuperar la pose 6D ────────────────────────────────────────────
        pose_cam = pose_solver.solve_pose_from_match(
            match=best_match,
            db=self.db,
            roi_offset=roi_offset,
            R_cam_plane=R_cam_plane,
            t_cam_plane=t_cam_plane,
            plane_z=self.ground_plane_z,
        )
        if pose_cam is None:
            return False, "no se pudo recuperar la pose desde el match"

        # ── Pose en frame óptico -> pose en base_link ───────────────────────
        T_cam_obj = pose_cam.as_matrix()
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R_cam_plane
        T_base_cam[:3, 3] = t_cam_plane
        T_base_obj = T_base_cam @ T_cam_obj

        R_base_obj = T_base_obj[:3, :3]
        t_base_obj = T_base_obj[:3, 3]
        qx, qy, qz, qw = pose_solver.rotation_to_quaternion(R_base_obj)

        # ── Publicación ──────────────────────────────────────────────────────
        self._publish_detection3d(t_base_obj, (qx, qy, qz, qw),
                                   pose_cam.similarity, best_det)
        self._publish_legacy_pose(t_base_obj, (qx, qy, qz, qw))
        self._broadcast_tf(t_base_obj, (qx, qy, qz, qw))
        self._publish_debug_image(best_det, best_match, pose_cam)

        info = (f"pose 6D estimada: pos=({t_base_obj[0]:.3f}, "
                f"{t_base_obj[1]:.3f}, {t_base_obj[2]:.3f}) "
                f"sim={pose_cam.similarity:.0f} "
                f"tpl={pose_cam.template_id} src={trigger_source}")
        self.get_logger().info(info)
        return True, info

    # ════════════════════════════════════════════════════════════════════════
    # Utilidades
    # ════════════════════════════════════════════════════════════════════════

    def _crop_roi(self, image: np.ndarray, detection):
        """Recorta el ROI del bbox YOLO con un margen de seguridad."""
        h, w = image.shape[:2]
        cx = detection.bbox.center.position.x
        cy = detection.bbox.center.position.y
        sx = detection.bbox.size_x
        sy = detection.bbox.size_y

        x0 = int(max(0, cx - sx / 2 - self.bbox_padding))
        y0 = int(max(0, cy - sy / 2 - self.bbox_padding))
        x1 = int(min(w, cx + sx / 2 + self.bbox_padding))
        y1 = int(min(h, cy + sy / 2 + self.bbox_padding))

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, (0, 0)

        roi = image[y0:y1, x0:x1].copy()

        # cv2.linemod puede segfault con ROIs demasiado pequeñas.
        # Para mantener la escala aparente, no redimensionamos: centramos el ROI
        # dentro de un lienzo negro de tamaño mínimo seguro.
        roi_h, roi_w = roi.shape[:2]
        target_w = max(roi_w, self.min_roi_size)
        target_h = max(roi_h, self.min_roi_size)
        if target_w != roi_w or target_h != roi_h:
            pad_left = (target_w - roi_w) // 2
            pad_right = target_w - roi_w - pad_left
            pad_top = (target_h - roi_h) // 2
            pad_bottom = target_h - roi_h - pad_top
            roi = cv2.copyMakeBorder(
                roi, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # El ROI devuelto comienza antes por el padding añadido.
            x0 -= pad_left
            y0 -= pad_top

        return roi, (x0, y0)

    def _get_camera_to_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve la rotación/traslación base_link <- cámara óptica.

        Intenta tf2 primero (correcto y general). Si no hay transformada,
        usa el montaje cenital conocido como fallback: cámara en
        (cam_x, cam_y, cam_z) mirando recto hacia abajo.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_optical_frame,
                rclpy.time.Time())
            q = tf.transform.rotation
            t = tf.transform.translation
            R = self._quat_to_rot(q.x, q.y, q.z, q.w)
            return R, np.array([t.x, t.y, t.z], dtype=np.float64)
        except (LookupException, ConnectivityException, ExtrapolationException):
            # Fallback: cámara cenital perfecta. El frame óptico ROS
            # (Z fwd, X right, Y down) apuntando hacia -Z_world equivale a
            # una rotación fija respecto a base_link.
            self.get_logger().warn(
                "tf2 sin transformada base<-cámara; usando montaje cenital "
                "conocido como fallback.", throttle_duration_sec=10.0)
            cam_x = float(self.get_parameter("cam_x_world").value)
            cam_y = float(self.get_parameter("cam_y_world").value)
            cam_z = float(self.get_parameter("cam_z_world").value)
            # Óptico->base para cámara cenital mirando hacia abajo:
            #   X_optical -> -Y_base ,  Y_optical -> -X_base ,  Z_optical -> -Z_base
            R = np.array([[0.0, -1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0]], dtype=np.float64)
            t = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
            return R, t

    @staticmethod
    def _quat_to_rot(x, y, z, w) -> np.ndarray:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ], dtype=np.float64)

    # ── Publicadores ─────────────────────────────────────────────────────────

    def _publish_detection3d(self, t, quat, similarity, det2d) -> None:
        arr = Detection3DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = self.base_frame

        d3 = Detection3D()
        d3.header = arr.header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "tube"
        # Normalizamos la similitud LINEMOD [0..100] a un score [0..1].
        hyp.hypothesis.score = float(similarity) / 100.0
        hyp.pose.pose.position.x = float(t[0])
        hyp.pose.pose.position.y = float(t[1])
        hyp.pose.pose.position.z = float(t[2])
        hyp.pose.pose.orientation.x = quat[0]
        hyp.pose.pose.orientation.y = quat[1]
        hyp.pose.pose.orientation.z = quat[2]
        hyp.pose.pose.orientation.w = quat[3]
        d3.results.append(hyp)
        d3.bbox.center.position.x = float(t[0])
        d3.bbox.center.position.y = float(t[1])
        d3.bbox.center.position.z = float(t[2])
        arr.detections.append(d3)
        self.pub_pose_array.publish(arr)

    def _publish_legacy_pose(self, t, quat) -> None:
        """Republica en /pose_target_cuaterniones para no romper el pipeline."""
        pose = Pose()
        pose.position.x = float(t[0])
        pose.position.y = float(t[1])
        pose.position.z = float(t[2])
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        self.pub_legacy_pose.publish(pose)

    def _broadcast_tf(self, t, quat) -> None:
        from geometry_msgs.msg import TransformStamped
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.base_frame
        tf.child_frame_id = self.object_frame
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = quat[0]
        tf.transform.rotation.y = quat[1]
        tf.transform.rotation.z = quat[2]
        tf.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(tf)

    def _publish_debug_image(self, det2d, match, pose_cam) -> None:
        if self.pub_debug.get_subscription_count() == 0:
            return
        vis = self.latest_image.copy()
        u, v = pose_cam.matched_uv
        cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0),
                       cv2.MARKER_CROSS, 22, 2)
        cv2.putText(vis, f"sim={pose_cam.similarity:.0f} tpl={pose_cam.template_id}",
                    (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        except CvBridgeError as exc:
            self.get_logger().error(f"debug image: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LinemodPoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
