#!/usr/bin/env python3
"""
yolo26n_camera_subscriber.py

Suscribe a la imagen de la cámara cenital, ejecuta YOLO de detección normal y publica:
  - /vision/detections       (vision_msgs/Detection2DArray)  bboxes + clase + score
  - /vision/annotated_image  (sensor_msgs/Image)             imagen anotada para debug

Este nodo es equivalente a yolo_camera_subscriber.py, pero usa el modelo best.pt
como detector YOLO normal y no publica máscara.
"""

import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class Yolo26nDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo26n_detector_node')

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('model_path', 'best.pt')
        # Tópico de la imagen de entrada (configurable para soportar tanto la
        # cámara cenital como la NavCam estéreo del pipeline ICP).
        self.declare_parameter('image_topic',
                               '/navcam/left/navcam_left/image_raw')
        # Optimización: procesar solo 1 de cada N frames
        self.declare_parameter('skip_frames', 10)
        # Publicar imagen anotada de debug (desactivar en producción)
        self.declare_parameter('publish_debug_image', False)

        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.model_path = str(self.get_parameter('model_path').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.skip_frames = int(self.get_parameter('skip_frames').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.frame_count = 0  # Contador para saltar frames

        # ── Ruta al modelo ───────────────────────────────────────────────────
        models_dir = os.path.join(get_package_share_directory('parol6_vision'), 'models')
        candidate_path = self.model_path if os.path.isfile(self.model_path) else os.path.join(models_dir, self.model_path)
        self.model_path = candidate_path

        if not os.path.isfile(self.model_path):
            self.get_logger().warn(
                f'Modelo no encontrado en {self.model_path}. '
                f'Ultralytics intentará descargarlo o fallará al cargar.'
            )
        else:
            self.get_logger().info(f'Modelo cargado desde: {self.model_path}')

        # ── Modelo ──────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        _prev_cwd = os.getcwd()
        os.chdir(models_dir)
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info('Modelo YOLO de detección cargado correctamente.')
        except ModuleNotFoundError as e:
            self.get_logger().error('Falta una dependencia requerida por ultralytics/YOLO.')
            raise e
        finally:
            os.chdir(_prev_cwd)

        # ── Suscriptor ──────────────────────────────────────────────────────
        self.subscription_camera = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.subscription_activate = self.create_subscription(
            String,
            'orchestrator/vision_activate',
            self.activate_callback,
            10,
        )

        # ── Publicadores ────────────────────────────────────────────────────
        self.pub_detections = self.create_publisher(
            Detection2DArray,
            '/vision/detections',
            10,
        )
        self.pub_annotated = self.create_publisher(
            Image,
            '/vision/annotated_image',
            10,
        )

        # ── Estado de activación ────────────────────────────────────────────
        self.is_active = False

        self.get_logger().info(
            f'Yolo26nDetectorNode iniciado. Suscrito a {self.image_topic}. '
            f'Procesando 1 de cada {self.skip_frames} frames. '
            f'Debug image: {"ON" if self.publish_debug_image else "OFF"}.'
        )

    # ────────────────────────────────────────────────────────────────────────

    def activate_callback(self, msg: String):
        """Activa o desactiva el procesamiento de imágenes."""
        if msg.data == 'activate':
            self.is_active = True
            self.get_logger().info('Procesamiento de imágenes ACTIVADO.')
        elif msg.data == 'deactivate':
            self.is_active = False
            self.get_logger().info('Procesamiento de imágenes DESACTIVADO.')

    def image_callback(self, msg: Image) -> None:
        if not self.is_active:
            return

        # ── Optimización: procesar solo 1 de cada N frames ──────────────────
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return  # Saltar este frame

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error CvBridge: {e}')
            return

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f'Error ejecutando YOLO: {e}')
            return

        result = results[0]

        det_array = Detection2DArray()
        det_array.header = msg.header

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                det = Detection2D()
                det.header = msg.header

                class_idx = int(box.cls.item())
                if isinstance(result.names, dict):
                    class_name = result.names.get(class_idx, str(class_idx))
                else:
                    class_name = result.names[class_idx] if class_idx < len(result.names) else str(class_idx)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = class_name
                hyp.hypothesis.score = float(box.conf.item())
                det.results.append(hyp)

                xywh = box.xywh[0].tolist()
                det.bbox.center.position.x = float(xywh[0])
                det.bbox.center.position.y = float(xywh[1])
                det.bbox.size_x = float(xywh[2])
                det.bbox.size_y = float(xywh[3])

                det_array.detections.append(det)

        self.pub_detections.publish(det_array)

        # ── Publicar imagen anotada solo si está configurado (debug) ───────
        if self.publish_debug_image:
            annotated = result.plot()
            try:
                self.pub_annotated.publish(
                    self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                )
            except CvBridgeError as e:
                self.get_logger().error(f'Error publicando imagen anotada: {e}')


# ────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Yolo26nDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
