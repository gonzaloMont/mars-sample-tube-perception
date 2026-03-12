#!/usr/bin/env python3
"""
yolo_detector_node.py

Suscribe a la imagen de la cámara cenital, ejecuta YOLOE y publica:
  - /vision/detections       (vision_msgs/Detection2DArray)  bboxes + clase + score
  - /vision/annotated_image  (sensor_msgs/Image)             imagen anotada para debug
  - /vision/mask_image       (sensor_msgs/Image)             máscara binaria para pose_estimator
"""

import os 
import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs import msg
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLOE
from vision_msgs import msg
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # ── Parámetros ──────────────────────────────────────────────────────
        self.declare_parameter('camera_topic',   '/overhead_camera/overhead_cam/image_raw')
        self.declare_parameter('conf_threshold', 0.2)
        self.declare_parameter('model_path',     'yoloe-26s-seg.pt')
        self.declare_parameter('target_classes', ['red tube'])

        self.camera_topic   = self.get_parameter('camera_topic').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.model_path     = self.get_parameter('model_path').value
        self.target_classes = list(self.get_parameter('target_classes').value)

        # ── Ruta al modelo ───────────────────────────────────────────────────
        models_dir = os.path.join(
            get_package_share_directory('parol6_vision'), 'models'
        )
        model_file = self.get_parameter('model_path').value  # solo el nombre
        self.model_path = os.path.join(models_dir, model_file)

        if not os.path.isfile(self.model_path):
            self.get_logger().warn(
                f'Modelo no encontrado en {self.model_path}. '
                f'Ultralytics intentará descargarlo.'
            )
        else:
            self.get_logger().info(f'Modelo cargado desde: {self.model_path}')


        # ── Modelo ──────────────────────────────────────────────────────────
        # Cambiamos temporalmente el cwd a models/ para que YOLOE encuentre
        # MobileClip  sin descargarlo.
        self.bridge = CvBridge()
        _prev_cwd = os.getcwd()
        os.chdir(models_dir)
        try:
            self.model = YOLOE(self.model_path)
            self.model.set_classes(self.target_classes)
            self.get_logger().info(f'Clases objetivo: {self.target_classes}')
        except ModuleNotFoundError as e:
            self.get_logger().error('Falta la dependencia "clip" para set_classes().')
            raise e
        finally:
            os.chdir(_prev_cwd)  # restaura el cwd original
        
            

        # ── Suscriptor ──────────────────────────────────────────────────────
        self.subscription_camera = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10,
        )

        self.subscription_activate = self.create_subscription(
            String,
            '/vision/activate',
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
        self.pub_mask = self.create_publisher(
            Image,
            '/vision/mask_image',
            10,
        )

        # ── Estado de activación ────────────────────────────────────────
        self.is_active = False

        self.get_logger().info(
            f'YoloDetectorNode iniciado. Suscrito a {self.camera_topic}.'
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
        # Solo procesar si está activado
        if not self.is_active:
            return

        # Convertir imagen ROS → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error CvBridge: {e}')
            return

        # Inferencia YOLO
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )
        result = results[0]

        # ── 1. Detection2DArray ──────────────────────────────────────────────
        det_array = Detection2DArray()
        det_array.header = msg.header  # mismo timestamp y frame_id que la imagen

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                det = Detection2D()
                det.header = msg.header

                # Clase y confianza
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = self.target_classes[int(box.cls.item())]
                hyp.hypothesis.score    = float(box.conf.item())
                det.results.append(hyp)

                # Bounding box en píxeles (xywh = centro x, centro y, ancho, alto)
                xywh = box.xywh[0].tolist()
                det.bbox.center.position.x = xywh[0]
                det.bbox.center.position.y = xywh[1]
                det.bbox.size_x            = xywh[2]
                det.bbox.size_y            = xywh[3]

                det_array.detections.append(det)

        self.pub_detections.publish(det_array)

        # ── 2. Imagen anotada (debug) ────────────────────────────────────────
        annotated = result.plot()
        try:
            self.pub_annotated.publish(
                self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            )
        except CvBridgeError as e:
            self.get_logger().error(f'Error publicando imagen anotada: {e}')

        # ── 3. Máscara binaria (para pose_estimator) ─────────────────────────
        # Si YOLOE devuelve máscaras de segmentación, combinamos todas en una
        # imagen binaria del tamaño del frame original.
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if result.masks is not None:
            for mask_tensor in result.masks.data:
                # mask_tensor es un tensor HxW con valores 0-1, en el tamaño
                # de inferencia — resize al tamaño original del frame
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (w, h), interpolation=cv2.INTER_NEAREST
                )
                combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8) * 255)

        try:
            self.pub_mask.publish(
                self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
            )
        except CvBridgeError as e:
            self.get_logger().error(f'Error publicando máscara: {e}')



# ────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()