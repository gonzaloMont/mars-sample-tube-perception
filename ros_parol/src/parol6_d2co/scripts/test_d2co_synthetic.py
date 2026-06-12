#!/usr/bin/env python3
"""Test sintético end-to-end del nodo D2CO sin Gazebo.

Publica CameraInfo (NavCam), una imagen mono8 con un rectángulo oscuro que
emula el tubo (0.186 m a z=0.7 m, horizontal, centrado) y la detección YOLO
correspondiente. Escucha /vision/tube_pose y compara con la verdad-terreno.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

W = H = 1024
FX = FY = 803.6
CX = CY = 512.0
Z_TRUE = 0.70
TUBE_L, TUBE_D = 0.186, 0.0208
FRAME = "navcam_left_optical_frame"


class SyntheticFeeder(Node):
    def __init__(self):
        super().__init__("d2co_synthetic_feeder")
        qos = QoSProfile(depth=5,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_info = self.create_publisher(
            CameraInfo, "/navcam/left/navcam_left/camera_info", qos)
        self.pub_img = self.create_publisher(
            Image, "/navcam/left/navcam_left/image_raw", qos)
        self.pub_det = self.create_publisher(
            Detection2DArray, "/vision/detections", 10)
        self.sub_pose = self.create_subscription(
            PoseStamped, "/vision/tube_pose", self.pose_cb, 10)

        # Imagen: fondo claro con ruido suave + "tubo" oscuro horizontal.
        len_px = FX * TUBE_L / Z_TRUE     # ~213 px
        dia_px = FX * TUBE_D / Z_TRUE     # ~24 px
        rng = np.random.default_rng(7)
        img = np.full((H, W), 170, np.uint8)
        img = (img + rng.normal(0, 3, (H, W))).clip(0, 255).astype(np.uint8)
        x0 = int(CX - len_px / 2); x1 = int(CX + len_px / 2)
        y0 = int(CY - dia_px / 2); y1 = int(CY + dia_px / 2)
        img[y0:y1, x0:x1] = 60
        self.img = img
        self.bbox = (CX, CY, len_px + 16, dia_px + 16)
        self.len_px, self.dia_px = len_px, dia_px

        self.best = None
        self.n_poses = 0
        self.timer = self.create_timer(0.2, self.tick)

    def tick(self):
        stamp = self.get_clock().now().to_msg()

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = FRAME
        info.width, info.height = W, H
        info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        info.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME
        msg.height, msg.width = H, W
        msg.encoding = "mono8"
        msg.step = W
        msg.data = self.img.tobytes()
        self.pub_img.publish(msg)

        det_arr = Detection2DArray()
        det_arr.header.stamp = stamp
        det_arr.header.frame_id = FRAME
        det = Detection2D()
        det.header = det_arr.header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "tube"
        hyp.hypothesis.score = 0.95
        det.results.append(hyp)
        cx, cy, sx, sy = self.bbox
        det.bbox.center.position.x = cx
        det.bbox.center.position.y = cy
        det.bbox.size_x = sx
        det.bbox.size_y = sy
        det_arr.detections.append(det)
        self.pub_det.publish(det_arr)

    def pose_cb(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        # Eje del tubo (Z del modelo) en frame de cámara
        zx = 2.0 * (q.x * q.z + q.w * q.y)
        zy = 2.0 * (q.y * q.z - q.w * q.x)
        zz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        ang_x = math.degrees(math.acos(min(1, abs(zx))))  # vs eje X cámara
        self.n_poses += 1
        err = math.sqrt(p.x ** 2 + p.y ** 2 + (p.z - Z_TRUE) ** 2)
        print(f"POSE[{self.n_poses}] frame={msg.header.frame_id} "
              f"x={p.x:+.3f} y={p.y:+.3f} z={p.z:.3f} | err3D={err*1000:.1f}mm "
              f"| eje vs X-cam: {ang_x:.1f} deg (eje=({zx:+.2f},{zy:+.2f},{zz:+.2f}))",
              flush=True)


def main():
    rclpy.init()
    node = SyntheticFeeder()
    t_end = time.time() + 25.0
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f"RESULTADO: {node.n_poses} poses recibidas "
          f"(tubo sintetico: centro=(0,0,{Z_TRUE}), eje=X cam, "
          f"{node.len_px:.0f}x{node.dia_px:.0f} px)", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
