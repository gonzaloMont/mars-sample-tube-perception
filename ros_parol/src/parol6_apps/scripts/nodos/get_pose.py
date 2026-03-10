#!/usr/bin/env python3
''' 
Suscribirse a TF para obtener la pose actual del efector final (L6_link) respecto a base_link, 
e imprimirla por consola.'''

from tf2_ros import Buffer, TransformListener
import rclpy
from rclpy.node import Node
import time 

class EEPositionListener(Node):
    def __init__(self):
        super().__init__('ee_position_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', 'L6_link', rclpy.time.Time())
            pos = trans.transform.translation
            ori = trans.transform.rotation
            print(
                f'Posición: x={pos.x:.5f}, y={pos.y:.5f}, z={pos.z:.5f} \n' 
                f'Orientación (cuaternion): x={ori.x:.5f}, y={ori.y:.5f}, z={ori.z:.5f}, w={ori.w:.5f} \n'
            )
            time.sleep(2)  # Para evitar imprimir demasiado rápido
        except Exception as e:
            self.get_logger().warn(f'No se pudo obtener la transformación: {e}')

def main():
    rclpy.init()
    node = EEPositionListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()