#!/usr/bin/env python3
'''
udp_pose_receiver.py    
'''
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

class UDPBridge(Node):

    def __init__(self):
        super().__init__('udp_pose_bridge')
        #self.publisher = self.create_publisher(Pose, '/pose_target_cuaterniones', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))

        self.get_logger().info("Esperando datos UDP...")

        #self.timer = self.create_timer(0.01, self.receive_data)

    # comentar receive data
    # 
    """
    def receive_data(self):
        try:
            self.sock.setblocking(False)
            data, _ = self.sock.recvfrom(1024)
            data = data.decode().strip()
            values = list(map(float, data.split()))

            if len(values) == 7:
                msg = Pose()
                msg.position.x = values[0]
                msg.position.y = values[1]
                msg.position.z = values[2]
                msg.orientation.x = values[3]
                msg.orientation.y = values[4]
                msg.orientation.z = values[5]
                msg.orientation.w = values[6]

                self.publisher.publish(msg)

        except BlockingIOError:
            pass
    """

def main():
    rclpy.init()
    node = UDPBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
