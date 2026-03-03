#!/usr/bin/env python3
"""
pose_ik_subscriber.py

Recibe una Pose absoluta (respecto a base_link), resuelve la cinemática inversa
con el servicio /compute_ik de MoveIt, y ejecuta el movimiento mediante
JointConstraints (igual que move_to_joints en test_moveit.py).

Esto garantiza que una misma pose siempre produzca el mismo resultado,
independientemente de donde esté el robot en ese momento.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
    RobotState,
    PositionIKRequest,
)
from moveit_msgs.srv import GetPositionIK


JOINT_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
GROUP_NAME = 'arm_group'
TIP_LINK = 'L6_link'
BASE_FRAME = 'base_link'


class PoseIKSubscriber(Node):
    def __init__(self):
        super().__init__('pose_ik_subscriber')

        # Cliente para resolver IK
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        # Cliente para ejecutar movimiento por joints
        self._action_client = ActionClient(self, MoveGroup, '/move_action')

        # Suscripción a la pose objetivo
        self.create_subscription(Pose, '/pose_target_cuaterniones', self._pose_cb, 10)

        self.get_logger().info('Esperando servicio /compute_ik...')
        self._ik_client.wait_for_service()
        self.get_logger().info('Esperando action server /move_action...')
        self._action_client.wait_for_server()
        self.get_logger().info('Listo. Esperando poses en /pose_target_cuaterniones...')

    def _pose_cb(self, msg: Pose):
        self.get_logger().info(
            f'Pose recibida: pos=({msg.position.x:.3f}, {msg.position.y:.3f}, {msg.position.z:.3f})'
        )
        self._resolve_ik_and_move(msg)

    def _resolve_ik_and_move(self, pose: Pose):
        # Construir la petición de IK
        # Cambiar posiciones x, z y orientación x para que coincida con la del robot real
        pose.position.x -= 0.062
        pose.position.z += 0.046
        pose.orientation.x = -pose.orientation.x

        ik_request = PositionIKRequest()
        ik_request.group_name = GROUP_NAME
        ik_request.ik_link_name = TIP_LINK
        ik_request.avoid_collisions = True
        ik_request.timeout.sec = 5

        # Pose objetivo en el frame del robot
        from geometry_msgs.msg import PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = BASE_FRAME
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        ik_request.pose_stamped = pose_stamped

        # Llamar al servicio de IK
        request = GetPositionIK.Request()
        request.ik_request = ik_request

        future = self._ik_client.call_async(request)
        future.add_done_callback(self._ik_response_cb)

    def _ik_response_cb(self, future):
        response = future.result()
        error_code = response.error_code.val

        if error_code != 1:  # 1 = SUCCESS
            self.get_logger().error(f'IK fallida (código: {error_code}). Pose inalcanzable.')
            return

        # Extraer las posiciones de joints de la solución IK
        joint_state: JointState = response.solution.joint_state
        joint_positions = {}
        for name, pos in zip(joint_state.name, joint_state.position):
            joint_positions[name] = pos

        target = [joint_positions.get(j, 0.0) for j in JOINT_NAMES]
        self.get_logger().info(f'IK resuelta: {[f"{v:.3f}" for v in target]}')

        self._move_to_joints(target)

    def _move_to_joints(self, target_positions, velocity_scaling=0.3, acceleration_scaling=0.3):
        """Ejecuta el movimiento a las posiciones articulares obtenidas por IK."""
        goal = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name = GROUP_NAME
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling

        # Goal en espacio de joints
        
        constraints = Constraints()
        for name, position in zip(JOINT_NAMES, target_positions):

            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movimiento rechazado por move_group')
            return
        self.get_logger().info('Movimiento aceptado, ejecutando...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('Movimiento completado OK')
        else:
            self.get_logger().error(f'Movimiento fallido (código: {result.error_code.val})')


def main():
    rclpy.init()
    node = PoseIKSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()