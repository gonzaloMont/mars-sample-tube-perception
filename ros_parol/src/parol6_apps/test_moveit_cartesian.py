#!/usr/bin/env python3
"""
Test: mueve el PAROL6 a una posición cartesiana (x, y, z, orientación como cuaternión) usando rclpy + moveit_msgs directamente, sin moveit_commander.
No controla la pinza, solo el brazo. Ejemplo sencillo.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
)
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
import time


class Parol6CartesianController(Node):
    def __init__(self):
        super().__init__('parol6_cartesian_controller')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._joint_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        self._current_joint_state = None
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.get_logger().info('Esperando al action server /move_action...')
        self._action_client.wait_for_server()
        self.get_logger().info('Conectado a move_group.')

    def _joint_state_cb(self, msg):
        self._current_joint_state = msg

    def get_current_joints(self):
        if self._current_joint_state is None:
            return None
        positions = {}
        for name, pos in zip(self._current_joint_state.name, self._current_joint_state.position):
            positions[name] = pos
        return [positions.get(j, 0.0) for j in self._joint_names]


    def move_to_pose(self, x, y, z, qx, qy, qz, qw, velocity_scaling=0.3, acceleration_scaling=0.3, constraints_mode='both'):
        """Planifica y ejecuta un movimiento a una pose cartesiana objetivo.
        constraints_mode: 'both', 'position', 'orientation', 'none'"""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'arm_group'
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        constraints = Constraints()
        if constraints_mode in ('both', 'position'):
            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = 'base_link'
            pos_constraint.link_name = 'L6_link'
            pos_constraint.target_point_offset.x = 0.0
            pos_constraint.target_point_offset.y = 0.0
            pos_constraint.target_point_offset.z = 0.0
            bv = BoundingVolume()
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.08, 0.08, 0.08]  # Más tolerancia
            bv.primitives.append(box)
            bv.primitive_poses.append(pose.pose)
            pos_constraint.constraint_region = bv
            pos_constraint.weight = 1.0
            constraints.position_constraints.append(pos_constraint)
        if constraints_mode in ('both', 'orientation'):
            ori_constraint = OrientationConstraint()
            ori_constraint.header.frame_id = 'base_link'
            ori_constraint.link_name = 'L6_link'
            ori_constraint.orientation = pose.pose.orientation
            ori_constraint.absolute_x_axis_tolerance = 0.5
            ori_constraint.absolute_y_axis_tolerance = 0.5
            ori_constraint.absolute_z_axis_tolerance = 0.5
            ori_constraint.weight = 1.0
            constraints.orientation_constraints.append(ori_constraint)
        if constraints_mode != 'none':
            req.goal_constraints.append(constraints)
        else:
            # fallback: no constraints, solo para debug
            pass

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f'Moviendo a pose: pos=({x:.3f}, {y:.3f}, {z:.3f}), quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}), constraints={constraints_mode}')
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movimiento rechazado por move_group')
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('Movimiento completado OK')
            return True
        else:
            self.get_logger().error(f'Movimiento fallido (código: {result.error_code.val})')
            if hasattr(result, 'error_code'):
                self.get_logger().error(f'Error code: {result.error_code.val}')
            if hasattr(result, 'error_message'):
                self.get_logger().error(f'Error message: {result.error_message}')
            return False

def main():
    rclpy.init()
    controller = Parol6CartesianController()
    controller.get_logger().info('Esperando joint_states...')
    while controller.get_current_joints() is None:
        rclpy.spin_once(controller, timeout_sec=0.1)
    print(f'Posición actual del brazo: {controller.get_current_joints()}')

    # Solo mover a la pose que sabemos que funciona
    x, y, z = 0.3, 0.2, 0.3
    qx, qy, qz, qw = 0.0, 0.0, 0.707, 0.707
    print(f'\nMoviendo a la pose conocida: pos=({x},{y},{z}), quat=({qx},{qy},{qz},{qw})')
    ok = controller.move_to_pose(x, y, z, qx, qy, qz, qw, constraints_mode='both')
    if ok:
        print('¡Movimiento exitoso!')
        print(f'Brazo: {controller.get_current_joints()}')
    else:
        print('Fallo en el movimiento.')
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Brazo: {controller.get_current_joints()}')

    # 2. Volver a home (opcional)
    print('\n--- Paso 2: Volver a home ---')
    x, y, z = 0.2, 0.0, 0.327
    qx, qy, qz, qw = 0.707, 0.0, -0.707, 0.0
    print(f'\nMoviendo a home: pos=({x},{y},{z}), quat=({qx},{qy},{qz},{qw})')
    ok = controller.move_to_pose(x, y, z, qx, qy, qz, qw, constraints_mode='both')
    if ok:
        print('¡Movimiento exitoso!')
        print(f'Brazo: {controller.get_current_joints()}')
    else:
        print('Fallo en el movimiento.')

    print('\nTest completado.')
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
