#!/usr/bin/env python3
"""Test: mueve el PAROL6 a una posición articular, abre/cierra el gripper y vuelve a home.
Usa rclpy + moveit_msgs directamente, sin moveit_commander."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
    RobotState,
)
from sensor_msgs.msg import JointState
import time


class Parol6Controller(Node):
    def __init__(self):
        super().__init__('parol6_controller')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._joint_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        self._gripper_joint_names = ['gripper_joint', 'gripper_right_joint']
        self._current_joint_state = None

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        self.get_logger().info('Esperando al action server /move_action...')
        self._action_client.wait_for_server()
        self.get_logger().info('Conectado a move_group.')

    def _joint_state_cb(self, msg):
        self._current_joint_state = msg

    def get_current_joints(self):
        """Devuelve las posiciones articulares actuales del brazo."""
        if self._current_joint_state is None:
            return None
        positions = {}
        for name, pos in zip(self._current_joint_state.name, self._current_joint_state.position):
            positions[name] = pos
        return [positions.get(j, 0.0) for j in self._joint_names]

    def get_current_gripper(self):
        """Devuelve las posiciones actuales del gripper."""
        if self._current_joint_state is None:
            return None
        positions = {}
        for name, pos in zip(self._current_joint_state.name, self._current_joint_state.position):
            positions[name] = pos
        return [positions.get(j, 0.0) for j in self._gripper_joint_names]

    def move_gripper(self, position, velocity_scaling=0.5, acceleration_scaling=0.5):
        """Abre o cierra el gripper.
        position: 0.0 (cerrado) a 0.025 (abierto)"""
        goal = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name = 'gripper'
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling

        constraints = Constraints()
        for name in self._gripper_joint_names:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        action = 'Cerrando' if position < 0.01 else 'Abriendo'
        self.get_logger().info(f'{action} gripper (posición: {position})')
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movimiento del gripper rechazado')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('Gripper movido OK')
            return True
        else:
            self.get_logger().error(f'Gripper fallo (código: {result.error_code.val})')
            return False

    def move_to_joints(self, target_positions, velocity_scaling=0.3, acceleration_scaling=0.3):
        """Planifica y ejecuta un movimiento a posiciones articulares."""
        goal = MoveGroup.Goal()

        # Motion plan request
        req = MotionPlanRequest()
        req.group_name = 'arm_group'
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity_scaling
        req.max_acceleration_scaling_factor = acceleration_scaling

        # Joint constraints (posición objetivo)
        constraints = Constraints()
        for name, position in zip(self._joint_names, target_positions):
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
        goal.planning_options.plan_only = False  # Planificar Y ejecutar
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.get_logger().info(f'Moviendo a: {target_positions}')
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movimiento rechazado por move_group')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:  # SUCCESS
            self.get_logger().info('Movimiento completado OK')
            return True
        else:
            self.get_logger().error(f'Movimiento fallido (código: {result.error_code.val})')
            return False


def main():
    rclpy.init()
    controller = Parol6Controller()

    # Esperar a recibir el primer joint_state
    controller.get_logger().info('Esperando joint_states...')
    while controller.get_current_joints() is None:
        rclpy.spin_once(controller, timeout_sec=0.1)

    print(f'Posición brazo: {controller.get_current_joints()}')
    print(f'Posición gripper: {controller.get_current_gripper()}')

    # 1. Abrir gripper
    print('\n--- Paso 1: Abrir gripper ---')
    controller.move_gripper(0.025)
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Gripper: {controller.get_current_gripper()}')

    # 2. Mover brazo a posición de recogida
    print('\n--- Paso 2: Mover brazo a posición de recogida ---')
    target = [0.5, 0.3, -0.5, 0.0, 0.5, 0.0]
    controller.move_to_joints(target)
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Brazo: {controller.get_current_joints()}')

    # 3. Cerrar gripper (agarrar tubo)
    print('\n--- Paso 3: Cerrar gripper (agarrar tubo) ---')
    controller.move_gripper(0.005)  # Casi cerrado, dejando hueco para el tubo
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Gripper: {controller.get_current_gripper()}')

    # 4. Mover brazo a posición de entrega
    print('\n--- Paso 4: Mover brazo a posición de entrega ---')
    entrega = [-0.5, 0.2, -0.3, 0.0, 0.4, 0.0]
    controller.move_to_joints(entrega)
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Brazo: {controller.get_current_joints()}')

    # 5. Abrir gripper (soltar tubo)
    print('\n--- Paso 5: Abrir gripper (soltar tubo) ---')
    controller.move_gripper(0.025)
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Gripper: {controller.get_current_gripper()}')

    # 6. Volver a home
    print('\n--- Paso 6: Volver a home ---')
    home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    controller.move_to_joints(home)
    time.sleep(0.5)
    rclpy.spin_once(controller, timeout_sec=0.5)
    print(f'Brazo: {controller.get_current_joints()}')
    print(f'Gripper: {controller.get_current_gripper()}')

    print('\nTest completado.')
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
