#!/usr/bin/env python3
"""
grasp_executor.py — Ejecutor de agarres con fallback automático primary→alt.

Toma las poses publicadas por `tube_grasp_pose_node`:

    /grasp/pick_pose            geometry_msgs/PoseStamped   (primary)
    /grasp/approach_pose        geometry_msgs/PoseStamped   (primary)
    /grasp/pick_pose_alt        geometry_msgs/PoseStamped   (rotada 180° en z_g)
    /grasp/approach_pose_alt    geometry_msgs/PoseStamped   (rotada 180° en z_g)

y ejecuta la secuencia approach → pick en el brazo mediante el servicio
/compute_ik + la action /move_action de MoveIt. Cuando una variante falla
(la IK no resuelve, los joints quedan fuera de límites, etc.) se intenta
AUTOMÁTICAMENTE la variante alt — geométricamente equivalente para un
cilindro pero correspondiente a otra configuración de joints, que el
PAROL6 sí acepta a menudo cuando la primary es inalcanzable.

╔══════════════════════════════════════════════════════════════════════════╗
║  POR QUÉ EXISTE ESTE NODO                                                  ║
║                                                                            ║
║  `pose_ik_subscriber.py` recibe UNA pose y la ejecuta tal cual. No tiene   ║
║  conocimiento de la simetría del cilindro ni del topic _alt publicado     ║
║  por `tube_grasp_pose_node`, así que cuando la IK rechaza la primary el   ║
║  agarre se da por imposible aunque la variante de 180° sí sería válida.   ║
║                                                                            ║
║  `grasp_executor` cierra ese hueco para el pipeline productivo. El test   ║
║  ya hacía esta lógica internamente; aquí la encapsulamos en un nodo       ║
║  reusable que el orquestador puede llamar como un servicio.               ║
╚══════════════════════════════════════════════════════════════════════════╝

INTERFACES

  Servicio:
    /grasp/execute    std_srvs/Trigger
        Ejecuta UNA vez la secuencia approach → pick con las últimas poses
        recibidas. Bloquea hasta que la operación termina (o todas las
        variantes fallan). Respuesta:
            success=True,  message="done:primary"   ← OK con la primary
            success=True,  message="done:alt"       ← primary falló, OK con alt
            success=False, message="failed:no_poses"      ← faltan poses
            success=False, message="failed:busy"          ← ya hay una en curso
            success=False, message="failed:all_variants"  ← ninguna variante OK

  Topic publicado:
    /grasp/execution_status   std_msgs/String
        Emite eventos durante la ejecución para monitorización:
            "executing:approach:primary", "executing:pick:primary",
            "executing:approach:alt",     "executing:pick:alt",
            "done:primary" | "done:alt"
            "failed:no_poses" | "failed:approach:primary:<code>" | ...

PARÁMETROS
  arm_group_name           default 'arm_group'
  tip_link                 default 'gripper_base_link'      (coincide con el SRDF
                                                              del arm_group)
  velocity_scaling         default 0.3
  acceleration_scaling     default 0.3
  planning_attempts        default 5
  allowed_planning_time    default 5.0   s
  motion_timeout_s         default 15.0  s   tiempo máx. de ejecución por pose
  try_alt_on_failure       default True       fallback alt si la primary falla
  require_approach_success default True       si False, intenta pick aunque
                                              approach haya fallado
  pose_freshness_s         default 5.0   s   ignorar poses recibidas hace más
                                              de este tiempo (evita ejecutar
                                              grasps obsoletos). Se mide con
                                              time.monotonic() del nodo, NO
                                              con `header.stamp`, para no
                                              romperse si la pose viene en
                                              sim_time. <= 0 desactiva el
                                              chequeo.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    PositionIKRequest,
)
from moveit_msgs.srv import GetPositionIK
from std_msgs.msg import String
from std_srvs.srv import Trigger


ARM_JOINT_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']


# QoS coherente con los publishers del tube_grasp_pose_node.
GRASP_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


class GraspExecutor(Node):

    def __init__(self) -> None:
        super().__init__('grasp_executor')

        # ── Parámetros ───────────────────────────────────────────────────────
        self.declare_parameter('arm_group_name',          'arm_group')
        self.declare_parameter('tip_link',                'gripper_base_link')
        self.declare_parameter('velocity_scaling',         0.3)
        self.declare_parameter('acceleration_scaling',     0.3)
        self.declare_parameter('planning_attempts',        5)
        self.declare_parameter('allowed_planning_time',    5.0)
        self.declare_parameter('motion_timeout_s',        15.0)
        self.declare_parameter('try_alt_on_failure',      True)
        self.declare_parameter('require_approach_success', True)
        self.declare_parameter('pose_freshness_s',         5.0)

        gp = self.get_parameter
        self._arm_group         = gp('arm_group_name').value
        self._tip_link          = gp('tip_link').value
        self._vel_scale         = float(gp('velocity_scaling').value)
        self._acc_scale         = float(gp('acceleration_scaling').value)
        self._plan_attempts     = int(gp('planning_attempts').value)
        self._plan_time         = float(gp('allowed_planning_time').value)
        self._motion_timeout    = float(gp('motion_timeout_s').value)
        self._try_alt           = bool(gp('try_alt_on_failure').value)
        self._req_approach_ok   = bool(gp('require_approach_success').value)
        self._pose_freshness    = float(gp('pose_freshness_s').value)

        # ── Callback groups: el service de ejecución vive en un grupo
        # exclusivo y bloqueante, mientras que las subs y las llamadas
        # asíncronas (IK service + MoveGroup action) viven en grupos
        # reentrantes para que sus futures se resuelvan mientras el
        # service callback espera. Sin esto habría deadlock.
        self._sub_group     = ReentrantCallbackGroup()
        self._exec_group    = MutuallyExclusiveCallbackGroup()
        self._client_group  = ReentrantCallbackGroup()

        # ── Estado: poses recibidas (cacheadas) ──────────────────────────────
        # Junto a cada pose guardamos el `time.monotonic()` del momento en
        # que llegó. Eso evita depender del `header.stamp`, que puede venir
        # en sim_time, en wall-time o vacío según quién la publique — algo
        # que dio falsos positivos de "stale_pose" en sim con sim_time != ROS.
        self._lock = threading.Lock()
        self._last_pick:         Optional[PoseStamped] = None
        self._last_approach:     Optional[PoseStamped] = None
        self._last_pick_alt:     Optional[PoseStamped] = None
        self._last_approach_alt: Optional[PoseStamped] = None
        self._last_pick_recv_t:     Optional[float] = None
        self._last_approach_recv_t: Optional[float] = None
        self._busy = False

        # ── Subscriptores ────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/grasp/pick_pose',
            self._pick_cb, GRASP_QOS, callback_group=self._sub_group)
        self.create_subscription(
            PoseStamped, '/grasp/approach_pose',
            self._approach_cb, GRASP_QOS, callback_group=self._sub_group)
        self.create_subscription(
            PoseStamped, '/grasp/pick_pose_alt',
            self._pick_alt_cb, GRASP_QOS, callback_group=self._sub_group)
        self.create_subscription(
            PoseStamped, '/grasp/approach_pose_alt',
            self._approach_alt_cb, GRASP_QOS, callback_group=self._sub_group)

        # ── Publicadores ─────────────────────────────────────────────────────
        self._pub_status = self.create_publisher(
            String, '/grasp/execution_status', 10)

        # ── Clientes MoveIt ──────────────────────────────────────────────────
        self._ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self._client_group)
        self._move_action_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self._client_group)

        # ── Servicio de disparo ──────────────────────────────────────────────
        self.create_service(
            Trigger, '/grasp/execute', self._execute_srv_cb,
            callback_group=self._exec_group)

        self.get_logger().info(
            "GraspExecutor iniciado.\n"
            f"  arm_group={self._arm_group}  tip_link={self._tip_link}\n"
            f"  try_alt_on_failure={self._try_alt}  "
            f"motion_timeout={self._motion_timeout:.1f}s\n"
            f"  Servicio: /grasp/execute  Status: /grasp/execution_status")

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks de pose
    # ─────────────────────────────────────────────────────────────────────────

    def _pick_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_pick = msg
            self._last_pick_recv_t = time.monotonic()

    def _approach_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_approach = msg
            self._last_approach_recv_t = time.monotonic()

    def _pick_alt_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_pick_alt = msg

    def _approach_alt_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_approach_alt = msg

    # ─────────────────────────────────────────────────────────────────────────
    # Publicador del estado
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_status.publish(msg)
        self.get_logger().info(f"[grasp_executor] {text}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: esperar el resultado de un Future sin spin manual (el executor
    # multi-thread procesa las respuestas en otros threads).
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _wait_future(fut, timeout_s: float):
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout_s):
            return None
        return fut.result()

    # ─────────────────────────────────────────────────────────────────────────
    # IK + MoveGroup para una pose. Devuelve 'done' o 'error:<reason>'.
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_pose(self, pose: Pose, base_frame: str = 'base_link') -> str:
        # 1) Resolver IK
        if not self._ik_client.wait_for_service(timeout_sec=2.0):
            return "error:ik_service_unavailable"

        ik_req = GetPositionIK.Request()
        ikr = ik_req.ik_request
        ikr.group_name = self._arm_group
        ikr.ik_link_name = self._tip_link
        ikr.avoid_collisions = True
        ikr.timeout.sec = 5
        ps = PoseStamped()
        ps.header.frame_id = base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        ikr.pose_stamped = ps

        fut = self._ik_client.call_async(ik_req)
        resp = self._wait_future(fut, timeout_s=8.0)
        if resp is None:
            return "error:ik_timeout"
        if resp.error_code.val != 1:
            return f"error:ik_code_{resp.error_code.val}"

        # Extraer las posiciones articulares del solucion IK
        joint_state = resp.solution.joint_state
        positions = {n: p for n, p in zip(joint_state.name, joint_state.position)}
        try:
            target = [positions[j] for j in ARM_JOINT_NAMES]
        except KeyError as exc:
            return f"error:missing_joint:{exc.args[0]}"

        # 2) Enviar joint goal a /move_action
        if not self._move_action_client.wait_for_server(timeout_sec=2.0):
            return "error:move_action_unavailable"

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name                      = self._arm_group
        req.num_planning_attempts           = self._plan_attempts
        req.allowed_planning_time           = self._plan_time
        req.max_velocity_scaling_factor     = self._vel_scale
        req.max_acceleration_scaling_factor = self._acc_scale

        constraints = Constraints()
        for jname, pos in zip(ARM_JOINT_NAMES, target):
            jc = JointConstraint()
            jc.joint_name      = jname
            jc.position        = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request                          = req
        goal.planning_options                 = PlanningOptions()
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3

        send_fut = self._move_action_client.send_goal_async(goal)
        handle = self._wait_future(send_fut, timeout_s=5.0)
        if handle is None:
            return "error:send_timeout"
        if not handle.accepted:
            return "error:rejected"

        result_fut = handle.get_result_async()
        result = self._wait_future(result_fut, timeout_s=self._motion_timeout)
        if result is None:
            return "error:exec_timeout"
        code = result.result.error_code.val
        return "done" if code == 1 else f"error:exec_code_{code}"

    # ─────────────────────────────────────────────────────────────────────────
    # Callback del servicio /grasp/execute
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_srv_cb(self, request: Trigger.Request,
                        response: Trigger.Response) -> Trigger.Response:
        # 1) No permitir ejecuciones concurrentes.
        if self._busy:
            response.success = False
            response.message = "failed:busy"
            self._publish_status(response.message)
            return response

        # 2) Comprobar que tenemos al menos la pareja primary, y opcionalmente
        # también la alt si va a intentarse el fallback.
        with self._lock:
            pick = self._last_pick
            approach = self._last_approach
            pick_alt = self._last_pick_alt
            approach_alt = self._last_approach_alt
            pick_recv_t = self._last_pick_recv_t
            approach_recv_t = self._last_approach_recv_t

        if pick is None or approach is None:
            response.success = False
            response.message = "failed:no_poses"
            self._publish_status(response.message)
            return response

        # 3) Comprobar frescura: cuánto tiempo hace que LLEGÓ la pose al nodo
        # (NO usamos `header.stamp` porque puede venir en sim_time o en cero
        # y dar falsos positivos al compararlo con el clock del executor).
        # `pose_freshness_s <= 0` desactiva el chequeo.
        if self._pose_freshness > 0.0:
            now_mono = time.monotonic()
            for label, recv_t in (("pick", pick_recv_t),
                                  ("approach", approach_recv_t)):
                if recv_t is None:
                    continue
                age = now_mono - recv_t
                if age > self._pose_freshness:
                    response.success = False
                    response.message = (f"failed:stale_pose:{label}"
                                        f"(age={age:.1f}s)")
                    self._publish_status(response.message)
                    return response

        # 4) Lista de variantes a intentar en orden.
        variants: list[tuple[str, PoseStamped, PoseStamped]] = [
            ("primary", approach, pick),
        ]
        if self._try_alt and pick_alt is not None and approach_alt is not None:
            variants.append(("alt", approach_alt, pick_alt))

        self._busy = True
        try:
            last_error = ""
            for label, app_ps, pick_ps in variants:
                # Approach
                self._publish_status(f"executing:approach:{label}")
                base_frame = app_ps.header.frame_id or 'base_link'
                appr_status = self._execute_pose(app_ps.pose, base_frame)
                if appr_status != "done":
                    last_error = f"approach:{label}:{appr_status}"
                    self._publish_status(f"failed:{last_error}")
                    if self._req_approach_ok:
                        continue  # descarta variante, prueba la siguiente
                # Pick
                self._publish_status(f"executing:pick:{label}")
                base_frame = pick_ps.header.frame_id or 'base_link'
                pick_status = self._execute_pose(pick_ps.pose, base_frame)
                if pick_status == "done":
                    self._publish_status(f"done:{label}")
                    response.success = True
                    response.message = f"done:{label}"
                    return response
                last_error = f"pick:{label}:{pick_status}"
                self._publish_status(f"failed:{last_error}")

            # Todas las variantes fallaron.
            response.success = False
            response.message = (f"failed:all_variants:{last_error}"
                                if last_error else "failed:all_variants")
            self._publish_status(response.message)
            return response
        finally:
            self._busy = False


# ════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspExecutor()
    # MultiThreadedExecutor: necesario para que el callback del service pueda
    # esperar futures (IK + MoveGroup) que se resuelven en otros threads.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
