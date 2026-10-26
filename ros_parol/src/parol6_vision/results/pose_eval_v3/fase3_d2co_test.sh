#!/usr/bin/env bash
# Fase 3: D2CO (nodo real de la rama d2co, build instalado en install/parol6_d2co)
# sobre test, UNA vez por configuración. D2CO no tiene parámetros que ajustar
# en esta evaluación, así que no pasa por train/val.
#   d2co_gt_umbral   bbox GT visible, umbral del nodo (max_avg_dist 4.5)
#   d2co_gt_sinumbral bbox GT visible, max_avg_dist 100 (sin rechazo)
#   d2co_yolo_umbral bbox YOLO (detector nuevo), umbral del nodo
# Necesita DISPLAY (contexto OpenGL de GLFW) o xvfb-run.
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-63}   # aislado de cualquier simulación abierta
export ROS_LOCALHOST_ONLY=1
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
DS=~/parol6_pose_eval_dataset_v3
PREFIX=$(ros2 pkg prefix parol6_d2co)
YAML=$PREFIX/share/parol6_d2co/config/d2co_pose_params.yaml
NODE_BIN=$PREFIX/lib/parol6_d2co/d2co_pose_estimator_node   # binario directo: kill llega al nodo
cd "$OUT"
NODE_PID=
stop_node() { [ -n "$NODE_PID" ] && kill "$NODE_PID" 2>/dev/null && wait "$NODE_PID" 2>/dev/null; NODE_PID=; true; }
trap stop_node EXIT
d2co() {  # d2co <nombre> <max_avg_dist> <args del harness...>
  local name=$1 thr=$2; shift 2
  if pgrep -f "$NODE_BIN" >/dev/null; then echo "ya hay un d2co_pose_estimator_node corriendo: páralo" >&2; exit 1; fi
  "$NODE_BIN" --ros-args --params-file "$YAML" \
      -p tracking_max_age_s:=0.0 -p 'output_frame:=""' -p broadcast_tf:=false \
      -p publish_debug_overlay:=false -p process_rate:=0.1 -p detection_max_age_s:=30.0 \
      -p max_avg_dist:=$thr > test/$name.node.log 2>&1 &
  NODE_PID=$!
  sleep 5
  ros2 run parol6_vision eval_pose_dataset.py --dataset $DS --split test --estimator d2co_node \
      --csv test/$name.csv "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn' | tee test/$name.log
  stop_node
}
d2co d2co_gt_umbral     4.5   --bbox gt_visib
d2co d2co_gt_sinumbral  100.0 --bbox gt_visib
d2co d2co_yolo_umbral   4.5   --bbox yolo --yolo-json $OUT/yolo/v3_detnuevo.json
