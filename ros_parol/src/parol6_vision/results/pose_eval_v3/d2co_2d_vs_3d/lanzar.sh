#!/usr/bin/env bash
# D2CO (nodo real) sobre las 60 primeras instancias de test, sin umbral
# (max_avg_dist 100) y con la caja GT, guardando la pose estimada (T_est):
#   d2co_malla_test60     modelo del paquete (mars2020tube_scaled.stl)
#   d2co_cilindro_test60  cilindro simple del mismo tamaño (cilindro_simple.stl)
# Necesita DISPLAY (contexto OpenGL de D2CO). ~3 min por corrida.
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-63} ROS_LOCALHOST_ONLY=1
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$(ros2 pkg prefix parol6_d2co)
YAML=$PREFIX/share/parol6_d2co/config/d2co_pose_params.yaml
BIN=$PREFIX/lib/parol6_d2co/d2co_pose_estimator_node
cd "$HERE"
run() {  # run <nombre> [stl]
  local name=$1 stl=${2:-}; local extra=(); [ -n "$stl" ] && extra=(-p stl_path:=$stl)
  if pgrep -f "$BIN --ros-args" >/dev/null; then echo "ya hay un d2co_pose_estimator_node corriendo" >&2; exit 1; fi
  "$BIN" --ros-args --params-file "$YAML" -p tracking_max_age_s:=0.0 -p 'output_frame:=""' \
      -p broadcast_tf:=false -p publish_debug_overlay:=false -p process_rate:=0.1 \
      -p detection_max_age_s:=30.0 -p max_avg_dist:=100.0 "${extra[@]}" > $name.node.log 2>&1 &
  local pid=$!; sleep 5
  ros2 run parol6_vision eval_pose_dataset.py --dataset ~/parol6_pose_eval_dataset_v3 --split test --limit 60 \
      --estimator d2co_node --bbox gt_visib --csv $name.csv --overlay-dir overlays_$name --overlay-n 60 \
      2>&1 | grep -v -E 'UserWarning|warnings.warn' > $name.log
  kill $pid; wait $pid 2>/dev/null || true
}
run d2co_malla_test60
run d2co_cilindro_test60 "$HERE/cilindro_simple.stl"
