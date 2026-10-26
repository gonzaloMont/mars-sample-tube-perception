#!/usr/bin/env bash
# Fase 0: detecciones YOLO en caché (detector nuevo y antiguo × base v3 y base antigua).
# El detector antiguo es models/best.pt en 15ac096 (el que usó la evaluación anterior, anterior/).
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
NEW=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/models/best.pt   # sha256 a07ff5df…
OLD=${OLD_WEIGHTS:-/tmp/best_15ac096.pt}                                 # sha256 bf9a8353…
[ -f "$OLD" ] || git -C ~/parol6_ws/src/PAROL6 show 15ac096:ros_parol/src/parol6_vision/models/best.pt > "$OLD"
V3=~/parol6_pose_eval_dataset_v3
V1=~/parol6_pose_eval_dataset
cd "$OUT"
ros2 run parol6_vision cache_yolo_detections.py --dataset $V3 --model "$NEW" --out yolo/v3_detnuevo.json   2>&1 | tee yolo/v3_detnuevo.log
ros2 run parol6_vision cache_yolo_detections.py --dataset $V3 --model "$OLD" --out yolo/v3_detantiguo.json 2>&1 | tee yolo/v3_detantiguo.log
ros2 run parol6_vision cache_yolo_detections.py --dataset $V1 --model "$NEW" --out yolo/v1_detnuevo.json   2>&1 | tee yolo/v1_detnuevo.log
ros2 run parol6_vision cache_yolo_detections.py --dataset $V1 --model "$OLD" --out yolo/v1_detantiguo.json 2>&1 | tee yolo/v1_detantiguo.log
