#!/usr/bin/env bash
# Fase 4: corridas CRUZADAS para separar efectos (datos / detector / configuración)
# en train+val de ambas bases de datos (ninguna tiene muestras 'hard' en train+val).
#   v3 + detector ANTIGUO + config A   → efecto del detector a datos y config fijos
#   v1 (base antigua) + config FINAL   → efecto de los datos a config fija (caja GT y YOLO nuevo)
# La config A sobre v1 ya existe (anterior/*.csv, filtrando train+val).
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
V3=~/parol6_pose_eval_dataset_v3
V1=~/parol6_pose_eval_dataset
C=$OUT/configs
FINAL=${FINAL:-B_defaults_nodo}
cd "$OUT"
run() {  # run <nombre> <dataset> <args...>
  local name=$1 ds=$2; shift 2
  ros2 run parol6_vision eval_pose_dataset.py --dataset $ds --split train val --estimator icp_lib \
      --csv cross/$name.csv "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn' | tee cross/$name.log
}
run v3_trval_icp_depthgt_yoloant_A_yaml_actual $V3 --cloud depth_gt --bbox yolo --yolo-json yolo/v3_detantiguo.json --icp-params $C/icp_A_yaml_actual.yaml
run v3_trval_icp_sgbm_yoloant_A_yaml_actual    $V3 --cloud sgbm     --bbox yolo --yolo-json yolo/v3_detantiguo.json --icp-params $C/icp_A_yaml_actual.yaml --stereo-params $C/stereo_wls8000.yaml
run v1_trval_icp_depthgt_gt_$FINAL     $V1 --cloud depth_gt --bbox gt_visib                                --icp-params $C/icp_$FINAL.yaml
run v1_trval_icp_sgbm_gt_$FINAL        $V1 --cloud sgbm     --bbox gt_visib                                --icp-params $C/icp_$FINAL.yaml --stereo-params $C/stereo_wls8000.yaml
run v1_trval_icp_depthgt_yolo_$FINAL   $V1 --cloud depth_gt --bbox yolo --yolo-json yolo/v1_detnuevo.json  --icp-params $C/icp_$FINAL.yaml
run v1_trval_icp_sgbm_yolo_$FINAL      $V1 --cloud sgbm     --bbox yolo --yolo-json yolo/v1_detnuevo.json  --icp-params $C/icp_$FINAL.yaml --stereo-params $C/stereo_wls8000.yaml
