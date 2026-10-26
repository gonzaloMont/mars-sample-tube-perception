#!/usr/bin/env bash
# Fase 3: CIFRA FINAL de ICP en test (222 instancias), UNA sola vez, con la
# configuración elegida en val (analysis/val.md): ICP = B (defaults del nodo),
# estéreo = wls_lambda 500 (configs/stereo_wls500.yaml).
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
DS=~/parol6_pose_eval_dataset_v3
YOLO=$OUT/yolo/v3_detnuevo.json
C=$OUT/configs
FINAL=B_defaults_nodo
STEREO_ARGS="--stereo-params $C/stereo_wls500.yaml"
cd "$OUT"
run() {  # run <nombre> <args...>
  local name=$1; shift
  ros2 run parol6_vision eval_pose_dataset.py --dataset $DS --split test --estimator icp_lib \
      --csv test/$name.csv --overlay-dir test/overlays_$name --overlay-n 30 "$@" \
      2>&1 | grep -v -E 'UserWarning|warnings.warn' | tee test/$name.log
}
run icp_depthgt_gt_$FINAL   --cloud depth_gt --bbox gt_visib               --icp-params $C/icp_$FINAL.yaml
run icp_sgbmwls500_gt_$FINAL   --cloud sgbm  --bbox gt_visib               --icp-params $C/icp_$FINAL.yaml $STEREO_ARGS
run icp_depthgt_yolo_$FINAL    --cloud depth_gt --bbox yolo --yolo-json $YOLO --icp-params $C/icp_$FINAL.yaml
run icp_sgbmwls500_yolo_$FINAL --cloud sgbm  --bbox yolo --yolo-json $YOLO --icp-params $C/icp_$FINAL.yaml $STEREO_ARGS
