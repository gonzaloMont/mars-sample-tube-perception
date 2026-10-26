#!/usr/bin/env bash
# Fase 2: CONFIRMACIÓN en val (142 instancias) de las configuraciones candidatas,
# con su rechazo real (sin filtrado a posteriori), en los cuatro escenarios de ICP.
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
DS=~/parol6_pose_eval_dataset_v3
YOLO=$OUT/yolo/v3_detnuevo.json
C=$OUT/configs
cd "$OUT"
run() {  # run <nombre> <args...>
  local name=$1; shift
  ros2 run parol6_vision eval_pose_dataset.py --dataset $DS --split val --estimator icp_lib \
      --csv val/$name.csv "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn' | tee val/$name.log
}
for cfg in A_yaml_actual B_defaults_nodo; do
  run icp_depthgt_gt_$cfg   --cloud depth_gt --bbox gt_visib                 --icp-params $C/icp_$cfg.yaml
  run icp_sgbm_gt_$cfg      --cloud sgbm     --bbox gt_visib                 --icp-params $C/icp_$cfg.yaml --stereo-params $C/stereo_wls8000.yaml
  run icp_depthgt_yolo_$cfg --cloud depth_gt --bbox yolo --yolo-json $YOLO   --icp-params $C/icp_$cfg.yaml
  run icp_sgbm_yolo_$cfg    --cloud sgbm     --bbox yolo --yolo-json $YOLO   --icp-params $C/icp_$cfg.yaml --stereo-params $C/stereo_wls8000.yaml
done
# Candidata secundaria salida de train (+6.9 pp en SGBM): estéreo con wls_lambda 500
run icp_sgbmwls500_gt_B_defaults_nodo   --cloud sgbm --bbox gt_visib               --icp-params $C/icp_B_defaults_nodo.yaml --stereo-params $C/stereo_wls500.yaml
run icp_sgbmwls500_yolo_B_defaults_nodo --cloud sgbm --bbox yolo --yolo-json $YOLO --icp-params $C/icp_B_defaults_nodo.yaml --stereo-params $C/stereo_wls500.yaml
