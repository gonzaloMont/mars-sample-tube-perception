#!/usr/bin/env bash
# Fase 1: AJUSTE en train (365 instancias). icp_lib sin rechazo (min_fitness 0,
# max_rmse 1 m): el rechazo es el último paso y no cambia la pose, así que las
# configuraciones A (yaml anterior) y B (defaults del nodo), y cualquier otro
# umbral, se obtienen después filtrando por fitness/rmse_mm (analysis/analiza.py).
# Lo que sí cambia la pose es la segmentación: remove_plane y plane_dist_thresh.
set -eo pipefail
source /opt/ros/humble/setup.bash
source ~/parol6_ws/install/setup.bash
OUT=~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3
DS=~/parol6_pose_eval_dataset_v3
YOLO=$OUT/yolo/v3_detnuevo.json
cd "$OUT"
run() {  # run <nombre> <args...>
  local name=$1; shift
  ros2 run parol6_vision eval_pose_dataset.py --dataset $DS --split train --estimator icp_lib \
      --csv train/$name.csv "$@" 2>&1 | grep -v -E 'UserWarning|warnings.warn' | tee train/$name.log
}
C=$OUT/configs
# bbox GT visible
run icp_depthgt_gt_rpF          --cloud depth_gt --bbox gt_visib --icp-params $C/icp_norej_rpF.yaml
run icp_depthgt_gt_rpT006       --cloud depth_gt --bbox gt_visib --icp-params $C/icp_norej_rpT_pd006.yaml
run icp_depthgt_gt_rpT004       --cloud depth_gt --bbox gt_visib --icp-params $C/icp_norej_rpT_pd004.yaml
run icp_depthgt_gt_rpT004_rep   --cloud depth_gt --bbox gt_visib --icp-params $C/icp_norej_rpT_pd004.yaml   # repetición: ¿determinista?
run icp_sgbm_gt_rpF             --cloud sgbm     --bbox gt_visib --icp-params $C/icp_norej_rpF.yaml --stereo-params $C/stereo_wls8000.yaml
run icp_sgbm_gt_rpT006          --cloud sgbm     --bbox gt_visib --icp-params $C/icp_norej_rpT_pd006.yaml --stereo-params $C/stereo_wls8000.yaml
run icp_sgbm_gt_rpT004          --cloud sgbm     --bbox gt_visib --icp-params $C/icp_norej_rpT_pd004.yaml --stereo-params $C/stereo_wls8000.yaml
run icp_sgbmwls500_gt_rpT004    --cloud sgbm     --bbox gt_visib --icp-params $C/icp_norej_rpT_pd004.yaml --stereo-params $C/stereo_wls500.yaml
# bbox YOLO (detector nuevo)
run icp_depthgt_yolo_rpF        --cloud depth_gt --bbox yolo --yolo-json $YOLO --icp-params $C/icp_norej_rpF.yaml
run icp_depthgt_yolo_rpT004     --cloud depth_gt --bbox yolo --yolo-json $YOLO --icp-params $C/icp_norej_rpT_pd004.yaml
run icp_sgbm_yolo_rpF           --cloud sgbm     --bbox yolo --yolo-json $YOLO --icp-params $C/icp_norej_rpF.yaml --stereo-params $C/stereo_wls8000.yaml
run icp_sgbm_yolo_rpT004        --cloud sgbm     --bbox yolo --yolo-json $YOLO --icp-params $C/icp_norej_rpT_pd004.yaml --stereo-params $C/stereo_wls8000.yaml
