# parol6_vision

Vision pipeline for PAROL6.

## Purpose

Consumes overhead camera images, runs object detection/segmentation, and estimates target poses.

## Main executables

- `yolo_camera_subscriber.py`: YOLO-based detection, annotated image and mask publication.
- `pose_estimator_node.py`: pose estimation pipeline variant 1.
- `pose_estimator_node2.py`: pose estimation pipeline variant 2.

Run examples:

```bash
ros2 run parol6_vision yolo_camera_subscriber.py
ros2 run parol6_vision pose_estimator_node.py
```

## Main folders

- `models/`: model files and weights.
- `nodos/`: ROS 2 Python nodes.

## Notes

- The node expects overhead camera image topics to be available.
- Some model artifacts may have their own usage constraints independent from source code license.



## Generación de la base de datos LINEMOD

La base de datos de templates LINEMOD se genera offline y **NO se versiona** 
(ver `.gitignore`).

Para generar la base de datos:

```bash
python3 ros_parol/src/parol6_vision/scripts/train_linemod_templates.py \
    --stl ros_parol/src/parol6_gazebo/meshes/mars2020tube.stl \
    --out ros_parol/src/parol6_vision/models/linemod \
    --distance 0.72

## Evaluación de los estimadores de pose (ICP / D2CO)

Protocolo, dataset de evaluación (par estéreo + depth GT + tubos en la huella
agarrable) y harness común en `docs/pose_evaluation.md`. Scripts:

- `compute_grasp_footprint.py`: huella de agarre por FK → `config/grasp_footprint_cells.json`.
- `cache_yolo_detections.py`: detecciones YOLO offline → `yolo_detections.json`.
- `eval_pose_dataset.py`: evaluación por instancia (ADD-S, radial/lateral/eje, cobertura).
- `compare_pose_results.py`: tabla comparativa emparejada de varias corridas.
- `parol6_vision/stereo_utils.py`: núcleo SGBM compartido por `stereo_pointcloud_node.py` y la evaluación offline.
