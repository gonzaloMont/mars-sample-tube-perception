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
