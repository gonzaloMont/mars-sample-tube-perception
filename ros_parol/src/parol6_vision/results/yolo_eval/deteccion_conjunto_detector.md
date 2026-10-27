# YOLO26n en su conjunto de datos (`~/parol6_synth_dataset_v3`)

Modelo `models/best.pt`. Fuente: `eval_detector.py` → `deteccion_conjunto_detector.json`.

## Métricas de Ultralytics (`model.val`, imgsz 640, conf 0.001)

| partición | imágenes | precisión | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| val | 200 | 0.989 | 0.914 | 0.967 | 0.835 |
| test | 100 | 0.932 | 0.915 | 0.974 | 0.834 |

Precisión y recall en el punto de máximo F1 (criterio de Ultralytics).

## Punto de operación del nodo en test (conf 0.3, IoU ≥ 0.5)

| subconjunto | imágenes | tubos | TP | FP | FN | precisión | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| tubo en cualquier parte | 63 | 75 | 71 | 3 | 4 | 0.959 | 0.947 | 0.953 |
| zona de trabajo | 23 | 30 | 25 | 3 | 5 | 0.893 | 0.833 | 0.862 |
| sin tubo (tanda general) | 7 | 0 | 0 | 0 | 0 | — | — | — |
| sin tubo (zona de trabajo) | 7 | 0 | 0 | 0 | 0 | — | — | — |
| **total** | 100 | 105 | 96 | 6 | 9 | 0.941 | 0.914 | 0.928 |
