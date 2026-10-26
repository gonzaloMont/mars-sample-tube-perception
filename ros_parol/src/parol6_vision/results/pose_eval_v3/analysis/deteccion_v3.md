Dataset: /home/gonzalo/parol6_pose_eval_dataset_v3

| detector | partición | instancias | recall IoU≥0.3 % (IC) | recall IoU≥0.5 % | imágenes | detecciones | FP (IoU<0.3) | FP/imagen | top-1 sobre un tubo % |
|---|---|---|---|---|---|---|---|---|---|
| nuevo (a07ff5) | train | 365 | 84.7 (80.6–88.0) | 82.7 | 300 | 322 | 10 | 0.03 | 86.7 |
| nuevo (a07ff5) | val | 142 | 88.0 (81.7–92.4) | 85.2 | 120 | 133 | 7 | 0.06 | 87.5 |
| nuevo (a07ff5) | test | 222 | 84.2 (78.9–88.4) | 84.2 | 180 | 194 | 6 | 0.03 | 86.1 |
| nuevo (a07ff5) | todas | 729 | 85.2 (82.4–87.6) | 83.7 | 600 | 649 | 23 | 0.04 | 86.7 |
| antiguo (15ac096) | train | 365 | 28.8 (24.4–33.6) | 19.5 | 300 | 307 | 198 | 0.66 | 30.7 |
| antiguo (15ac096) | val | 142 | 38.0 (30.5–46.2) | 28.2 | 120 | 151 | 97 | 0.81 | 38.3 |
| antiguo (15ac096) | test | 222 | 34.7 (28.7–41.2) | 24.3 | 180 | 201 | 121 | 0.67 | 36.1 |
| antiguo (15ac096) | todas | 729 | 32.4 (29.1–35.9) | 22.6 | 600 | 659 | 416 | 0.69 | 33.8 |

Recall IoU≥0.3 por visibilidad (todas las particiones):

| detector | visib 0.00–0.25 | visib 0.25–0.50 | visib 0.50–0.75 | visib 0.75–0.90 | visib 0.90–1.00 |
|---|---|---|---|---|---|
| nuevo (a07ff5) | 28.6 (n=56) | 54.0 (n=113) | 86.0 (n=93) | 95.7 (n=46) | 99.8 (n=421) |
| antiguo (15ac096) | 0.0 (n=56) | 0.0 (n=113) | 5.4 (n=93) | 19.6 (n=46) | 52.7 (n=421) |
