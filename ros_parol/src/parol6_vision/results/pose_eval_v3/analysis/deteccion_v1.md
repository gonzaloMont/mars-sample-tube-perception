Dataset: /home/gonzalo/parol6_pose_eval_dataset

| detector | partición | instancias | recall IoU≥0.3 % (IC) | recall IoU≥0.5 % | imágenes | detecciones | FP (IoU<0.3) | FP/imagen | top-1 sobre un tubo % |
|---|---|---|---|---|---|---|---|---|---|
| nuevo (a07ff5) | train | 345 | 86.1 (82.0–89.3) | 84.9 | 284 | 315 | 11 | 0.04 | 86.6 |
| nuevo (a07ff5) | val | 137 | 89.8 (83.6–93.8) | 86.1 | 115 | 132 | 5 | 0.04 | 91.3 |
| nuevo (a07ff5) | test | 247 | 76.9 (71.3–81.7) | 75.7 | 201 | 201 | 7 | 0.03 | 78.6 |
| nuevo (a07ff5) | todas | 729 | 83.7 (80.8–86.2) | 82.0 | 600 | 648 | 23 | 0.04 | 84.8 |
| antiguo (15ac096) | train | 345 | 29.9 (25.3–34.9) | 17.4 | 284 | 271 | 167 | 0.59 | 31.0 |
| antiguo (15ac096) | val | 137 | 31.4 (24.2–39.6) | 23.4 | 115 | 119 | 76 | 0.66 | 32.2 |
| antiguo (15ac096) | test | 247 | 20.2 (15.7–25.7) | 11.7 | 201 | 161 | 110 | 0.55 | 21.4 |
| antiguo (15ac096) | todas | 729 | 26.9 (23.8–30.2) | 16.6 | 600 | 551 | 353 | 0.59 | 28.0 |

Recall IoU≥0.3 por visibilidad (todas las particiones):

| detector | visib 0.00–0.25 | visib 0.25–0.50 | visib 0.50–0.75 | visib 0.75–0.90 | visib 0.90–1.00 |
|---|---|---|---|---|---|
| nuevo (a07ff5) | 17.6 (n=85) | 65.3 (n=101) | 90.2 (n=102) | 90.9 (n=44) | 100.0 (n=397) |
| antiguo (15ac096) | 0.0 (n=85) | 0.0 (n=101) | 1.0 (n=102) | 22.7 (n=44) | 46.6 (n=397) |
