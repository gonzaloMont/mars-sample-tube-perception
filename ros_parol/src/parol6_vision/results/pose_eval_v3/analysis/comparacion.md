# Comparación con la evaluación anterior

v1 = `~/parol6_pose_eval_dataset` (resultados en `anterior/`), v3 = `~/parol6_pose_eval_dataset_v3`. Config A = yaml anterior (`remove_plane: false`), B = defaults del nodo. Detector antiguo = best.pt de 15ac096, nuevo = best.pt actual. Umbral ADD-S 18.69 mm en todas las corridas.

## 1. Test: cifra final v3 frente a test v1

v1: config A, detector antiguo, estéreo `wls_lambda` 8000 (corridas de `anterior/` filtradas a `split == test`). v3: config B, detector nuevo, estéreo `wls_lambda` 500. Cambian a la vez datos, detector y configuración: la sección 2 los separa.

| escenario | v1 test (todas) | v1 test sin hard | v3 test | v3 − v1 sin hard (pp) | IC 95 % |
|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 44.5 (38.5–50.8), n=247 | 51.2 (44.5–57.8), n=211 | 84.2 (78.9–88.4), n=222 | +33.0 | +24.5 … +41.0 |
| ICP, SGBM, caja GT | 16.6 (12.5–21.7), n=247 | 18.5 (13.8–24.3), n=211 | 23.0 (17.9–28.9), n=222 | +4.5 | -3.2 … +12.1 |
| ICP, profundidad real, caja YOLO | 8.9 (6.0–13.1), n=247 | 10.4 (7.0–15.3), n=211 | 80.2 (74.4–84.9), n=222 | +69.8 | +62.2 … +75.6 |
| ICP, SGBM, caja YOLO (cadena completa) | 3.2 (1.7–6.3), n=247 | 3.3 (1.6–6.7), n=211 | 23.0 (17.9–28.9), n=222 | +19.7 | +13.6 … +25.9 |
| D2CO con umbral, caja GT | 0.8 (0.2–2.9), n=247 | 0.9 (0.3–3.4), n=211 | 0.5 (0.1–2.5), n=222 | -0.5 | -3.0 … +1.7 |
| D2CO sin umbral, caja GT | 0.8 (0.2–2.9), n=247 | 0.9 (0.3–3.4), n=211 | 0.9 (0.2–3.2), n=222 | -0.0 | -2.6 … +2.4 |
| D2CO con umbral, caja YOLO | — | — | 0.5 (0.1–2.5), n=222 | — | — |

Cobertura (fracción con pose publicada), v1 test sin hard frente a v3 test:

| escenario | v1 test sin hard % | v3 test % | diferencia (pp) | IC 95 % |
|---|---|---|---|---|
| ICP, profundidad real, caja GT | 91.9 | 98.6 | +6.7 | +2.8 … +11.3 |
| ICP, SGBM, caja GT | 95.7 | 86.5 | -9.2 | -14.7 … -3.9 |
| ICP, profundidad real, caja YOLO | 17.5 | 83.8 | +66.2 | +58.4 … +72.5 |
| ICP, SGBM, caja YOLO (cadena completa) | 22.7 | 73.4 | +50.7 | +42.0 … +58.1 |
| D2CO con umbral, caja GT | 54.5 | 48.6 | -5.9 | -15.1 … +3.5 |
| D2CO sin umbral, caja GT | 85.8 | 82.9 | -2.9 | -9.7 … +4.0 |

Resumen completo de v3 test:

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 222 | 98.6 | **84.2** (78.9–88.4) | 85.4 | 2.1 | 0.6 | -6.9 | 2.3 | 0.1 | 371 |
| ICP, SGBM, caja GT | 222 | 86.5 | **23.0** (17.9–28.9) | 26.6 | 43.5 | 39.0 | -1.5 | 40.0 | 22.9 | 452 |
| ICP, profundidad real, caja YOLO | 222 | 83.8 | **80.2** (74.4–84.9) | 95.7 | 2.1 | 0.5 | -0.7 | 2.2 | 0.1 | 355 |
| ICP, SGBM, caja YOLO (cadena completa) | 222 | 73.4 | **23.0** (17.9–28.9) | 31.3 | 37.5 | 31.7 | +8.9 | 37.1 | 15.6 | 476 |
| D2CO con umbral, caja GT | 222 | 48.6 | **0.5** (0.1–2.5) | 0.9 | 430.8 | 446.3 | +798.9 | 37.0 | 45.1 | 1994 |
| D2CO sin umbral, caja GT | 222 | 82.9 | **0.9** (0.2–3.2) | 1.1 | 311.8 | 342.9 | +542.2 | 35.0 | 41.0 | 2577 |
| D2CO con umbral, caja YOLO | 222 | 38.3 | **0.5** (0.1–2.5) | 1.2 | 254.7 | 269.3 | +563.7 | 30.0 | 45.4 | 2519 |

Resumen de v1 test sin hard (mismas métricas):

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 211 | 91.9 | **51.2** (44.5–57.8) | 55.7 | 15.3 | 14.0 | -17.9 | 20.3 | 6.8 | 6282 |
| ICP, SGBM, caja GT | 211 | 95.7 | **18.5** (13.8–24.3) | 19.3 | 70.4 | 66.8 | +45.4 | 40.0 | 22.3 | 6385 |
| ICP, profundidad real, caja YOLO | 211 | 17.5 | **10.4** (7.0–15.3) | 59.5 | 2.5 | 1.0 | -29.2 | 7.3 | 0.3 | 12662 |
| ICP, SGBM, caja YOLO (cadena completa) | 211 | 22.7 | **3.3** (1.6–6.7) | 14.6 | 70.0 | 71.7 | +46.1 | 66.6 | 25.9 | 18073 |
| D2CO con umbral, caja GT | 211 | 54.5 | **0.9** (0.3–3.4) | 1.7 | 349.1 | 392.2 | +684.7 | 34.5 | 44.7 | 3053 |
| D2CO sin umbral, caja GT | 211 | 85.8 | **0.9** (0.3–3.4) | 1.1 | 293.6 | 324.7 | +510.3 | 30.7 | 44.6 | 3108 |

Muestras 'hard' de v1 test: 36 instancias (de 247); visibilidad mediana 0.032; aciertos ICP profundidad real + caja GT: 2.

## 2. Descomposición en train+val (ninguna base tiene muestras hard en train+val)

v3 train = corrida sin rechazo con el rechazo de la config aplicado a posteriori; v3 val = corrida real.


### ICP, profundidad real, caja GT

| efecto | recall antes % | recall después % | diferencia (pp) | IC 95 % | tipo |
|---|---|---|---|---|---|
| datos v1 → v3 (config A) | 46.3 (n=482) | 49.9 (n=507) | +3.6 | -2.6 … +9.8 | independiente (Newcombe) |
| config A → B (v3) | 49.9 (n=507) | 85.2 | +35.3 | +31.2 … +39.4 | emparejado, McNemar p=5e-48 |
| config A → B (v1, réplica) | 46.3 (n=482) | 85.5 | +39.2 | +34.7 … +43.8 | emparejado, McNemar p=6.2e-52 |
| datos v1 → v3 (config B) | 85.5 (n=482) | 85.2 (n=507) | -0.3 | -4.7 … +4.2 | independiente (Newcombe) |

Recalls: v1 A 46.3 (41.9–50.7), n=482; v3 A 49.9 (45.6–54.2), n=507; v3 B 85.2 (81.9–88.0), n=507; v1 B 85.5 (82.1–88.3), n=482.

### ICP, SGBM, caja GT

| efecto | recall antes % | recall después % | diferencia (pp) | IC 95 % | tipo |
|---|---|---|---|---|---|
| datos v1 → v3 (config A) | 19.5 (n=482) | 13.4 (n=507) | -6.1 | -10.7 … -1.5 | independiente (Newcombe) |
| config A → B (v3) | 13.4 (n=507) | 14.6 | +1.2 | -1.6 … +4.0 | emparejado, McNemar p=0.5 |
| config A → B (v1, réplica) | 19.5 (n=482) | 23.2 | +3.7 | +0.8 … +6.8 | emparejado, McNemar p=0.02 |
| datos v1 → v3 (config B) | 23.2 (n=482) | 14.6 (n=507) | -8.6 | -13.5 … -3.8 | independiente (Newcombe) |
| estéreo wls_lambda 8000 → 500 (v3, config B) | 14.6 (n=507) | 23.7 | +9.1 | +5.1 … +13.0 | emparejado, McNemar p=1.6e-05 |

Recalls: v1 A 19.5 (16.2–23.3), n=482; v3 A 13.4 (10.7–16.7), n=507; v3 B 14.6 (11.8–17.9), n=507; v1 B 23.2 (19.7–27.2), n=482.

### ICP, profundidad real, caja YOLO

| efecto | recall antes % | recall después % | diferencia (pp) | IC 95 % | tipo |
|---|---|---|---|---|---|
| datos v1 → v3 (config A, detector antiguo) | 10.8 (n=482) | 12.6 (n=507) | +1.8 | -2.2 … +5.9 | independiente (Newcombe) |
| detector antiguo → nuevo (v3, config A) | 12.6 (n=507) | 45.0 | +32.3 | +28.0 … +36.8 | emparejado, McNemar p=3.2e-41 |
| config A → B (v3, detector nuevo) | 45.0 (n=507) | 79.9 | +34.9 | +30.9 … +39.1 | emparejado, McNemar p=4.7e-52 |
| datos v1 → v3 (config B, detector nuevo) | 81.7 (n=482) | 79.9 (n=507) | -1.9 | -6.8 … +3.1 | independiente (Newcombe) |

Recalls: v1 A 10.8 (8.3–13.9), n=482; v3 A 45.0 (40.7–49.3), n=507; v3 B 79.9 (76.2–83.1), n=507; v1 B 81.7 (78.0–84.9), n=482.

### ICP, SGBM, caja YOLO (cadena completa)

| efecto | recall antes % | recall después % | diferencia (pp) | IC 95 % | tipo |
|---|---|---|---|---|---|
| datos v1 → v3 (config A, detector antiguo) | 5.2 (n=482) | 5.1 (n=507) | -0.1 | -2.9 … +2.8 | independiente (Newcombe) |
| detector antiguo → nuevo (v3, config A) | 5.1 (n=507) | 15.0 | +9.9 | +6.9 … +12.9 | emparejado, McNemar p=1e-11 |
| config A → B (v3, detector nuevo) | 15.0 (n=507) | 17.9 | +3.0 | +0.0 … +5.9 | emparejado, McNemar p=0.072 |
| datos v1 → v3 (config B, detector nuevo) | 23.9 (n=482) | 17.9 (n=507) | -5.9 | -11.0 … -0.8 | independiente (Newcombe) |

Recalls: v1 A 5.2 (3.5–7.5), n=482; v3 A 15.0 (12.1–18.4), n=507; v3 B 17.9 (14.9–21.5), n=507; v1 B 23.9 (20.3–27.9), n=482.

## 3. Iluminación aleatoria dentro de v3

v1 tenía la difusa fija del mundo, (0.9, 0.85, 0.8) (`mars_optimized.world`), es decir g = 0.9; el 78 % de las escenas de v3 están más oscuras. Difusa del sol g ~ U(0.55, 1.0) por escena; terciles g < 0.691, 0.691–0.853, > 0.853. Todas las particiones disponibles (train con rechazo a posteriori).

| serie | g bajo | g medio | g alto |
|---|---|---|---|
| detección YOLO nuevo (IoU ≥ 0.3) | 83.6 (78–88), n=244 | 83.6 (78–88), n=244 | 88.4 (84–92), n=241 |
| ICP B, profundidad real, caja GT (train+val+test) | 84.0 (79–88), n=244 | 84.4 (79–88), n=244 | 86.3 (81–90), n=241 |
| ICP B, SGBM 8000 + YOLO (train+val) | 11.4 (7–17), n=166 | 18.9 (14–26), n=169 | 23.3 (18–30), n=172 |
| ICP A, SGBM 8000, caja GT (train+val) | 9.6 (6–15), n=166 | 15.4 (11–22), n=169 | 15.1 (11–21), n=172 |
| ICP B, SGBM 8000, caja GT (train+val) | 12.0 (8–18), n=166 | 18.3 (13–25), n=169 | 13.4 (9–19), n=172 |
| ICP B, SGBM 500, caja GT (train+val) | 21.7 (16–29), n=166 | 23.1 (17–30), n=169 | 26.2 (20–33), n=172 |
| D2CO sin umbral, caja GT (test) | 1.3 (0–7), n=78 | 1.3 (0–7), n=75 | 0.0 (0–5), n=69 |
