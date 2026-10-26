# Fase 1 — ajuste en train (365 instancias, bbox GT visible salvo indicación)

Corridas sin rechazo; configuraciones reconstruidas aplicando el rechazo del nodo a posteriori (`analysis/fase1_train.py`). Umbral ADD-S 18.69 mm.


## Profundidad real, bbox GT

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) | 365 | 98.1 | **47.7** (42.6–52.8) | 48.6 | 20.0 | 16.0 | -10.3 | 20.5 | 16.9 | 567 |
| A+rp (yaml + remove_plane) | 365 | 96.7 | **81.4** (77.1–85.0) | 84.1 | 2.3 | 1.0 | -12.2 | 2.8 | 0.3 | 380 |
| B con min_fitness 0.20 | 365 | 98.1 | **84.4** (80.3–87.7) | 86.0 | 2.3 | 1.0 | -17.0 | 2.7 | 0.3 | 415 |
| B (defaults del nodo) | 365 | 96.4 | **84.1** (80.0–87.5) | 87.2 | 2.3 | 0.9 | -16.2 | 2.6 | 0.2 | 415 |
| remove_plane T, plane 4 mm, SIN rechazo | 365 | 98.1 | **84.4** (80.3–87.7) | 86.0 | 2.3 | 1.0 | -17.0 | 2.7 | 0.3 | 415 |

Comparaciones emparejadas (mismas instancias; IC 95 % bootstrap por imagen, McNemar exacto):

| A → B | n | recall A % | recall B % | B − A (pp) | IC 95 % | solo A | solo B | p McNemar |
|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) → A+rp (yaml + remove_plane) | 365 | 47.7 | 81.4 | +33.7 | +28.5 … +38.5 | 5 | 128 | 6.1e-32 |
| A+rp (yaml + remove_plane) → B con min_fitness 0.20 | 365 | 81.4 | 84.4 | +3.0 | +1.3 … +5.1 | 1 | 12 | 0.0034 |
| B con min_fitness 0.20 → B (defaults del nodo) | 365 | 84.4 | 84.1 | -0.3 | -0.8 … +0.0 | 1 | 0 | 1 |
| A (yaml anterior) → B (defaults del nodo) | 365 | 47.7 | 84.1 | +36.4 | +31.2 … +41.2 | 2 | 135 | 1.1e-37 |

Curva de rechazo (remove_plane T, plane 4 mm, max_rmse 6 mm):

| min_fitness | cobertura % | recall total % | recall aceptadas % |
|---|---|---|---|
| 0.00 | 98.1 | 84.4 | 86.0 |
| 0.20 | 98.1 | 84.4 | 86.0 |
| 0.30 | 97.3 | 84.4 | 86.8 |
| 0.45 | 96.4 | 84.1 | 87.2 |
| 0.60 | 95.6 | 83.6 | 87.4 |
| 0.75 | 91.5 | 80.5 | 88.0 |

Estratos (A frente a B):

| estrato | n | A: recall % (IC) | B: recall % (IC) |
|---|---|---|---|
| visib 0.00–0.25 | 29 | 13.8 (5–31) | 13.8 (5–31) |
| visib 0.25–0.50 | 58 | 41.4 (30–54) | 63.8 (51–75) |
| visib 0.50–0.75 | 48 | 64.6 (50–77) | 89.6 (78–95) |
| visib 0.75–0.90 | 30 | 53.3 (36–70) | 93.3 (79–98) |
| visib 0.90–1.00 | 200 | 49.5 (43–56) | 97.5 (94–99) |
| modo buried | 67 | 37.3 (27–49) | 89.6 (80–95) |
| modo grounded | 298 | 50.0 (44–56) | 82.9 (78–87) |
| brazo home | 84 | 44.0 (34–55) | 90.5 (82–95) |
| brazo random | 281 | 48.8 (43–55) | 82.2 (77–86) |

## SGBM, bbox GT

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) | 365 | 96.2 | **12.3** (9.3–16.1) | 12.8 | 88.8 | 92.5 | +45.9 | 39.9 | 29.9 | 435 |
| A+rp (yaml + remove_plane) | 365 | 86.8 | **14.0** (10.8–17.9) | 16.1 | 75.5 | 72.8 | +21.8 | 50.5 | 30.1 | 381 |
| B con min_fitness 0.20 | 365 | 90.7 | **14.0** (10.8–17.9) | 15.4 | 73.5 | 72.9 | +26.0 | 53.9 | 31.8 | 378 |
| B (defaults del nodo) | 365 | 88.5 | **14.0** (10.8–17.9) | 15.8 | 73.5 | 70.9 | +24.9 | 53.8 | 31.1 | 378 |
| remove_plane T, plane 4 mm, SIN rechazo | 365 | 91.2 | **14.0** (10.8–17.9) | 15.3 | 73.5 | 72.9 | +26.2 | 53.8 | 31.8 | 378 |
| B min_fitness 0.20 + SGBM wls_lambda 500 | 365 | 92.6 | **21.4** (17.5–25.9) | 23.1 | 45.8 | 38.0 | -3.5 | 41.9 | 20.3 | 420 |
| B min_fitness 0.45 + SGBM wls_lambda 500 | 365 | 85.8 | **20.8** (17.0–25.3) | 24.3 | 46.2 | 41.6 | -4.7 | 42.5 | 20.3 | 420 |

Comparaciones emparejadas (mismas instancias; IC 95 % bootstrap por imagen, McNemar exacto):

| A → B | n | recall A % | recall B % | B − A (pp) | IC 95 % | solo A | solo B | p McNemar |
|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) → A+rp (yaml + remove_plane) | 365 | 12.3 | 14.0 | +1.6 | -1.9 … +5.1 | 19 | 25 | 0.45 |
| A+rp (yaml + remove_plane) → B con min_fitness 0.20 | 365 | 14.0 | 14.0 | +0.0 | -2.4 … +2.5 | 11 | 11 | 1 |
| B con min_fitness 0.20 → B (defaults del nodo) | 365 | 14.0 | 14.0 | +0.0 | +0.0 … +0.0 | 0 | 0 | 1 |
| A (yaml anterior) → B (defaults del nodo) | 365 | 12.3 | 14.0 | +1.6 | -1.8 … +5.0 | 17 | 23 | 0.43 |

Curva de rechazo (remove_plane T, plane 4 mm, max_rmse 6 mm):

| min_fitness | cobertura % | recall total % | recall aceptadas % |
|---|---|---|---|
| 0.00 | 91.2 | 14.0 | 15.3 |
| 0.20 | 90.7 | 14.0 | 15.4 |
| 0.30 | 90.4 | 14.0 | 15.5 |
| 0.45 | 88.5 | 14.0 | 15.8 |
| 0.60 | 85.8 | 13.4 | 15.7 |
| 0.75 | 78.1 | 12.3 | 15.8 |

Estratos (A frente a B):

| estrato | n | A: recall % (IC) | B: recall % (IC) |
|---|---|---|---|
| visib 0.00–0.25 | 29 | 0.0 (0–12) | 0.0 (0–12) |
| visib 0.25–0.50 | 58 | 3.4 (1–12) | 5.2 (2–14) |
| visib 0.50–0.75 | 48 | 14.6 (7–27) | 14.6 (7–27) |
| visib 0.75–0.90 | 30 | 3.3 (1–17) | 6.7 (2–21) |
| visib 0.90–1.00 | 200 | 17.5 (13–23) | 19.5 (15–26) |
| modo buried | 67 | 10.4 (5–20) | 11.9 (6–22) |
| modo grounded | 298 | 12.8 (9–17) | 14.4 (11–19) |
| brazo home | 84 | 11.9 (7–21) | 15.5 (9–25) |
| brazo random | 281 | 12.5 (9–17) | 13.5 (10–18) |

## Profundidad real, bbox YOLO

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) | 365 | 82.2 | **44.4** (39.4–49.5) | 54.0 | 15.1 | 13.8 | -2.4 | 17.0 | 8.4 | 534 |
| B con min_fitness 0.20 | 365 | 84.1 | **78.9** (74.4–82.8) | 93.8 | 2.2 | 0.7 | -4.7 | 2.3 | 0.1 | 373 |
| B (defaults del nodo) | 365 | 83.6 | **78.6** (74.1–82.5) | 94.1 | 2.2 | 0.7 | -4.5 | 2.3 | 0.1 | 373 |
| remove_plane T, plane 4 mm, SIN rechazo | 365 | 84.1 | **78.9** (74.4–82.8) | 93.8 | 2.2 | 0.7 | -4.7 | 2.3 | 0.1 | 373 |

Comparaciones emparejadas (mismas instancias; IC 95 % bootstrap por imagen, McNemar exacto):

| A → B | n | recall A % | recall B % | B − A (pp) | IC 95 % | solo A | solo B | p McNemar |
|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) → B con min_fitness 0.20 | 365 | 44.4 | 78.9 | +34.5 | +29.5 … +39.3 | 1 | 127 | 7.6e-37 |
| B con min_fitness 0.20 → B (defaults del nodo) | 365 | 78.9 | 78.6 | -0.3 | -0.8 … +0.0 | 1 | 0 | 1 |

Curva de rechazo (remove_plane T, plane 4 mm, max_rmse 6 mm):

| min_fitness | cobertura % | recall total % | recall aceptadas % |
|---|---|---|---|
| 0.00 | 84.1 | 78.9 | 93.8 |
| 0.20 | 84.1 | 78.9 | 93.8 |
| 0.30 | 84.1 | 78.9 | 93.8 |
| 0.45 | 83.6 | 78.6 | 94.1 |
| 0.60 | 83.0 | 78.4 | 94.4 |
| 0.75 | 80.3 | 75.9 | 94.5 |

## SGBM, bbox YOLO

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) | 365 | 80.3 | **14.2** (11.0–18.2) | 17.7 | 71.9 | 75.0 | +60.9 | 36.6 | 23.2 | 472 |
| B con min_fitness 0.20 | 365 | 75.9 | **18.1** (14.5–22.4) | 23.8 | 60.8 | 55.7 | +38.2 | 45.3 | 23.3 | 385 |
| B (defaults del nodo) | 365 | 74.2 | **18.1** (14.5–22.4) | 24.4 | 60.2 | 55.7 | +36.0 | 44.8 | 23.0 | 385 |
| remove_plane T, plane 4 mm, SIN rechazo | 365 | 76.4 | **18.1** (14.5–22.4) | 23.7 | 60.8 | 55.7 | +38.2 | 45.3 | 23.6 | 385 |

Comparaciones emparejadas (mismas instancias; IC 95 % bootstrap por imagen, McNemar exacto):

| A → B | n | recall A % | recall B % | B − A (pp) | IC 95 % | solo A | solo B | p McNemar |
|---|---|---|---|---|---|---|---|---|
| A (yaml anterior) → B con min_fitness 0.20 | 365 | 14.2 | 18.1 | +3.8 | +0.5 … +7.1 | 14 | 28 | 0.044 |
| B con min_fitness 0.20 → B (defaults del nodo) | 365 | 18.1 | 18.1 | +0.0 | +0.0 … +0.0 | 0 | 0 | 1 |

Curva de rechazo (remove_plane T, plane 4 mm, max_rmse 6 mm):

| min_fitness | cobertura % | recall total % | recall aceptadas % |
|---|---|---|---|
| 0.00 | 76.4 | 18.1 | 23.7 |
| 0.20 | 75.9 | 18.1 | 23.8 |
| 0.30 | 75.9 | 18.1 | 23.8 |
| 0.45 | 74.2 | 18.1 | 24.4 |
| 0.60 | 72.3 | 17.5 | 24.2 |
| 0.75 | 65.5 | 17.3 | 26.4 |

## Repetibilidad

Misma corrida dos veces (profundidad real, bbox GT, remove_plane T, plane 4 mm, sin rechazo): 79 de 365 instancias con ADD-S distinto (> 0.01 mm); recall 84.4 % frente a 84.1 % (solo en la 1.ª: 3, solo en la 2.ª: 2).

Sin `remove_plane` (sin RANSAC; primeras 60 instancias de train, repetidas): 31 con ADD-S distinto; |ΔADD-S| mediana 0.02 mm, p90 1.81 mm, máximo 10.6 mm; 1 cambio(s) de acierto. El harness no es determinista bit a bit ni sin RANSAC (causa no aislada; previsiblemente el paralelismo de Open3D); el efecto sobre el recall es de décimas de punto, muy por debajo del IC muestral.

