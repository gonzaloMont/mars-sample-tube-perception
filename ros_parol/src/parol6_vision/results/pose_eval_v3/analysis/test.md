# Fase 3 — cifra final en test (222 instancias, una sola corrida por configuración)

ICP: config B (defaults del nodo) + estéreo `wls_lambda` 500 (elegidas en val). D2CO: nodo de la rama `d2co` (45743e7). Detector: best.pt actual. Umbral ADD-S 18.69 mm.

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT | 222 | 98.6 | **84.2** (78.9–88.4) | 85.4 | 2.1 | 0.6 | -6.9 | 2.3 | 0.1 | 371 |
| ICP, SGBM, caja GT | 222 | 86.5 | **23.0** (17.9–28.9) | 26.6 | 43.5 | 39.0 | -1.5 | 40.0 | 22.9 | 452 |
| ICP, profundidad real, caja YOLO | 222 | 83.8 | **80.2** (74.4–84.9) | 95.7 | 2.1 | 0.5 | -0.7 | 2.2 | 0.1 | 355 |
| ICP, SGBM, caja YOLO (cadena completa) | 222 | 73.4 | **23.0** (17.9–28.9) | 31.3 | 37.5 | 31.7 | +8.9 | 37.1 | 15.6 | 476 |
| D2CO con umbral, caja GT | 222 | 48.6 | **0.5** (0.1–2.5) | 0.9 | 430.8 | 446.3 | +798.9 | 37.0 | 45.1 | 1994 |
| D2CO sin umbral, caja GT | 222 | 82.9 | **0.9** (0.2–3.2) | 1.1 | 311.8 | 342.9 | +542.2 | 35.0 | 41.0 | 2577 |
| D2CO con umbral, caja YOLO | 222 | 38.3 | **0.5** (0.1–2.5) | 1.2 | 254.7 | 269.3 | +563.7 | 30.0 | 45.4 | 2519 |

## Estratos (recall ADD-S sobre el total, IC 95 % Wilson)

ICP:

| estrato | n | profundidad real, caja GT: recall % (IC) | SGBM, caja GT: recall % (IC) | profundidad real, caja YOLO: recall % (IC) | SGBM, caja YOLO (cadena completa): recall % (IC) |
|---|---|---|---|---|---|
| visib 0.00–0.25 | 19 | 26.3 (12–49) | 0.0 (0–17) | 21.1 (9–43) | 0.0 (0–17) |
| visib 0.25–0.50 | 37 | 59.5 (43–74) | 8.1 (3–21) | 43.2 (29–59) | 5.4 (1–18) |
| visib 0.50–0.75 | 20 | 80.0 (58–92) | 15.0 (5–36) | 75.0 (53–89) | 20.0 (8–42) |
| visib 0.75–0.90 | 12 | 100.0 (76–100) | 8.3 (1–35) | 100.0 (76–100) | 16.7 (5–45) |
| visib 0.90–1.00 | 134 | 98.5 (95–100) | 32.8 (25–41) | 97.8 (94–99) | 32.1 (25–40) |
| modo buried | 54 | 87.0 (76–94) | 20.4 (12–33) | 85.2 (73–92) | 22.2 (13–35) |
| modo grounded | 168 | 83.3 (77–88) | 23.8 (18–31) | 78.6 (72–84) | 23.2 (17–30) |
| brazo home | 54 | 92.6 (82–97) | 24.1 (15–37) | 92.6 (82–97) | 25.9 (16–39) |
| brazo random | 168 | 81.5 (75–87) | 22.6 (17–30) | 76.2 (69–82) | 22.0 (16–29) |

D2CO:

| estrato | n | con umbral, caja GT: recall % (IC) | sin umbral, caja GT: recall % (IC) | con umbral, caja YOLO: recall % (IC) |
|---|---|---|---|---|
| visib 0.00–0.25 | 19 | 0.0 (0–17) | 0.0 (0–17) | 0.0 (0–17) |
| visib 0.25–0.50 | 37 | 0.0 (0–9) | 0.0 (0–9) | 0.0 (0–9) |
| visib 0.50–0.75 | 20 | 0.0 (0–16) | 0.0 (0–16) | 0.0 (0–16) |
| visib 0.75–0.90 | 12 | 0.0 (0–24) | 0.0 (0–24) | 0.0 (0–24) |
| visib 0.90–1.00 | 134 | 0.7 (0–4) | 1.5 (0–5) | 0.7 (0–4) |
| modo buried | 54 | 1.9 (0–10) | 1.9 (0–10) | 1.9 (0–10) |
| modo grounded | 168 | 0.0 (0–2) | 0.6 (0–3) | 0.0 (0–2) |
| brazo home | 54 | 0.0 (0–7) | 0.0 (0–7) | 0.0 (0–7) |
| brazo random | 168 | 0.6 (0–3) | 1.2 (0–4) | 0.6 (0–3) |

## Comparaciones emparejadas en test

| comparación | n | recall 1.ª % | recall 2.ª % | diferencia (pp) | IC 95 % | solo 1.ª | solo 2.ª | p McNemar |
|---|---|---|---|---|---|---|---|---|
| ICP, profundidad real, caja GT → ICP, SGBM, caja GT | 222 | 84.2 | 23.0 | -61.3 | -68.0 … -54.6 | 138 | 2 | 1.4e-38 |
| ICP, profundidad real, caja GT → ICP, profundidad real, caja YOLO | 222 | 84.2 | 80.2 | -4.1 | -7.6 … -0.5 | 13 | 4 | 0.049 |
| ICP, SGBM, caja GT → ICP, SGBM, caja YOLO (cadena completa) | 222 | 23.0 | 23.0 | +0.0 | -4.2 … +4.1 | 11 | 11 | 1 |
| ICP, profundidad real, caja YOLO → ICP, SGBM, caja YOLO (cadena completa) | 222 | 80.2 | 23.0 | -57.2 | -63.2 … -50.9 | 127 | 0 | 1.2e-38 |
| ICP, SGBM, caja GT → D2CO con umbral, caja GT | 222 | 23.0 | 0.5 | -22.5 | -28.2 … -16.9 | 51 | 1 | 2.4e-14 |
| ICP, SGBM, caja GT → D2CO sin umbral, caja GT | 222 | 23.0 | 0.9 | -22.1 | -27.7 … -16.4 | 50 | 1 | 4.6e-14 |
| ICP, SGBM, caja YOLO (cadena completa) → D2CO con umbral, caja YOLO | 222 | 23.0 | 0.5 | -22.5 | -28.3 … -17.0 | 51 | 1 | 2.4e-14 |
| D2CO con umbral, caja GT → D2CO sin umbral, caja GT | 222 | 0.5 | 0.9 | +0.5 | +0.0 … +1.4 | 0 | 1 | 1 |

## Coherencia D2CO con/sin umbral

La corrida sin umbral publica 184 poses; 115 tienen avg_dist ≤ 4.5 (lo que el nodo con umbral aceptaría). La corrida con umbral aceptó 108; coinciden en 104. En las 104 instancias con pose en ambas, |Δz| mediana 0.0 mm, máximo 0 mm: cuando las dos corridas publican, la pose es idéntica. Pero 15 instancias cambian de resultado entre corridas (no por el umbral: el avg_dist final es otro), compatible con el atajo de tracking del nodo (`tracking_fast_dist` / `full_search_every`), que depende del contador de ciclos y este avanza también con el timer. avg_dist de las poses sin umbral: mediana 3.97, p90 7.03.
