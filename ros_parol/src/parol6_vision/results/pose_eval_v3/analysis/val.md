# Fase 2 — confirmación en val (142 instancias)

Configuraciones con su rechazo real. Umbral ADD-S 18.69 mm.

| corrida | n | cobertura % | recall ADD-S % (IC 95 % Wilson) | recall aceptadas % | ADD-S med mm | \|radial\| med mm | sesgo radial medio mm | lateral med mm | eje med ° | latencia med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Profundidad real, caja GT — A | 142 | 93.0 | **55.6** (47.4–63.5) | 59.8 | 12.5 | 9.3 | -6.1 | 21.3 | 3.9 | 484 |
| Profundidad real, caja GT — B | 142 | 99.3 | **88.0** (81.7–92.4) | 88.7 | 2.1 | 0.6 | -6.0 | 2.4 | 0.1 | 368 |
| SGBM, caja GT — A | 142 | 95.1 | **16.2** (11.0–23.1) | 17.0 | 77.8 | 75.1 | +36.9 | 38.8 | 21.5 | 483 |
| SGBM, caja GT — B | 142 | 85.2 | **16.2** (11.0–23.1) | 19.0 | 59.6 | 51.3 | +22.9 | 48.5 | 26.5 | 399 |
| Profundidad real, caja YOLO — A | 142 | 81.7 | **46.5** (38.5–54.7) | 56.9 | 13.2 | 9.9 | -3.5 | 18.6 | 9.4 | 473 |
| Profundidad real, caja YOLO — B | 142 | 87.3 | **83.1** (76.1–88.4) | 95.2 | 2.0 | 0.5 | -1.2 | 2.2 | 0.1 | 389 |
| SGBM, caja YOLO (cadena completa) — A | 142 | 87.3 | **16.9** (11.6–23.9) | 19.4 | 66.5 | 65.3 | +51.7 | 37.1 | 17.5 | 480 |
| SGBM, caja YOLO (cadena completa) — B | 142 | 78.9 | **17.6** (12.2–24.7) | 22.3 | 48.4 | 41.4 | +39.8 | 44.8 | 19.9 | 429 |
| SGBM wls_lambda 500, caja GT — B | 142 | 88.7 | **31.0** (24.0–39.0) | 34.9 | 32.9 | 24.2 | -7.5 | 34.0 | 14.8 | 505 |
| SGBM wls_lambda 500, caja YOLO — B | 142 | 78.2 | **23.9** (17.7–31.6) | 30.6 | 33.5 | 25.4 | +19.4 | 32.6 | 15.0 | 453 |

## A → B (emparejado)

| comparación | n | recall 1.ª % | recall 2.ª % | diferencia (pp) | IC 95 % | solo 1.ª | solo 2.ª | p McNemar |
|---|---|---|---|---|---|---|---|---|
| Profundidad real, caja GT | 142 | 55.6 | 88.0 | +32.4 | +23.9 … +40.7 | 3 | 49 | 1e-11 |
| SGBM, caja GT | 142 | 16.2 | 16.2 | +0.0 | -5.1 … +5.1 | 7 | 7 | 1 |
| Profundidad real, caja YOLO | 142 | 46.5 | 83.1 | +36.6 | +29.0 … +44.4 | 0 | 52 | 4.4e-16 |
| SGBM, caja YOLO (cadena completa) | 142 | 16.9 | 17.6 | +0.7 | -5.3 … +6.5 | 9 | 10 | 1 |

**Decisión A/B según el criterio fijado:** B (diferencias ≥ 0 en los cuatro: True; IC en profundidad real + caja GT excluye 0: True).

## B → B + wls_lambda 500 (emparejado)

| comparación | n | recall 1.ª % | recall 2.ª % | diferencia (pp) | IC 95 % | solo 1.ª | solo 2.ª | p McNemar |
|---|---|---|---|---|---|---|---|---|
| SGBM, caja GT | 142 | 16.2 | 31.0 | +14.8 | +6.4 … +23.1 | 8 | 29 | 0.00075 |
| SGBM, caja YOLO (cadena completa) | 142 | 17.6 | 23.9 | +6.3 | -0.7 … +13.5 | 11 | 20 | 0.15 |

**Decisión wls_lambda según el criterio fijado:** adoptar 500 (mejora en ambos: True; IC en SGBM + caja GT excluye 0: True).

## Estratos, profundidad real + caja GT

| estrato | n | A: recall % (IC) | B: recall % (IC) |
|---|---|---|---|
| visib 0.00–0.25 | 8 | 12.5 (2–47) | 0.0 (0–32) |
| visib 0.25–0.50 | 18 | 77.8 (55–91) | 72.2 (49–88) |
| visib 0.50–0.75 | 25 | 72.0 (52–86) | 88.0 (70–96) |
| visib 0.75–0.90 | 4 | 25.0 (5–70) | 100.0 (51–100) |
| visib 0.90–1.00 | 87 | 51.7 (41–62) | 98.9 (94–100) |
| modo buried | 18 | 50.0 (29–71) | 94.4 (74–99) |
| modo grounded | 124 | 56.5 (48–65) | 87.1 (80–92) |
| brazo home | 35 | 51.4 (36–67) | 91.4 (78–97) |
| brazo random | 107 | 57.0 (48–66) | 86.9 (79–92) |

## Estratos, cadena completa (SGBM + YOLO)

| estrato | n | A: recall % (IC) | B: recall % (IC) |
|---|---|---|---|
| visib 0.00–0.25 | 8 | 0.0 (0–32) | 0.0 (0–32) |
| visib 0.25–0.50 | 18 | 5.6 (1–26) | 11.1 (3–33) |
| visib 0.50–0.75 | 25 | 16.0 (6–35) | 16.0 (6–35) |
| visib 0.75–0.90 | 4 | 0.0 (0–49) | 0.0 (0–49) |
| visib 0.90–1.00 | 87 | 21.8 (14–32) | 21.8 (14–32) |
| modo buried | 18 | 11.1 (3–33) | 11.1 (3–33) |
| modo grounded | 124 | 17.7 (12–25) | 18.5 (13–26) |
| brazo home | 35 | 20.0 (10–36) | 17.1 (8–33) |
| brazo random | 107 | 15.9 (10–24) | 17.8 (12–26) |
