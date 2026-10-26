# Atribución de fallos — evaluación anterior (base antigua, config A, detector antiguo), split test

## ICP: cadena completa (SGBM + YOLO), escalera ideal → real

| causa | n | % de 247 |
|---|---|---|
| 1 detección (YOLO no encuentra el tubo) | 197 | 79.8 |
| 2 segmentación/registro (falla con profundidad real y caja GT) | 26 | 10.5 |
| 3 reconstrucción estéreo (acierta con profundidad real) | 20 | 8.1 |
| 4 caja de YOLO (acierta con caja GT) | 1 | 0.4 |
| 5 acierto de la cadena completa | 3 | 1.2 |

Subtipos:

| subtipo | n | % de 247 |
|---|---|---|
| SGBM: eje < 10°, |radial| ≥ umbral | 2 | 0.8 |
| SGBM: eje ≥ 10° | 17 | 6.9 |
| pose con eje ≥ 10° | 19 | 7.7 |
| sin pose (rechazada) | 7 | 2.8 |
| sin pose con SGBM | 1 | 0.4 |

No monótonas (aciertan con SGBM + YOLO pero fallan en algún peldaño ideal): 5.

SGBM + caja GT, poses con eje < 10° (n=48): error radial mediano +3.1 mm, media +19.8 mm, |radial| < umbral en 47.9 %.

Puntos del cluster tras la limpieza (profundidad real, caja GT): mediana 480 en aciertos, 462 en fallos con pose.

Causa por visibilidad:

| visibilidad | n | 1 detección (YOLO no encuentra el tubo) | 2 segmentación/registro (falla con profundidad real y caja GT) | 3 reconstrucción estéreo (acierta con profundidad real) | 4 caja de YOLO (acierta con caja GT) | 5 acierto de la cadena completa |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 48 | 48 | 0 | 0 | 0 | 0 |
| 0.25–0.50 | 31 | 31 | 0 | 0 | 0 | 0 |
| 0.50–0.75 | 30 | 30 | 0 | 0 | 0 | 0 |
| 0.75–0.90 | 12 | 11 | 0 | 1 | 0 | 0 |
| 0.90–1.00 | 126 | 77 | 26 | 19 | 1 | 3 |

## d2co_asis_gtbbox

| resultado | n | % de 247 |
|---|---|---|
| aceptada, falla: eje | 1 | 0.4 |
| aceptada, falla: lateral+eje | 1 | 0.4 |
| aceptada, falla: profundidad | 2 | 0.8 |
| aceptada, falla: profundidad+eje | 25 | 10.1 |
| aceptada, falla: profundidad+lateral | 7 | 2.8 |
| aceptada, falla: profundidad+lateral+eje | 96 | 38.9 |
| acierto | 2 | 0.8 |
| rechazada (avg_dist > umbral) | 74 | 30.0 |
| sin candidato (todas fuera de la banda de profundidad) | 39 | 15.8 |

Poses publicadas (n=134): |radial| < umbral 2.2 %, lateral < umbral 22.4 %, eje < 10° 7.5 %; radial mediano +434 mm (sobreestima la profundidad en 78.4 %).

## d2co_relaxed_gtbbox

| resultado | n | % de 247 |
|---|---|---|
| aceptada, falla: eje | 1 | 0.4 |
| aceptada, falla: lateral+eje | 3 | 1.2 |
| aceptada, falla: profundidad | 5 | 2.0 |
| aceptada, falla: profundidad+eje | 41 | 16.6 |
| aceptada, falla: profundidad+lateral | 14 | 5.7 |
| aceptada, falla: profundidad+lateral+eje | 137 | 55.5 |
| acierto | 2 | 0.8 |
| rechazada (avg_dist > umbral) | 2 | 0.8 |
| sin candidato (todas fuera de la banda de profundidad) | 42 | 17.0 |

Poses publicadas (n=203): |radial| < umbral 2.5 %, lateral < umbral 24.1 %, eje < 10° 9.9 %; radial mediano +314 mm (sobreestima la profundidad en 75.9 %).
