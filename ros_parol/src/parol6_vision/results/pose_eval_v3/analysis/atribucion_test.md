# Atribución de fallos — test v3 (config B, estéreo wls_lambda 500, detector nuevo)

## ICP: cadena completa (SGBM + YOLO), escalera ideal → real

| causa | n | % de 222 |
|---|---|---|
| 1 detección (YOLO no encuentra el tubo) | 35 | 15.8 |
| 2 segmentación/registro (falla con profundidad real y caja GT) | 12 | 5.4 |
| 3 reconstrucción estéreo (acierta con profundidad real) | 126 | 56.8 |
| 4 caja de YOLO (acierta con caja GT) | 9 | 4.1 |
| 5 acierto de la cadena completa | 40 | 18.0 |

Subtipos:

| subtipo | n | % de 222 |
|---|---|---|
| SGBM: eje < 10°, otro | 3 | 1.4 |
| SGBM: eje < 10°, |radial| ≥ umbral | 13 | 5.9 |
| SGBM: eje ≥ 10° | 92 | 41.4 |
| eje < 10°, error de traslación | 6 | 2.7 |
| pose con eje ≥ 10° | 5 | 2.3 |
| sin pose (rechazada) | 1 | 0.5 |
| sin pose con SGBM | 18 | 8.1 |

No monótonas (aciertan con SGBM + YOLO pero fallan en algún peldaño ideal): 11.

SGBM + caja GT, poses con eje < 10° (n=58): error radial mediano -0.5 mm, media +5.4 mm, |radial| < umbral en 58.6 %.

Puntos del cluster tras la limpieza (profundidad real, caja GT): mediana 468 en aciertos, 175 en fallos con pose.

Causa por visibilidad:

| visibilidad | n | 1 detección (YOLO no encuentra el tubo) | 2 segmentación/registro (falla con profundidad real y caja GT) | 3 reconstrucción estéreo (acierta con profundidad real) | 4 caja de YOLO (acierta con caja GT) | 5 acierto de la cadena completa |
|---|---|---|---|---|---|---|
| 0.00–0.25 | 19 | 13 | 4 | 2 | 0 | 0 |
| 0.25–0.50 | 37 | 18 | 4 | 13 | 0 | 2 |
| 0.50–0.75 | 20 | 4 | 2 | 11 | 1 | 2 |
| 0.75–0.90 | 12 | 0 | 0 | 11 | 0 | 1 |
| 0.90–1.00 | 134 | 0 | 2 | 89 | 8 | 35 |

## d2co_gt_umbral

| resultado | n | % de 222 |
|---|---|---|
| aceptada, falla: profundidad | 2 | 0.9 |
| aceptada, falla: profundidad+eje | 23 | 10.4 |
| aceptada, falla: profundidad+lateral | 3 | 1.4 |
| aceptada, falla: profundidad+lateral+eje | 79 | 35.6 |
| acierto | 1 | 0.5 |
| rechazada (avg_dist > umbral) | 70 | 31.5 |
| sin candidato (todas fuera de la banda de profundidad) | 44 | 19.8 |

Poses publicadas (n=108): |radial| < umbral 0.9 %, lateral < umbral 24.1 %, eje < 10° 5.6 %; radial mediano +446 mm (sobreestima la profundidad en 86.1 %).

## d2co_gt_sinumbral

| resultado | n | % de 222 |
|---|---|---|
| aceptada, falla: profundidad | 5 | 2.3 |
| aceptada, falla: profundidad+eje | 44 | 19.8 |
| aceptada, falla: profundidad+lateral | 9 | 4.1 |
| aceptada, falla: profundidad+lateral+eje | 124 | 55.9 |
| acierto | 2 | 0.9 |
| rechazada (avg_dist > umbral) | 2 | 0.9 |
| sin candidato (todas fuera de la banda de profundidad) | 36 | 16.2 |

Poses publicadas (n=184): |radial| < umbral 1.1 %, lateral < umbral 27.2 %, eje < 10° 8.2 %; radial mediano +285 mm (sobreestima la profundidad en 75.5 %).

## d2co_yolo_umbral

| resultado | n | % de 222 |
|---|---|---|
| aceptada, falla: eje | 1 | 0.5 |
| aceptada, falla: lateral+eje | 2 | 0.9 |
| aceptada, falla: profundidad+eje | 21 | 9.5 |
| aceptada, falla: profundidad+lateral | 2 | 0.9 |
| aceptada, falla: profundidad+lateral+eje | 58 | 26.1 |
| acierto | 1 | 0.5 |
| rechazada (avg_dist > umbral) | 73 | 32.9 |
| sin candidato (todas fuera de la banda de profundidad) | 29 | 13.1 |
| sin detección YOLO | 35 | 15.8 |

Poses publicadas (n=85): |radial| < umbral 4.7 %, lateral < umbral 27.1 %, eje < 10° 3.5 %; radial mediano +254 mm (sobreestima la profundidad en 76.5 %).
