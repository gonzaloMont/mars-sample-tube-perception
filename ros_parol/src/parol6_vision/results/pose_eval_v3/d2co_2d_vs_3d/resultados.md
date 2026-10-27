# D2CO: registro 2D frente a error 3D

Malla del paquete, 60 primeras instancias de test, caja GT, sin umbral: 51 poses publicadas de 60. Fuente: `d2co_malla_test60.csv` → `metrica_2d_malla.csv` (`analiza.py`).

| Métrica 2D (px a 1024) | Mediana | < 5 px | < 10 px |
|---|---|---|---|
| Unidireccional (D2CO → real; indulgente) | 1.4 | 78.4 % | 90.2 % |
| **Bidireccional** (cubre también el tubo real) | **7.3** | **25.5 %** | 64.7 % |

- Ángulo entre los ejes del tubo en la imagen: mediana 2.4°; < 5° en el 64.7 %.
- Longitud proyectada D2CO / real: mediana 0.73 (p25 0.49, p75 0.92).
- Poses con registro 2D completo (bidireccional < 5 px): 13; de ellas cumplen ADD-S (< 18.69 mm): 1; ADD-S mediano 114 mm.

## Malla del paquete frente a cilindro simple (mismas 60 instancias)

| modelo | n | cobertura % | recall ADD-S % | \|radial\| med mm | radial med mm | lateral med mm | lateral < umbral % | eje med ° | eje < 10° % | \|radial\| < umbral % |
|---|---|---|---|---|---|---|---|---|---|---|
| malla (`d2co_malla_test60.csv`) | 60 | 85.0 | 1.7 (1) | 248 | +213 | 35.5 | 23.5 | 44.5 | 3.9 | 2.0 |
| cilindro (`d2co_cilindro_test60.csv`) | 60 | 88.3 | 3.3 (2) | 301 | +301 | 16.1 | 52.8 | 42.9 | 17.0 | 5.7 |
