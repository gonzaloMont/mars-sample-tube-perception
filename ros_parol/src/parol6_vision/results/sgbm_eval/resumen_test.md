# SGBM frente a la profundidad real — partición test (180 pares)

Puntos de SGBM proyectados en la imagen izquierda; error = z_SGBM − z_real. Cobertura = media por imagen de la fracción de celdas (2×2 px) de la región con algún punto. Errores sobre todos los puntos de la región. Fuente: `eval_sgbm.py` → `sgbm_<config>_test.csv`.

| configuración | región | cobertura % | error mediano (sesgo) mm | \|error\| mediano mm | < 10 mm % | < 30 mm % | > 100 mm % | puntos |
|---|---|---|---|---|---|---|---|---|
| stereo_wls8000 | imagen | 79.6 | +17.2 | 43.3 | 14.7 | 37.4 | 13.6 | 27706132 |
| stereo_wls8000 | zona de trabajo | 96.1 | +20.2 | 31.8 | 22.8 | 48.3 | 14.1 | 8022756 |
| stereo_wls8000 | tubo | 97.6 | +7.3 | 15.9 | 39.7 | 60.8 | 15.3 | 70386 |
| stereo_wls8000 | tiempo por par (SGBM + WLS, 512²) | 52 ms | | | | | | |
| stereo_wls500 | imagen | 79.8 | +4.4 | 9.0 | 53.1 | 89.0 | 0.8 | 27759858 |
| stereo_wls500 | zona de trabajo | 96.1 | +1.7 | 8.4 | 54.4 | 83.8 | 2.0 | 8022727 |
| stereo_wls500 | tubo | 97.6 | +5.4 | 9.0 | 54.0 | 78.5 | 4.3 | 70386 |
| stereo_wls500 | tiempo por par (SGBM + WLS, 512²) | 49 ms | | | | | | |

## Por iluminación (mediana por imagen del \|error\| en la región; terciles de la difusa g)

| configuración | región | g < 0.691 | 0.691–0.853 | > 0.853 |
|---|---|---|---|---|
| stereo_wls8000 | zona de trabajo | 32.4 mm (cob. 96 %, n=60) | 33.2 mm (cob. 96 %, n=60) | 29.8 mm (cob. 96 %, n=60) |
| stereo_wls8000 | tubo | 21.8 mm (cob. 98 %, n=60) | 16.8 mm (cob. 97 %, n=60) | 14.9 mm (cob. 97 %, n=60) |
| stereo_wls500 | zona de trabajo | 8.1 mm (cob. 96 %, n=60) | 8.0 mm (cob. 96 %, n=60) | 8.3 mm (cob. 96 %, n=60) |
| stereo_wls500 | tubo | 10.2 mm (cob. 98 %, n=60) | 10.3 mm (cob. 97 %, n=60) | 8.6 mm (cob. 97 %, n=60) |

Sensibilidad: f = 401.84 px (resolución de trabajo), B = 0.150 m; a la distancia mediana cámara–tubo de esta partición (1.236 m) la disparidad es 48.8 px y 1 px de disparidad equivale a 25.3 mm de profundidad.
