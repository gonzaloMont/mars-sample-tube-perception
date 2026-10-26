# Calidad de SGBM en los píxeles del tubo (sin ICP)

Estéreo: `wls_lambda` 8000 (el default cuando se corrió; equivale a `configs/stereo_wls8000.yaml`). Medianas por instancia; solo instancias con visibilidad ≥ 0.5 para no mezclar oclusión. Fuente: `analysis/calidad_sgbm.csv`.

| grupo | n | brillo medio imagen | cobertura del tubo % | error z mediano mm | \|error z\| mediano mm | celdas con \|error\| < 10 mm % |
|---|---|---|---|---|---|---|
| v1 (todas) | 543 | 129.7 | 91.5 | +7.3 | 10.1 | 49.5 |
| v3 (todas) | 560 | 115.9 | 91.5 | +7.6 | 12.2 | 41.9 |
| v3, g < 0.690 | 179 | 98.7 | 91.3 | +8.9 | 14.4 | 35.8 |
| v3, 0.690–0.852 | 191 | 114.9 | 92.4 | +7.7 | 11.5 | 44.3 |
| v3, 0.852–0.9 | 77 | 127.1 | 91.5 | +7.2 | 11.8 | 45.4 |
| v3, g ≥ 0.9 (más luz que v1) | 113 | 135.1 | 90.2 | +7.3 | 10.7 | 48.5 |

Spearman (v3) entre g y fracción de celdas con |error| < 10 mm: ρ = +0.19 (IC 95 % +0.11 … +0.27, n = 559).
Spearman (v3) entre g y |error z| mediano: ρ = -0.18 (IC 95 % -0.26 … -0.10, n = 559).
Fracción media de celdas buenas, v3 g < 0.69 − v1: -10.6 pp (IC 95 % bootstrap -14.4 … -6.7).
Fracción media de celdas buenas, v3 g ≥ 0.9 − v1: +0.7 pp (IC 95 % bootstrap -3.8 … +5.5).
