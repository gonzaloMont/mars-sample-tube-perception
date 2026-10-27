# D2CO: buen registro en la imagen, profundidad errónea

Complemento de `../README.md` (§3, «D2CO: ambigüedad de profundidad»).
Responde a dos preguntas: si el ~1 % de recall de D2CO se debe a un fallo de
implementación, y por qué sus registros dibujados sobre la imagen parecen
buenos.

## Qué se ha medido

D2CO (nodo real de `parol6_d2co`) sobre las 60 primeras instancias de test de
`~/parol6_pose_eval_dataset_v3`, con la caja GT y sin umbral de rechazo, con
dos modelos: la malla del paquete y un cilindro simple del mismo tamaño
(`cilindro_simple.stl`). El banco de pruebas guarda la pose estimada completa
(columna `T_est`), lo que permite comparar el error en la imagen con el error 3D.

## Resultados

Tablas completas en `resultados.md` (`analiza.py`) y `raster_composicion.md`
(`raster_sim.py`).

**Registro 2D frente a 3D** (malla, 51 poses publicadas de 60):

| Métrica | Valor |
|---|---|
| Ángulo entre los ejes del tubo en la imagen | mediana 2.4°; < 5° en el 64.7 % |
| Longitud proyectada D2CO / real | mediana 0.73 |
| Error 2D bidireccional (px a 1024) | mediana 7.3; < 5 px en el 25.5 % |
| Poses con registro 2D completo (< 5 px) | 13, de las que 1 cumple ADD-S; ADD-S mediano 114 mm |

D2CO acierta en la imagen la dirección del tubo y sus bordes laterales. Lo
habitual es que lo proyecte más corto, porque lo coloca más lejos. Incluso
cuando la proyección coincide de punta a punta, la pose 3D está a decenas de
centímetros: a ~1.2 m, desplazar el tubo 20 cm en profundidad e inclinarlo
cambia su anchura en la imagen en ~1 px, y es lo único que distingue esas
poses (`figura_d2co_2d_vs_3d.png`: imagen a la izquierda, vista cenital a la
derecha; verde = pose real, rojo = D2CO).

Una advertencia sobre la métrica: la variante unidireccional (para cada punto
de D2CO, distancia al punto real más cercano) da una mediana de 1.4 px y un
78.4 % de poses < 5 px, porque un tubo proyectado más corto, o que cruza al
real, puntúa bien. No sirve como medida del registro.

**Malla del paquete frente a cilindro simple** (mismas 60 instancias):

| Modelo | Recall ADD-S | Lateral mediano | Lateral < 18.69 mm | Eje < 10° | Radial mediano |
|---|---|---|---|---|---|
| Malla | 1.7 % (1) | 35.5 mm | 23.5 % | 3.9 % | +213 mm |
| Cilindro | 3.3 % (2) | 16.1 mm | 52.8 % | 17.0 % | +301 mm |

El cilindro reduce a la mitad el error lateral y mejora el eje, pero la
profundidad sigue igual de mal.

## Revisión del código

Revisados el nodo (`parol6_d2co/src/d2co_pose_estimator_node.cpp`) y las
partes de la biblioteca que usa (`external/d2co`: `chamfer_matching`,
`template_matching`, `raster_object_model3D` y `cv_ext::PinholeCameraModel`).

**Correcto:**
- `setSizeScaleFactor(2)` divide fx, fy, cx y cy, y `fx()` devuelve el valor
  escalado (401.84 px a 512).
- Modelo en metros, origen en el centro de la caja envolvente (el de
  `tube_link`, el del GT) y eje en Z. El STL escalado
  (`make_scaled_tube_stl.py`) aplica la misma escala que Gazebo e ICP.
- El optimizador se reconstruye en cada llamada con el tensor de distancias de
  la imagen actual (`allVisiblePoints()` es falso para el modelo 3D).
- El residuo es el del D2CO original (tensor direccional, Huber, LM). El
  cuaternión se normaliza al proyectar (`ceres::QuaternionRotatePoint`).
- El parche ya existente en `getDistance` corrige lecturas fuera del tensor y
  penaliza los puntos fuera de la imagen.

**Defectos encontrados (no corregidos: D2CO queda como referencia):**
1. **El modelo de bordes está dominado por la cabeza del tubo.** La malla
   tiene 5838 aristas vivas, casi todas en la cabeza y la tapa. En las poses
   de test, la cabeza aporta el 41–58 % de los puntos (en ~12 px de la imagen
   de trabajo), la tapa inferior ~1000 más, y la silueta del cuerpo, que es lo
   que se ve en la imagen, solo el 6.5–13.7 % (`raster_composicion.md`). Con un
   cilindro simple mejoran el lateral y el eje, no la profundidad.
2. **`width_bias_px` = 1 px está mal calibrado.** La anchura medida tras
   restarlo es de 6.0–6.15 px (mediana), frente a ~6.8 px esperados a la
   distancia mediana (`../analysis/d2co_mecanismo.md`). Cuando se aplica la
   corrección por anchura, aleja el tubo un ~10 % de más.
3. **El cuaternión no se normaliza antes de puntuar la pose refinada**
   (`getAvgDistance` y la corrección por anchura); solo se normaliza al
   publicar. Efecto pequeño.
4. **En el optimizador, los puntos fuera de la imagen siguen costando 0** (el
   parche solo afecta a la puntuación). Aquí no influye: el tubo siempre queda
   dentro.

Ninguno explica el ~1 %: la causa es la ambigüedad de profundidad del caso
monocular.

## Reproducción

```bash
cd ~/parol6_ws/src/PAROL6/ros_parol/src/parol6_vision/results/pose_eval_v3/d2co_2d_vs_3d
python3 crear_cilindro.py      # cilindro_simple.stl
./lanzar.sh                    # 2 corridas de D2CO (~3 min cada una, necesita DISPLAY)
python3 analiza.py             # → resultados.md, metrica_2d_malla.csv
python3 raster_sim.py          # → raster_composicion.md
python3 figura_2d_vs_3d.py     # → figura_d2co_2d_vs_3d.png
```

Los CSV versionados se obtuvieron con el mismo nodo (build anterior de la rama
`d2co`, con el código fuente idéntico) y con el banco de pruebas parcheado con
la columna `T_est`, que ahora ya está en `eval_pose_dataset.py`. La corrida del
cilindro es anterior a ese parche y no tiene `T_est`. D2CO no es determinista
entre corridas (`../README.md` §3): al repetir, unas pocas instancias pueden
cambiar.
