# parol6_d2co

Estimación de pose 6D del tubo mediante **D2CO** (*Direct Directional Chamfer
Optimization*, Imperoli & Pretto, ICVS 2015) integrada en el pipeline de
visión de PAROL6.

A diferencia del pipeline ICP (`parol6_vision/icp_pose_estimator_node.py`),
D2CO es **edge-based y monocular**: registra el modelo CAD del tubo contra los
bordes de UNA imagen (NavCam izquierda), sin necesidad de nube de puntos
estéreo. Publica el **mismo contrato** que el nodo ICP, por lo que el resto
del pipeline (`tube_grasp_pose_node`, `grasp_executor`, …) funciona sin
cambios.

```
 /navcam/left/.../image_raw ─┐
 /navcam/left/.../camera_info├─► d2co_pose_estimator_node ─► /vision/tube_pose (PoseStamped)
 /vision/detections (YOLO) ──┘            │                  /vision/d2co_overlay (debug)
                                          └─► TF: <output_frame> → tube_link_estimated
```

## Arquitectura del nodo

1. **YOLO** (`yolo26n_camera_subscriber.py`, reutilizado de `parol6_vision`)
   publica el bbox del tubo en `/vision/detections`.
2. El bbox (+margen) se convierte en **máscara del detector de bordes**
   (Canny/LSD): el tensor de distancias solo contiene bordes del entorno del
   tubo. Es el equivalente funcional del ROI, sin los problemas de
   reinicialización OpenGL (ver más abajo).
3. Se calcula el **tensor direccional de distancias** (DT3V del paper:
   `num_directions` canales, suavizado con peso `tensor_lambda`).
4. Se generan **hipótesis iniciales** desde el bbox: rayo de retroproyección
   del centro, profundidad estimada por el tamaño aparente
   (`z ≈ fx·L/px`, acotada a `[z_min, z_max]` y barrida con `z_offsets`) y
   orientaciones del eje del cilindro deducidas del aspecto del bbox
   (+inclinaciones `tilt_angles_deg`). Si hay una pose reciente aceptada
   (tracking), se usa como primera semilla.
5. Las hipótesis se puntúan con `getAvgDistance` (barato), las `refine_top_k`
   mejores se refinan con `DirectionalChamferMatching::refinePosition`
   (Levenberg-Marquardt de Ceres, `optimizer_iterations` iteraciones) y se
   acepta la mejor si su distancia media queda bajo `max_avg_dist`.
6. Publicación idéntica al nodo ICP: `PoseStamped` en el frame óptico o
   transformada a `output_frame` vía TF2, TF `tube_link_estimated`, overlay de
   depuración y `Pose` legacy opcional.

### Por qué no se usa el ROI nativo de D2CO

`cv_ext::PinholeCameraModel` soporta ROI (`setRegionOfInterest` +
`enableRegionOfInterest`: las proyecciones pasan a coordenadas del recorte y
la intrínseca se ajusta internamente — no hay que modificar K a mano). Pero
`RasterObjectModel3D::computeRaster()` crea su ventana OpenGL (GLFW) **una
única vez** y congela ahí el modelo de cámara: con un bbox que cambia cada
frame habría que recrear el contexto GL continuamente (fuga de ventanas).
Por eso este nodo procesa la imagen completa **reescalada**
(`image_scale`, def. 0.5 → 512×512, la intrínseca la escala
`setSizeScaleFactor` de forma consistente) y aplica el bbox como **máscara de
bordes**, que tiene el mismo efecto práctico sobre el coste y la robustez.

## Compilación

### 1. Librería externa libd2co (una sola vez)

La librería está **vendorizada dentro del paquete**, en `external/d2co/`
(no es un paquete ROS; colcon no entra en ella). Incluye:

- las fuentes de D2CO con los parches de compatibilidad para
  Ubuntu 22.04 / OpenCV 4 / C++14 (`AutoDiff→NumericDiff` en
  `chamfer_matching.cpp`, `CV_DIST_L2→cv::DIST_L2`, etc.) — estos parches
  **no** están en el repo original de alberto_pretto;
- `externals/cv_ext` — dependencia de Bitbucket hoy desaparecida: esta es la
  única copia de las fuentes (¡no borrar!);
- `externals/ceres_1_14/install` — Ceres 1.14 precompilado (el 2.0 del
  sistema es incompatible; su config de CMake es relocatable).

```bash
cd ~/parol6_ws/src/PAROL6/ros_parol/src/parol6_d2co/external/d2co

# cv_ext (estática)
cd externals/cv_ext && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 ..
make -j4

# libd2co.so
cd ../../.. && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
      -DCeres_DIR=$PWD/../externals/ceres_1_14/install/lib/cmake/Ceres ..
make -j4 d2co
```

> El RPATH del nodo se hornea al compilar apuntando a `external/d2co/lib` y
> al Ceres local: si se mueve el repo de sitio, basta recompilar en orden
> (cv_ext → libd2co → colcon) para regenerarlo.

> **Importante:** se fija `OpenCV_DIR` al OpenCV 4.5.4 del sistema (el mismo
> que usa `cv_bridge`). En esta máquina hay un OpenCV 4.14 en `/usr/local`
> que `find_package` encontraría primero; mezclar dos OpenCV en el mismo
> proceso corrompe el ABI.

### 1b. Dependencias de sistema (OBLIGATORIAS — sin ellas el nodo no arranca)

D2CO necesita estas librerías instaladas **en el sistema**. Sin ellas el
paquete compila pero el nodo aborta al lanzarse, porque el enlazado dinámico
no encuentra las bibliotecas en tiempo de ejecución.

```bash
sudo apt update
sudo apt install -y \
    libgoogle-glog-dev \
    libunwind-dev \
    libglfw3-dev \
    libdime-dev \
    libglm-dev \
    libglew-dev \
    libpcl-dev \
    libeigen3-dev
```

| Librería | Para qué | Quién la pide |
|---|---|---|
| `libgoogle-glog-dev` | logging de Ceres | **Ceres 1.14 vendorizado** (dep. transitiva) |
| `libunwind-dev` | trazas de glog | glog |
| `libglfw3-dev` | contexto OpenGL oculto | `RasterObjectModel3D::computeRaster()` |
| `libdime-dev` | lectura de DXF | libd2co |
| `libglm-dev`, `libglew-dev` | matemáticas y extensiones GL | libd2co |
| `libpcl-dev`, `libeigen3-dev` | estructuras geométricas | libd2co |

**OpenMesh 8.1** no está empaquetado en Ubuntu 22.04 y hay que instalarlo
aparte (es quien carga el STL del modelo):

```bash
# Si ya está compilado en el prefijo local, basta reinstalarlo en /usr/local:
sudo cmake --install ~/parol6_ws/.d2co_deps/OpenMesh-8.1/build --prefix /usr/local
sudo ldconfig

# En una máquina nueva: descargar de https://www.openmesh.org y compilar
#   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
#   make -j4 && sudo make install && sudo ldconfig
```

> **Error esperado y NO problemático** al reinstalar desde el build local:
>
> ```
> CMake Error at .../src/OpenMesh/Tools/cmake_install.cmake:46 (file):
>   file INSTALL cannot find ".../Build/lib/libOpenMeshTools.so.8.1"
> ```
>
> En ese build solo se compiló el target `Core`, nunca `Tools`. D2CO enlaza
> únicamente `libOpenMeshCore`, así que el fallo es inocuo: la instalación
> de `Core` (librerías + cabeceras) ya ha terminado correctamente antes de
> llegar a esa línea. Verifícalo con `ldconfig -p | grep OpenMesh`.

#### El fallo típico y por qué ocurre

```
d2co_pose_estimator_node: error while loading shared libraries:
    libglog.so.0: cannot open shared object file: No such file or directory
```

Suele desconcertar porque el `RUNPATH` del ejecutable **sí** incluye las rutas
donde está `libglog`. La razón es que `DT_RUNPATH` solo se consulta para las
dependencias **directas** del objeto que lo lleva: no se propaga a las
dependencias de sus dependencias. Y `libglog` no la pide el nodo, la pide
`libceres.so.1`. Por eso `libglog` **tiene que estar visible para `ldconfig`**,
es decir, instalada en el sistema. No basta con que exista en el árbol del
proyecto.

> **Solución de emergencia** (solo si no puedes instalar paquetes): exportar
> la ruta antes de lanzar el nodo o cualquier script que lo use. No es la vía
> recomendada; hay que repetirlo en cada terminal.
>
> ```bash
> export LD_LIBRARY_PATH=$HOME/parol6_ws/.d2co_deps/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
> ```

#### Comprobar que está todo

```bash
ldd $(ros2 pkg prefix parol6_d2co)/lib/parol6_d2co/d2co_pose_estimator_node \
    | grep "not found"       # sin salida = correcto
```

### 2. Modelo CAD pre-escalado (una sola vez)

D2CO (OpenMesh) solo admite escala uniforme y el `mars2020tube.stl` original
necesita la escala anisótropa `(0.022939, 0.022939, 0.033212)`:

```bash
python3 scripts/make_scaled_tube_stl.py
# -> models/mars2020tube_scaled.stl (0.186 m de largo, Ø0.021 m, centrado)
```

### 3. Paquete ROS 2

```bash
cd ~/parol6_ws
colcon build --packages-select parol6_d2co
```

## Uso

```bash
# Gazebo + MoveIt con la NavCam:
ros2 launch parol6_gazebo gazebo_moveit_mars.launch.py camera:=navcam

# Pipeline D2CO (YOLO + estimador):
ros2 launch parol6_d2co d2co_pose.launch.py

# Activar YOLO la primera vez:
ros2 topic pub --once /orchestrator/vision_activate std_msgs/String '{data: activate}'

# Forzar una estimación puntual:
ros2 service call /d2co_pose_estimator_node/estimate_now std_srvs/srv/Trigger

# Pipeline completo de agarre con D2CO como estimador
# (sin el ICP: enable_vision:=false y D2CO aparte):
ros2 launch parol6_apps tube_grasp_pipeline.launch.py enable_vision:=false &
ros2 launch parol6_d2co d2co_pose.launch.py
```

El nodo necesita un servidor gráfico (la rasterización del modelo usa una
ventana GLFW oculta). En sesiones sin display: `xvfb-run -a ros2 launch …`.

## Parámetros principales (`config/d2co_pose_params.yaml`)

| Parámetro | Default | Significado |
|---|---|---|
| `image_scale` | 0.5 | Escala de la imagen de trabajo (512² desde 1024²). |
| `num_directions` | 60 | Canales de orientación del tensor (valor del paper). |
| `tensor_lambda` | 6.0 | Peso del término direccional λ (valor del paper). |
| `optimizer_iterations` | 50 | Iteraciones máx. de LM (Ceres). |
| `edge_detector` | CANNY | `CANNY` o `LSD`. |
| `bbox_margin_px` | 24 | Margen de la máscara alrededor del bbox YOLO. |
| `z_min/z_max/z_factors` | 0.3/2.0/×{0.8–1.8} | Barrido de profundidad (multiplicativo sobre z_est). |
| `tilt_angles_deg` | ±40° | Inclinaciones fuera de plano del eje del tubo. |
| `refine_top_k` | 3 | Hipótesis refinadas con LM por ciclo. |
| `max_avg_dist` | 4.5 | Umbral de aceptación (ver sección de validación). |
| `process_rate` | 1.0 Hz | Frecuencia del pipeline. |

## Verificación rápida en RViz

- `Image` → `/vision/d2co_overlay`: puntos amarillos = modelo proyectado en la
  pose estimada; rectángulo rojo = bbox YOLO. Deben solaparse con el tubo.
- `TF` → frame `tube_link_estimated`.
- `ros2 topic echo /vision/tube_pose` para la pose numérica.

## Evaluación sistemática (oct-2026): buen registro 2D, profundidad errónea

La evaluación de referencia está en
`parol6_vision/results/pose_eval_v3/README.md` (dataset
`~/parol6_pose_eval_dataset_v3`, harness `eval_pose_dataset.py --estimator
d2co_node`, umbral ADD-S 18.69 mm). En test (222 instancias, caja GT):

| | con umbral (`max_avg_dist` 4.5) | sin umbral |
|---|---|---|
| Cobertura | 48.6 % | 82.9 % |
| Recall ADD-S | 0.5 % | 0.9 % |
| Error radial (profundidad) mediano | +446 mm | +285 mm |

En la imagen el registro es mucho mejor que en 3D. Sobre 60 instancias de
test (`parol6_vision/results/pose_eval_v3/d2co_2d_vs_3d/`):
- D2CO acierta la dirección del tubo en la imagen (ángulo mediano de 2.4° con
  el eje real).
- Pero lo suele proyectar más corto (longitud mediana de 0.73 respecto a la
  real), porque lo coloca más lejos.
- Error 2D bidireccional mediano de 7.3 px a 1024 px. El 27.5 % de las poses
  son un registro 2D completo (< 5 px), y aun así solo 1 de ellas cumple ADD-S
  (mediana de 143 mm).

Una pose a otra profundidad y con otra inclinación proyecta casi la misma
silueta: es la ambigüedad de profundidad del caso monocular, no un fallo de
implementación (revisión del código en ese mismo informe). Las secciones siguientes son validaciones
anteriores con pocas poses escogidas a mano; quedan como histórico.

## Validación en simulación (jun-2026, mundo mars_optimized)

Verdad-terreno vía `gz model`; tubo estático; bbox de YOLO real cacheada y
re-publicada (ver abajo). Dos escenas:

| Escena | Posición | Eje | Ciclo |
|---|---|---|---|
| Zona clara, tubo a 1.51 m | **53.5 mm** | ~17–22° | ~1.5 s (tracking) |
| Zona oscura (sin tapas visibles), 1.48 m | **79.9 mm** | ~21° | ~3.5 s (corrección) |

Mecanismos calibrados en esta validación (todos con parámetro):
- **Semillas por bordes**: centroide+PCA de la mayor componente conexa del
  Canny enmascarado (los bbox YOLO llegan holgados y descentrados; score 0.32–0.45).
- **Banda física de profundidad**: z ≤ z_est = fx·L/l_px (el escorzo solo acorta).
- **Corrección por anchura de raíles** (z = fx·Ø/w_px, inmune al tilt), aplicada
  SOLO si avg_dist > `width_correction_gate` (ajuste pobre = tapas sin contraste
  = deslizamiento por el valle de profundidad). Con buen ajuste el optimizador
  es más preciso que la medida (±0.5 px ≈ ±10 % z).
- `max_avg_dist 4.5`: la DT saturada a 30 px suma ~3 por rasgos ausentes sin
  implicar desalineación.

### Ejecutar la validación con pocos recursos
YOLO congela el equipo si corre continuo. Estrategia validada:
1. Arrancar YOLO, capturar UNA detección y matarlo:
   `python3 ~/parol6_ws/.d2co_validation/bbox_cache_replay.py capture`
2. Re-publicarla a 1 Hz (coste ~0): `... bbox_cache_replay.py replay`
3. Matar `gzclient` (las cámaras renderizan en gzserver).
4. Comparar con verdad-terreno: `... d2co_sim_validate.py 60`.

Límite conocido: la inclinación fuera de plano del eje queda en ~17–22° (poco
observable en monocular con tapas débiles); la posición es el dato robusto.

## Test multi-pose con imágenes de debug (`scripts/test_d2co_pose.py`)

Equivalente a `parol6_vision/test_icp_pose.py` pero para D2CO. Spawnea el tubo
en una serie de poses conocidas, mide la estimación de `/vision/tube_pose`
contra la verdad-terreno (`/model_states`) y **guarda dos imágenes de debug
por pose** para evaluación visual.

**No necesita YOLO** (que congela el equipo): como el test conoce la pose
verdad-terreno del tubo que spawnea, **calcula la bbox proyectando el modelo 3D
del tubo en la imagen** y la publica él mismo en `/vision/detections`. Esto
aísla el algoritmo D2CO del ruido de YOLO y no consume recursos. La pose de la
cámara se lee de **TF** (`world`→`navcam_left_optical_frame`), así que el test
funciona con cualquier configuración de mástil/mundo sin tocar constantes.

```bash
# 1) Gazebo (mundo optimized, va más fluido) con la NavCam:
ros2 launch parol6_gazebo gazebo_moveit_mars_optimized.launch.py camera:=navcam
# 2) Nodo D2CO SIN YOLO (el test provee las bbox), salida en world:
ros2 launch parol6_d2co d2co_pose.launch.py enable_yolo:=false output_frame:=world
# 3) (recomendado) matar gzclient: las cámaras renderizan en gzserver y el
#    GUI ralentiza mucho los ciclos del optimizador.
kill $(pgrep -x gzclient)
# 4) Test con imágenes de debug:
ros2 run parol6_d2co test_d2co_pose.py --debug-dir /tmp/d2co_test
```

Por cada pose `{name}` se guardan:
- `{name}_overlay.png` — el overlay PROPIO del nodo (`/vision/d2co_overlay`):
  raster real del STL en la pose estimada (amarillo) + bbox (rojo). Solo
  aparece en las poses cuya estimación fue aceptada (`avg_dist < max_avg_dist`).
- `{name}_annot.png` — imagen anotada por el test: eje GT (verde), estimación
  D2CO media + cilindro proyectado (azul/amarillo), bbox usada (rojo), línea de
  error GT→D2CO y panel con métricas. Permite comparar GT vs estimación de un
  vistazo.

Opciones útiles: `--samples N` (muestras por pose), `--grace S` (debe cubrir
≥1 ciclo del nodo, def. 10 s), `--bbox-margin-px`/`--bbox-jitter-px` (emular
bbox holgada/descentrada de YOLO), `--poses-file` (YAML `{poses: [...]}`),
`--csv`. Umbrales de PASS por defecto realistas para monocular: 80 mm / 30°.

> **Nota sobre los resultados (barrido de 12 poses, jun-2026).** Este test
> sistemático muestra que la precisión de D2CO es **muy dependiente de la
> posición**: los 53/79 mm de la sección anterior eran dos puntos escogidos a
> mano (mejor caso). En el barrido completo: **0/12 PASS**, error medio de
> posición **604 mm**, 5/12 poses sin estimación aceptada (`avg_dist > 4.5`).
> El patrón es consistente: la anchura de los raíles se mide ~3.4 px en vez de
> ~6 px (a resolución de trabajo) y la corrección por anchura empuja la
> profundidad estimada a ~1.7–2.1 m frente a los ~1.4 m reales. Las imágenes
> `_annot.png` lo muestran nítido: la bbox y el eje GT caen sobre el tubo, pero
> el cilindro estimado queda más pequeño (más lejos). Una bbox más holgada
> (`--bbox-margin-px 60`) mejora algún caso (p.ej. 1200 → 233 mm) pero no lo
> resuelve. Es el límite monocular ya documentado, ahora reproducible de forma
> sistemática — **no es un fallo del test sino del algoritmo en esta escena**.
> Resultados completos + 23 imágenes en `~/parol6_ws/.d2co_validation/test_poses/`.
