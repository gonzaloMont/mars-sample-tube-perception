#!/usr/bin/env python3
"""
icp_utils.py — Utilidades de geometría 3D para la estimación de pose 6D
del tubo mediante YOLOv8 + ICP (Open3D).

Este módulo NO depende de ROS: contiene únicamente funciones puras de
NumPy / Open3D, de forma que se puede probar y depurar de forma aislada
(``python3 -m parol6_vision.icp_utils <ruta_al_stl>``).

╔══════════════════════════════════════════════════════════════════════════╗
║  REQUISITO GEOMÉTRICO CRÍTICO                                              ║
║                                                                            ║
║  El .stl original (mars2020tube.stl) tiene proporciones incorrectas.       ║
║  ANTES de muestrear la nube de puntos de referencia para el ICP hay que    ║
║  multiplicar sus vértices por la escala NO UNIFORME:                       ║
║                                                                            ║
║         scale = (0.022939, 0.022939, 0.033212)   # (X, Y, Z)               ║
║                                                                            ║
║  Verificación (bbox del .stl medido sobre el fichero):                     ║
║      bbox crudo    ≈ [0.9055, 0.9681, 5.6004]                              ║
║      bbox escalado ≈ [0.02077, 0.02221, 0.18600]  m                        ║
║                                                                            ║
║  Esto coincide con el cilindro de colisión del SDF (mars2020tube.sdf):     ║
║      radius = 0.01016 m  →  Ø 0.02032 m                                    ║
║      length = 0.186   m                                                    ║
║                                                                            ║
║  Si este escalado no se aplica, la nube de referencia no tiene el tamaño   ║
║  del objeto real y el ICP converge a una pose sin sentido.                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Se requiere 'open3d' para la estimacion de pose por ICP.\n"
        "Instalalo con:  pip3 install open3d"
    ) from exc


# ── Constantes geométricas (única fuente de verdad del proyecto) ────────────
# Escala NO UNIFORME obligatoria del .stl (X, Y, Z).
DEFAULT_MESH_SCALE: tuple[float, float, float] = (0.022939, 0.022939, 0.033212)


# ════════════════════════════════════════════════════════════════════════════
# 1. NUBE DE REFERENCIA — generada a partir del .stl ESCALADO
# ════════════════════════════════════════════════════════════════════════════

def load_reference_cloud(
    stl_path: str,
    scale: tuple[float, float, float] = DEFAULT_MESH_SCALE,
    n_points: int = 5000,
    recenter: bool = True,
    estimate_normals: bool = True,
) -> "o3d.geometry.PointCloud":
    """Carga el .stl, le aplica la escala NO UNIFORME y muestrea una nube.

    Parámetros
    ----------
    stl_path  : ruta a mars2020tube.stl
    scale     : escala (X, Y, Z) que se multiplica a cada vértice de la malla.
    n_points  : nº de puntos a muestrear uniformemente sobre la superficie.
    recenter  : si True, traslada la malla para que su centro de bounding box
                quede en el origen. La pose 6D estimada será entonces la del
                CENTRO geométrico del tubo (equivale al frame `tube_link`,
                ver mars2020tube.sdf <visual><pose>).
    estimate_normals : calcula normales (necesarias para el ICP point-to-plane).

    Devuelve
    --------
    open3d.geometry.PointCloud  — el modelo del tubo en su frame de objeto.
    """
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        raise ValueError(f"La malla '{stl_path}' esta vacia o no se pudo cargar.")

    # ── PASO OBLIGATORIO: escalado NO UNIFORME de los vértices ──────────────
    # Open3D.scale() solo admite escala uniforme, así que escalamos el array
    # de vértices manualmente con NumPy (esto sí permite (sx, sy, sz)).
    scale_vec = np.asarray(scale, dtype=np.float64)
    if scale_vec.shape != (3,):
        raise ValueError("`scale` debe tener 3 componentes (X, Y, Z).")
    verts = np.asarray(mesh.vertices, dtype=np.float64) * scale_vec
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # Normales coherentes tras el reescalado.
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ── Recentrar en el origen → la pose ICP será la del centro del tubo ────
    if recenter:
        bbox = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-bbox.get_center())

    # ── Muestreo uniforme de la superficie ──────────────────────────────────
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))

    if estimate_normals:
        # Radio de estimación ~ 3 × tamaño de voxel típico para este objeto.
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,
                                                              max_nn=30))
    return pcd


def reference_cloud_dimensions(pcd: "o3d.geometry.PointCloud") -> np.ndarray:
    """Devuelve las extensiones (dx, dy, dz) de la nube de referencia [m]."""
    bbox = pcd.get_axis_aligned_bounding_box()
    return np.asarray(bbox.get_extent(), dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
# 2. CONVERSIÓN DE SENSORES → NumPy
# ════════════════════════════════════════════════════════════════════════════

def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Convierte un sensor_msgs/PointCloud2 a un array NumPy (N, 3) float32.

    Implementación genérica que no depende de la versión de `sensor_msgs_py`:
    lee directamente los offsets de los campos x, y, z del buffer binario.
    Filtra automáticamente NaN / Inf.
    """
    # Localizar offsets de x, y, z
    offsets = {f.name: f.offset for f in msg.fields}
    for axis in ("x", "y", "z"):
        if axis not in offsets:
            raise ValueError(f"PointCloud2 sin campo '{axis}'.")

    n_points = msg.width * msg.height
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw.reshape(n_points, msg.point_step)

    def _col(off):
        # Cada coordenada es float32 (4 bytes) en PointCloud2 estándar.
        return raw[:, off:off + 4].copy().view(np.float32).reshape(-1)

    xyz = np.column_stack([_col(offsets["x"]),
                           _col(offsets["y"]),
                           _col(offsets["z"])]).astype(np.float32)

    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def depth_to_xyz(
    depth: np.ndarray,
    K: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Deproyecta una imagen de profundidad ORGANIZADA a una nube (H*W, 3).

    El resultado mantiene el orden de los píxeles (fila-mayor), de modo que
    el recorte por bounding box es un simple `slicing` 2D.

    Parámetros
    ----------
    depth       : imagen de profundidad (H, W). Puede ser float32 en metros
                  o uint16 en milímetros.
    K           : matriz intrínseca 3x3.
    depth_scale : factor para pasar a metros. 1.0 si la imagen ya está en m,
                  0.001 si está en mm (uint16).

    Devuelve
    --------
    np.ndarray (H*W, 3) float32. Los píxeles sin profundidad válida llevan NaN.
    """
    h, w = depth.shape[:2]
    z = depth.astype(np.float32) * float(depth_scale)
    z[z <= 0.0] = np.nan

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 3. RECORTE DE LA NUBE POR LA CAJA 2D DE YOLO
# ════════════════════════════════════════════════════════════════════════════

def crop_cloud_by_bbox(
    xyz: np.ndarray,
    K: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    margin_px: float = 0.0,
) -> np.ndarray:
    """Recorta una nube de puntos 3D usando la caja 2D de YOLO.

    Reproyecta cada punto 3D al plano imagen con la matriz intrínseca `K`
    y conserva únicamente los puntos cuya proyección cae dentro de la caja.
    Esto descarta de un plumazo todo lo que YOLO no detectó como tubo
    (suelo, robot, otros objetos) y reduce drásticamente el coste del ICP.

    IMPORTANTE — coherencia de frames:
      `xyz` y `K` deben corresponder a la MISMA cámara. La nube debe estar
      expresada en el frame ÓPTICO de esa cámara (convención ROS: +Z hacia
      adelante, +X derecha, +Y abajo). El nodo de stereo publica la nube en
      `navcam_left_optical_frame`, que cumple esta convención.

    Parámetros
    ----------
    xyz       : nube de la escena (N, 3) en el frame óptico de la cámara.
    K         : matriz intrínseca 3x3 de esa cámara.
    bbox_xyxy : (x_min, y_min, x_max, y_max) en píxeles.
    margin_px : margen extra (px) que se añade a la caja antes de filtrar.
                Útil para no perder los bordes del tubo si YOLO recorta justo.

    Devuelve
    --------
    np.ndarray (M, 3) — subconjunto de `xyz` dentro de la caja.
    """
    if xyz.shape[0] == 0:
        return xyz

    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min -= margin_px
    y_min -= margin_px
    x_max += margin_px
    y_max += margin_px

    # Solo puntos delante de la cámara pueden proyectarse.
    z = xyz[:, 2]
    in_front = z > 1e-6

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Proyección pinhole: u = fx*X/Z + cx ,  v = fy*Y/Z + cy
    u = np.full(xyz.shape[0], -1.0, dtype=np.float64)
    v = np.full(xyz.shape[0], -1.0, dtype=np.float64)
    u[in_front] = fx * xyz[in_front, 0] / z[in_front] + cx
    v[in_front] = fy * xyz[in_front, 1] / z[in_front] + cy

    inside = (
        in_front
        & (u >= x_min) & (u <= x_max)
        & (v >= y_min) & (v <= y_max)
    )
    return xyz[inside]


# ════════════════════════════════════════════════════════════════════════════
# 4. LIMPIEZA / SEGMENTACIÓN DE LA NUBE DE LA ESCENA
# ════════════════════════════════════════════════════════════════════════════

def make_o3d_cloud(xyz: np.ndarray) -> "o3d.geometry.PointCloud":
    """Crea un open3d.geometry.PointCloud a partir de un array (N, 3)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


def preprocess_scene_cloud(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.003,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    remove_plane: bool = True,
    plane_dist_thresh: float = 0.004,
    cluster_eps: float = 0.012,
    cluster_min_points: int = 30,
) -> "o3d.geometry.PointCloud":
    """Limpia la nube recortada y deja, idealmente, solo los puntos del tubo.

    Pasos:
      1. Voxel downsampling          → uniformiza densidad y acelera el ICP.
      2. Statistical outlier removal → elimina puntos sueltos / ruido.
      3. (opcional) Segmentación de plano por RANSAC → quita la mesa/suelo
         que pudiera haber quedado dentro de la caja de YOLO.
      4. (opcional) Clustering DBSCAN → se queda con el cluster más grande,
         que debe ser el tubo.

    Devuelve la nube filtrada (con normales estimadas).
    """
    # 1. Downsampling
    work = pcd.voxel_down_sample(voxel_size) if voxel_size > 0 else pcd

    if len(work.points) == 0:
        return work

    # 2. Outlier removal
    if sor_neighbors > 0 and len(work.points) > sor_neighbors:
        work, _ = work.remove_statistical_outlier(
            nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)

    # 3. Quitar el plano dominante (mesa / suelo)
    if remove_plane and len(work.points) > 10:
        try:
            _, inliers = work.segment_plane(
                distance_threshold=plane_dist_thresh,
                ransac_n=3,
                num_iterations=200)
            # Solo lo eliminamos si el plano no se "come" todo el tubo:
            # un tubo tumbado no debería ser > ~55 % de un plano detectado.
            if 0 < len(inliers) < 0.9 * len(work.points):
                work = work.select_by_index(inliers, invert=True)
        except RuntimeError:
            pass  # nube degenerada: seguimos sin segmentar

    # 4. Clustering: nos quedamos con el grupo MÁS CILÍNDRICO (no el más
    #    grande). Cuando el bbox de YOLO captura terreno cercano al tubo,
    #    el cluster del terreno puede llegar a tener más puntos que el tubo
    #    y "ganar" si elegimos por tamaño. La forma fiable de distinguirlos
    #    es la elongación PCA: el tubo (≈20×20×186 mm) tiene un autovalor
    #    dominante mucho mayor que los otros dos, mientras que un parche de
    #    terreno tiene los tres autovalores parecidos.
    if cluster_eps > 0 and len(work.points) > cluster_min_points:
        labels = np.asarray(work.cluster_dbscan(
            eps=cluster_eps, min_points=cluster_min_points))
        if labels.max() >= 0:
            pts_np = np.asarray(work.points, dtype=np.float64)
            best_idx = -1
            best_score = -1.0
            for cid in range(int(labels.max()) + 1):
                idx = np.where(labels == cid)[0]
                if idx.size < cluster_min_points:
                    continue
                cp = pts_np[idx]
                # PCA: autovalores de la covarianza (ascendentes)
                cov = np.cov(cp.T)
                ev = np.sort(np.linalg.eigvalsh(cov))
                if ev[1] <= 1e-12:
                    continue
                # Elongación = √(λ_max/λ_mid). Un tubo de 186/20 ≈ 9:1 da
                # ~9; un parche de suelo da ~1. Multiplicamos por log(N+1)
                # para evitar elegir un cluster microscópico muy alargado.
                elong = float(np.sqrt(ev[2] / ev[1]))
                score = elong * float(np.log10(idx.size + 1.0))
                if score > best_score:
                    best_score = score
                    best_idx = cid
            if best_idx < 0:
                # Sin candidatos válidos: fallback al cluster más poblado
                counts = np.bincount(labels[labels >= 0])
                best_idx = int(np.argmax(counts))
            work = work.select_by_index(np.where(labels == best_idx)[0])

    # Normales para el ICP point-to-plane
    if len(work.points) > 0:
        work.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 3.0 if voxel_size > 0 else 0.01,
                max_nn=30))
    return work


# ════════════════════════════════════════════════════════════════════════════
# 5. ALINEAMIENTO GRUESO (PCA) + REGISTRO ICP
# ════════════════════════════════════════════════════════════════════════════

def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz de rotación 3x3 que lleva el vector unitario `a` sobre `b`."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:                       # ya alineados
        return np.eye(3)
    if c < -1.0 + 1e-8:                      # anti-paralelos
        # rotación de 180° alrededor de cualquier eje perpendicular a `a`
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return _axis_angle_to_mat(axis, np.pi)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: eje (unitario) + ángulo → matriz de rotación 3x3."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def pca_principal_axis(pcd: "o3d.geometry.PointCloud") -> tuple[np.ndarray, np.ndarray]:
    """Centroide y eje principal (mayor varianza) de la nube vía PCA."""
    pts = np.asarray(pcd.points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)        # ascendente
    principal = eigvecs[:, -1]                    # mayor autovalor
    return centroid, principal / np.linalg.norm(principal)


def _initial_guesses(
    scene: "o3d.geometry.PointCloud",
) -> list[np.ndarray]:
    """Genera hipótesis iniciales (objeto→cámara) a partir del PCA de la escena.

    El modelo de referencia tiene su eje LARGO sobre +Z. Alineamos ese eje con
    el eje principal del tubo observado. Como un cilindro es casi simétrico, se
    prueban ambos sentidos del eje y varios giros (`roll`) alrededor de él; el
    nodo ejecuta el ICP desde cada hipótesis y se queda con la mejor.

    Devuelve una lista de matrices homogéneas 4x4 T_cam_obj (pose del objeto
    expresada en el frame de la cámara).
    """
    centroid, axis = pca_principal_axis(scene)
    model_axis = np.array([0.0, 0.0, 1.0])

    guesses: list[np.ndarray] = []
    for sign in (+1.0, -1.0):
        R_align = _rotation_between(model_axis, sign * axis)
        for roll in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            R_roll = _axis_angle_to_mat(sign * axis, roll)
            T = np.eye(4)
            T[:3, :3] = R_roll @ R_align
            T[:3, 3] = centroid
            guesses.append(T)
    return guesses


def register_tube_icp(
    scene: "o3d.geometry.PointCloud",
    model: "o3d.geometry.PointCloud",
    voxel_size: float = 0.003,
    max_corr_dist: float | None = None,
    use_point_to_plane: bool = True,
    max_iter: int = 80,
    use_global_registration: bool = False,
) -> dict:
    """Registra la nube de la escena contra el modelo del tubo (ICP).

    Convenio ICP
    ------------
    ``source = scene`` (vista parcial)  y  ``target = model`` (modelo completo).
    Se elige así porque cada punto de la vista parcial SÍ tiene un vecino
    correcto en el modelo completo; al revés, los puntos de la cara oculta del
    modelo no tendrían correspondencia y degradarían el ajuste.

    Open3D devuelve ``T`` tal que  ``T · scene ≈ model``  →  ``T = T_obj_cam``.
    La pose del objeto en la cámara es por tanto  ``T_cam_obj = inv(T)``.

    Estrategia multi-arranque
    -------------------------
    Se lanza el ICP desde cada hipótesis del PCA (sentido del eje + roll) y se
    conserva la de mayor `fitness` (desempate por menor `inlier_rmse`). Esto
    resuelve la ambigüedad de 180° propia de un objeto casi simétrico.

    Devuelve
    --------
    dict con:
      'T_cam_obj' : np.ndarray 4x4 — pose del tubo en el frame de la cámara.
      'fitness'   : float [0..1]   — fracción de puntos con correspondencia.
      'rmse'      : float          — error cuadrático medio de los inliers [m].
      'n_scene'   : int            — nº de puntos de la escena usados.
    """
    if max_corr_dist is None:
        max_corr_dist = voxel_size * 3.0

    n_scene = len(scene.points)
    if n_scene < 10:
        return {"T_cam_obj": None, "fitness": 0.0, "rmse": float("inf"),
                "n_scene": n_scene}

    if use_point_to_plane:
        if not scene.has_normals():
            scene.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max_corr_dist, max_nn=30))
        if not model.has_normals():
            model.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max_corr_dist, max_nn=30))
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=int(max_iter))

    # ── Hipótesis iniciales ─────────────────────────────────────────────────
    # T_obj_cam = pose objeto→cámara. El ICP necesita la init en convenio
    # source→target = scene→model = inv(T_obj_cam).
    candidates = _initial_guesses(scene)

    if use_global_registration:
        candidates = [_global_registration(scene, model, voxel_size)] + candidates

    best = {"T_cam_obj": None, "fitness": -1.0, "rmse": float("inf"),
            "n_scene": n_scene}

    for T_cam_obj_guess in candidates:
        if T_cam_obj_guess is None:
            continue
        init = np.linalg.inv(T_cam_obj_guess)            # scene → model
        reg = o3d.pipelines.registration.registration_icp(
            scene, model, max_corr_dist, init, estimator, criteria)

        better = (
            reg.fitness > best["fitness"] + 1e-6
            or (abs(reg.fitness - best["fitness"]) <= 1e-6
                and reg.inlier_rmse < best["rmse"])
        )
        if better:
            best = {
                "T_cam_obj": np.linalg.inv(reg.transformation),  # model → cam
                "fitness": float(reg.fitness),
                "rmse": float(reg.inlier_rmse),
                "n_scene": n_scene,
            }
    return best


def _global_registration(
    scene: "o3d.geometry.PointCloud",
    model: "o3d.geometry.PointCloud",
    voxel_size: float,
) -> np.ndarray | None:
    """Registro global RANSAC sobre features FPFH (hipótesis inicial robusta).

    Opcional: útil si el PCA falla (p.ej. nube muy parcial). Devuelve una
    matriz T_cam_obj o None si no converge.
    """
    radius_feat = voxel_size * 5.0
    try:
        scene_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            scene, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat,
                                                        max_nn=100))
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            model, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat,
                                                        max_nn=100))
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            scene, model, scene_fpfh, model_fpfh, True,
            voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                 voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return np.linalg.inv(result.transformation)  # model → cam
    except RuntimeError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 6. CONVERSIONES DE POSE
# ════════════════════════════════════════════════════════════════════════════

def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Matriz de rotación 3x3 → cuaternión (x, y, z, w). Algoritmo de Shepperd."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return tuple(q / np.linalg.norm(q))


# ════════════════════════════════════════════════════════════════════════════
# Diagnóstico por línea de comandos
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":      # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        print("Uso: python3 -m parol6_vision.icp_utils <ruta_al_mars2020tube.stl>")
        raise SystemExit(1)

    ref = load_reference_cloud(sys.argv[1])
    dims = reference_cloud_dimensions(ref)
    print("Nube de referencia generada correctamente.")
    print(f"  nº de puntos     = {len(ref.points)}")
    print(f"  extents (X,Y,Z)  = {dims.round(5)} m")
    print(f"  longitud (eje Z) = {dims[2]:.4f} m   (esperado ~0.186 m)")
    print(f"  diametro (X/Y)   = {dims[0]:.4f} / {dims[1]:.4f} m "
          f"(esperado ~0.0203 m)")
