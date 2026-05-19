#!/usr/bin/env python3
"""
mesh_utils.py — Carga y normalización geométrica de la malla del tubo.

╔══════════════════════════════════════════════════════════════════════════╗
║  REQUISITO GEOMÉTRICO CRÍTICO                                              ║
║                                                                            ║
║  El .stl original (mars2020tube.stl) tiene proporciones incorrectas.       ║
║  ANTES de renderizar cualquier template hay que multiplicar sus vértices   ║
║  por la escala NO UNIFORME:                                                ║
║                                                                            ║
║         scale = (0.022939, 0.022939, 0.033212)   # (X, Y, Z)               ║
║                                                                            ║
║  Verificación (bbox del .stl crudo medido sobre el fichero):               ║
║      bbox crudo  ≈ [0.9055, 0.9681, 5.6004]                                ║
║      bbox escalado ≈ [0.0208, 0.0222, 0.1860]  m                           ║
║                                                                            ║
║  Esto coincide EXACTAMENTE con el cilindro de colisión del SDF             ║
║  (mars2020tube.sdf):  radius = 0.01016 m  →  Ø 0.02032 m                    ║
║                       length = 0.186   m                                  ║
║                                                                            ║
║  Si este escalado no se aplica, el template renderizado no se parece al    ║
║  objeto que ve la cámara en Gazebo y LINEMOD no hace match.                ║
╚══════════════════════════════════════════════════════════════════════════╝

Además del escalado, esta utilidad lleva la malla al frame `tube_link`
(el mismo que usa Gazebo) replicando el offset del <visual><pose> del SDF.
Así la pose 6D que estime el nodo de inferencia ya está expresada en el
frame del objeto que el resto del sistema (TF, MoveIt) espera.
"""

from __future__ import annotations

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - dependencia offline
    raise ImportError(
        "Se requiere 'trimesh' para cargar y escalar el .stl.\n"
        "Instálalo con:  pip install trimesh"
    ) from exc


# ── Constantes geométricas (única fuente de verdad del proyecto) ────────────
# Escala NO UNIFORME obligatoria (X, Y, Z).
DEFAULT_MESH_SCALE: tuple[float, float, float] = (0.022939, 0.022939, 0.033212)

# Offset del <visual><pose> de mars2020tube.sdf. Lleva la malla escalada
# al origen del frame `tube_link` (centro geométrico del cilindro).
DEFAULT_FRAME_OFFSET: tuple[float, float, float] = (-0.010497, -0.010489, -0.093000)


class TubeModel:
    """Encapsula la malla del tubo ya escalada y centrada en `tube_link`.

    Atributos
    ---------
    mesh        : trimesh.Trimesh   malla lista para renderizar.
    length      : float             longitud del tubo (eje largo, Z) [m].
    radius      : float             radio del tubo [m].
    keypoints   : np.ndarray (N,3)  puntos 3D estables para refino PnP,
                                    expresados en el frame `tube_link`.
    """

    def __init__(self, mesh: "trimesh.Trimesh") -> None:
        self.mesh = mesh
        ext = np.asarray(mesh.extents, dtype=np.float64)  # (dx, dy, dz)
        # El eje largo del tubo es Z tras el escalado.
        self.length = float(ext[2])
        self.radius = float(max(ext[0], ext[1]) * 0.5)
        self.keypoints = self._build_keypoints()

    def _build_keypoints(self) -> np.ndarray:
        """Puntos 3D bien repartidos y geométricamente estables.

        Se usan para el refino opcional por PnP (cv2.solvePnP) una vez
        LINEMOD ha dado una hipótesis de rotación. Para un cilindro elegimos
        los dos extremos del eje y cuatro puntos del contorno radial.
        """
        hl = self.length * 0.5
        r = self.radius
        return np.array(
            [
                [0.0, 0.0, +hl],   # punta +Z
                [0.0, 0.0, -hl],   # punta -Z
                [+r, 0.0, 0.0],    # borde +X
                [-r, 0.0, 0.0],    # borde -X
                [0.0, +r, 0.0],    # borde +Y
                [0.0, -r, 0.0],    # borde -Y
                [0.0, 0.0, 0.0],   # centro
            ],
            dtype=np.float64,
        )


def load_tube_mesh(
    stl_path: str,
    scale: tuple[float, float, float] = DEFAULT_MESH_SCALE,
    frame_offset: tuple[float, float, float] | None = DEFAULT_FRAME_OFFSET,
) -> TubeModel:
    """Carga el .stl del tubo, aplica el escalado obligatorio y lo centra.

    Parámetros
    ----------
    stl_path     : ruta al fichero mars2020tube.stl
    scale        : escala NO UNIFORME (X, Y, Z). Por defecto la del proyecto.
    frame_offset : traslación posterior para llevar la malla al frame
                   `tube_link`. Si es None, se recentra por el centroide
                   del bounding box.

    Devuelve
    --------
    TubeModel con la malla lista para renderizar.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"La malla '{stl_path}' está vacía o no se pudo cargar.")

    # ── PASO OBLIGATORIO: escalado no uniforme de los vértices ──────────────
    scale_vec = np.asarray(scale, dtype=np.float64)
    if scale_vec.shape != (3,):
        raise ValueError("`scale` debe ser una tupla/array de 3 elementos (X, Y, Z).")
    # trimesh acepta un vector de 3 componentes => escala no uniforme.
    mesh.apply_scale(scale_vec)

    # ── Llevar la malla al frame tube_link ──────────────────────────────────
    if frame_offset is not None:
        mesh.apply_translation(np.asarray(frame_offset, dtype=np.float64))
    else:
        mesh.apply_translation(-mesh.bounding_box.centroid)

    # Normales coherentes tras el escalado (importante para el renderer).
    mesh.fix_normals()
    return TubeModel(mesh)


def intrinsics_from_hfov(width: int, height: int, hfov_rad: float) -> np.ndarray:
    """Construye la matriz intrínseca K a partir del FOV horizontal.

    Útil para generar templates con las MISMAS intrínsecas que la cámara
    cenital de Gazebo cuando no se dispone del camera_info en disco.

    overhead_camera.urdf.xacro define:
        <horizontal_fov>1.3962634</horizontal_fov>
        <image><width>640</width><height>480</height></image>
    """
    fx = (width * 0.5) / np.tan(hfov_rad * 0.5)
    fy = fx  # píxel cuadrado (Gazebo no distorsiona la relación de aspecto)
    cx = width * 0.5
    cy = height * 0.5
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


if __name__ == "__main__":  # diagnóstico rápido por línea de comandos
    import sys

    if len(sys.argv) != 2:
        print("Uso: python3 mesh_utils.py <ruta_al_mars2020tube.stl>")
        raise SystemExit(1)

    model = load_tube_mesh(sys.argv[1])
    print("Malla cargada y escalada correctamente.")
    print(f"  extents (X,Y,Z) = {model.mesh.extents}")
    print(f"  longitud        = {model.length:.4f} m")
    print(f"  radio           = {model.radius:.4f} m")
    print(f"  nº keypoints    = {len(model.keypoints)}")
