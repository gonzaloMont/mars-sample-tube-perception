#!/usr/bin/env python3
"""
rendering.py — Muestreo de viewpoints y renderer offline para LINEMOD.

Este módulo SOLO se usa en la fase offline (train_linemod_templates.py).
Genera, para cada orientación muestreada del tubo, una imagen RGB sintética
y su máscara, que `linemod_db.add_template()` convertirá en un template.

Estrategia de muestreo
──────────────────────
La cámara cenital es fija y el tubo descansa sobre el suelo. Por tanto el
eje largo del tubo es prácticamente horizontal y la variación dominante es:

  • yaw   — rotación del tubo alrededor de la vertical (0..360°). DENSO.
  • tilt  — pequeñas inclinaciones (suelo no perfectamente plano, cámara no
            perfectamente cenital). POCAS muestras.

La rotación del cilindro sobre su PROPIO eje largo es irrelevante (simetría
de revolución), así que NO se muestrea: esto reduce drásticamente el número
de templates sin perder cobertura.

Convención de frames
─────────────────────
Se usa la convención óptica de ROS para la cámara:
    +Z  hacia la escena (delante)
    +X  derecha,  +Y  abajo
Cada viewpoint se expresa como T_cam_obj (pose del objeto en el frame de la
cámara). El renderer dibuja el objeto en esa pose. La pose se ALMACENA junto
al template: es la "etiqueta" que LINEMOD por sí solo no produce.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# pyrender necesita un backend EGL/OSMesa para render sin display.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import pyrender  # noqa: E402
    import trimesh   # noqa: E402  (solo para el type hint de la malla)
except ImportError as exc:  # pragma: no cover - dependencia offline
    raise ImportError(
        "El renderer offline requiere 'pyrender' y 'trimesh'.\n"
        "Instálalos con:  pip install pyrender trimesh\n"
        "Si no hay GPU, exporta antes:  export PYOPENGL_PLATFORM=osmesa"
    ) from exc


# ── Helpers de rotación (sin dependencia de scipy) ──────────────────────────
def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass
class Viewpoint:
    """Una orientación muestreada del tubo, con metadatos para depuración."""
    rotation: np.ndarray  # R_cam_obj (3x3) — orientación del objeto en cámara
    yaw_deg: float
    tilt_deg: float


def sample_tube_viewpoints(
    yaw_steps: int = 36,
    tilt_steps: int = 3,
    tilt_max_deg: float = 18.0,
) -> list[Viewpoint]:
    """Genera la rejilla de orientaciones (yaw × tilt) para el tubo.

    Con los valores por defecto: 36 yaws × 3 tilts = 108 templates, que es
    un buen compromiso cobertura/velocidad para una cámara cenital fija.

    Rotación base
    -------------
    El frame del objeto tiene el eje LARGO en +Z (ver mesh_utils). Para que
    el tubo aparezca "tumbado" bajo una cámara cenital, su eje largo debe
    quedar contenido en el plano de imagen. Rotamos 90° en X para llevar
    el +Z del objeto al +Y de la cámara, y luego aplicamos yaw/tilt.
    """
    base = _rot_x(np.pi / 2.0)  # tubo tumbado: eje largo -> plano imagen

    if tilt_steps <= 1:
        tilts = [0.0]
    else:
        tilts = list(np.linspace(-tilt_max_deg, tilt_max_deg, tilt_steps))

    viewpoints: list[Viewpoint] = []
    for tilt_deg in tilts:
        tilt = np.radians(tilt_deg)
        for i in range(yaw_steps):
            yaw_deg = 360.0 * i / yaw_steps
            yaw = np.radians(yaw_deg)
            # R = yaw (alrededor del eje óptico) ∘ tilt ∘ base
            R = _rot_z(yaw) @ _rot_x(tilt) @ base
            viewpoints.append(Viewpoint(rotation=R, yaw_deg=yaw_deg, tilt_deg=tilt_deg))
    return viewpoints


@dataclass
class RenderResult:
    """Salida de un render: imagen, máscara y centroide proyectado."""
    color: np.ndarray          # BGR uint8 (HxWx3) — formato OpenCV/LINEMOD
    mask: np.ndarray           # uint8 (HxW) — 255 dentro del objeto
    centroid_uv: tuple[float, float]  # centroide del objeto en píxeles


class TemplateRenderer:
    """Renderer offscreen para generar las imágenes de los templates.

    Renderiza el tubo a UNA distancia fija (la distancia real cámara-suelo
    del montaje), porque LINEMOD basado en color no es robusto a cambios de
    escala. Con cámara cenital fija esa distancia es esencialmente constante,
    lo que convierte la limitación de LINEMOD en algo irrelevante para este
    proyecto.
    """

    def __init__(
        self,
        tube_mesh: "trimesh.Trimesh",
        K: np.ndarray,
        width: int,
        height: int,
        render_distance: float,
    ) -> None:
        """
        Parámetros
        ----------
        tube_mesh       : malla YA escalada y centrada (ver mesh_utils).
        K               : matriz intrínseca 3x3 de la cámara cenital.
        width, height   : resolución de la imagen de la cámara.
        render_distance : distancia cámara→objeto [m] a la que se renderiza
                          (≈ altura de la cámara sobre el plano del tubo).
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.render_distance = float(render_distance)

        self._pyrender_mesh = pyrender.Mesh.from_trimesh(tube_mesh, smooth=False)

        # Cámara con intrínsecas exactas de la cámara cenital de Gazebo.
        self._camera = pyrender.IntrinsicsCamera(
            fx=self.K[0, 0], fy=self.K[1, 1],
            cx=self.K[0, 2], cy=self.K[1, 2],
            znear=0.01, zfar=10.0,
        )
        self._renderer = pyrender.OffscreenRenderer(self.width, self.height)

    def render(self, R_cam_obj: np.ndarray) -> RenderResult | None:
        """Renderiza el tubo con orientación R_cam_obj a la distancia fija.

        Devuelve None si el objeto no es visible (no debería ocurrir con un
        muestreo razonable, pero se contempla por robustez).
        """
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=[0.35, 0.35, 0.35])

        # Pose del objeto en el frame de la cámara: rotación muestreada +
        # traslación fija sobre el eje óptico (objeto centrado en imagen).
        T_cam_obj = np.eye(4, dtype=np.float64)
        T_cam_obj[:3, :3] = R_cam_obj
        T_cam_obj[:3, 3] = [0.0, 0.0, self.render_distance]
        scene.add(self._pyrender_mesh, pose=T_cam_obj)

        # pyrender usa convención OpenGL (cámara mira a -Z); la convención
        # óptica de ROS mira a +Z. La conversión es una rotación de 180° en X.
        gl_from_cv = np.diag([1.0, -1.0, -1.0, 1.0])
        scene.add(self._camera, pose=gl_from_cv)

        # Iluminación variada para gradientes de color ricos.
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=gl_from_cv)

        color, depth = self._renderer.render(scene)

        mask = (depth > 0).astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            return None

        # Centroide del objeto proyectado (para recuperar traslación luego).
        ys, xs = np.nonzero(mask)
        centroid_uv = (float(xs.mean()), float(ys.mean()))

        # pyrender devuelve RGB; LINEMOD/OpenCV trabajan en BGR.
        color_bgr = color[:, :, ::-1].copy()
        return RenderResult(color=color_bgr, mask=mask, centroid_uv=centroid_uv)

    def close(self) -> None:
        """Libera el contexto OpenGL del renderer."""
        self._renderer.delete()
