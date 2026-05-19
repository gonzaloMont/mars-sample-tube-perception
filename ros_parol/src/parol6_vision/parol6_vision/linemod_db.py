#!/usr/bin/env python3
"""
linemod_db.py — Detector cv2.linemod + base de datos de templates y poses.

Este módulo es el CORAZÓN compartido entre entrenamiento e inferencia:

  • make_color_detector()  — crea el detector LINEMOD con una configuración
                             idéntica en ambas fases (imprescindible: un
                             detector entrenado debe leerse con la MISMA
                             pirámide T).

  • LinemodDatabase        — empaqueta el detector + la tabla
                             template_id → pose 6D. LINEMOD por sí solo NO
                             guarda poses; esa tabla es lo que convierte un
                             detector de matching en un estimador de pose.

Nota sobre modalidades
──────────────────────
La cámara cenital de overhead_camera.urdf.xacro es RGB (sensor `camera`,
no `depth`). Por eso se usa SOLO la modalidad ColorGradient. Si en el futuro
se cambia el sensor a `depth`, se puede añadir DepthNormal para mayor
robustez (ver README).

Nota sobre la API de cv2.linemod
─────────────────────────────────
La API Python de cv2.linemod varía ligeramente entre versiones de OpenCV
y requiere opencv-contrib. Este módulo aísla esas diferencias en un único
sitio y degrada con mensajes claros si el módulo no está disponible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Se requiere OpenCV (cv2).") from exc

if not hasattr(cv2, "linemod"):
    raise ImportError(
        "El módulo cv2.linemod no está disponible.\n"
        "Instala OpenCV con contrib:  pip install opencv-contrib-python"
    )


# Pirámide de muestreo T de LINEMOD. Dos niveles es el valor habitual del
# sample oficial; debe ser IDÉNTICO en entrenamiento e inferencia.
DEFAULT_T_PYRAMID: tuple[int, int] = (5, 8)


def make_color_detector(t_pyramid: tuple[int, int] = DEFAULT_T_PYRAMID):
    """Crea un detector LINEMOD con una sola modalidad de color (RGB)."""
    # El nombre del constructor de la modalidad cambió entre versiones.
        # Parámetros optimizados para un cilindro liso (tubo):
        #   - weak_threshold=10: más permisivo con gradientes débiles
        #   - num_features=20: reducido de 50 (tubos lisos tienen pocos features)
        #   - strong_threshold=40: menos restrictivo para capturar más características
        # Estos valores se eligieron tras observar que ~72% de templates se rechazaban
        # con los valores anteriores. Ver LINEMOD_DIAGNOSTICO.md para detalles.
    if hasattr(cv2.linemod, "ColorGradient_create"):
        modality = cv2.linemod.ColorGradient_create(
            weak_threshold=10, num_features=20, strong_threshold=40
        )
    else:
        modality = cv2.linemod.ColorGradient()
    return cv2.linemod.Detector([modality], list(t_pyramid))


@dataclass
class TemplateEntry:
    """Metadatos de un template: su pose y dónde queda el objeto en él.

    LINEMOD almacena el template (gradientes) internamente; aquí guardamos
    lo que LINEMOD NO sabe:
      • la pose 6D con la que se renderizó (rotación = etiqueta de la vista),
      • el offset del centroide del objeto respecto a la esquina del bbox
        del template, necesario para recuperar la traslación tras un match.
    """
    template_id: int
    rotation: np.ndarray              # R_cam_obj (3x3)
    centroid_offset: tuple[float, float]  # (du, dv) centroide - esquina bbox
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0


@dataclass
class LinemodDatabase:
    """Detector LINEMOD + tabla de poses, con persistencia a disco."""

    class_id: str = "tube"
    t_pyramid: tuple[int, int] = DEFAULT_T_PYRAMID
    render_distance: float = 0.72
    image_size: tuple[int, int] = (640, 480)   # (width, height)
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    entries: dict[int, TemplateEntry] = field(default_factory=dict)
    detector: object = None  # cv2.linemod.Detector

    # ── Construcción ────────────────────────────────────────────────────────
    def ensure_detector(self):
        """Crea el detector si aún no existe."""
        if self.detector is None:
            self.detector = make_color_detector(self.t_pyramid)
        return self.detector

    def add_template(
        self,
        color_bgr: np.ndarray,
        mask: np.ndarray,
        rotation: np.ndarray,
        centroid_uv: tuple[float, float],
        yaw_deg: float = 0.0,
        tilt_deg: float = 0.0,
    ) -> int | None:
        """Añade un template renderizado a la base de datos.

        Devuelve el template_id asignado, o None si LINEMOD lo rechaza
        (suele ocurrir si el objeto tiene muy pocos gradientes detectables).
        """
        det = self.ensure_detector()
        # addTemplate -> (template_id, bounding_box). El bbox es (x, y, w, h)
        # del template dentro de la imagen renderizada.
        ret = det.addTemplate([color_bgr], self.class_id, mask)
        template_id, bbox = ret[0], ret[1]
        if template_id < 0:
            return None

        bx, by = float(bbox[0]), float(bbox[1])
        offset = (centroid_uv[0] - bx, centroid_uv[1] - by)
        self.entries[template_id] = TemplateEntry(
            template_id=template_id,
            rotation=np.asarray(rotation, dtype=np.float64),
            centroid_offset=offset,
            yaw_deg=yaw_deg,
            tilt_deg=tilt_deg,
        )
        return template_id

    # ── Persistencia ────────────────────────────────────────────────────────
    def save(self, out_dir: str) -> None:
        """Guarda el detector (templates) y la tabla de poses en `out_dir`.

        Se generan:
          • templates_<class_id>.yml  — templates LINEMOD (writeClasses).
          • linemod_meta.json         — config del detector + intrínsecas.
          • linemod_poses.npz         — tabla template_id -> pose/offsets.
        """
        os.makedirs(out_dir, exist_ok=True)
        det = self.ensure_detector()

        # 1) Templates LINEMOD. writeClasses usa un patrón con %s = class_id.
        templates_fmt = os.path.join(out_dir, "templates_%s.yml")
        det.writeClasses(templates_fmt)

        # 2) Metadatos del detector e intrínsecas.
        meta = {
            "class_id": self.class_id,
            "t_pyramid": list(self.t_pyramid),
            "render_distance": self.render_distance,
            "image_size": list(self.image_size),
            "K": np.asarray(self.K, dtype=np.float64).tolist(),
            "num_templates": len(self.entries),
        }
        with open(os.path.join(out_dir, "linemod_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        # 3) Tabla de poses (lo que LINEMOD no guarda).
        ids = sorted(self.entries.keys())
        rotations = np.stack([self.entries[i].rotation for i in ids], axis=0)
        offsets = np.array([self.entries[i].centroid_offset for i in ids], dtype=np.float64)
        yaws = np.array([self.entries[i].yaw_deg for i in ids], dtype=np.float64)
        tilts = np.array([self.entries[i].tilt_deg for i in ids], dtype=np.float64)
        np.savez(
            os.path.join(out_dir, "linemod_poses.npz"),
            template_ids=np.array(ids, dtype=np.int64),
            rotations=rotations,
            centroid_offsets=offsets,
            yaws=yaws,
            tilts=tilts,
        )

    @classmethod
    def load(cls, in_dir: str) -> "LinemodDatabase":
        """Reconstruye una LinemodDatabase desde disco para la inferencia."""
        with open(os.path.join(in_dir, "linemod_meta.json"), "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        db = cls(
            class_id=meta["class_id"],
            t_pyramid=tuple(meta["t_pyramid"]),
            render_distance=float(meta["render_distance"]),
            image_size=tuple(meta["image_size"]),
            K=np.array(meta["K"], dtype=np.float64),
        )

        # Recrear el detector con la MISMA pirámide T y leer los templates.
        det = make_color_detector(db.t_pyramid)
        templates_fmt = os.path.join(in_dir, "templates_%s.yml")
        # readClasses necesita conocer las class_ids a leer.
        try:
            det.readClasses([db.class_id], templates_fmt)
        except TypeError:
            # Algunas versiones aceptan solo el patrón.
            det.readClasses(templates_fmt)
        db.detector = det

        # Cargar la tabla de poses.
        data = np.load(os.path.join(in_dir, "linemod_poses.npz"))
        ids = data["template_ids"]
        rotations = data["rotations"]
        offsets = data["centroid_offsets"]
        yaws = data["yaws"]
        tilts = data["tilts"]
        for k, tid in enumerate(ids):
            db.entries[int(tid)] = TemplateEntry(
                template_id=int(tid),
                rotation=rotations[k],
                centroid_offset=(float(offsets[k][0]), float(offsets[k][1])),
                yaw_deg=float(yaws[k]),
                tilt_deg=float(tilts[k]),
            )
        return db

    # ── Inferencia ──────────────────────────────────────────────────────────
    def match(self, color_bgr: np.ndarray, threshold: float = 80.0):
        """Ejecuta LINEMOD sobre una imagen (normalmente el ROI del bbox YOLO).

        Devuelve la lista de matches ordenada por similitud descendente.
        Cada match tiene atributos: x, y, similarity, class_id, template_id.
        """
        det = self.ensure_detector()
        img = np.asarray(color_bgr)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"LINEMOD espera una imagen BGR HxWx3, recibida shape={img.shape}."
            )
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        if img.size == 0:
            raise ValueError("LINEMOD recibió una imagen vacía.")

        # cv2.linemod es una extensión nativa y puede segfault si recibe una
        # matriz no contigua o con dtype inesperado. Normalizamos antes.
        out = det.match([img], float(threshold), [self.class_id])
        matches = out[0] if out else []
        return sorted(matches, key=lambda m: m.similarity, reverse=True)
