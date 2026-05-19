#!/usr/bin/env python3
"""
pose_solver.py — Recuperación de la pose 6D a partir de un match LINEMOD.

LINEMOD devuelve (template_id, x, y, similarity). Este módulo traduce eso a
una pose 6D del tubo:

  ROTACIÓN
    Viene directamente del `template_id`: la tabla de poses de la
    LinemodDatabase guarda la R_cam_obj con la que se renderizó ese template.

  TRASLACIÓN
    Se recupera retroproyectando el centroide del objeto (deducido del match)
    contra el plano del suelo, cuya altura es conocida (cámara cenital fija).
    Esto es robusto y coherente con la geometría que ya usa el proyecto.

  REFINO OPCIONAL (PnP)
    Si se dispone de una máscara/keypoints fiables, cv2.solvePnP afina la
    pose. Por defecto está desactivado: para una cámara cenital y un objeto
    de revolución, la pose LINEMOD + retroproyección suele ser suficiente.

Todas las poses que devuelve este módulo están en el FRAME ÓPTICO de la
cámara. El nodo ROS las transforma a `base_link` con tf2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Se requiere OpenCV (cv2).") from exc


@dataclass
class PoseEstimate:
    """Resultado de la estimación de pose en el frame óptico de la cámara."""
    R_cam_obj: np.ndarray      # rotación 3x3 objeto->cámara
    t_cam_obj: np.ndarray      # traslación (3,) en metros
    similarity: float          # similitud LINEMOD [0..100]
    template_id: int
    matched_uv: tuple[float, float]  # centroide del objeto en la imagen completa

    def as_matrix(self) -> np.ndarray:
        """Devuelve la pose como matriz homogénea 4x4."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R_cam_obj
        T[:3, 3] = self.t_cam_obj
        return T


def backproject_to_plane(
    uv: tuple[float, float],
    K: np.ndarray,
    R_cam_plane: np.ndarray,
    t_cam_plane: np.ndarray,
    plane_z: float = 0.0,
) -> np.ndarray | None:
    """Retroproyecta un píxel hasta su intersección con un plano horizontal.

    Parámetros
    ----------
    uv          : píxel (u, v) en la imagen.
    K           : intrínseca 3x3.
    R_cam_plane,
    t_cam_plane : transformación frame_del_plano <- cámara (p.ej. base<-cam).
    plane_z     : altura del plano en el frame del plano [m].

    Devuelve
    --------
    El punto 3D (3,) en el frame del plano, o None si el rayo es paralelo.
    """
    u, v = uv
    # Rayo en el frame de la cámara.
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    # Rayo y origen de cámara expresados en el frame del plano.
    ray_plane = R_cam_plane @ ray_cam
    origin_plane = t_cam_plane

    if abs(ray_plane[2]) < 1e-9:
        return None  # rayo paralelo al plano

    # origin.z + s * ray.z = plane_z  ->  s = (plane_z - origin.z) / ray.z
    s = (plane_z - origin_plane[2]) / ray_plane[2]
    if s <= 0.0:
        return None  # intersección detrás de la cámara
    return origin_plane + s * ray_plane


def solve_pose_from_match(
    match,
    db,
    roi_offset: tuple[int, int],
    R_cam_plane: np.ndarray,
    t_cam_plane: np.ndarray,
    plane_z: float,
) -> PoseEstimate | None:
    """Convierte un match LINEMOD en una pose 6D en el frame de la cámara.

    Parámetros
    ----------
    match       : objeto match de cv2.linemod (atributos x, y, template_id...).
    db          : LinemodDatabase con la tabla de poses.
    roi_offset  : (x0, y0) — esquina del ROI recortado del bbox YOLO, para
                  pasar las coordenadas del match a la imagen completa.
    R_cam_plane,
    t_cam_plane : transformación plano <- cámara (base_link <- cámara óptica).
    plane_z     : altura del plano del suelo en el frame del plano.

    Devuelve
    --------
    PoseEstimate en el frame óptico de la cámara, o None si falla.
    """
    entry = db.entries.get(int(match.template_id))
    if entry is None:
        return None

    # ── Centroide del objeto en la imagen completa ──────────────────────────
    # match.(x,y) es la esquina del template en el ROI; sumamos el offset del
    # centroide (guardado al entrenar) y el offset del ROI.
    du, dv = entry.centroid_offset
    cu = match.x + du + roi_offset[0]
    cv_ = match.y + dv + roi_offset[1]

    # ── Traslación: retroproyección del centroide al plano del suelo ────────
    p_plane = backproject_to_plane((cu, cv_), db.K, R_cam_plane, t_cam_plane, plane_z)
    if p_plane is None:
        return None

    # El punto está en el frame del plano; lo pasamos al frame de la cámara
    # para mantener PoseEstimate coherente (R_cam_obj ya está en cámara).
    R_plane_cam = R_cam_plane.T
    t_plane_cam = -R_plane_cam @ t_cam_plane
    t_cam_obj = R_plane_cam @ p_plane + t_plane_cam

    return PoseEstimate(
        R_cam_obj=entry.rotation.copy(),
        t_cam_obj=t_cam_obj,
        similarity=float(match.similarity),
        template_id=int(match.template_id),
        matched_uv=(float(cu), float(cv_)),
    )


def refine_with_pnp(
    pose: PoseEstimate,
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    K: np.ndarray,
) -> PoseEstimate:
    """Refina una PoseEstimate con cv2.solvePnP (refino iterativo).

    Úsalo solo cuando dispongas de correspondencias 2D fiables (p.ej. de la
    máscara de segmentación de YOLO). Si PnP falla, se devuelve la pose
    original sin modificar.

    keypoints_3d : (N,3) puntos en el frame del objeto (TubeModel.keypoints).
    keypoints_2d : (N,2) sus proyecciones observadas en la imagen.
    """
    if len(keypoints_3d) < 4 or len(keypoints_2d) < 4:
        return pose

    rvec0, _ = cv2.Rodrigues(pose.R_cam_obj)
    tvec0 = pose.t_cam_obj.reshape(3, 1).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(keypoints_3d, dtype=np.float64),
        np.asarray(keypoints_2d, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        distCoeffs=None,
        rvec=rvec0, tvec=tvec0,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return pose

    R_ref, _ = cv2.Rodrigues(rvec)
    return PoseEstimate(
        R_cam_obj=R_ref,
        t_cam_obj=tvec.reshape(3),
        similarity=pose.similarity,
        template_id=pose.template_id,
        matched_uv=pose.matched_uv,
    )


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convierte una matriz de rotación 3x3 a cuaternión (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
