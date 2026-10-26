#!/usr/bin/env python3
"""
stereo_utils.py — Núcleo de la reconstrucción estéreo de la NavCam (SGBM +
matcher derecho + filtro WLS + fusión de la zona muerta), SIN ROS.

Es la ÚNICA implementación: `stereo_pointcloud_node.py` la envuelve en ROS y
`eval_pose_dataset.py` la ejecuta offline sobre los pares del dataset. Así la
nube que ve ICP en la evaluación offline es, paso a paso, la misma que en el
pipeline en vivo (en Gazebo las cámaras no tienen ruido, así que la fusión
multi-frame del nodo no añade información: un par único basta).

Convenciones
------------
* `R`, `T`: pose de la cámara IZQUIERDA vista desde la DERECHA, convención de
  cv2.stereoRectify:  P_right = R · P_left + T.  Es exactamente
  `cam_R_l2r` / `cam_t_l2r` de `scene_camera.json` (dataset v2.1). El nodo la
  obtiene invirtiendo la TF left←right.
* Las imágenes de entrada son las ORIGINALES (1024²); el reescalado
  `scale_factor` se aplica dentro (INTER_AREA) y K se escala en consecuencia.
* La nube sale en el frame óptico izquierdo (X derecha, Y abajo, Z delante),
  en metros, como PointCloud2 del nodo.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Se requiere OpenCV (cv2) con módulos contrib.") from exc


# Parámetros por defecto del nodo estéreo (única fuente de verdad; el nodo
# los declara a partir de este dict y config/stereo_pointcloud_params.yaml
# los repite para poder editarlos sin tocar código).
DEFAULT_PARAMS = {
    'min_disparity': 0,
    'num_disparities': 96,     # múltiplo de 16
    'block_size': 7,           # impar, 3–11
    'uniqueness_ratio': 5,
    'speckle_window': 100,
    'speckle_range': 2,
    'disp12_max_diff': 1,
    'min_depth': 0.30,         # m
    'max_depth': 2.0,          # m
    'scale_factor': 0.5,       # factor de reducción de la imagen
    'wls_lambda': 8000.0,
    'wls_sigma': 1.5,
    'accumulate_frames': 4,    # solo lo usa el nodo (fusión multi-frame)
}


def require_ximgproc() -> None:
    """Aborta con mensaje claro si falta opencv-contrib (WLS + RightMatcher)."""
    if not hasattr(cv2, 'ximgproc'):
        raise RuntimeError(
            'cv2.ximgproc no disponible (instala opencv-contrib-python). '
            'Es obligatorio: WLS, mapa de confianza y fusión de zona '
            'muerta dependen de él para que ICP reciba una nube fiable.')


class StereoMatcher:
    """SGBM (3WAY) + matcher derecho + WLS + relleno geométrico de la zona
    muerta, con rectificación calculada una sola vez.

    Parámetros
    ----------
    K1, K2       : intrínsecas 3x3 de las imágenes ORIGINALES (izq., der.).
    D1, D2       : distorsión (None/[] => sin distorsión).
    R, T         : P_right = R · P_left + T  (ver cabecera del módulo).
    img_size     : (ancho, alto) de las imágenes ORIGINALES.
    params       : dict con las claves de DEFAULT_PARAMS (las que falten se
                   toman del default).
    """

    def __init__(self, K1, D1, K2, D2, R, T, img_size, params=None):
        require_ximgproc()
        p = dict(DEFAULT_PARAMS)
        p.update(params or {})
        self.p = p
        scale = float(p['scale_factor'])
        self.scale = scale
        w, h = int(img_size[0]), int(img_size[1])
        self.img_size = (int(w * scale), int(h * scale))

        K1 = np.array(K1, dtype=np.float64).reshape(3, 3).copy()
        K2 = np.array(K2, dtype=np.float64).reshape(3, 3).copy()
        if scale != 1.0:
            K1[:2] *= scale
            K2[:2] *= scale
        D1 = np.array(D1 if D1 is not None else [], dtype=np.float64)
        D2 = np.array(D2 if D2 is not None else [], dtype=np.float64)
        if D1.size < 4:
            D1 = np.zeros(5, dtype=np.float64)
        if D2.size < 4:
            D2 = np.zeros(5, dtype=np.float64)
        R = np.array(R, dtype=np.float64).reshape(3, 3)
        T = np.array(T, dtype=np.float64).reshape(3, 1)

        # Rectificación: alpha=0.0 recorta solo a píxeles válidos, evita bordes
        R1, R2, P1_m, P2_m, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, self.img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
        self.Q = Q
        self.map1l, self.map2l = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1_m, self.img_size, cv2.CV_32FC1)
        self.map1r, self.map2r = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2_m, self.img_size, cv2.CV_32FC1)

        blk = int(p['block_size'])
        self.num_disp = int(p['num_disparities'])
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=int(p['min_disparity']),
            numDisparities=self.num_disp,
            blockSize=blk,
            P1=8 * blk * blk,     # penalización suave (1 canal grayscale)
            P2=32 * blk * blk,    # penalización fuerte
            disp12MaxDiff=int(p['disp12_max_diff']),
            uniquenessRatio=int(p['uniqueness_ratio']),
            speckleWindowSize=int(p['speckle_window']),
            speckleRange=int(p['speckle_range']),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.sgbm_r = cv2.ximgproc.createRightMatcher(self.sgbm)
        self.wls = cv2.ximgproc.createDisparityWLSFilter(self.sgbm)
        self.wls.setLambda(float(p['wls_lambda']))
        self.wls.setSigmaColor(float(p['wls_sigma']))

    # ── Construcción desde el dataset v2.1 ──────────────────────────────
    @classmethod
    def from_scene_camera(cls, cam_entry: dict, img_size, params=None):
        """Crea el matcher a partir de una entrada de `bop/scene_camera.json`
        con `cam_K`, `cam_K_right`, `cam_R_l2r`, `cam_t_l2r`."""
        return cls(cam_entry['cam_K'], None, cam_entry['cam_K_right'], None,
                   cam_entry['cam_R_l2r'], cam_entry['cam_t_l2r'], img_size, params)

    # ── Procesado de un par ─────────────────────────────────────────────
    def preprocess(self, left_gray: np.ndarray, right_gray: np.ndarray):
        """Reescala (INTER_AREA) y rectifica el par (mono8)."""
        left = left_gray
        right = right_gray
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if right.ndim == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            left = cv2.resize(left, self.img_size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, self.img_size, interpolation=cv2.INTER_AREA)
        left_rect = cv2.remap(left, self.map1l, self.map2l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map1r, self.map2r, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> dict:
        """Disparidad L→R filtrada (WLS) con la zona muerta SGBM sustituida
        por la disparidad R→L, más el mapa de confianza y el peso de fusión.

        Devuelve dict(disp, conf, weight, left_rect, right_rect), todo a la
        resolución de trabajo (`img_size`).
        """
        left_rect, right_rect = self.preprocess(left_gray, right_gray)

        # Disparidad L→R, R→L y suavizado WLS (matchers obligatorios)
        disp_l_raw = self.sgbm.compute(left_rect, right_rect)
        disp_r_raw = self.sgbm_r.compute(right_rect, left_rect)
        disp_f_raw = self.wls.filter(disp_l_raw, left_rect, None, disp_r_raw)

        disp = disp_f_raw.astype(np.float32) / 16.0   # disparidad filtrada L→R
        disp_r = disp_r_raw.astype(np.float32) / 16.0  # disparidad R→L

        # Fusión geométrica de la zona muerta SGBM: las primeras
        # `num_disparities` columnas L→R son inválidas por geometría; se
        # sustituyen de forma determinista por R→L (ver nodo).
        nd = self.num_disp
        disp[:, :nd] = disp_r[:, :nd]

        # Mapa de confianza WLS → pesos de fusión. En la zona muerta WLS nunca
        # evaluó nada: peso neutro (media de la región válida).
        conf = self.wls.getConfidenceMap().astype(np.float32)
        conf[:, :nd] = float(conf[:, nd:].mean()) if conf.shape[1] > nd else 1.0
        weight = np.clip(conf, 1.0, None)   # ≥ 1 → ningún frame pesa cero
        return {'disp': disp, 'conf': conf, 'weight': weight,
                'left_rect': left_rect, 'right_rect': right_rect}

    def reproject(self, disp: np.ndarray, return_organized: bool = False):
        """Disparidad (resolución de trabajo) → puntos 3D en el frame óptico
        izquierdo, filtrados por disparidad válida y [min_depth, max_depth].

        Devuelve (N,3) float32, o (pts3d HxWx3, valid HxW) si
        `return_organized`.
        """
        pts3d = cv2.reprojectImageTo3D(disp, self.Q)    # (H, W, 3)
        min_d = float(self.p['min_disparity'])
        z = pts3d[:, :, 2]
        valid = ((disp > min_d + 1.0) & np.isfinite(z)
                 & (z > float(self.p['min_depth']))
                 & (z < float(self.p['max_depth'])))
        if return_organized:
            return pts3d, valid
        return pts3d[valid].astype(np.float32)

    def compute_points(self, left_gray, right_gray) -> np.ndarray:
        """Atajo offline: par original → nube (N,3) de UN solo frame."""
        return self.reproject(self.compute(left_gray, right_gray)['disp'])


def disparity_debug_image(disp: np.ndarray, min_disparity: float = 0.0) -> np.ndarray:
    """Disparidad normalizada a mono8 (solo píxeles válidos) para depuración."""
    valid = disp > min_disparity + 1.0
    norm = np.zeros_like(disp, dtype=np.uint8)
    if valid.any():
        d_min, d_max = float(disp[valid].min()), float(disp[valid].max())
        if d_max > d_min:
            norm[valid] = ((disp[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return norm
