#!/usr/bin/env python3
"""
compute_grasp_footprint.py — Huella de agarre top-down del PAROL6 y banda
operativa de la NavCam, por cinemática directa muestreada del URDF.

Responde a "¿dónde puede estar un tubo apoyado en el suelo para que el brazo
lo agarre desde arriba?" y, proyectando esa huella con la pose del mástil,
"¿a qué distancia y en qué zona de la imagen lo verá la NavCam?".

Método (sin ROS):
  * Se parsea la cadena base_link→L1…L6 de parol6.urdf (origen, eje, límites)
    y el joint fijo L6→gripper_base_link del gripper.
  * Se muestrean configuraciones uniformes dentro de los límites y se
    conservan las que dejan el eje Z de gripper_base_link vertical (±ang_tol)
    con el centro del tubo (gripper − finger_offset·z_g) a la altura de un
    tubo apoyado (±z_tol). Es la convención de tube_grasp_pose_node.py.
  * La huella son las celdas XY (frame base_link) con al menos una solución,
    menos la placa de la base (bbox de base_link.STL + margen).
  * Se proyecta cada celda con la NavCam en mástil (misma pose que los launch)
    para dar la distancia cámara→tubo y la posición en imagen.

Salida: JSON con las celdas (lo consume synthetic_dataset_generator_v2.py en
`placement_mode: footprint`) y, opcionalmente, una figura.

Uso:
    ros2 run parol6_vision compute_grasp_footprint.py \
        --out ros_parol/src/parol6_vision/config/grasp_footprint_cells.json --png huella.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_mat(*rpy)
    T[:3, 3] = xyz
    return T


def parse_arm_chain(urdf_path):
    """Cadena de joints revolutos base_link→…→L6_link del URDF, en orden."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall('joint'):
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        a = j.find('axis')
        axis = [float(v) for v in (a.get('xyz') if a is not None else '0 0 1').split()]
        lim = j.find('limit')
        lo = float(lim.get('lower')) if lim is not None and lim.get('lower') else -math.pi
        hi = float(lim.get('upper')) if lim is not None and lim.get('upper') else math.pi
        joints[parent] = dict(name=j.get('name'), type=j.get('type'), child=child,
                              xyz=xyz, rpy=rpy, axis=axis, lo=lo, hi=hi)
    chain, link = [], 'base_link'
    while link in joints:
        j = joints[link]
        if j['type'] in ('revolute', 'continuous'):
            chain.append(j)
        link = j['child']
    if len(chain) != 6:
        raise RuntimeError(f'Se esperaban 6 joints revolutos en la cadena, hay {len(chain)}')
    return chain


def parse_gripper_joint(xacro_path):
    txt = open(xacro_path).read()
    m = re.search(r'<joint name="gripper_base_joint"[^>]*>.*?<origin xyz="([^"]+)" rpy="([^"]+)"',
                  txt, re.S)
    if not m:
        raise RuntimeError('No encuentro gripper_base_joint en ' + xacro_path)
    return [float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()]


def fk_batch(chain, T_grip, Q):
    """FK vectorizada: Q (N,6) → R (N,3,3), p (N,3) de gripper_base_link."""
    N = Q.shape[0]
    M = np.tile(np.eye(4), (N, 1, 1))
    for k, j in enumerate(chain):
        axis = np.asarray(j['axis'], dtype=float)
        if abs(abs(axis[2]) - 1.0) > 1e-6 or abs(axis[0]) > 1e-6 or abs(axis[1]) > 1e-6:
            raise RuntimeError(f"Joint {j['name']}: solo se soportan ejes ±Z")
        th = np.sign(axis[2]) * Q[:, k]
        c, s = np.cos(th), np.sin(th)
        Rz = np.zeros((N, 4, 4))
        Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
        Rz[:, 2, 2] = 1; Rz[:, 3, 3] = 1
        M = M @ make_T(j['xyz'], j['rpy']) @ Rz
    M = M @ T_grip
    return M[:, :3, :3], M[:, :3, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--urdf', default=None, help='parol6.urdf (default: share de parol6_description)')
    ap.add_argument('--gripper-xacro', default=None, help='gripper.urdf.xacro (default: mismo share)')
    ap.add_argument('--samples', type=int, default=8_000_000)
    ap.add_argument('--ang-tol-deg', type=float, default=10.0, help='tolerancia de verticalidad del gripper')
    ap.add_argument('--z-tol', type=float, default=0.02, help='tolerancia de altura del centro del tubo [m]')
    ap.add_argument('--finger-offset', type=float, default=0.035, help='gripper_base_link → centro entre dedos [m]')
    ap.add_argument('--tube-radius', type=float, default=0.01016)
    ap.add_argument('--cell', type=float, default=0.02, help='tamaño de celda [m]')
    ap.add_argument('--base-bbox', type=float, nargs=4, default=[-0.178, 0.05, -0.066, 0.059],
                    help='xmin xmax ymin ymax de la placa base_link [m] (de base_link.STL)')
    ap.add_argument('--base-margin', type=float, default=0.02)
    ap.add_argument('--mast', type=float, nargs=6, default=[-0.3, -0.3, 1.11, 0.0, 0.91, 0.3],
                    help='pose del rig NavCam en base_link: x y z roll pitch yaw')
    ap.add_argument('--hfov', type=float, default=1.1344640137963142, help='FOV horizontal NavCam [rad]')
    ap.add_argument('--img', type=int, nargs=2, default=[1024, 1024])
    ap.add_argument('--stereo-dead-px', type=int, default=192,
                    help='columnas sin disparidad SGBM (num_disparities/scale)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, help='JSON de salida con las celdas')
    ap.add_argument('--png', default=None, help='figura opcional')
    args = ap.parse_args()

    if args.urdf is None or args.gripper_xacro is None:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('parol6_description')
        args.urdf = args.urdf or os.path.join(share, 'urdf', 'parol6.urdf')
        args.gripper_xacro = args.gripper_xacro or os.path.join(share, 'urdf', 'gripper.urdf.xacro')

    chain = parse_arm_chain(args.urdf)
    g_xyz, g_rpy = parse_gripper_joint(args.gripper_xacro)
    T_grip = make_T(g_xyz, g_rpy)
    print('cadena:', ' → '.join(j['name'] for j in chain), '| gripper offset', g_xyz, g_rpy)

    rng = np.random.default_rng(args.seed)
    lo = np.array([j['lo'] for j in chain])
    hi = np.array([j['hi'] for j in chain])
    z_tube = args.tube_radius
    ang_tol = math.radians(args.ang_tol_deg)
    hits = []
    chunk = 500_000
    for _ in range(max(1, args.samples // chunk)):
        Q = lo + (hi - lo) * rng.random((chunk, 6))
        R, p = fk_batch(chain, T_grip, Q)
        zax = R[:, :, 2]
        up = np.arccos(np.clip(zax[:, 2], -1, 1)) < ang_tol
        tube = p - args.finger_offset * zax
        sel = up & (np.abs(tube[:, 2] - z_tube) < args.z_tol)
        hits.append(tube[sel][:, :2])
    xy = np.vstack(hits)
    print(f'configuraciones muestreadas: {args.samples}  → agarres top-down válidos: {len(xy)}')

    cells = set(map(tuple, np.floor(xy / args.cell).astype(int)))
    C = np.array(sorted(cells)) * args.cell + args.cell / 2
    bx0, bx1, by0, by1 = args.base_bbox
    m = args.base_margin
    in_base = ((C[:, 0] > bx0 - m) & (C[:, 0] < bx1 + m)
               & (C[:, 1] > by0 - m) & (C[:, 1] < by1 + m))
    C = C[~in_base]
    r = np.hypot(C[:, 0], C[:, 1])
    print(f'celdas de {args.cell*100:.0f} cm: {len(cells)} → sin placa base: {len(C)} '
          f'(área ≈ {len(C) * args.cell ** 2:.3f} m²)')
    print(f'radio desde base_link: min {r.min():.2f}  p5 {np.percentile(r, 5):.2f}  '
          f'med {np.median(r):.2f}  p95 {np.percentile(r, 95):.2f}  max {r.max():.2f} m')

    # Proyección con la NavCam en mástil (frames como en los launch)
    T_b_rig = make_T(args.mast[:3], args.mast[3:])
    T_rig_l = make_T([0, 0.075, 0], [0, 0, 0])
    T_l_o = make_T([0, 0, 0], [-1.5707963267949, 0, -1.5707963267949])
    T_b_o = T_b_rig @ T_rig_l @ T_l_o
    T_o_b = np.linalg.inv(T_b_o)
    W, H = args.img
    fx = (W / 2) / math.tan(args.hfov / 2)
    cx, cy = W / 2 + 0.5, H / 2 + 0.5
    P = np.column_stack([C, np.full(len(C), z_tube)])
    Pc = (T_o_b[:3, :3] @ P.T).T + T_o_b[:3, 3]
    d = np.linalg.norm(Pc, axis=1)
    u = fx * Pc[:, 0] / Pc[:, 2] + cx
    v = fx * Pc[:, 1] / Pc[:, 2] + cy
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Pc[:, 2] > 0)
    print(f'distancia cámara→tubo: min {d.min():.3f}  p5 {np.percentile(d, 5):.3f}  '
          f'med {np.median(d):.3f}  p95 {np.percentile(d, 95):.3f}  max {d.max():.3f} m')
    print(f'en imagen: {inside.mean()*100:.1f}% dentro; u [{u[inside].min():.0f},{u[inside].max():.0f}] '
          f'v [{v[inside].min():.0f},{v[inside].max():.0f}]; '
          f'en zona muerta estéreo (u<{args.stereo_dead_px}): {(u[inside] < args.stereo_dead_px).mean()*100:.1f}%')

    out = {
        'cell_size_m': args.cell,
        'z_tube_center_m': z_tube,
        'frame': 'base_link',
        'method': (f'FK muestreada ({args.samples} configs), gripper z vertical ±{args.ang_tol_deg}°, '
                   f'centro del tubo a {z_tube} m ±{args.z_tol}, finger_offset {args.finger_offset}; '
                   f'excluida la placa de la base {args.base_bbox} +{args.base_margin} m'),
        'camera_distance_m': {'min': round(float(d.min()), 3), 'median': round(float(np.median(d)), 3),
                              'max': round(float(d.max()), 3)},
        'cells_xy': [[round(float(x), 3), round(float(y), 3)] for x, y in C],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('guardado', args.out)

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sc = ax[0].scatter(C[:, 0], C[:, 1], c=d, s=8, cmap='viridis')
        plt.colorbar(sc, ax=ax[0], label='dist cámara→tubo [m]')
        ax[0].plot(0, 0, 'k^', ms=9, label='base_link')
        ax[0].plot(args.mast[0], args.mast[1], 'bs', ms=7, label='NavCam (x,y)')
        ax[0].set_aspect('equal'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        ax[0].set_xlabel('x base_link [m]'); ax[0].set_ylabel('y base_link [m]')
        ax[0].set_title('Huella de agarre top-down')
        ax[1].scatter(u[inside], v[inside], c=d[inside], s=8, cmap='viridis')
        ax[1].axvline(args.stereo_dead_px, color='r', ls='--', label='zona muerta estéreo')
        ax[1].set_xlim(0, W); ax[1].set_ylim(H, 0); ax[1].set_aspect('equal'); ax[1].legend(fontsize=8)
        ax[1].set_title('Huella en la imagen izquierda'); ax[1].set_xlabel('u [px]'); ax[1].set_ylabel('v [px]')
        fig.tight_layout(); fig.savefig(args.png, dpi=110)
        print('figura', args.png)
    return 0


if __name__ == '__main__':
    sys.exit(main())
