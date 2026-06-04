#!/usr/bin/env python3
"""
Genera un "recorte" optimizado del terreno marciano para pruebas de visión.

Toma la malla original (worlds/mars/mars_topografi.dae, 50x25 m / 81 200 tri) y
produce un parche ligero centrado en la posición del robot:

    - mars_patch.obj / .mtl : solo los triángulos a +-RADIUS m del robot,
                              ya "horneados" en coordenadas de mundo (pose
                              identidad en el .world). Reduccion tipica ~90 %.
    - model_texture.jpg     : la textura original reescalada (14 MB -> ~1 MB).

No depende de Blender/MeshLab/pycollada: parsea el COLLADA con la stdlib + numpy
y respeta la matriz de escala 0.25 del nodo Root del export de Blender.

Uso:
    python3 generate_mars_patch.py                 # +-6 m alrededor de (0,0)
    python3 generate_mars_patch.py --radius 8
    python3 generate_mars_patch.py --cx 10.3 --cy 5.0 --radius 6   # robot reubicado
    python3 generate_mars_patch.py --maxpx 2048    # textura mas/menos nitida
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "mars")
DEFAULT_DAE = os.path.join(SRC_DIR, "mars_topografi.dae")
DEFAULT_TEX = os.path.join(SRC_DIR, "model_texture.jpg")

# Pose del modelo 'my_mesh' en el mars.world original: x y z + yaw=pi
MODEL_T = np.array([20.311, 20.066, -2.36])
MODEL_YAW = math.pi


def _tag(e):
    return e.tag.split("}")[-1]


def load_collada(path):
    """Devuelve (positions[N,3], normals[M,3], uvs[K,2], p[ntri*3,stride], offsets, node_matrix)."""
    root = ET.parse(path).getroot()
    pos = nor = uv = None
    for fa in root.iter():
        if _tag(fa) != "float_array":
            continue
        i = fa.get("id", "")
        if "positions" in i:
            pos = np.fromstring(fa.text, sep=" ").reshape(-1, 3)
        elif "normals" in i:
            nor = np.fromstring(fa.text, sep=" ").reshape(-1, 3)
        elif "map-0" in i or "map-1" in i or "texcoord" in i.lower():
            uv = np.fromstring(fa.text, sep=" ").reshape(-1, 2)

    p = offsets = None
    for tg in root.iter():
        if _tag(tg) != "triangles":
            continue
        offsets = {inp.get("semantic"): int(inp.get("offset"))
                   for inp in tg if _tag(inp) == "input"}
        stride = max(offsets.values()) + 1
        ptext = [c for c in tg if _tag(c) == "p"][0].text
        p = np.fromstring(ptext, sep=" ", dtype=np.int64).reshape(-1, stride)
        break
    if p is None:
        raise SystemExit("No se encontro <triangles> en el DAE")

    node = np.eye(4)
    for nd in root.iter():
        if _tag(nd) == "node" and nd.get("id") == "Root":
            for m in nd:
                if _tag(m) == "matrix":
                    node = np.fromstring(m.text, sep=" ").reshape(4, 4)
    return pos, nor, uv, p, offsets, node


def to_world(pos, nor, node):
    """Aplica la matriz del nodo (escala 0.25) y la pose del modelo (yaw=pi)."""
    cz, sz = math.cos(MODEL_YAW), math.sin(MODEL_YAW)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Rn = node[:3, :3]
    Pw = (Rz @ ((Rn @ pos.T).T + node[:3, 3]).T).T + MODEL_T
    # Normales: solo rotacion (la escala uniforme no cambia direccion)
    Nl = (Rn @ nor.T).T
    Nl /= np.linalg.norm(Nl, axis=1, keepdims=True) + 1e-12
    Nw = (Rz @ Nl.T).T
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12
    return Pw, Nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dae", default=DEFAULT_DAE)
    ap.add_argument("--tex", default=DEFAULT_TEX)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--cx", type=float, default=0.0, help="centro X del recorte (world)")
    ap.add_argument("--cy", type=float, default=0.0, help="centro Y del recorte (world)")
    ap.add_argument("--radius", type=float, default=6.0, help="medio-lado del recorte [m]")
    ap.add_argument("--maxpx", type=int, default=2048, help="lado maximo de la textura [px]")
    args = ap.parse_args()

    pos, nor, uv, p, off, node = load_collada(args.dae)
    Pw, Nw = to_world(pos, nor, node)

    vidx = p[:, off["VERTEX"]]
    nidx = p[:, off.get("NORMAL", off["VERTEX"])]
    tidx = p[:, off.get("TEXCOORD", off["VERTEX"])]
    cP, cN, cT = Pw[vidx], Nw[nidx], uv[tidx]          # por-esquina
    faces = np.arange(len(vidx)).reshape(-1, 3)

    cen = cP[faces].mean(axis=1)
    keep = (np.abs(cen[:, 0] - args.cx) <= args.radius) & \
           (np.abs(cen[:, 1] - args.cy) <= args.radius)
    kf = faces[keep]
    if not len(kf):
        raise SystemExit("El recorte quedo vacio: revisa --cx/--cy/--radius")

    # Dedup de vertices (posicion+uv+normal) -> indices OBJ
    corners = kf.reshape(-1)
    key = np.column_stack([cP[corners].round(4), cN[corners].round(3), cT[corners].round(5)])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    vert_src = corners[first[order]]
    V, N, T = cP[vert_src], cN[vert_src], cT[vert_src]
    F = remap[inv].reshape(-1, 3)

    os.makedirs(args.out, exist_ok=True)
    obj_path = os.path.join(args.out, "mars_patch.obj")
    mtl_path = os.path.join(args.out, "mars_patch.mtl")
    tex_out = os.path.join(args.out, "model_texture.jpg")

    with open(obj_path, "w") as f:
        f.write("# Recorte optimizado del terreno marciano (generado)\n")
        f.write("mtllib mars_patch.mtl\no mars_patch\n")
        np.savetxt(f, V, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, T, fmt="vt %.6f %.6f")
        np.savetxt(f, N, fmt="vn %.5f %.5f %.5f")
        f.write("usemtl mars\ns off\n")
        f.write("\n".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}"
                          for a, b, c in (F + 1)))
        f.write("\n")

    with open(mtl_path, "w") as f:
        f.write("newmtl mars\n")
        f.write("Ka 0.4 0.4 0.4\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write("d 1.0\nillum 1\n")
        f.write("map_Kd model_texture.jpg\n")

    # Textura: reescalar manteniendo aspecto (UVs intactas -> sin riesgo de volteo)
    im = Image.open(args.tex)
    w, h = im.size
    s = min(1.0, args.maxpx / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    im.convert("RGB").save(tex_out, quality=85, optimize=True)

    zc = V[np.hypot(V[:, 0] - args.cx, V[:, 1] - args.cy) <= max(2.0, args.radius * 0.3)][:, 2]
    print(f"[OK] {obj_path}")
    print(f"     triangulos: {len(F)}  (de {len(faces)} -> {100*len(F)/len(faces):.1f} %)")
    print(f"     vertices  : {len(V)}")
    print(f"     bbox world: min {V.min(0).round(2)}  max {V.max(0).round(2)}")
    if len(zc):
        print(f"     suelo bajo el centro (z mediana): {np.median(zc):.2f} m  "
              f"-> usa esa altura para el plano de colision")
    print(f"[OK] {tex_out}  ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()
