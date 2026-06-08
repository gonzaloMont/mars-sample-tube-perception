#!/usr/bin/env python3
"""
Genera un parche de terreno "egg-crate" (montañitas sinusoidales) con textura
marciana, pensado para que un tubo fino (r~1 cm) se quede atrapado entre bultos
y NO salga rodando, conservando distintas poses de tip/tilt/yaw.

Superficie:  z = A * ( sin(2*pi*x/lambda) + sin(2*pi*y/lambda) )
  - El campo es de media nula (sin pendiente global -> no hay deriva).
  - Los valles también ondulan a lo largo, así que el tubo queda atrapado en
    CUALQUIER orientación (no solo perpendicular a un canal).
  - Barrera de rodadura ~= A; con A=5 mm frente a un tubo de r=10 mm, basta.

Salida (en este directorio):
  - mars_bumpy.obj / .mtl : malla visual + colisión (coords de mundo, pose identidad)
  - model_texture.jpg     : recorte cuadrado de la textura marciana original

Uso:
  python3 generate_bumpy_patch.py                      # 1.2x1.2 m, lambda=10cm, A=1cm
  python3 generate_bumpy_patch.py --wavelength 0.05 --amplitude 0.004
  python3 generate_bumpy_patch.py --half 0.8 --res 0.0125
"""
import argparse
import math
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_TEX = os.path.join(HERE, "..", "mars", "model_texture.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=float, default=0.6,
                    help="medio-lado del parche [m] (patch = 2*half cuadrado)")
    ap.add_argument("--wavelength", type=float, default=0.1, help="lambda de las montañitas [m]")
    ap.add_argument("--amplitude", type=float, default=0.01, help="amplitud A [m]")
    ap.add_argument("--res", type=float, default=0.015, help="paso de malla [m]")
    ap.add_argument("--tex", default=SRC_TEX)
    ap.add_argument("--texsize", type=int, default=1024, help="lado de la textura de salida [px]")
    ap.add_argument("--texcrop", type=int, default=1400, help="lado del recorte de la textura fuente [px]")
    ap.add_argument("--ground_half", type=float, default=3.0,
                    help="medio-lado del suelo Mars grande de fondo [m] (0 = no generarlo)")
    ap.add_argument("--ground_tile", type=float, default=2.0,
                    help="metros por repeticion de textura en el suelo (tile de 2 m -> 2x2 en 4 m)")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    A, lam, half = args.amplitude, args.wavelength, args.half
    k = 2.0 * math.pi / lam

    xs = np.arange(-half, half + 1e-9, args.res)
    ys = np.arange(-half, half + 1e-9, args.res)
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = A * (np.sin(k * X) + np.sin(k * Y))

    # Normales analíticas
    dzx = A * k * np.cos(k * X)
    dzy = A * k * np.cos(k * Y)
    Nx, Ny, Nz = -dzx, -dzy, np.ones_like(Z)
    Nn = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
    Nx, Ny, Nz = Nx / Nn, Ny / Nn, Nz / Nn

    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    Nrm = np.column_stack([Nx.ravel(), Ny.ravel(), Nz.ravel()])
    U = (X.ravel() + half) / (2 * half)
    Vt = (Y.ravel() + half) / (2 * half)
    UV = np.column_stack([U, Vt])

    # Caras (2 triángulos por celda); índice de vértice = i*ny + j
    i = np.arange(nx - 1)[:, None]
    j = np.arange(ny - 1)[None, :]
    v00 = (i * ny + j).ravel()
    v01 = (i * ny + (j + 1)).ravel()
    v10 = ((i + 1) * ny + j).ravel()
    v11 = ((i + 1) * ny + (j + 1)).ravel()
    F = np.vstack([np.column_stack([v00, v10, v11]),
                   np.column_stack([v00, v11, v01])])

    os.makedirs(args.out, exist_ok=True)
    obj_path = os.path.join(args.out, "mars_bumpy.obj")
    mtl_path = os.path.join(args.out, "mars_bumpy.mtl")
    tex_out = os.path.join(args.out, "model_texture.jpg")

    with open(obj_path, "w") as f:
        f.write("# Terreno egg-crate marciano (montañitas sinusoidales, generado)\n")
        f.write(f"# half={half} lambda={lam} A={A} res={args.res}\n")
        f.write("mtllib mars_bumpy.mtl\no mars_bumpy\n")
        np.savetxt(f, V, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, UV, fmt="vt %.6f %.6f")
        np.savetxt(f, Nrm, fmt="vn %.5f %.5f %.5f")
        f.write("usemtl mars\ns off\n")
        f.write("\n".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}"
                          for a, b, c in (F + 1)))
        f.write("\n")

    with open(mtl_path, "w") as f:
        f.write("newmtl mars\nKa 0.4 0.4 0.4\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
        f.write("d 1.0\nillum 1\nmap_Kd model_texture.jpg\n")

    # Textura: recorte cuadrado central de la textura marciana (UV 0..1, sin tiling)
    im = Image.open(args.tex).convert("RGB")
    w, h = im.size
    c = min(args.texcrop, w, h)
    left, top = (w - c) // 2, (h - c) // 2
    im = im.crop((left, top, left + c, top + c)).resize((args.texsize, args.texsize), Image.LANCZOS)
    im.save(tex_out, quality=85, optimize=True)

    # Suelo Mars grande y plano (visual de fondo) que rellena el campo de visión.
    # Comparte la misma textura, repetida cada ground_tile metros (z=0; el .world
    # lo coloca ligeramente por debajo del parche de montañitas).
    if args.ground_half > 0:
        gh = args.ground_half
        rep = (2 * gh) / args.ground_tile
        with open(os.path.join(args.out, "mars_ground.obj"), "w") as f:
            f.write("# Suelo Mars de fondo (quad plano texturizado, generado)\n")
            f.write("mtllib mars_bumpy.mtl\no mars_ground\n")
            for x, y in ((-gh, -gh), (gh, -gh), (gh, gh), (-gh, gh)):
                f.write(f"v {x:.4f} {y:.4f} 0.0\n")
            for u, v in ((0, 0), (rep, 0), (rep, rep), (0, rep)):
                f.write(f"vt {u:.4f} {v:.4f}\n")
            f.write("vn 0 0 1\n")
            f.write("usemtl mars\ns off\n")
            f.write("f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
        print(f"[OK] {os.path.join(args.out, 'mars_ground.obj')}  "
              f"({2*gh:.0f}x{2*gh:.0f} m, textura x{rep:.0f})")

    print(f"[OK] {obj_path}")
    print(f"     malla: {nx}x{ny} = {len(V)} vértices, {len(F)} triángulos")
    print(f"     patch: {2*half:.2f}x{2*half:.2f} m  z in [{Z.min():.3f},{Z.max():.3f}]  "
          f"(barrera {2*A*1000:.0f} mm pico-valle)")
    print(f"     pendiente máx: {math.degrees(math.atan(A*k)):.1f}°")
    print(f"[OK] {tex_out}  ({args.texsize}x{args.texsize})")


if __name__ == "__main__":
    main()
