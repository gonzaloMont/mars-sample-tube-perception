#!/usr/bin/env python3
"""
generate_dataset_multiworld.py — Genera UN dataset sintético repartido entre
VARIOS mundos de Gazebo en una sola ejecución.

Gazebo Classic no puede cambiar de mundo en caliente (un gzserver = un
mundo), así que este orquestador lanza secuencialmente
`dataset_generation.launch.py` una vez por mundo, todas sobre el MISMO
`output_dir` en modo append (`start_index`): los ids de imagen son globales y
contiguos, los JSON BOP se acumulan, y cada muestra registra su mundo en
`meta.jsonl` (campo `world`). Cada tanda hace su propio split 70/20/10, por lo
que la proporción de mundos se mantiene dentro de train/val/test.

Uso:
    ros2 run parol6_vision generate_dataset_multiworld.py \
        --worlds mars_optimized=0.6,mars_bumpy=0.4 \
        --num_samples 1000 --output_dir ~/parol6_synth_dataset --seed 0

    # si una generación larga se interrumpe, continúa donde iba:
    ros2 run parol6_vision generate_dataset_multiworld.py \
        --worlds mars_optimized=0.6,mars_bumpy=0.4 \
        --num_samples 1000 --output_dir ~/parol6_synth_dataset --resume

Notas:
  * Las fracciones se normalizan si no suman 1; el reparto usa el método del
    mayor resto para que el total sea exacto.
  * Cada mundo usa `seed + 1000·k` (tandas decorreladas pero reproducibles).
  * Entre tandas se espera a que gzserver muera; si queda colgado se mata
    (solo gzserver/gzclient).
  * `--resume` cuenta por mundo lo ya generado (campo `world` de meta.jsonl)
    y genera solo lo que falte, con la MISMA especificación de --worlds.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def parse_worlds(spec):
    """'a=0.6,b=0.4' → [('a', 0.6), ('b', 0.4)] normalizado."""
    pairs = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise SystemExit(f"--worlds: falta '=' en {item!r} (mundo=fracción)")
        name, frac = item.split('=', 1)
        pairs.append((name.strip(), float(frac)))
    if not pairs:
        raise SystemExit('--worlds vacío')
    total = sum(f for _, f in pairs)
    if total <= 0:
        raise SystemExit('--worlds: las fracciones deben sumar > 0')
    if abs(total - 1.0) > 1e-6:
        print(f'[multiworld] fracciones suman {total:.3f}: se normalizan.')
    return [(n, f / total) for n, f in pairs]


def largest_remainder(fracs, n):
    raw = [f * n for f in fracs]
    base = [int(x) for x in raw]
    rest = n - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rest]:
        base[i] += 1
    return base


def read_meta(path):
    metas = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))
    return metas


def gazebo_running():
    return subprocess.run(['pgrep', '-x', 'gzserver'],
                          capture_output=True).returncode == 0


def preflight_no_foreign_gazebo(args):
    """Aborta si ya hay un gzserver corriendo ANTES de empezar.

    Si arrancamos con una simulación ajena viva, nuestro gzserver muere por
    conflicto de puerto y el generador puede engancharse en silencio a la
    simulación equivocada (mundo desconocido → dataset inválido)."""
    if not gazebo_running():
        return
    if args.kill_existing_gazebo:
        print('[multiworld] --kill_existing_gazebo: cerrando el gzserver '
              'existente…')
        subprocess.run(['pkill', '-x', 'gzserver'])
        time.sleep(5.0)
        if gazebo_running():
            subprocess.run(['pkill', '-9', '-x', 'gzserver'])
            subprocess.run(['pkill', '-9', '-x', 'gzclient'])
            time.sleep(2.0)
        return
    raise SystemExit(
        '[multiworld] ✘ Ya hay un gzserver corriendo (¿otra simulación '
        'abierta, p.ej. gazebo_moveit_*.launch.py?). Ciérrala primero o '
        'pasa --kill_existing_gazebo para cerrarla automáticamente.')


def wait_gazebo_dead(timeout=20.0):
    """Espera a que muera el gzserver de NUESTRA tanda (tras el pre-flight,
    cualquier gzserver vivo aquí es nuestro y es seguro rematarlo)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not gazebo_running():
            return
        time.sleep(1.0)
    print('[multiworld] ⚠ el gzserver de la tanda no murió con el shutdown: '
          'pkill -9 gzserver/gzclient.')
    subprocess.run(['pkill', '-9', '-x', 'gzserver'])
    subprocess.run(['pkill', '-9', '-x', 'gzclient'])
    time.sleep(2.0)


def run_world(world, count, start_index, seed, args):
    cmd = [
        'ros2', 'launch', 'parol6_gazebo', 'dataset_generation.launch.py',
        f'world:={world}',
        f'num_samples:={count}',
        f'start_index:={start_index}',
        f'output_dir:={args.output_dir}',
        f'seed:={seed}',
        'shutdown_when_done:=true',
    ] + [a for a in args.launch_arg]
    print(f'\n[multiworld] ▶ {world}: {count} muestras '
          f'(ids {start_index}…{start_index + count - 1})\n  {" ".join(cmd)}')
    meta_path = os.path.join(args.output_dir, 'meta.jsonl')
    n_before = len(read_meta(meta_path))

    proc = subprocess.Popen(cmd, start_new_session=True)
    last_n, last_progress = n_before, time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(5.0)
            n_now = len(read_meta(meta_path))
            if n_now > last_n:
                last_n, last_progress = n_now, time.monotonic()
            elif time.monotonic() - last_progress > args.stall_timeout:
                print(f'[multiworld] ✘ sin progreso en {args.stall_timeout:.0f}s '
                      f'({last_n - n_before}/{count}): se aborta la tanda.')
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                wait_gazebo_dead()
                return False
    except KeyboardInterrupt:
        print('\n[multiworld] interrumpido: cerrando el launch…')
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        wait_gazebo_dead()
        raise

    wait_gazebo_dead()
    n_after = len(read_meta(meta_path))
    made = n_after - n_before
    if made < count:
        print(f'[multiworld] ✘ {world}: generó {made}/{count} muestras '
              f'(launch rc={proc.returncode}). Revisa el log; puedes relanzar '
              f'con --resume.')
        return False
    print(f'[multiworld] ✔ {world}: {made} muestras.')
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--worlds', required=True,
                    help="p.ej. 'mars_optimized=0.6,mars_bumpy=0.4'")
    ap.add_argument('--num_samples', type=int, default=1000,
                    help='total de muestras del dataset combinado')
    ap.add_argument('--output_dir', default='~/parol6_synth_dataset')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--launch_arg', action='append', default=[],
                    metavar='key:=val',
                    help='argumento extra para TODOS los launch (repetible)')
    ap.add_argument('--resume', action='store_true',
                    help='continuar un dataset existente (misma spec --worlds)')
    ap.add_argument('--kill_existing_gazebo', action='store_true',
                    help='si ya hay un gzserver corriendo, cerrarlo en vez '
                         'de abortar')
    ap.add_argument('--stall_timeout', type=float, default=300.0,
                    help='s sin muestras nuevas antes de abortar una tanda')
    ap.add_argument('--dry_run', action='store_true',
                    help='muestra el plan de tandas sin ejecutar nada')
    args = ap.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    worlds = parse_worlds(args.worlds)
    counts = largest_remainder([f for _, f in worlds], args.num_samples)
    targets = {w: c for (w, _), c in zip(worlds, counts)}

    meta_path = os.path.join(args.output_dir, 'meta.jsonl')
    metas = read_meta(meta_path)
    if metas and not args.resume:
        raise SystemExit(
            f'{args.output_dir} ya contiene {len(metas)} muestras. Usa '
            f'--resume para continuar (misma --worlds) u otro --output_dir.')

    done = {w: 0 for w, _ in worlds}
    next_id = 0
    if metas:
        next_id = max(m['im_id'] for m in metas) + 1
        unknown = 0
        for m in metas:
            w = m.get('world', '')
            if w in done:
                done[w] += 1
            else:
                unknown += 1
        if unknown:
            print(f'[multiworld] ⚠ {unknown} muestras previas sin mundo '
                  f'reconocido: se conservan pero no cuentan para los cupos.')

    plan = []
    for k, (world, _) in enumerate(worlds):
        remaining = targets[world] - done[world]
        if remaining > 0:
            plan.append((k, world, remaining))
        elif args.resume:
            print(f'[multiworld] {world}: cupo cubierto '
                  f'({done[world]}/{targets[world]}), se omite.')

    print(f'[multiworld] objetivo: {args.num_samples} muestras en '
          f'{args.output_dir}')
    for k, world, remaining in plan:
        print(f'  - {world}: {remaining} muestras (seed {args.seed + 1000 * k})')
    if args.dry_run:
        return
    if not plan:
        print('[multiworld] nada que generar.')
        return

    preflight_no_foreign_gazebo(args)
    t0 = time.monotonic()
    for k, world, remaining in plan:
        ok = run_world(world, remaining, next_id, args.seed + 1000 * k, args)
        if not ok:
            sys.exit(1)
        next_id = max(m['im_id'] for m in read_meta(meta_path)) + 1

    metas = read_meta(meta_path)
    per_world = {}
    for m in metas:
        per_world[m.get('world', '?')] = per_world.get(m.get('world', '?'), 0) + 1
    print(f'\n[multiworld] COMPLETADO en {(time.monotonic() - t0) / 60:.1f} min: '
          f'{len(metas)} muestras {per_world}')
    print('[multiworld] QA recomendado:')
    print(f'  ros2 run parol6_vision visualize_dataset_samples.py '
          f'--dataset_dir {args.output_dir} --n 20 --check')
    print(f'  ros2 run parol6_vision dataset_stats.py '
          f'--dataset_dir {args.output_dir}')


if __name__ == '__main__':
    main()
