"""Profile solver time and peak memory over (number of samples used, cores).

For each pool, take the first k samples, keep the unique strings, solve with each core count,
and record one CSV row per call. This is the raw data for the cost model.

Usage (from scripts/):
  python profile_solver.py                       # defaults: all pools, samples 100..1500, cores 1 2 4, 3 repeats
  python profile_solver.py --pools N2_1.00 N2_1.30 --samples 100 500 1000 --cores 1 2 --repeats 1
Output: data/profile.csv (rows are appended per run; delete the file to start fresh).
Time is measured inside the solver process only; the ~1 s process start-up is not included.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from trial_sqd import solve_isolated

FIELDS = ["pool", "n_samples", "unique_strings", "dim", "cores", "repeat", "solver_s", "peak_mem_mb"]


def load_pool(path):
    z = np.load(path)
    p = dict(hcore=z["hcore"], eri=z["eri"], ecore=float(z["ecore"]), norb=int(z["norb"]),
             nelec=tuple(int(x) for x in z["nelec"]))
    return p, z["alpha"], z["beta"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=None, help="pool names, e.g. N2_1.00 (default: all)")
    ap.add_argument("--samples", nargs="*", type=int, default=[100, 200, 400, 600, 800, 1000, 1250, 1500])
    ap.add_argument("--cores", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    pool_dir = root / "data" / "pools"
    names = args.pools or sorted(f.stem for f in pool_dir.glob("*.npz"))
    out = Path(args.out) if args.out else root / "data" / "profile.csv"
    new_file = not out.exists()
    total = len(names) * len(args.samples) * len(args.cores) * args.repeats
    done = 0
    t_start = time.time()

    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for name in names:
            p, a, b = load_pool(pool_dir / f"{name}.npz")
            for k in args.samples:
                strs = np.unique(np.concatenate([a[:k], b[:k]])).astype(np.int64)
                dim = len(strs) ** 2
                for cores in args.cores:
                    for rep in range(args.repeats):
                        _, dt, mem = solve_isolated(strs, p, cores)
                        w.writerow(dict(pool=name, n_samples=k, unique_strings=len(strs), dim=dim,
                                        cores=cores, repeat=rep, solver_s=dt, peak_mem_mb=mem))
                        fh.flush()
                        done += 1
                        if done % 10 == 0 or done == total:
                            el = time.time() - t_start
                            print(f"[{done}/{total}] {name} k={k} dim={dim} cores={cores} "
                                  f"t={dt:.2f}s  elapsed {el/60:.1f} min", flush=True)
    print(f"wrote {out}")
