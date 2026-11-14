"""Fit the shared cost model from profile CSV(s) and test how well per-job calibration predicts new geometries.

Model (d = subspace dimension = unique_strings**2, n = unique_strings, p = cores):
  time:   t(d, p) = scale_job * d**alpha * (s + (1 - s) / p)
            alpha, s are SHARED across jobs; scale_job differs per job (about 4x between geometries) and is
            learned online from the job's own first solver calls (see sched/cost_model.py).
  memory: m(n, p) = (m0 + b * n**g) * (1 + k * (p - 1))      MB, peak above baseline; same for all jobs

Only the tuning pools in data/split.json are used for fitting. Held-out pools are used only to report error.
Rows with dim < 10,000 are ignored: they are dominated by fixed overhead, not by the size of the problem.

Usage (from scripts/):
  python fit_cost_model.py                      # reads ../data/profile.csv
  python fit_cost_model.py a.csv b.csv          # several CSVs are combined (repeats are median-ed)
Writes ../data/cost_model.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

MIN_DIM = 10_000


def log_time(lc, alpha, s, d, p):
    return lc + alpha * np.log(d) + np.log(s + (1 - s) / p)


def fit_time(df):
    pools = sorted(df.pool.unique())
    ix = {p: i for i, p in enumerate(pools)}
    d, p = df.dim.values.astype(float), df.cores.values.astype(float)
    y = np.log(df.solver_s.values)
    lc_idx = np.array([ix[q] for q in df.pool])

    def resid(th):
        return log_time(th[2:][lc_idx], th[0], th[1], d, p) - y

    r = least_squares(resid, [0.75, 0.2] + [-7.0] * len(pools),
                      bounds=([0.3, 0.0] + [-30.0] * len(pools), [2.0, 1.0] + [5.0] * len(pools)))
    return float(r.x[0]), float(r.x[1])


def fit_memory(df):
    n, p = df.unique_strings.values.astype(float), df.cores.values.astype(float)

    def resid(th):
        return (th[0] + np.exp(th[1]) * n ** th[2]) * (1 + th[3] * (p - 1)) - df.peak_mem_mb.values

    r = least_squares(resid, [5, -3, 1.2, 0.01], bounds=([0, -20, 0.5, 0.0], [30, 5, 3, 0.2]))
    return dict(m0=float(r.x[0]), b=float(np.exp(r.x[1])), g=float(r.x[2]), k=float(r.x[3]))


def mem_pred(mem, n, p):
    return (mem["m0"] + mem["b"] * np.asarray(n, float) ** mem["g"]) * (1 + mem["k"] * (np.asarray(p, float) - 1))


def pct(a, b):
    e = np.abs(a - b) / b * 100
    return f"median {np.median(e):4.1f}%   worst {e.max():5.1f}%"


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    paths = [Path(a) for a in sys.argv[1:]] or [root / "data" / "profile.csv"]
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    split = json.loads((root / "data" / "split.json").read_text())
    m = raw.groupby(["pool", "n_samples", "unique_strings", "dim", "cores"], as_index=False).median(numeric_only=True)
    m = m[m.dim >= MIN_DIM]
    tune, held = m[m.pool.isin(split["tuning"])], m[m.pool.isin(split["held_out"])]
    if tune.empty:
        raise SystemExit("No profile rows for the tuning pools in data/split.json.")

    alpha, s = fit_time(tune)
    mem = fit_memory(tune)
    print(f"time:   t = scale_job * d^{alpha:.3f} * ({s:.3f} + {1 - s:.3f}/p)")
    print(f"        s = {s:.2f}: about {s*100:.0f}% of the work does not speed up with cores "
          f"(best possible speedup {1/s:.1f}x; at 8 cores {1/(s+(1-s)/8):.1f}x)")
    print(f"memory: m = ({mem['m0']:.1f} + {mem['b']:.2g} * n^{mem['g']:.2f}) * (1 + {mem['k']:.3f}*(p-1))  MB")

    def calibrate(pool_df, k_calls, cal_cores=2):
        """scale from the job's first k solver calls (smallest sizes, run at cal_cores)."""
        c = pool_df[(pool_df.cores == cal_cores) & (pool_df.dim >= MIN_DIM)].sort_values("dim").head(k_calls)
        lc = np.mean(np.log(c.solver_s.values) - log_time(0.0, alpha, s, c.dim.values.astype(float), cal_cores))
        return lc, c.dim.max()

    print("\nTime prediction on held-out pools: scale learned from the job's first k solver calls, then predicting"
          "\nevery LARGER size at ALL core counts (error = |predicted - measured| / measured):")
    for pool, g in held.groupby("pool"):
        out = []
        for k in (1, 2, 3):
            lc, dmax = calibrate(g, k)
            r = g[g.dim > dmax]
            pred = np.exp(log_time(lc, alpha, s, r.dim.values.astype(float), r.cores.values.astype(float)))
            out.append(f"k={k}: {pct(pred, r.solver_s.values)}")
        print(f"  {pool}   " + "   ".join(out))
    if not held.empty:
        print("\nMemory prediction on held-out pools (no calibration needed):")
        print("  " + pct(mem_pred(mem, held.unique_strings, held.cores), held.peak_mem_mb.values))

    print("\nPer-pool time scale on tuning pools (shows how much jobs differ; not used for held-out pools):")
    for pool, g in tune.groupby("pool"):
        lc, _ = calibrate(g, 3)
        print(f"  {pool}: scale = {np.exp(lc):.5f}")

    (root / "data" / "cost_model.json").write_text(json.dumps(
        dict(time=dict(alpha=alpha, s=s), memory=mem, min_dim=MIN_DIM, fitted_on=split["tuning"],
             note="time scale is per job and is learned online; see sched/cost_model.py"), indent=1))
    print("\nwrote data/cost_model.json")
