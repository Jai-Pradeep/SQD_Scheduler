"""Generate saved, deterministic sample pools: one per N2 bond length.

A "sample request" in the simulator will just read the next chunk of a pool, so every policy sees
exactly the same samples and runs are reproducible.

Files written (relative to this script's parent folder):
  data/pools/N2_<r>.npz   problem integrals + ordered samples. The scheduler may read these.
  data/ref/N2_<r>.json    exact reference energy. EVALUATION ONLY: scheduler code must not read this.
  data/split.json         which geometries are for tuning policies and which are held out for testing.

Usage (from scripts/):  python make_pools.py [n_samples]
The sampler is still the stand-in from trial_sqd.py (not a circuit); swap in circuit samples later.
"""
import json
import sys
from pathlib import Path

import numpy as np

from trial_sqd import FakeSampler, build_problem, exact_reference

GEOMETRIES = [1.0, 1.1, 1.3, 1.6, 2.0]  # N-N distance in Angstrom
TUNING = [1.0, 1.3, 2.0]  # used to tune policies and fit models
HELD_OUT = [1.1, 1.6]  # touched only for final testing

if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    root = Path(__file__).resolve().parent.parent
    (root / "data" / "pools").mkdir(parents=True, exist_ok=True)
    (root / "data" / "ref").mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(GEOMETRIES):
        name = f"N2_{r:.2f}"
        p = build_problem(atom=f"N 0 0 0; N 0 0 {r}")
        e_ref, vec = exact_reference(p)
        a, b = FakeSampler(p, vec, seed=1000 + i).draw(n_samples)
        np.savez_compressed(
            root / "data" / "pools" / f"{name}.npz",
            hcore=p["hcore"], eri=p["eri"], ecore=p["ecore"],
            norb=p["norb"], nelec=np.array(p["nelec"]),
            alpha=a.astype(np.int64), beta=b.astype(np.int64),
        )
        (root / "data" / "ref" / f"{name}.json").write_text(json.dumps({"e_ref": float(e_ref)}))
        print(f"{name}: FCI E={e_ref:.6f}  samples={n_samples}")

    (root / "data" / "split.json").write_text(
        json.dumps({"tuning": [f"N2_{r:.2f}" for r in TUNING],
                    "held_out": [f"N2_{r:.2f}" for r in HELD_OUT]}, indent=1))
    print("wrote pools, references and split.json")
