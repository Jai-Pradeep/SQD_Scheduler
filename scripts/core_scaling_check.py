"""Does changing the core count between solver calls change solver time?

Same subspace, solved with 1 and 2 cores, 3 repeats each. The machine has 2 cores, so this only
shows the mechanism works; the real scaling curve needs a bigger machine.
"""
import sys

import numpy as np
from pyscf.fci import cistring

from trial_sqd import build_problem, solve_isolated

if __name__ == "__main__":
    p = build_problem()
    all_strs = cistring.make_strings(range(p["norb"]), p["nelec"][0])
    strs = np.sort(all_strs[:600]).astype(np.int64)  # dim = 600^2 = 360k
    print(f"subspace dim = {len(strs) ** 2}")
    core_list = [int(x) for x in sys.argv[1:]] or [1, 2]  # usage: python core_scaling_check.py 1 2 4 8
    for cores in core_list:
        times = [solve_isolated(strs, p, cores)[1] for _ in range(3)]
        print(f"cores={cores}: " + ", ".join(f"{t:.2f}s" for t in times) + f"  median={np.median(times):.2f}s")
