"""Trial: one SQD job as an iterative loop, logging what the scheduler will later use.

Each round: request a batch of samples -> add to the pool -> keep unique alpha/beta
CI strings -> call Qiskit's solve_fermion -> log energy, unique configs, subspace
dimension, solver time and peak memory.

Stand-in sampler: draws determinants from the exact FCI |c|^2 distribution mixed with
uniform noise. This is NOT a quantum circuit; it only imitates "samples that get
repeated and carry diminishing information". Replace with the ffsim LUCJ simulator later.
The exact FCI energy is the reference and is used for reporting only.
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from pyscf import ao2mo, cc, fci, gto, mcscf, scf
from pyscf.fci import cistring
from qiskit_addon_sqd.fermion import solve_fermion


def build_problem(atom="N 0 0 0; N 0 0 1.1", basis="6-31g", norb=12, nelec_total=10):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    nelec = (nelec_total // 2, nelec_total // 2)
    mc = mcscf.CASCI(mf, norb, nelec_total)
    hcore, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), norb)
    return dict(hcore=hcore, eri=eri, ecore=ecore, norb=norb, nelec=nelec)


def exact_reference(p):
    e, vec = fci.direct_spin1.kernel(p["hcore"], p["eri"], p["norb"], p["nelec"])
    return e + p["ecore"], vec


def _worker(strs, hcore, eri, cores, conn):
    """Runs in a fresh spawned child (fork can deadlock after OpenMP use) so peak memory and core count are clean per call."""
    import os
    import threading

    from threadpoolctl import threadpool_limits

    # Linux: VmHWM is the exact peak RSS of this process (unlike ru_maxrss, it is not
    # inherited from the parent). Elsewhere (macOS): sample RSS in a thread with psutil.
    # SQD_NO_PROC=1 forces the psutil path so it can be tested on Linux.
    use_proc = os.path.exists("/proc/self/status") and not os.environ.get("SQD_NO_PROC")

    def hwm_kb():
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return int(line.split()[1])

    if use_proc:
        base = hwm_kb()
    else:
        import psutil

        proc = psutil.Process()
        base = proc.memory_info().rss
        state = {"peak": base, "stop": False}

        def sampler():
            while not state["stop"]:
                state["peak"] = max(state["peak"], proc.memory_info().rss)
                time.sleep(0.002)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()

    with threadpool_limits(limits=cores):
        t0 = time.perf_counter()
        e, _, _, _ = solve_fermion((strs, strs), hcore, eri)
        dt = time.perf_counter() - t0

    if use_proc:
        mem_mb = (hwm_kb() - base) / 1024.0  # KB -> MB
    else:
        state["stop"] = True
        thread.join()
        state["peak"] = max(state["peak"], proc.memory_info().rss)
        mem_mb = (state["peak"] - base) / 1e6  # bytes -> MB
    conn.send((float(e), dt, mem_mb))
    conn.close()


def solve_isolated(strs, p, cores):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(strs, p["hcore"], p["eri"], cores, child))
    proc.start()
    out = parent.recv()
    proc.join()
    return out


class FakeSampler:
    def __init__(self, p, fci_vec, noise=0.3, seed=0):
        self.rng = np.random.default_rng(seed)
        n = p["norb"]
        self.sa = cistring.make_strings(range(n), p["nelec"][0])
        self.sb = cistring.make_strings(range(n), p["nelec"][1])
        prob = (np.asarray(fci_vec) ** 2).ravel()
        prob = (1 - noise) * prob / prob.sum() + noise / prob.size
        self.prob = prob / prob.sum()
        self.shape = np.asarray(fci_vec).shape

    def draw(self, k):
        idx = self.rng.choice(self.prob.size, size=k, p=self.prob)
        ia, ib = np.unravel_index(idx, self.shape)
        return self.sa[ia], self.sb[ib]


def run_job(name, p, e_ref, fci_vec, batch=200, iters=12, seed=0, cores=2, out_dir=None):
    sampler = FakeSampler(p, fci_vec, seed=seed)
    seen_a, seen_b = set(), set()
    cum_samples, prev_e, log = 0, None, []
    for k in range(1, iters + 1):
        a, b = sampler.draw(batch)
        cum_samples += batch
        seen_a.update(a.tolist())
        seen_b.update(b.tolist())
        before = len(seen_a | seen_b)
        # closed-shell style: same string set for alpha and beta
        strs = np.array(sorted(seen_a | seen_b), dtype=np.int64)
        e, dt, mem_mb = solve_isolated(strs, p, cores)
        e_tot = float(e) + p["ecore"]
        row = dict(
            job=name, iter=k, samples_requested=batch, cum_samples=cum_samples,
            unique_configs=int(before), subspace_dim=int(len(strs) ** 2),
            energy=e_tot, gain=None if prev_e is None else prev_e - e_tot,
            cores=cores, solver_s=dt, peak_mem_mb=mem_mb,
            # evaluation-only field; the scheduler must never read this
            _err_to_ref_mha=(e_tot - e_ref) * 1e3,
        )
        prev_e = e_tot
        log.append(row)
        print(f"{name} it{k:02d} dim={row['subspace_dim']:>6} E={e_tot:.6f} "
              f"gain={row['gain'] if row['gain'] is None else round(row['gain']*1e3,3)} mHa "
              f"t={dt:.2f}s mem={mem_mb:.0f}MB err={row['_err_to_ref_mha']:.3f} mHa")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{name}.json").write_text(json.dumps(log, indent=1))
    return log


if __name__ == "__main__":
    import sys

    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # usage: python trial_sqd.py [cores]
    p = build_problem()
    e_ref, vec = exact_reference(p)
    print(f"norb={p['norb']} nelec={p['nelec']} FCI reference E={e_ref:.6f}  (solver cores={cores})")
    run_job("N2_1.10", p, e_ref, vec, batch=100, iters=15, cores=cores, out_dir="logs")
