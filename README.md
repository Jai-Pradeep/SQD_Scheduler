# SQD scheduler: trial setup

This folder holds the first trial for the progress-aware scheduling project. It checks that
everything runs on your machine before we build the scheduler.

- `scripts/trial_sqd.py`: runs one SQD job as a loop of 15 rounds and logs, for each round, the
  energy, energy gain, unique configurations, subspace size, solver time, cores and peak memory.
- `scripts/core_scaling_check.py`: solves the same problem with different core counts to check
  that changing cores between solver calls really changes the solver time.

The sampler in the trial is a stand-in (see the top of `trial_sqd.py`), not a quantum circuit.

## 1. What you need

- **OS:** Linux, macOS, or Windows through WSL2. PySCF does not install natively on Windows.
- **Python:** 3.10 to 3.13 (tested on 3.13). Check with `python3 --version`.
- **RAM:** the trial needs well under 1 GB. Keep 8 GB or more free for the later experiments.
- **Disk:** about 1 GB for the packages.

## 2. Set up

Run these from the folder that contains this README.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows (WSL) and macOS/Linux all use this line
pip install --upgrade pip
pip install -r requirements.txt
```

Check the install:

```bash
python -c "import qiskit_addon_sqd, pyscf, ffsim, qiskit; print('ok')"
```

It should print `ok`.

## 3. Find out how many cores you have

```bash
python -c "import os; print(os.cpu_count())"
```

Use at most this many cores in the commands below. Physical cores matter more than the
logical count, so if it reports 8 on a 4-core laptop, stay at 4 or below.

## 4. Run

Run both commands from inside `scripts/`, because `core_scaling_check.py` imports `trial_sqd.py`.

```bash
cd scripts

# Run 1: the trial loop (about 1 to 2 minutes). The number is the cores for the solver.
python trial_sqd.py 2

# Run 2: core scaling. Give it the core counts to compare, up to your own core count.
python core_scaling_check.py 1 2 4
```

Run 1 writes `scripts/logs/N2_1.10.json`, one record per round.

## 5. What a good result looks like

Your exact numbers will differ (see "Known limits"), but you should see these patterns.

**Run 1**

- First line: `norb=12 nelec=(5, 5) FCI reference E=-108.999941`. This reference should match exactly.
- Subspace size (`dim`) grows every round, from about 6,000 to about 320,000.
- Solver time `t` and memory `mem` grow with it. Expect roughly 0.3 s to 5 s on 2 cores and
  roughly 5 MB to 70 MB. A faster machine gives smaller times.
- `err` falls from about 50 mHa to under about 10 mHa. It falls unevenly: some rounds gain almost
  nothing and one later round can gain a lot. That unevenness is expected.

**Run 2**

- `subspace dim = 360000`.
- Time falls as cores increase, but by less than the core ratio. On the 2-core test machine,
  1 core took about 7 to 8 s and 2 cores took about 5 to 6 s (about 1.3 to 1.4 times faster).

## 6. If something goes wrong

| Symptom                              | Likely cause and fix                                                                                           |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `pip install` fails building `pyscf` | Python version too new or too old. Use 3.11 or 3.12 and reinstall. On Windows, use WSL2.                       |
| Script hangs with no output          | Do not change `get_context("spawn")` in `solve_isolated` to `fork`. Forking after OpenMP has run can deadlock. |
| `ModuleNotFoundError: threadpoolctl` | `pip install threadpoolctl`                                                                                    |
| `ModuleNotFoundError: trial_sqd`     | You are not inside `scripts/`. Run `cd scripts` first.                                                         |
| Memory always prints 0 MB            | Send us the output. On Linux it reads `/proc/self/status`; elsewhere it uses `psutil`.                         |
| Very slow or out of memory           | Close other programs, or lower `iters` in the `run_job(...)` call at the bottom of `trial_sqd.py`.             |

## 7. Known limits of this trial

- **Not exactly reproducible yet:** two runs with the same seed give slightly different energies and
  sizes. Later we will generate each sample pool once and save it to disk.
- **The sampler is a stand-in:** it draws from the exact ground-state distribution mixed with 30% uniform
  noise. The circuit-based sampler comes next.
- **Memory is measured per solver call:** each call runs in its own short-lived process so peak memory and
  core count are clean. This adds about 1.2 s of start-up per call on the test machine, which is
  not counted in the reported solver time (so Run 1 takes longer in wall-clock than the `t=` values add up to).
- **The reference energy** (`_err_to_ref_mha` in the log) is for evaluation only. The scheduler must never read it.

## 8. What to send back

Paste the full output of both runs plus: your OS, `python3 --version`, core count, and RAM.

## 9. Next stage: sample pools, profiling, cost model

Run from inside `scripts/`, in this order. Install the new requirement first: `pip install -r ../requirements.txt`.

```bash
# 1. Build saved sample pools for 5 N2 bond lengths (about 1 minute). Do this ONCE, then share
#    data/pools, data/ref and data/split.json with the team so everyone uses identical samples.
python make_pools.py

# 2. Profile solver time and memory (about 20 to 40 minutes with the defaults; do not run other heavy
#    programs meanwhile, or the timings get noisy). Pick core counts up to your PHYSICAL core count.
python profile_solver.py --cores 1 2 4 8

# 3. Fit the cost model and check it on the held-out geometries
python fit_cost_model.py
```

- `data/ref/` holds the exact energies and is for evaluation only; scheduler code must not read it.
- `data/split.json` fixes which geometries are for tuning (1.00, 1.30, 2.00) and which are held out
  for final testing (1.10, 1.60). Do not change it after looking at results.
