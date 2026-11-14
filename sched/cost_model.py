"""Cost model used by the scheduler and simulator. Reads data/cost_model.json written by fit_cost_model.py.

Usage:
    cm = CostModel.load("data/cost_model.json")
    job = cm.new_job()
    job.observe(dim=20449, cores=2, seconds=0.50)        # after each real (or replayed) solver call
    job.predict_seconds(dim=60000, cores=4)               # time for the next call
    cm.predict_memory_mb(unique_strings=250, cores=4)     # same for every job

Until a job has been observed once, predict_seconds uses the default scale (average over tuning pools).
The scheduler must only feed it measured solver times, never reference energies.
"""
import json
import math
from pathlib import Path


class JobCost:
    def __init__(self, model, default_log_scale):
        self.model = model
        self.obs = []  # list of (log residual of the scale) per observed call
        self.default = default_log_scale

    def _shape(self, dim, cores):
        t = self.model["time"]
        return t["alpha"] * math.log(dim) + math.log(t["s"] + (1 - t["s"]) / cores)

    def observe(self, dim, cores, seconds):
        if dim < self.model["min_dim"] or seconds <= 0:
            return  # too small: dominated by fixed overhead
        self.obs.append(math.log(seconds) - self._shape(dim, cores))

    @property
    def log_scale(self):
        return sum(self.obs) / len(self.obs) if self.obs else self.default

    def predict_seconds(self, dim, cores):
        return math.exp(self.log_scale + self._shape(dim, cores))


class CostModel:
    def __init__(self, model):
        self.model = model
        # rough default: scale of the middle geometry; replaced as soon as the job reports a solver call
        self.default_log_scale = model.get("default_log_scale", math.log(0.0012))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def new_job(self):
        return JobCost(self.model, self.default_log_scale)

    def predict_memory_mb(self, unique_strings, cores):
        m = self.model["memory"]
        return (m["m0"] + m["b"] * unique_strings ** m["g"]) * (1 + m["k"] * (cores - 1))
