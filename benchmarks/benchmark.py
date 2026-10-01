"""Benchmark ergmx against R's ergm on the same networks and models.

Run benchmarks/benchmark.R first, then, from the repository root:

    python benchmarks/benchmark.py

Both sides use their defaults, including the log-likelihood estimate. The
single-threaded column runs in a subprocess limited to one thread, like R.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import igraph as ig

import ergmx

DATA = Path(__file__).parent / "data"
SEEDS = (1, 2, 3)
reference = json.loads((DATA / "r_benchmark.json").read_text())


def fit_times(name: str, **options) -> tuple[list[float], list[dict]]:
    model = reference["models"][name]
    g = ig.Graph.Read_GraphML(str(DATA / f"{model['network']}.graphml"))
    times, coefs = [], []
    for seed in SEEDS:
        t0 = time.perf_counter()
        fit = ergmx.ergm(g, model["formula"], seed=seed, **options)
        times.append(time.perf_counter() - t0)
        coefs.append(fit.coef)
    return times, coefs


if len(sys.argv) == 3 and sys.argv[1] == "--single-thread":
    times, _ = fit_times(sys.argv[2], n_chains=1)
    print(statistics.median(times))
    sys.exit()

print(f"R ergm {reference['ergm_version']} vs ergmx {ergmx.__version__}, median of {len(SEEDS)} seeds, "
      "both with their default log-likelihood estimate\n")
header = (f"{'model':9} {'n':>5}  {'R ergm':>8}  {'ergmx, 1 thread':>16}  {'ergmx, all threads':>19}"
          f"  {'max |diff| / SE':>15}")
print(header)
print("-" * len(header))
for name, model in reference["models"].items():
    r_runs = model["runs"]
    r_time = statistics.median(run["seconds"] for run in r_runs)
    r_coef = {k: statistics.median(run["coef"][k] for run in r_runs) for k in r_runs[0]["coef"]}
    r_se = {k: statistics.median(run["se"][k] for run in r_runs) for k in r_runs[0]["se"]}
    single = float(subprocess.run(
        [sys.executable, __file__, "--single-thread", name], capture_output=True, text=True,
        env={**os.environ, "RAYON_NUM_THREADS": "1"}, check=True,
    ).stdout.split()[-1])
    times, coefs = fit_times(name)
    diff = max(abs(c[k] - r_coef[k]) / r_se[k] for c in coefs for k in r_coef)
    parallel = statistics.median(times)
    print(f"{name:9} {model['n']:>5}  {r_time:7.2f}s  {single:8.2f}s ({r_time / single:4.1f}x)"
          f"  {parallel:10.2f}s ({r_time / parallel:4.1f}x)  {diff:15.2f}", flush=True)
