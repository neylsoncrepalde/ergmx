"""Benchmark ergmx against R's ergm on the same networks and models.

Run benchmarks/benchmark.R first, then, from the repository root:

    python benchmarks/benchmark.py
"""

import json
import statistics
import time
from pathlib import Path

import igraph as ig
import numpy as np

import ergmx

DATA = Path(__file__).parent / "data"
reference = json.loads((DATA / "r_benchmark.json").read_text())

print(f"R ergm {reference['ergm_version']} vs ergmx {ergmx.__version__}, median of 3 seeds\n")
header = f"{'model':9} {'n':>5}  {'R ergm':>8}  {'ergmx, 1 chain':>15}  {'ergmx, 4 chains':>16}  {'max |diff| / SE':>15}"
print(header)
print("-" * len(header))
for name, model in reference["models"].items():
    g = ig.Graph.Read_GraphML(str(DATA / f"{model['network']}.graphml"))
    r_runs = model["runs"]
    r_time = statistics.median(run["seconds"] for run in r_runs)
    r_coef = {k: statistics.median(run["coef"][k] for run in r_runs) for k in r_runs[0]["coef"]}
    r_se = {k: statistics.median(run["se"][k] for run in r_runs) for k in r_runs[0]["se"]}
    times, diffs = {}, []
    for chains in (1, 4):
        runs = []
        for seed in (1, 2, 3):
            t0 = time.perf_counter()
            fit = ergmx.ergm(g, model["formula"], seed=seed, n_chains=chains)
            runs.append(time.perf_counter() - t0)
            diffs += [abs(fit.coef[k] - r_coef[k]) / r_se[k] for k in fit.names]
        times[chains] = statistics.median(runs)
    print(f"{name:9} {model['n']:>5}  {r_time:7.2f}s  {times[1]:7.2f}s ({r_time / times[1]:4.1f}x)"
          f"  {times[4]:7.2f}s ({r_time / times[4]:4.1f}x)  {max(diffs):15.2f}")
