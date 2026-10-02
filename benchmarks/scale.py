"""The scale benchmark: ergmx against R's ergm and ergm.multi on networks of
1,461 to 10,000 vertices and on 500 classrooms. Run benchmarks/make_scale.py
and benchmarks/scale.R first, then, from the repository root:

    python benchmarks/scale.py

Both sides use their defaults, including the log-likelihood estimate; ergmx's
single-threaded column runs in a subprocess limited to one thread, like R.
"""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import igraph as ig

import ergmx

DATA = Path(__file__).parent / "data"
reference = json.loads((DATA / "r_scale.json").read_text())


def read(name: str) -> ig.Graph:
    with tempfile.TemporaryDirectory() as tmp, gzip.open(DATA / f"{name}.graphml.gz") as f:
        path = Path(tmp) / "g.graphml"
        path.write_bytes(f.read())
        return ig.Graph.Read_GraphML(str(path))


def network(name: str):
    if name == "magnolia_1461":
        return ig.Graph.Read_GraphML(str(DATA / "faux.magnolia.high.graphml"))
    g = read(name)
    if name != "classrooms":
        return g
    rooms = sorted(set(g.vs["classroom"]))
    return ergmx.Networks([g.induced_subgraph(g.vs.select(classroom=k)) for k in rooms])


def times(name: str) -> tuple[float, float, dict]:
    g, formula = network(name), reference["models"][name]["formula"]
    t0 = time.perf_counter()
    ergmx.ergm(g, formula, estimate="MPLE")
    t1 = time.perf_counter()
    fit = ergmx.ergm(g, formula, seed=1)
    return t1 - t0, time.perf_counter() - t1, fit.coef


if len(sys.argv) == 3 and sys.argv[1] == "--single-thread":
    print(times(sys.argv[2])[1])
    sys.exit()

print(f"R ergm {reference['ergm_version']} (ergm.multi {reference['ergm_multi_version']}) vs "
      f"ergmx {ergmx.__version__}, one seed, both with their default log-likelihood estimate\n")
header = (f"{'network':15} {'n':>6}  {'MPLE: R':>8} {'ergmx':>14}  {'MLE: R':>8} {'ergmx, 1 thread':>17}"
          f" {'all threads':>17}  {'max |diff| / SE':>15}")
print(header)
print("-" * len(header))
for name, r in reference["models"].items():
    single = float(subprocess.run(
        [sys.executable, __file__, "--single-thread", name], capture_output=True, text=True,
        env={**os.environ, "RAYON_NUM_THREADS": "1"}, check=True,
    ).stdout.split()[-1])
    mple, mle, coef = times(name)
    diff = max(abs(coef[k] - r["coef"][k]) / r["se"][k] for k in r["coef"])
    print(f"{name:15} {r['n']:>6}  {r['mple_seconds']:7.1f}s {mple:6.1f}s ({r['mple_seconds'] / mple:4.0f}x)"
          f"  {r['mle_seconds']:7.1f}s {single:8.1f}s ({r['mle_seconds'] / single:4.1f}x)"
          f" {mle:8.1f}s ({r['mle_seconds'] / mle:4.1f}x)  {diff:15.2f}", flush=True)
