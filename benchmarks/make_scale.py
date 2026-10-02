"""Networks for the scale benchmark (benchmarks/scale.R and scale.py), in
benchmarks/data. Run from the repository root:

    python benchmarks/make_scale.py

- magnolia_<n>.graphml.gz: networks like faux.magnolia.high with 2,500, 5,000 and
  10,000 vertices, simulated from R's fit of the magnolia benchmark model,
  with the attributes drawn from faux.magnolia.high's and the edges
  coefficient shifted by -log(n / 1461), which keeps the mean degree about the
  same.
- classrooms.graphml.gz: 500 classrooms of 20 students (10,000 vertices), each
  with a gender, simulated from a model with homophily and transitivity; the
  vertex attribute "classroom" numbers them.
"""

import gzip
import json
import tempfile
from pathlib import Path

import igraph as ig
import numpy as np

import ergmx

DATA = Path(__file__).parent / "data"
reference = json.loads((DATA / "r_benchmark.json").read_text())["models"]["magnolia"]
magnolia = ig.Graph.Read_GraphML(str(DATA / "faux.magnolia.high.graphml"))
coef = list(reference["runs"][0]["coef"].values())
rng = np.random.default_rng(2026)


def write(g: ig.Graph, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.graphml"
        g.write_graphml(str(path))
        (DATA / f"{name}.graphml.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))


for n in (2500, 5000, 10000):
    pick = rng.integers(0, magnolia.vcount(), n)
    g = ig.Graph(n=n)
    for a in ("Grade", "Race", "Sex"):
        g.vs[a] = [magnolia.vs[int(k)][a] for k in pick]
    theta = [coef[0] - np.log(n / magnolia.vcount()), *coef[1:]]
    (h,) = ergmx.simulate(g, reference["formula"], theta, nsim=1, seed=n, burnin=1000 * n)
    write(h, f"magnolia_{n}")
    print(f"magnolia_{n}: {h.ecount()} edges, mean degree {2 * h.ecount() / n:.2f}")

classes = []
for k in range(500):
    g = ig.Graph(n=20)
    g.vs["gender"] = rng.choice(["F", "M"], 20).tolist()
    (h,) = ergmx.simulate(g, "edges + nodematch('gender') + gwesp(0.5, fixed=TRUE)", [-2.5, 1.0, 0.6],
                          nsim=1, seed=k, burnin=20_000)
    classes.append(h)
combined = ig.disjoint_union(classes)
combined.vs["classroom"] = [k + 1 for k in range(500) for _ in range(20)]
write(combined, "classrooms")
print(f"classrooms: {combined.ecount()} edges in 500 classrooms")
