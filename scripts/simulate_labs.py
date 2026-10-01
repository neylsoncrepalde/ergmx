"""Simulates labs_sim, a multilevel network of 120 researchers and 30
laboratories from a known model, and writes it as
python/ergmx/data/labs_sim.graphml.gz (ergmx.datasets.load("labs_sim")).

The design is that of Lazega et al.'s (2008) researchers and laboratories,
which Wang, Robins, Pattison and Lazega (2013) modelled with MPNet: each
researcher belongs to a laboratory, some to two, and the ties among
researchers and among laboratories depend on each other through these
memberships. The memberships are drawn first; the ties within each level are
then simulated from an ERGM given them (the memberships fixed with blocks()),
with the coefficients of TRUTH.

Run from the root of the ergmx repository:
    uv run python scripts/simulate_labs.py
"""

import gzip
import shutil
import tempfile
from pathlib import Path

import igraph as ig
import numpy as np

import ergmx

N_RESEARCHERS, N_LABS, SEED = 120, 30, 1

MODEL = (
    "S(~edges + gwesp(0.693147, fixed=TRUE), ~level == 'researcher')"
    " + S(~edges, ~level == 'laboratory')"
    " + txbx('level') + txax('level') + c4axb('level')"
)
TRUTH = {
    'S(level=="researcher")~edges': -3.6,
    'S(level=="researcher")~gwesp.fixed.0.693147': 0.3,  # closure among researchers
    'S(level=="laboratory")~edges': -2.8,
    "TXBX.level": 1.5,  # researchers of the same laboratory are tied
    "TXAX.level": 1.5,  # laboratories that share a researcher are tied
    "C4AXB.level": 0.4,  # members of tied laboratories are tied
}
GIVEN = "blocks('level', levels2=2)"  # the laboratory-researcher dyads

rng = np.random.default_rng(SEED)
g = ig.Graph(n=N_RESEARCHERS + N_LABS)
g.vs["name"] = [f"R{i + 1}" for i in range(N_RESEARCHERS)] + [f"L{i + 1}" for i in range(N_LABS)]
g.vs["type"] = [False] * N_RESEARCHERS + [True] * N_LABS  # multinets' convention: True for the higher level
g.vs["level"] = ["researcher"] * N_RESEARCHERS + ["laboratory"] * N_LABS

# Memberships: two researchers in each laboratory, the others drawn by
# laboratory size, and a quarter of the researchers in a second laboratory.
size = rng.gamma(2.0, 1.0, N_LABS)
order = rng.permutation(N_RESEARCHERS)
lab = np.empty(N_RESEARCHERS, dtype=int)
lab[order[:2 * N_LABS]] = np.repeat(np.arange(N_LABS), 2)
lab[order[2 * N_LABS:]] = rng.choice(N_LABS, size=N_RESEARCHERS - 2 * N_LABS, p=size / size.sum())
memberships = {(r, N_RESEARCHERS + lab[r]) for r in range(N_RESEARCHERS)}
for r in rng.choice(N_RESEARCHERS, size=N_RESEARCHERS // 4, replace=False):
    other = rng.choice([k for k in range(N_LABS) if k != lab[r]])
    memberships.add((r, N_RESEARCHERS + other))
g.add_edges(sorted(memberships))

# The ties within each level, given the memberships, after a long burn-in.
g = ergmx.simulate(g, MODEL, TRUTH, constraints=GIVEN, seed=SEED, burnin=2_000_000)[0]

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "labs_sim.graphml"
    g.write_graphml(str(path))
    with open(path, "rb") as f, gzip.GzipFile("python/ergmx/data/labs_sim.graphml.gz", "wb", mtime=0) as out:
        shutil.copyfileobj(f, out)
print(g.summary())
print(ergmx.summary_stats(g, "nodemix('level', levels2=TRUE)"))
print(ergmx.summary_stats(g, MODEL))
