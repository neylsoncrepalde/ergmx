"""Exports Davis, Gardner and Gardner's Southern Women network, from networkx,
as python/ergmx/data/davis.graphml.gz (ergmx.datasets.load("davis")).

Run from the root of the ergmx repository:
    uv run python scripts/export_davis.py
"""

import gzip
import shutil
import tempfile
from pathlib import Path

import igraph as ig
import networkx as nx

G = nx.davis_southern_women_graph()  # the 18 women, then the 14 events
g = ig.Graph.from_networkx(G)
g.vs["name"] = list(G)
del g.vs["_nx_name"]
for attribute in g.attributes():  # networkx's "top" and "bottom" lists
    del g[attribute]
g.vs["type"] = [bool(b) for b in g.vs["bipartite"]]  # igraph's convention: True for events
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "davis.graphml"
    g.write_graphml(str(path))
    with open(path, "rb") as f, gzip.GzipFile("python/ergmx/data/davis.graphml.gz", "wb", mtime=0) as out:
        shutil.copyfileobj(f, out)
print(g.summary())
