import json
from pathlib import Path

import igraph as ig
import pytest

DATA = Path(__file__).parent / "data"

#: Results of R's ergm on the same networks and models (scripts/r_reference.R).
REFERENCE = json.loads((DATA / "r_reference.json").read_text())["models"]


def load(name: str) -> ig.Graph:
    """One of the networks exported from R's ergm package."""
    g = ig.Graph.Read_GraphML(str(DATA / f"{name}.graphml"))
    del g.vs["id"]  # added by the GraphML reader
    return g


@pytest.fixture(params=sorted(REFERENCE))
def reference(request):
    """Each reference model, with its network loaded."""
    model = REFERENCE[request.param]
    return request.param, model, load(model["network"])
