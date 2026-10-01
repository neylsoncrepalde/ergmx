import json
from pathlib import Path

import igraph as ig
import numpy as np
import pytest

import ergmx

DATA = Path(__file__).parent / "data"

#: Results of R's ergm on the same networks and models (scripts/r_reference.R).
REFERENCE = json.loads((DATA / "r_reference.json").read_text())["models"]


#: Networks combined from several, by the names scripts/r_reference.R gives them.
COMBINED = {
    "samplk123.Networks": lambda: ergmx.Networks(*(load(f"samplk{k}") for k in (1, 2, 3))),
    "samplk123.NetSeries": lambda: ergmx.NetSeries(*(load(f"samplk{k}") for k in (1, 2, 3))),
    "samplk12.NetSeries": lambda: ergmx.NetSeries(load("samplk1"), load("samplk2")),
    "Goeyvaerts.weekday": lambda: ergmx.Networks(
        [g for g in ergmx.datasets.load("Goeyvaerts") if g["included"] and g["weekday"]]),
}


def load(name: str):
    """One of the networks exported from R's ergm package, with its dyadic
    covariates (tests/data/<name>__<covariate>.csv) as graph attributes, or
    networks combined from several (COMBINED)."""
    if name in COMBINED:
        return COMBINED[name]()
    g = ig.Graph.Read_GraphML(str(DATA / f"{name}.graphml"))
    del g.vs["id"]  # added by the GraphML reader
    for path in DATA.glob(f"{name}__*.csv"):
        g[path.stem.split("__", 1)[1]] = np.loadtxt(path, delimiter=",")
    return g


def options(model: dict) -> dict:
    """The constraints, offset coefficients and bipartite modes of a reference
    model, as ergm() arguments."""
    # jsonlite writes R's NULL as {}.
    out = {k: model[k] for k in ("constraints", "offset_coef") if model.get(k) not in (None, {}, [])}
    if model.get("bipartite") is True:
        out["bipartite"] = "type"
    return out


def without_overflow(stats: dict) -> dict:
    """Statistics without the overflow bins of curved terms, which ergm doesn't have."""
    return {k: v for k, v in stats.items() if "#>" not in k}


def estimated(model: dict) -> list[str]:
    """The reference model's estimated (not offset) coefficients."""
    return [name for name in model["mle"] if not name.startswith("offset(")]


def models_with(check: str) -> list[str]:
    return sorted(name for name, m in REFERENCE.items() if check in m["checks"])


@pytest.fixture(params=sorted(REFERENCE))
def reference(request):
    """Each reference model, with its network loaded."""
    model = REFERENCE[request.param]
    return request.param, model, load(model["network"])
