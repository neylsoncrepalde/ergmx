"""Networks from R's ergm package, for examples and tests.

>>> from ergmx import datasets
>>> g = datasets.load("faux.mesa.high")
>>> g.vcount(), g.ecount()
(205, 203)

They are the networks of ergm's documentation and tutorials, distributed
with ergm under the GPL-3, multinets' multilevel network ``linked_sim``, and
Davis's Southern Women, from networkx.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

_DESCRIPTIONS = {
    "flomarriage": (
        "Marriage ties among 16 Renaissance Florentine families (Padgett 1994; "
        "Breiger and Pattison 1986). Undirected. Vertex attributes: wealth "
        "(thousands of lira, 1427), priorates (seats on the civic council) and "
        "totalties (ties in the full dataset)."
    ),
    "flobusiness": (
        "Business ties (loans, credits, partnerships) among the same 16 Florentine "
        "families, with the same attributes as flomarriage. Undirected."
    ),
    "samplk1": (
        "Liking among 18 novice monks at a New England monastery (Sampson 1968), "
        "at the first of three times: each monk named the three he liked most. "
        "Directed. Vertex attributes: group (Turks, Loyal, Outcasts), group3, "
        "group4 and cloisterville (attended the minor seminary of Cloisterville)."
    ),
    "samplk2": "Sampson's monks, liking at the second time. See samplk1.",
    "samplk3": "Sampson's monks, liking at the third time, just before the expulsions. See samplk1.",
    "faux.mesa.high": (
        "A simulated friendship network of 205 students in a high school in the "
        "rural western US, from the Add Health study design (Resnick et al. 1997). "
        "Undirected. Vertex attributes: Grade (7 to 12), Race and Sex."
    ),
    "faux.dixon.high": (
        "A simulated directed friendship network of 248 students in a high school "
        "in the US South, from the Add Health study design: each student nominated "
        "up to 5 male and 5 female friends. Vertex attributes: grade (7 to 12), "
        "race and sex (1 = male, 2 = female)."
    ),
    "davis": (
        "Davis, Gardner and Gardner's (1941) Southern Women: the attendance of 18 "
        "women at 14 social events in Natchez, Mississippi, in the 1930s. Bipartite: "
        "the women are the first mode, the events the second. Vertex attributes: "
        "type (True for events, igraph's convention) and bipartite (0 or 1, "
        "networkx's). Fit it with bipartite=True."
    ),
    "linked_sim": (
        "A simulated multilevel network from the multinets package: 100 individuals "
        "(the lower level) and 50 organizations (the higher level), with ties among "
        "individuals, among organizations, and affiliation ties between the two. "
        "Undirected. Vertex attributes: type (True for organizations, as in multinets) "
        "and level ('individual' or 'organization')."
    ),
    "faux.magnolia.high": (
        "A simulated friendship network of 1,461 students in a high school in the "
        "US South, from the Add Health study design. Undirected. Vertex attributes: "
        "Grade (7 to 12), Race and Sex."
    ),
}


def names() -> list[str]:
    """Names of the bundled networks."""
    return list(_DESCRIPTIONS)


def describe(name: str) -> str:
    """What a bundled network is, its source and its vertex attributes."""
    _check(name)
    return _DESCRIPTIONS[name]


def load(name: str, backend: str = "igraph") -> Any:
    """Load a bundled network.

    Parameters
    ----------
    name : str
        One of :func:`names`, as in R's ergm (``"faux.mesa.high"``,
        ``"samplk3"``...).
    backend : {"igraph", "networkx"}
        Return an :class:`igraph.Graph` or a :class:`networkx.Graph` /
        :class:`networkx.DiGraph`. Vertex names are in the ``name`` attribute.

    Returns
    -------
    igraph.Graph or networkx.Graph
    """
    _check(name)
    path = resources.files("ergmx") / "data" / f"{name}.graphml.gz"
    with resources.as_file(path) as file:
        if backend == "igraph":
            import igraph as ig

            g = ig.Graph.Read(str(file), format="graphmlz")
            if "id" in g.vs.attributes():
                del g.vs["id"]  # added by the GraphML reader
            return g
        if backend == "networkx":
            import networkx as nx

            g = nx.read_graphml(file)
            return nx.relabel_nodes(g, {v: data["name"] for v, data in g.nodes(data=True)})
    raise ValueError(f"backend must be 'igraph' or 'networkx', not {backend!r}")


def _check(name: str) -> None:
    if name not in _DESCRIPTIONS:
        raise KeyError(f"no dataset {name!r}; available: {', '.join(_DESCRIPTIONS)}")
