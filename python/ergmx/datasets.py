"""Networks from R's ergm package, for examples and tests.

>>> from ergmx import datasets
>>> g = datasets.load("faux.mesa.high")
>>> g.vcount(), g.ecount()
(205, 203)

They are the networks of ergm's documentation and tutorials, distributed
with ergm under the GPL-3, ergm.multi's household networks ``Goeyvaerts`` (a
list of networks), multinets' multilevel network ``linked_sim``, Davis's
Southern Women, from networkx, ergm.count's valued karate club ``zach``,
bigergm's ``toyNet``, ergm's ``samplike`` and latentnet's ``tribes``, and
``labs_sim``, a multilevel network simulated by ergmx from a known model.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

_DESCRIPTIONS = {
    "flomarriage": (
        "Marriage ties among 16 Renaissance Florentine families (Padgett 1994, a paper to the Social Science History Association; "
        "Breiger and Pattison 1986, doi:10.1016/0378-8733(86)90006-7). Undirected. Vertex attributes: wealth "
        "(thousands of lira, 1427), priorates (seats on the civic council) and "
        "totalties (ties in the full dataset)."
    ),
    "flobusiness": (
        "Business ties (loans, credits, partnerships) among the same 16 Florentine "
        "families, with the same attributes as flomarriage. Undirected."
    ),
    "samplk1": (
        "Liking among 18 novice monks at a New England monastery (Sampson 1968, a PhD dissertation at Cornell University), "
        "at the first of three times: each monk named the three he liked most. "
        "Directed. Vertex attributes: group (Sampson's Turks, Loyal opposition, Outcasts and "
        "Waverers), group3 (the Waverers in the other groups), group4 and cloisterville "
        "(attended the minor seminary of Cloisterville)."
    ),
    "samplk2": "Sampson's monks, liking at the second time. See samplk1.",
    "samplk3": "Sampson's monks, liking at the third time, just before the expulsions. See samplk1.",
    "faux.mesa.high": (
        "A simulated friendship network of 205 students in a high school in the "
        "rural western US, from the Add Health study design (Resnick et al. 1997, doi:10.1001/jama.278.10.823). "
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
        "networkx's). Fit it with bipartite=True. From their book Deep South "
        "(University of Chicago Press; reissued 2022, doi:10.7208/chicago/9780226817996.001.0001)."
    ),
    "linked_sim": (
        "A simulated multilevel network from the multinets package: 100 individuals "
        "(the lower level) and 50 organizations (the higher level), with ties among "
        "individuals, among organizations, and affiliation ties between the two. "
        "Undirected. Vertex attributes: type (True for organizations, as in multinets) "
        "and level ('individual' or 'organization')."
    ),
    "labs_sim": (
        "A multilevel network of 120 researchers and 30 laboratories, simulated by ergmx from a known "
        "model (scripts/simulate_labs.py), with the design of Lazega et al.'s (2008, "
        "doi:10.1016/j.socnet.2008.02.001) researchers and laboratories: each researcher belongs to a "
        "laboratory, and a quarter to two (150 affiliations). Given the affiliations, the ties among "
        "researchers and among laboratories are from the model S(~edges + gwesp(0.693147, fixed=TRUE), "
        "~level == 'researcher') + S(~edges, ~level == 'laboratory') + txbx('level') + txax('level') + "
        "c4axb('level'), with coefficients -3.6, 0.3, -2.8, 1.5, 1.5 and 0.4, to fit with the "
        "constraints blocks('level', levels2=2). Undirected. Vertex attributes: type (True for "
        "laboratories, as in multinets) and level ('researcher' or 'laboratory')."
    ),
    "zach": (
        "Zachary's (1977, doi:10.1086/jfr.33.4.3629752) karate club, as R's ergm.count has it: 34 "
        "members of a university karate club, and the number of contexts (of 8: classes, the "
        "instructor's studio, bars, tournaments...) in which each pair interacted, in the edge "
        "attribute contexts. A valued network: fit it with response='contexts'. Undirected. Vertex "
        "attributes: club (the club the member joined after the split), faction (as Zachary recorded "
        "it) and faction.id (-2, strongly the instructor's, to 2, strongly the president's), and role "
        "(Instructor, President or Member)."
    ),
    "faux.magnolia.high": (
        "A simulated friendship network of 1,461 students in a high school in the "
        "US South, from the Add Health study design. Undirected. Vertex attributes: "
        "Grade (7 to 12), Race and Sex."
    ),
    "samplike": (
        "Sampson's (1968) monks, as R's ergm has them: the cumulative liking nominations of 18 novices "
        "in a monastery over the three waves (directed: a tie if a monk named the other in any wave), "
        "with the number of waves in the edge attribute nominations (1 to 3). Vertex attributes: group "
        "(Sampson's Loyal, Outcasts and Turks), group3, group4 and cloisterville (whether the monk "
        "attended the minor seminary Cloisterville). The network of latentnet's examples."
    ),
    "tribes": (
        "Read's (1954) Gahuku-Gama subtribes of the New Guinea highlands, as R's latentnet has them: 16 "
        "subtribes and their relations, as a complete undirected graph whose edge attributes are pos "
        "(1 for an alliance, 29), neg (1 for an enmity, 29), sign (-1, 0 or 1) and sign.012 (0, 1 or 2: "
        "enmity, neither, alliance). Fit it with response='pos' (a binary network of alliances) or "
        "response='sign.012' and the binomial family with 2 trials."
    ),
    "toyNet": (
        "bigergm's toyNet: a network of 200 vertices simulated from an ERGM with local dependence, in 4 "
        "blocks of 50 (vertex attribute block), whose ties between blocks are independent and whose "
        "ties within blocks depend on each other through triangles. Undirected. Vertex attributes: "
        "block, and the covariates x and y (whole numbers), for nodematch terms. From R's bigergm "
        "(Fritz, Schweinberger, Komatsu, Martínez Dahbura, Nishida and Mele)."
    ),
    "Goeyvaerts": (
        "A list of 318 networks: the contacts within households with at least one "
        "child aged 12 or under, in Flanders and Brussels, from contact diaries "
        "(Goeyvaerts et al. 2018, Proc. R. Soc. B 285: 20182201, doi:10.1098/rspb.2018.2201; curated by Pietro "
        "Coletti for R's ergm.multi; cite both when publishing). Undirected. Vertex "
        "attributes: age, gender ('F', 'M') and role ('Father', 'Mother', 'Child', "
        "'Grandmother'). Graph attributes: weekday (whether the diary was kept on a "
        "weekday) and included (whether Goeyvaerts et al. analysed it; two were not)."
    ),
}

#: Datasets that are lists of networks, stored as JSON.
_LISTS = {"Goeyvaerts"}


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
        A list of them for lists of networks (``"Goeyvaerts"``).
    """
    _check(name)
    if name in _LISTS:
        return _load_list(name, backend)
    if backend == "igraph":
        return _read_igraph(name).copy()
    path = resources.files("ergmx") / "data" / f"{name}.graphml.gz"
    with resources.as_file(path) as file:
        if backend == "networkx":
            import networkx as nx

            g = nx.read_graphml(file)
            return nx.relabel_nodes(g, {v: data["name"] for v, data in g.nodes(data=True)})
    raise ValueError(f"backend must be 'igraph' or 'networkx', not {backend!r}")


@lru_cache(maxsize=None)
def _read_igraph(name: str):
    """A bundled network read once: python-igraph's file reader leaves a C
    file stream open at each read, and Windows allows only 512 of them."""
    import igraph as ig

    with resources.as_file(resources.files("ergmx") / "data" / f"{name}.graphml.gz") as file:
        g = ig.Graph.Read(str(file), format="graphmlz")
    if "id" in g.vs.attributes():
        del g.vs["id"]  # added by the GraphML reader
    return g


def _load_list(name: str, backend: str) -> list:
    import gzip
    import json

    with resources.as_file(resources.files("ergmx") / "data" / f"{name}.json.gz") as file:
        with gzip.open(file, "rt") as f:
            records = json.load(f)
    graphs = []
    for r in records:
        edges = [tuple(e) for e in r["edges"]]
        if backend == "igraph":
            import igraph as ig

            g = ig.Graph(n=r["n"], edges=edges)
            for a, values in r["vertex"].items():
                g.vs[a] = values
            for a, value in r["graph"].items():
                g[a] = value
        elif backend == "networkx":
            import networkx as nx

            g = nx.Graph(**r["graph"])
            names = r["vertex"]["name"]
            g.add_nodes_from((names[v], {a: values[v] for a, values in r["vertex"].items() if a != "name"})
                             for v in range(r["n"]))
            g.add_edges_from((names[i], names[j]) for i, j in edges)
        else:
            raise ValueError(f"backend must be 'igraph' or 'networkx', not {backend!r}")
        graphs.append(g)
    return graphs


def _check(name: str) -> None:
    if name not in _DESCRIPTIONS:
        raise KeyError(f"no dataset {name!r}; available: {', '.join(_DESCRIPTIONS)}")
