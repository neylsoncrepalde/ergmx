"""Several networks combined into one, as R's ergm.multi (``Networks()``) and
tergm (``NetSeries()``) do: the networks are the blocks of a block-diagonal
network, with no ties between blocks, and the operators N(), Form(),
Persist(), Diss(), Cross() and Change() evaluate their terms on each block."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np

from ._network import Block, Network, _edge_array, _read, _with_modes


def _graphs(args, kind: str) -> tuple[list, list[str] | None]:
    """The graphs given as arguments, a list or a dict (whose keys name them)."""
    if len(args) == 1 and isinstance(args[0], dict):
        return list(args[0].values()), [str(k) for k in args[0]]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0]), None
    if not args:
        raise ValueError(f"{kind}() needs networks")
    return list(args), None


def _parts(graphs, bipartite, kind: str) -> list[Network]:
    parts = []
    for g in graphs:
        if isinstance(g, Network):
            if g.combined:
                raise ValueError(f"{kind}() takes single networks, not {g!r}")
            part = g if bipartite is None or g.mode is not None else _with_modes(g, bipartite)
        else:
            part = _with_modes(_read(g), bipartite)
        parts.append(part)
    if len({p.directed for p in parts}) > 1:
        raise ValueError(f"{kind}(): the networks must be all directed or all undirected")
    if len({p.bipartite for p in parts}) > 1:
        raise ValueError(f"{kind}(): the networks must be all bipartite or all not")
    return parts


def _graph_attributes(network: Network) -> dict[str, Any]:
    """Network-level attributes for N()'s linear model: scalars only, as R's
    network attributes in a data frame."""
    return {k: v for k, v in network.graph_attributes.items()
            if isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating))}


def _combine(parts: list[Network], attributes: list[dict], prev: list[Network] | None) -> Network:
    starts = np.cumsum([0] + [p.n for p in parts[:-1]])
    n = int(sum(p.n for p in parts))
    names = sorted({a for p in parts for a in p.attributes})
    vertex = {a: [v for p in parts for v in p.attributes.get(a, [None] * p.n)] for a in names}

    def stack(arrays):
        stacked = [a.astype(np.int64) + s for a, s in zip(arrays, starts) if len(a)]
        return np.ascontiguousarray(np.vstack(stacked).astype(np.uint32)) if stacked \
            else np.zeros((0, 2), dtype=np.uint32)

    mode = np.concatenate([p.mode for p in parts]) if parts[0].bipartite else None
    blocks = tuple(Block(int(s), p, a, None if prev is None else prev[k])
                   for k, (s, p, a) in enumerate(zip(starts, parts, attributes)))
    # A single network's graph attributes (dyadic covariates...) are the combination's too.
    graph = dict(parts[0].graph_attributes) if len(parts) == 1 else {}
    return Network(n, parts[0].directed, stack([p.edges for p in parts]), vertex, None, graph,
                   stack([p.missing for p in parts]), mode, blocks)


def Networks(*networks, bipartite=None) -> Network:  # noqa: N802 (R's name)
    """Several networks to model jointly, as R's ``ergm.multi::Networks()``.

    The networks keep their vertex attributes; inside :func:`~ergmx.terms.N`,
    each network's terms are evaluated on that network, and the operator's
    linear model can use network-level attributes (igraph's graph attributes
    or networkx's ``G.graph``), the network size ``n``, ``.NetworkID`` (1, 2,
    ...) and ``.NetworkName``. Terms outside N() are evaluated on the networks
    combined, as in ergm.multi.

    Parameters
    ----------
    *networks : igraph.Graph or networkx.Graph
        The networks, as arguments, a list, or a dict whose keys name them.
        They must be all directed or all undirected, and can differ in size.
    bipartite : str or bool, optional
        For bipartite networks, the vertex attribute with each vertex's mode,
        as in :func:`ergmx.ergm`.

    Examples
    --------
    >>> from ergmx import datasets
    >>> monks = ergmx.Networks(datasets.load("samplk1"), datasets.load("samplk2"))  # doctest: +SKIP
    >>> ergmx.ergm(monks, "N(~edges + mutual)")  # doctest: +SKIP
    """
    graphs, names = _graphs(networks, "Networks")
    parts = _parts(graphs, bipartite, "Networks")
    names = names or [str(k + 1) for k in range(len(parts))]
    attributes = [{**_graph_attributes(p), "n": p.n, ".NetworkID": k + 1, ".NetworkName": name}
                  for k, (p, name) in enumerate(zip(parts, names))]
    return _combine(parts, attributes, None)


IMPUTERS = ("next", "previous", "majority", "0", "1")


def _dyads(p: Network) -> int:
    if p.bipartite:
        b = int(np.sum(p.mode))
        return b * (p.n - b)
    return p.n * (p.n - 1) // (1 if p.directed else 2)


def _impute(nets: list[Network], imputers, after: Network) -> list[Network]:
    """tergm's imputation of the missing dyads of the networks transitioned
    from, with each of the imputers in turn: ``"next"`` (the dyad's value in
    the next network, ``after`` for the last), ``"previous"`` (in the previous
    one), ``"majority"`` (the more common value of the network's observed
    dyads) and ``"0"`` or ``"1"``."""
    if isinstance(imputers, str):
        imputers = [imputers]
    chosen = []
    for imputer in imputers:
        matches = [m for m in IMPUTERS if m.startswith(str(imputer))] if imputer else []
        if len(matches) != 1:
            raise ValueError(f"unknown NA imputation option {imputer!r}: give one or more of "
                             + ", ".join(repr(m) for m in IMPUTERS))
        chosen.append(matches[0])
    directed = nets[0].directed

    def key(pair):
        a, b = int(pair[0]), int(pair[1])
        return (a, b) if directed or a < b else (b, a)

    ties = [{key(e) for e in p.edges} for p in [*nets, after]]
    missing = [{key(e) for e in p.missing} for p in [*nets, after]]

    def fill(t, dyads, source):  # missing dyads of network t take network source's values
        ties[t] |= {d for d in dyads if d in ties[source]}
        missing[t] = {d for d in missing[t] if d not in dyads or d in missing[source]}

    tie = False
    for imputer in chosen:
        if imputer == "next":
            for t in reversed(range(len(nets))):
                fill(t, set(missing[t]), t + 1)
        elif imputer == "previous":
            for t in range(1, len(nets)):
                fill(t, set(missing[t]), t - 1)
        else:
            for t, p in enumerate(nets):
                if not missing[t]:
                    continue
                value = imputer == "1"
                if imputer == "majority":
                    density = len(ties[t]) / (_dyads(p) - len(missing[t]))
                    if density == 0.5:
                        tie = True
                        continue
                    value = density > 0.5
                if value:
                    ties[t] |= missing[t]
                missing[t] = set()
    left = [bool(m) for m in missing[:-1]]
    if "previous" in chosen and left[0]:
        warnings.warn("NA imputation 'previous' can't impute the dyads of the first network",
                      stacklevel=3)
    if "next" in chosen and left[-1]:
        warnings.warn("NA imputation 'next' can't impute the dyads of the last network transitioned "
                      "from that are missing in the last network too", stacklevel=3)
    if tie:
        warnings.warn("NA imputation 'majority': a network has as many ties as non-ties among its "
                      "observed dyads; its missing dyads stay missing", stacklevel=3)
    return [dataclasses.replace(p, edges=_edge_array(sorted(ties[t])), missing=_edge_array(sorted(missing[t])))
            for t, p in enumerate(nets)]


def NetSeries(*networks, times=None, bipartite=None, na_impute=None) -> Network:  # noqa: N802 (R's name)
    """A series of networks on the same vertices, as R's ``tergm::NetSeries()``,
    to model each transition conditional on the network before it.

    Use it with tergm's operators: :func:`~ergmx.terms.Form` (the ties that
    formed: terms of the union of the previous and the current network),
    :func:`~ergmx.terms.Persist` and :func:`~ergmx.terms.Diss` (the ties that
    persisted: the intersection), :func:`~ergmx.terms.Cross` (the current
    network) and :func:`~ergmx.terms.Change` (the dyads that changed). Their
    linear models can use ``.Time`` (the time of the current network),
    ``.TimeID`` (1 for the first transition, 2...) and ``.TimeDelta`` (the
    time since the previous network). :func:`ergmx.tergm` builds the series
    itself.

    Parameters
    ----------
    *networks : igraph.Graph or networkx.Graph
        The networks, in time order, as arguments or a list. They must have
        the same vertices, in the same order.
    times : list of numbers, optional
        The times the networks were observed: 0, 1, 2... by default.
    bipartite : str or bool, optional
        For bipartite networks, the vertex attribute with each vertex's mode.
    na_impute : str or list of str, optional
        How to impute the missing dyads of the networks transitioned from
        (all but the last), as tergm's ``NA.impute``: ``"next"`` (the dyad's
        value in the next network), ``"previous"`` (in the previous one),
        ``"majority"`` (the more common value among the network's observed
        dyads), ``"0"`` or ``"1"``; several are applied in turn. The networks
        transitioned to keep their missing dyads, which the fit treats as
        missing. Without it, missing dyads in the networks transitioned from
        are an error.
    """
    graphs, _ = _graphs(networks, "NetSeries")
    parts = _parts(graphs, bipartite, "NetSeries")
    if len(parts) < 2:
        raise ValueError("NetSeries() needs at least two networks")
    if len({p.n for p in parts}) > 1:
        raise ValueError("NetSeries(): the networks must have the same vertices")
    times = list(range(len(parts))) if times is None else list(times)
    if len(times) != len(parts) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("NetSeries(): times must be increasing, one per network")
    before = parts[:-1] if na_impute is None else _impute(parts[:-1], na_impute, parts[-1])
    left = [times[t] for t, p in enumerate(before) if len(p.missing)]
    if left:
        raise ValueError(f"NetSeries(): the network(s) at time(s) {', '.join(map(str, left))} have "
                         "missing dyads, and are transitioned from: impute them with na_impute= "
                         "(only the networks transitioned to can have missing dyads)")
    attributes = [{**_graph_attributes(p), "n": p.n, ".NetworkID": k, ".NetworkName": str(k),
                   ".Time": times[k], ".TimeID": k, ".TimeDelta": times[k] - times[k - 1]}
                  for k, p in enumerate(parts) if k > 0]
    return _combine(parts[1:], attributes, before)


def series_from(networks, times=None, bipartite=None, na_impute=None) -> Network:
    """A NetSeries from a NetSeries, or from a list of networks."""
    if isinstance(networks, Network) and networks.series:
        if na_impute is not None:
            raise ValueError("na_impute= is for a list of networks: give it to NetSeries()")
        return networks
    if isinstance(networks, Network) and networks.combined:
        raise ValueError("tergm() needs a NetSeries() or a list of networks, not Networks()")
    if not isinstance(networks, (list, tuple)):
        raise TypeError("tergm() needs a list of networks, in time order, or a NetSeries()")
    return NetSeries(list(networks), times=times, bipartite=bipartite, na_impute=na_impute)
