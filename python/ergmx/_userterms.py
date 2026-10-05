"""Terms written in Python. Subclass :class:`UserTerm` with the statistics'
names and their change on toggling a dyad; register the class to use it in
formula strings. A dyad-independent term's changes are tabulated once, for
every dyad, and are then as fast as ergmx's own terms; a dyad-dependent
term's change function is called at every MCMC proposal, which holds
Python's GIL, so its chains run one at a time and much slower: for trying a
statistic out before writing it in Rust."""

from __future__ import annotations

import contextlib

import numpy as np

from .terms import TERMS, Term

#: The Python terms of the model being bound (their bound callbacks), by index.
_COLLECTING: list[list] = []


@contextlib.contextmanager
def collecting():
    """Collects the callbacks of the Python terms whose specs are made inside."""
    callbacks: list = []
    _COLLECTING.append(callbacks)
    try:
        yield callbacks
    finally:
        _COLLECTING.pop()


class _EmptyView:
    """The network without ties, for tabulating a dyad-independent term."""

    def __init__(self, n: int, directed: bool):
        self.n, self.directed, self.n_edges = n, directed, 0

    def has_edge(self, i, j):
        return False

    def neighbours(self, i):
        return []

    out_neighbours = in_neighbours = neighbours

    def degree(self, i):
        return 0

    in_degree = degree

    def edges(self):
        return []


class _Bound:
    """A Python term bound to a network, as the Rust core calls it back."""

    def __init__(self, term: UserTerm, p: int):
        self.term, self.p = term, p

    def _change(self, view, i: int, j: int, adding: bool) -> list[float]:
        return [float(v) for v in np.atleast_1d(self.term.change(view, i, j, adding))]


class UserTerm(Term):
    """A term written in Python.

    Subclass it with:

    - ``names(network)``: the statistics' names (by default, the class's
      ``label``, or its name in lower case);
    - ``change(net, i, j, adding)``: the change of the statistics on toggling
      the dyad (i, j) of the network ``net``: adding the tie if ``adding``,
      removing it otherwise (``net.has_edge(i, j)`` is then true). ``net``
      has ``n``, ``directed``, ``n_edges``, ``has_edge(i, j)``,
      ``neighbours(i)`` (out-neighbours, if directed), ``in_neighbours(i)``,
      ``degree(i)``, ``in_degree(i)`` and ``edges()``, and is only valid during
      the call. ``self.network`` is the network the term is bound to, with
      its vertex attributes (``self.network.attributes``);
    - optionally ``empty(network)``: the statistics of the network without
      ties (0 by default);
    - ``dyad_independent = True`` if the change of each dyad doesn't depend
      on the rest of the network: its changes are then tabulated once
      (``net`` is the empty network), for fast sampling;
    - ``directed = True`` or ``False`` if it is only defined for one kind of
      network;
    - ``triadic = True`` if it counts triangles or shared partners, for
      ergmx's triadic proposals, which mix faster.

    Register it with :func:`register_term` to use it in formula strings.
    """

    dyad_independent = False

    @property
    def label(self) -> str:
        return getattr(type(self), "name", type(self).__name__.lower())

    def change(self, net, i: int, j: int, adding: bool):
        raise NotImplementedError("a UserTerm needs change(net, i, j, adding)")

    def empty(self, network):
        return np.zeros(len(self.names(network)))

    def full_spec(self, network):
        self.network = network
        p = len(self.names(network))
        empty = np.asarray(self.empty(network), dtype=float).reshape(-1)
        if len(empty) != p:
            raise ValueError(f"{self!r}: empty() gives {len(empty)} values for {p} statistics")
        if self.dyad_independent:
            if empty.any():
                raise ValueError(f"{self!r}: a dyad-independent term's empty network statistics are 0")
            n, view = network.n, _EmptyView(network.n, network.directed)
            table = np.zeros((n, n, p))
            for i in range(n):
                for j in range(i + 1 if not network.directed else 0, n):
                    if i != j:
                        table[i, j] = np.atleast_1d(self.change(view, i, j, True))
            return ("dyadtable", table.ravel().tolist(), [p], [])
        if not _COLLECTING:
            raise RuntimeError("a Python term's spec is only made while binding a model")
        _COLLECTING[-1].append(_Bound(self, p))
        return ("python", empty.tolist(), [len(_COLLECTING[-1]) - 1, p], [])

    def spec(self, network):
        return self.full_spec(network)[:3]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def register_term(name: str, factory) -> None:
    """Make a term (a :class:`UserTerm` subclass, or a function returning a
    term) available in formula strings, by ``name``."""
    if not callable(factory):
        raise TypeError("register_term() takes a UserTerm subclass or a function returning a term")
    if name in TERMS and not getattr(TERMS[name], "_user", False):
        raise ValueError(f"{name!r} is one of ergmx's terms")

    def make(*args, **kwargs):
        return factory(*args, **kwargs)

    make.__name__, make._user = name, True
    TERMS[name] = make
