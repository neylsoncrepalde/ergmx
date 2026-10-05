"""Multilayer networks, as R's ergm.multi's ``Layer()`` (`Krivitsky, Koehly
and Marcum 2020 <https://doi.org/10.1007/s11336-020-09720-7>`__): several
relations on the same vertices, as layers of one network. Terms sum over
the layers; ``L(formula, Ls)`` evaluates terms on observed or logical
layers, whose ties are functions of the observed layers' ties (ergm.multi's
Layer Logic), and the layer-aware terms (``CMBL``, ``twostarL``,
``mutualL``, ``espL``...) relate the layers."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np

from ._network import Network, as_network
from .terms import TERMS, ErgmDifferenceWarning, Term, as_formula

# -- Layer() ------------------------------------------------------------------------------------


def Layer(*networks, **named) -> Network:  # noqa: N802 (R's name)
    """Layers of one network, as R's ``ergm.multi::Layer()``: networks on the
    same vertices, as arguments (``Layer(g1, g2)``), named
    (``Layer(marriage=g1, business=g2)``) or in a list or dict.

    Terms outside ``L()`` sum over the layers (each layer's ties count in
    its own block, so no term sees ties of different layers together);
    ``L()`` and the layer-aware terms refer to the layers by name or by
    number in backticks, as in ``L(~edges, ~`1` & `2`)``. Ties only form
    within layers. Networks simulated from it are dicts of their layers by
    name, which ``Layer()`` takes back.
    """
    from ._multi import _combine

    if len(networks) == 1 and isinstance(networks[0], (list, tuple, dict)):
        if named:
            raise ValueError("Layer(): give the layers as arguments, names=networks, or one list or dict")
        networks = networks[0]
    if isinstance(networks, dict):
        named, networks = dict(networks), ()
    graphs = list(networks) + list(named.values())
    names = [str(k + 1) for k in range(len(networks))] + list(named)
    if len(graphs) < 2:
        raise ValueError("Layer() needs at least 2 layers")
    if len(set(names)) != len(names):
        raise ValueError("Layer(): duplicate layer names")
    parts = [as_network(g) for g in graphs]
    if len({p.n for p in parts}) > 1:
        raise ValueError("Layer(): the layers must have the same vertices")
    if len({p.directed for p in parts}) > 1 or any(p.bipartite or p.combined for p in parts):
        raise ValueError("Layer(): the layers must be all directed or all undirected, and one-mode")
    first = parts[0]
    # Every layer has the first one's vertex attributes, as ergm.multi's.
    parts = [Network(p.n, p.directed, p.edges, first.attributes, p.source, p.graph_attributes, p.missing)
             for p in parts]
    attributes = [{".LayerID": k + 1, ".LayerName": name, ".NetworkID": k + 1, ".NetworkName": name}
                  for k, name in enumerate(names)]
    combined = _combine(parts, attributes, None)
    combined.graph_attributes["_layers"] = names
    return combined


def _layer_names(network: Network) -> list[str]:
    names = network.graph_attributes.get("_layers") if network.combined else None
    if names is None:
        raise ValueError("layer-aware terms and L() need a network of layers, from Layer()")
    return list(names)


# -- Layer Logic --------------------------------------------------------------------------------

#: Operator codes of src/layers.rs.
_UNARY = {"!": 1, "neg": 2, "abs": 3, "sign": 4, "round": 5}
_BINARY = {"&": 10, "&&": 10, "|": 11, "||": 11, "xor": 12, "==": 13, "!=": 14, "<": 15, ">": 16, "<=": 17,
           ">=": 18, "+": 19, "-": 20, "*": 21, "/": 22, "%/%": 22, "%%": 23, "^": 24, "round2": 25}
_TOKEN = re.compile(r"\s*(?:(`[^`]*`)|(\d+(?:\.\d*)?)|([A-Za-z_.][\w.]*)|(%/%|%%|==|!=|<=|>=|&&|\|\||[-+*/^<>&|!(),~]))")


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"can't read the layer logic {text!r} at {text[pos:]!r}")
        out.append(next(g for g in m.groups() if g is not None))
        pos = m.end()
    return out


class _Parser:
    """R's grammar for Layer Logic, with R's operator precedence (from the
    lowest): | ||, & &&, !, comparisons, + -, * /, %% %/%, unary - +, ^."""

    def __init__(self, tokens: list[str]):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"layer logic: expected {expected or 'more'}, got {token!r}")
        self.pos += 1
        return token

    def parse(self):
        node = self.disjunction()
        if self.peek() is not None:
            raise ValueError(f"layer logic: unexpected {self.peek()!r}")
        return node

    def disjunction(self):
        node = self.conjunction()
        while self.peek() in ("|", "||"):
            node = (self.take(), node, self.conjunction())
        return node

    def conjunction(self):
        node = self.negation()
        while self.peek() in ("&", "&&"):
            node = (self.take(), node, self.negation())
        return node

    def negation(self):
        if self.peek() == "!":
            self.take()
            return ("!", self.negation())
        return self.comparison()

    def comparison(self):
        node = self.sum()
        if self.peek() in ("==", "!=", "<", ">", "<=", ">="):
            node = (self.take(), node, self.sum())
        return node

    def sum(self):
        node = self.product()
        while self.peek() in ("+", "-"):
            node = (self.take(), node, self.product())
        return node

    def product(self):
        node = self.special()
        while self.peek() in ("*", "/"):
            node = (self.take(), node, self.special())
        return node

    def special(self):
        node = self.unary()
        while self.peek() in ("%%", "%/%"):
            node = (self.take(), node, self.unary())
        return node

    def unary(self):
        if self.peek() in ("-", "+"):
            op = self.take()
            inner = self.unary()
            return ("neg", inner) if op == "-" else inner
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek() == "^":
            self.take()
            node = ("^", node, self.unary())
        return node

    def atom(self):
        token = self.take()
        if token == "(":
            node = self.disjunction()
            self.take(")")
            return node
        if token[0] == "`":
            return ("name", token[1:-1])
        if token[0].isdigit():
            return ("const", int(float(token)))
        if token in ("TRUE", "T"):
            return ("const", 1)
        if token in ("FALSE", "F"):
            return ("const", 0)
        if self.peek() == "(" and token in ("t", "abs", "sign", "round", "xor"):
            self.take("(")
            args = [self.disjunction()]
            while self.peek() == ",":
                self.take(",")
                args.append(self.disjunction())
            self.take(")")
            if token == "t":
                if len(args) != 1 or args[0][0] != "name":
                    raise ValueError("layer logic: t() applies to an observed layer only, as t(`1`)")
                return ("t", args[0][1])
            if token == "xor":
                if len(args) != 2:
                    raise ValueError("layer logic: xor() takes two arguments")
                return ("xor", *args)
            if token == "round" and len(args) == 2:
                return ("round2", *args)
            if len(args) != 1:
                raise ValueError(f"layer logic: {token}() takes one argument")
            return (token, args[0])
        if token[0].isalpha() or token[0] in "._":
            return ("name", token)
        raise ValueError(f"layer logic: unexpected {token!r}")


@dataclass
class LayerLogic:
    """A logical layer: its R text (for names) and its tree."""

    text: str
    tree: tuple

    @classmethod
    def parse(cls, text: str) -> LayerLogic:
        text = text.strip()
        if text.startswith("~"):
            text = text[1:].strip()
        return cls(text, _Parser(_tokens(text)).parse())

    def program(self, names: list[str]) -> list[int]:
        """The postfix program of src/layers.rs, with the layers numbered by `names`."""

        def layer(name: str) -> int:
            if name.isdigit():
                k = int(name)
                if not 1 <= k <= len(names):
                    raise ValueError(f"layer logic: no layer {k} of {len(names)}")
                return k
            if name not in names:
                raise ValueError(f"layer logic: no layer {name!r}; the layers are {names}")
            return names.index(name) + 1

        out: list[int] = []

        def emit(node):
            kind = node[0]
            if kind == "name":
                out.extend([1, layer(node[1])])
            elif kind == "t":
                out.extend([2, layer(node[1])])
            elif kind == "const":
                out.extend([3, node[1]])
            elif kind in _UNARY:
                emit(node[1])
                out.extend([4, _UNARY[kind]])
            else:
                emit(node[1])
                emit(node[2])
                out.extend([4, _BINARY[kind]])

        emit(self.tree)
        return out

    def label(self) -> str:
        """As ergm.multi names it: R's deparse without spaces (a lone layer
        without its backticks)."""
        if self.tree[0] == "name":
            return self.tree[1]
        return re.sub(r"\s+", "", self.text)

    def is_empty_on_empty(self, names) -> bool:
        """Whether the layer has no ties when the observed layers have none."""
        values = {"name": lambda n: 0, "t": lambda n: 0}

        def value(node):
            kind = node[0]
            if kind in values:
                return 0
            if kind == "const":
                return node[1]
            if kind in _UNARY:
                x = value(node[1])
                return {"!": int(x == 0), "neg": -x, "abs": abs(x), "sign": int(np.sign(x)), "round": x}[kind]
            x, y = value(node[1]), value(node[2])
            return {"&": int(bool(x and y)), "&&": int(bool(x and y)), "|": int(bool(x or y)),
                    "||": int(bool(x or y)), "xor": int(bool(x) != bool(y)), "==": int(x == y),
                    "!=": int(x != y), "<": int(x < y), ">": int(x > y), "<=": int(x <= y), ">=": int(x >= y),
                    "+": x + y, "-": x - y, "*": x * y}.get(kind, 0)

        return value(self.tree) == 0


def _split(text: str) -> list[str]:
    """R's c(...) or list(...) of formulas, as their texts (a formula as itself)."""
    text = text.strip()
    m = re.match(r"^(c|list)\s*\((.*)\)$", text, re.S)
    if m is None:
        return [text]
    items, depth, start, inner = [], 0, 0, m.group(2)
    for k, ch in enumerate(inner):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(inner[start:k])
            start = k + 1
    items.append(inner[start:])
    return [x.strip() for x in items if x.strip()]


def _layer_list(spec, default_all: bool = True) -> list[tuple[float, LayerLogic | None]]:
    """A list of (weight, logical layer) from L()'s Ls: "~." (every observed
    layer: None), one formula, or a list (R's c() or list(), or Python's),
    each formula possibly two-sided ("3 ~ m") for its weight."""
    if spec is None or (isinstance(spec, str) and spec.strip().lstrip("~").strip() == "."):
        return [(1.0, None)] if default_all else []
    items = _split(spec) if isinstance(spec, str) else [spec] if isinstance(spec, LayerLogic) else list(spec)
    out = []
    for item in items:
        if isinstance(item, tuple):
            weight, logic = item
            out.append((float(weight), logic if isinstance(logic, LayerLogic) else LayerLogic.parse(logic)))
        elif isinstance(item, LayerLogic):
            out.append((1.0, item))
        else:
            text = str(item).strip()
            lhs, _, rhs = text.partition("~") if not text.startswith("~") else ("", "~", text[1:])
            out.append((float(lhs) if lhs.strip() else 1.0, LayerLogic.parse(rhs)))
    return out


def _views(network: Network, layers) -> list[tuple[float, list[int], str]]:
    """The (weight, program, label) of each logical layer, expanding "~."
    to every observed layer."""
    names = _layer_names(network)
    out = []
    for weight, logic in layers:
        if logic is None:
            out.extend((weight, [1, k + 1], names[k]) for k in range(len(names)))
            continue
        if not logic.is_empty_on_empty(names):
            raise ValueError(f"layer logic {logic.text!r}: logical layers with ties where no layer has one "
                             "are not supported, as in ergm.multi")
        out.append((weight, logic.program(names), logic.label()))
    return out


def _references(tree) -> list[tuple[str, bool]]:
    """The layers a logical layer's tree reads: (name, reversed)."""
    if tree[0] in ("name", "t"):
        return [(tree[1], tree[0] == "t")]
    return [ref for child in tree[1:] if isinstance(child, tuple) for ref in _references(child)]


def _encode(programs: list[list[int]]) -> list[int]:
    return [x for p in programs for x in [len(p), *p]]


def _list_label(layers) -> str:
    """ergm.multi's label of a list of layers: (a,b), with weights 3~a."""
    labels = []
    for weight, logic in layers:
        text = "." if logic is None else logic.label() if logic.tree[0] == "name" else logic.label()
        labels.append(text if weight == 1 else f"{weight:g}~{text}")
    return "(" + ",".join(labels) + ")" if len(layers) > 1 else labels[0]


# -- L() ----------------------------------------------------------------------------------------


class L(Term):  # noqa: N801 (R's name)
    """ergm.multi's L(formula, Ls=~.): the formula's terms on logical layers,
    weighted and summed."""

    def __init__(self, formula, Ls="~."):  # noqa: N803 (R's name)
        self.formula = as_formula(formula)
        self.layers = _layer_list(Ls)

    triadic = property(lambda self: any(t.triadic for t in self.formula))

    @property
    def dyad_independent(self) -> bool:
        # A logical layer of two layers (or of a layer both ways) ties the
        # dyads of the combined network together; "~." is each layer alone.
        return all(t.dyad_independent for t in self.formula) and all(
            logic is None or len(set(_references(logic.tree))) <= 1 for _, logic in self.layers)

    @staticmethod
    def _layer(network: Network) -> Network:
        """A layer's network (the first's: its vertices and attributes)."""
        return network.blocks[0].network

    def check(self, network):
        _layer_names(network)
        for term in self.formula:
            term.check(self._layer(network))
            if term.curved:
                raise NotImplementedError("L() of curved terms is not supported yet: fix their decays")

    def _label(self, network) -> str:
        if len(self.layers) == 1 and self.layers[0][1] is None:
            return "."
        return _list_label(self.layers)

    def names(self, network):
        label = self._label(network)
        return [f"L({label})~{n}" for t in self.formula for n in t.names(self._layer(network))]

    def full_spec(self, network):
        views = _views(network, self.layers)
        layer = self._layer(network)
        return ("layerL", [w for w, _, _ in views], [len(views), *_encode([p for _, p, _ in views])],
                [t.full_spec(layer) for t in self.formula])

    def spec(self, network):
        return self.full_spec(network)[:3]

    def __repr__(self) -> str:
        return f"L({self.formula!r})"


# -- Layer-aware terms --------------------------------------------------------------------------


def _one_layer(spec) -> LayerLogic | None:
    if spec is None:
        return None
    if isinstance(spec, LayerLogic):
        return spec
    return LayerLogic.parse(str(spec))


def _items(spec) -> list:
    return _split(spec) if isinstance(spec, str) else [spec] if isinstance(spec, LayerLogic) else list(spec)


def _two_layers(spec) -> list[LayerLogic]:
    """Two logical layers (one given twice), from a formula or a list of one or two."""
    items = _items(spec)
    logics = [_one_layer(x) for x in items]
    if len(logics) == 1:
        logics = logics * 2
    if len(logics) != 2:
        raise ValueError("give one or two layers")
    return logics


class _LayerTerm(Term):
    """A term of logical layers of a Layer() network."""

    dyad_independent = False

    def check(self, network):
        _layer_names(network)

    def _programs(self, network, logics) -> list[int]:
        names = _layer_names(network)
        for logic in logics:
            if not logic.is_empty_on_empty(names):
                raise ValueError(f"layer logic {logic.text!r}: logical layers with ties where no layer has one "
                                 "are not supported, as in ergm.multi")
        return _encode([logic.program(names) for logic in logics])

    def spec(self, network):
        return self.full_spec(network)[:3]


class CMBL(_LayerTerm):
    """ergm.multi's CMBL(Ls=~.): Conway-Maxwell-binomial dependence among the
    layers: the sum over dyads of log(E! (R - E)! / R!), E the number of the
    R layers with a tie in the dyad."""

    def __init__(self, Ls="~."):  # noqa: N803
        self.layers = None if isinstance(Ls, str) and Ls.strip().lstrip("~").strip() == "." else \
            [_one_layer(x) for x in _items(Ls)]

    def names(self, network):
        if self.layers is None:
            return ["CMBL(~.)"]
        return ["CMBL(list(" + ",".join("~" + logic.label() for logic in self.layers) + "))"]

    def full_spec(self, network):
        names = _layer_names(network)
        logics = self.layers or [LayerLogic(f"`{k + 1}`", ("name", str(k + 1))) for k in range(len(names))]
        return ("layerCMB", [], [len(logics), *self._programs(network, logics)], [])

    def __repr__(self) -> str:
        return "CMBL()"


class TwostarL(_LayerTerm):
    """ergm.multi's twostarL(Ls, type, distinct=TRUE): two-stars whose ties are
    in the two logical layers: at a vertex i, ties i -- j in the first and
    i -- k in the second (directed: i -> j and i -> k, "out"; j -> i and
    k -> i, "in"; j -> i and i -> k, "path"), j and k distinct with
    ``distinct``. Ordered pairs of ties: a layer with itself counts each
    two-star twice."""

    TYPES = {"out": (1, "<>"), "in": (2, "><"), "path": (3, ">>")}

    def __init__(self, Ls, type: str = "out", distinct: bool = True):
        self.logics, self.type, self.distinct = _two_layers(Ls), type, bool(distinct)
        if type not in self.TYPES:
            raise ValueError(f"twostarL(type=): one of {', '.join(map(repr, self.TYPES))}")

    def names(self, network):
        separator = self.TYPES[self.type][1] if network.directed else "--"
        label = separator.join(logic.label() for logic in self.logics)
        return [f"twostarL({label}{',distinct' if self.distinct else ''})"]

    def full_spec(self, network):
        code = self.TYPES[self.type][0] if network.directed else 0
        return ("layerTwostar", [], [code, int(self.distinct), *self._programs(network, self.logics)], [])

    def __repr__(self) -> str:
        return f"twostarL({self.type!r})"


class MutualL(_LayerTerm):
    """ergm.multi's mutualL(same, by, diff, keep, Ls): ordered pairs (i, j)
    with i -> j in the first logical layer and j -> i in the second (a
    layer with itself counts each mutual pair twice), between vertices that
    match on ``same`` (by level with ``diff``), or by the level of each
    vertex (``by``)."""

    directed = True

    def __init__(self, same=None, by=None, diff: bool = False, keep=None, Ls=None):  # noqa: N803
        if Ls is None:
            raise ValueError("mutualL() needs Ls= (without it, use mutual())")
        self.same, self.by, self.diff, self.keep = same, by, bool(diff), keep
        self.logics = _two_layers(Ls)

    def _levels(self, network):
        attr = self.same if self.same is not None else self.by
        if attr is None:
            return None, None
        values = network.blocks[0].network.attribute(attr)  # a layer's vertices
        levels = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
        if self.keep is not None:
            levels = [levels[k - 1] for k in np.atleast_1d(self.keep)]
        return values, levels

    def names(self, network):
        a, b = self.logics
        label = a.label() if a.label() == b.label() else f"{a.label()},{b.label()}"
        values, levels = self._levels(network)
        if levels is None:
            inner = ["mutual"]
        elif self.same is not None:
            inner = [f"mutual.same.{self.same}.{v}" for v in levels] if self.diff else [f"mutual.{self.same}"]
        else:
            inner = [f"mutual.by.{self.by}.{v}" for v in levels]
        return [f"L({label})~{n}" for n in inner]

    def full_spec(self, network):
        values, levels = self._levels(network)
        if levels is None:
            mode, codes = 0, []
        else:
            position = {v: k for k, v in enumerate(levels)}
            codes = [position.get(v, -1 - k) for k, v in enumerate(values)]  # unmatched: never match
            mode = 3 if self.same is None else 2 if self.diff else 1
        return ("layerMutual", [], [mode, len(levels or []), *codes, *self._programs(network, self.logics)], [])

    def __repr__(self) -> str:
        return "mutualL()"


class SharedPartnersL(_LayerTerm):
    """ergm.multi's layer-aware shared partner terms: espL, dspL, nspL and
    their gw versions (despL... aliases). A shared partner k of (i, j) has
    its two ties in the path layers (one in each, in either order unless
    ``L.in_order``), and the base layer has the edges (esp), or not (nsp);
    dsp counts every dyad."""

    TYPE_CODES = {"UTP": 0, "OTP": 1, "ITP": 2, "OSP": 4, "ISP": 5}
    KINDS = {"esp": 0, "dsp": 1, "nsp": 2}

    def __init__(self, kind: str, d=None, decay=None, fixed: bool = False, cutoff: int = 30, type: str = "OTP",
                 L_base=None, Ls_path=None, L_in_order: bool = False):  # noqa: N803
        if decay is not None and not fixed:
            raise NotImplementedError(f"gw{kind}L with an estimated decay is not supported yet: fixed=TRUE")
        if type == "RTP":
            raise ValueError("layer-aware shared partner terms do not support reciprocated two-paths, as in ergm.multi")
        self.kind, self.d, self.decay, self.cutoff, self.type = kind, d, decay, int(cutoff), type
        self.base = _one_layer(L_base) if kind != "dsp" else None
        path = None if Ls_path is None else [_one_layer(x) for x in _items(Ls_path)]
        if path is None and self.base is None:
            raise ValueError(f"{kind}L() needs L.base or Ls.path")
        if path is None:
            path = [self.base, self.base]
        if len(path) == 1:
            path = path * 2
        self.path = path
        if self.base is None and kind != "dsp":
            self.base = path[0]
        self.in_order = bool(L_in_order)
        self.triadic = True

    def check(self, network):
        super().check(network)
        if network.directed and self.in_order and self.type in ("OSP", "ISP"):
            warnings.warn(
                f"{self.kind}L(type={self.type!r}, L.in_order=TRUE) puts the first tie at the first vertex of the "
                "pair (the base tie's tail), as ergm.multi documents; ergm.multi 0.3.0's shared partner cache "
                "doesn't keep the order, so its statistics depend on the order of the ties (and differ with "
                "cache.sp=FALSE).", ErgmDifferenceWarning, stacklevel=5)

    def _type(self, network) -> str:
        return self.type if network.directed else "UTP"

    def _ds(self) -> list[int]:
        return [int(x) for x in np.atleast_1d(self.d)]

    def names(self, network):
        pth = self.path[0].label() if self.path[0].label() == self.path[1].label() \
            else f"{self.path[0].label()},{self.path[1].label()}"
        bse = self.base.label() if self.base is not None and self.kind != "dsp" else ""
        wrap = f"L(pth=({pth}),bse={bse},inord={'TRUE' if self.in_order else 'FALSE'})~"
        t = "" if not network.directed else f".{self.type}"
        if self.decay is not None:
            return [f"{wrap}gw{self.kind}{t}.fixed.{self.decay:g}"]
        return [f"{wrap}{self.kind}{t}{d}" for d in self._ds()]

    def full_spec(self, network):
        t = self._type(network)
        any_order = t == "UTP" or not self.in_order
        logics = [*self.path, *([self.base] if self.kind != "dsp" else [])]
        reals = [] if self.decay is None else [float(self.decay)]
        ds = [] if self.decay is not None else self._ds()
        return ("layerSP", reals, [self.KINDS[self.kind], self.TYPE_CODES[t], int(any_order), self.cutoff, len(ds),
                                   *ds, len(logics), *self._programs(network, logics)], [])

    def __repr__(self) -> str:
        return f"{self.kind}L()"


def _sp(kind: str, gw: bool):
    def make(*args, type="OTP", L_base=None, Ls_path=None, L_in_order=False, **kwargs):  # noqa: N803
        if gw:
            decay = args[0] if args else kwargs.pop("decay")
            fixed = args[1] if len(args) > 1 else kwargs.pop("fixed", False)
            cutoff = args[2] if len(args) > 2 else kwargs.pop("cutoff", 30)
            if len(args) > 3:
                type = args[3]
            return SharedPartnersL(kind, decay=decay, fixed=fixed, cutoff=cutoff, type=type, L_base=L_base,
                                   Ls_path=Ls_path, L_in_order=L_in_order)
        d = args[0] if args else kwargs.pop("d")
        if len(args) > 1:
            type = args[1]
        return SharedPartnersL(kind, d=d, type=type, L_base=L_base, Ls_path=Ls_path, L_in_order=L_in_order)

    return make


#: The multilayer terms, by their R names.
LAYER_TERMS = {
    "L": L, "CMBL": CMBL, "twostarL": TwostarL, "mutualL": MutualL,
    "despL": _sp("esp", False), "espL": _sp("esp", False), "ddspL": _sp("dsp", False), "dspL": _sp("dsp", False),
    "dnspL": _sp("nsp", False), "nspL": _sp("nsp", False),
    "dgwespL": _sp("esp", True), "gwespL": _sp("esp", True), "dgwdspL": _sp("dsp", True),
    "gwdspL": _sp("dsp", True), "dgwnspL": _sp("nsp", True), "gwnspL": _sp("nsp", True),
}


TERMS.update(LAYER_TERMS)
