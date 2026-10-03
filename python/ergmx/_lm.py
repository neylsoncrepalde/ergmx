"""The linear models of N() and tergm's operators: R's one-sided ``lm()``
formulas over network-level attributes, such as ``~log(n) + weekday`` or
``~I(n <= 3)``, giving R's design matrix and column names (with treatment
contrasts for factors, logical and character attributes).

Supported: the intercept (``~1``, removed with ``0 +`` or ``- 1``),
attributes, arithmetic (``+ - * / ^ %% %/%``), comparisons, ``& | !``,
``I()``, ``log``, ``exp``, ``sqrt``, ``abs``, ``factor``, ``as.numeric``,
``as.logical`` and ``offset()``; and, in expressions, ``c()`` and ranges
(``1:3``). Interactions (``a:b``, ``a*b`` outside ``I()``) are not.
"""

from __future__ import annotations

import math
import re

import numpy as np

_TOKEN = re.compile(r"""
    \s*(?:
      (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)
    | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    | (?P<name>[A-Za-z.][A-Za-z0-9._]*)
    | (?P<op>%%|%/%|<=|>=|==|!=|&&|\|\||[-+*/^<>!&|(),~:])
    )""", re.VERBOSE)

# Binary operators by increasing precedence, as in R.
_BINARY = [("|", "||"), ("&", "&&"), ("==", "!=", "<", ">", "<=", ">="), ("+", "-"),
           ("*", "/"), ("%%", "%/%"), (":",)]
_SPACED = {"|", "||", "&", "&&", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "%%", "%/%"}


class LmError(ValueError):
    """The linear model formula can't be parsed or evaluated."""


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise LmError(f"can't parse {text!r} at {text[pos:]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _Parser:
    """Recursive descent over R's expression grammar; nodes are tuples:
    ("num", value, text), ("str", value), ("name", name), ("call", f, args),
    ("unary", op, x), ("binary", op, a, b), ("paren", x)."""

    def __init__(self, text: str):
        self.tokens, self.k, self.text = _tokens(text), 0, text

    def peek(self):
        return self.tokens[self.k] if self.k < len(self.tokens) else (None, None)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None or (value is not None and token[1] != value):
            raise LmError(f"can't parse {self.text!r}: expected {value or 'more'}")
        self.k += 1
        return token

    def parse(self):
        node = self.binary(0)
        if self.k != len(self.tokens):
            raise LmError(f"can't parse {self.text!r} at {self.tokens[self.k][1]!r}")
        return node

    def binary(self, level):
        if level == len(_BINARY):
            return self.unary()
        if level == 2 and self.peek() == ("op", "!"):
            # R's negation binds less tightly than comparisons: !a == b is !(a == b).
            self.take()
            return ("unary", "!", self.binary(2))
        node = self.binary(level + 1)
        while self.peek()[0] == "op" and self.peek()[1] in _BINARY[level]:
            op = self.take()[1]
            node = ("binary", op, node, self.binary(level + 1))
        return node

    def unary(self):
        kind, value = self.peek()
        if kind == "op" and value in ("-", "+"):
            self.take()
            return ("unary", value, self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            return ("binary", "^", node, self.unary())  # right-associative
        return node

    def atom(self):
        kind, value = self.take()
        if kind == "num":
            return ("num", float(value.rstrip("L")), value)
        if kind == "str":
            return ("str", value[1:-1])
        if kind == "name":
            if self.peek() == ("op", "("):
                self.take("(")
                args = []
                while self.peek() != ("op", ")"):
                    args.append(self.binary(0))
                    if self.peek() == ("op", ","):
                        self.take(",")
                self.take(")")
                return ("call", value, args)
            return ("name", value)
        if value == "(":
            node = self.binary(0)
            self.take(")")
            return ("paren", node)
        raise LmError(f"can't parse {self.text!r} at {value!r}")


def deparse(node) -> str:
    """The expression as R deparses it (R's names of the design's columns)."""
    kind = node[0]
    if kind == "num":
        value = node[1]
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else format(value, ".15g")
    if kind == "str":
        return '"' + node[1] + '"'
    if kind == "name":
        return node[1]
    if kind == "paren":
        return f"({deparse(node[1])})"
    if kind == "call":
        return f"{node[1]}({', '.join(map(deparse, node[2]))})"
    if kind == "unary":
        return node[1] + deparse(node[2])
    op = node[1]
    sep = f" {op} " if op in _SPACED else op
    return deparse(node[2]) + sep + deparse(node[3])


_FUNCTIONS = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
              "as.numeric": lambda x: np.asarray(x, dtype=float),
              "as.integer": lambda x: np.trunc(np.asarray(x, dtype=float)),
              "as.logical": lambda x: np.asarray(x, dtype=bool)}
_CONSTANTS = {"TRUE": True, "FALSE": False, "T": True, "F": False, "pi": math.pi}


def _evaluate(node, data: dict[str, np.ndarray]):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "str":
        return node[1]
    if kind == "name":
        if node[1] in data:
            return data[node[1]]
        if node[1] in _CONSTANTS:
            return _CONSTANTS[node[1]]
        raise LmError(f"no network attribute {node[1]!r} (available: {', '.join(sorted(data))})")
    if kind == "paren":
        return _evaluate(node[1], data)
    if kind == "call":
        name, args = node[1], node[2]
        if name in ("I", "factor") and len(args) == 1:
            return _evaluate(args[0], data)
        if name == "c":
            return np.concatenate([np.atleast_1d(_evaluate(a, data)) for a in args]) if args \
                else np.zeros(0)
        if name in _FUNCTIONS and len(args) == 1:
            return _FUNCTIONS[name](_evaluate(args[0], data))
        raise LmError(f"unsupported function in a linear model: {deparse(node)}")
    if kind == "unary":
        x = _evaluate(node[2], data)
        return np.logical_not(x) if node[1] == "!" else (-np.asarray(x) if node[1] == "-" else x)
    op, a, b = node[1], _evaluate(node[2], data), _evaluate(node[3], data)
    with np.errstate(divide="ignore", invalid="ignore"):
        if op in ("&", "&&"):
            return np.logical_and(a, b)
        if op in ("|", "||"):
            return np.logical_or(a, b)
        a, b = np.asarray(a), np.asarray(b)
        if op == ":":
            if a.ndim or b.ndim:
                raise LmError(f"unsupported in a linear model: {deparse(node)}")
            step = 1 if b >= a else -1
            return np.arange(float(a), float(b) + step / 2, step)
        return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
                "^": np.power, "%%": np.mod, "%/%": np.floor_divide, "==": np.equal,
                "!=": np.not_equal, "<": np.less, ">": np.greater, "<=": np.less_equal,
                ">=": np.greater_equal}[op](a, b)


def _terms(node) -> list[tuple[int, object]]:
    """The terms of a formula's right-hand side, with +1 or -1 (removed)."""
    if node[0] == "binary" and node[1] in ("+", "-"):
        sign = 1 if node[1] == "+" else -1
        return _terms(node[2]) + [(sign * s, t) for s, t in _terms(node[3])]
    if node[0] == "unary" and node[1] == "-":
        return [(-s, t) for s, t in _terms(node[2])]
    return [(1, node)]


def _levels(values: np.ndarray) -> list:
    """A factor's levels, sorted as R sorts them."""
    unique = set(values.tolist())
    return sorted(unique, key=lambda v: (not isinstance(v, (bool, np.bool_)), v))


def _level_label(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _arrays(columns: dict[str, list]) -> dict[str, np.ndarray]:
    """Columns of values as arrays: logical, numeric, or else of objects."""
    data = {}
    for k, values in columns.items():
        v = np.array(values, dtype=object)
        if len(v) and all(isinstance(x, (bool, np.bool_)) for x in v):
            v = v.astype(bool)
        elif len(v) and all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
                            for x in v):
            v = v.astype(float)
        data[k] = v
    return data


def _split_formula(text: str) -> tuple[str | None, str]:
    """The sides of an R formula: (None, rhs) if one-sided."""
    text = text.strip()
    if text.startswith("~"):
        return None, text[1:]
    depth, quote = 0, None
    for k, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"`":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "~" and depth == 0:
            return text[:k], text[k + 1:]
    return None, text


def vertex_sets(attrs, attributes: dict[str, list], n: int):
    """The vertices an S() selector picks, as ergm's: a one-sided formula
    (``~level == "A"``) picks one set; a two-sided one (``(level == "A") ~
    (level == "B")``) two, for a bipartite subgraph. Each side is an R
    expression of the vertex attributes, logical (recycled) or 1-based
    indices (negative ones deselect). Returns the sets (the second None if
    one-sided) and the label ergm gives them in names."""
    data = _arrays(attributes)
    if isinstance(attrs, str):
        lhs, rhs = _split_formula(attrs)
        sides = [rhs] if lhs is None else [lhs, rhs]
    else:
        sides = list(attrs) if isinstance(attrs, (list, tuple)) and len(attrs) == 2 and \
            not np.isscalar(attrs[0]) else [attrs]
    sets, labels = [], []
    for side in sides:
        if isinstance(side, str):
            node = _Parser(side).parse()
            labels.append(re.sub(r"\s", "", deparse(node)))
            value = _evaluate(node, data)
        else:
            value = np.asarray(side)
            labels.append("c(" + ",".join(str(v) for v in value.tolist()) + ")")
        value = np.asarray(value)
        if value.dtype == bool:
            chosen = np.flatnonzero(np.resize(value, n) if value.size else np.zeros(n, bool))
        elif value.dtype.kind in "if":
            idx = value.astype(int)
            if np.all(idx < 0):
                chosen = np.setdiff1d(np.arange(n), -idx - 1)
            elif np.all(idx > 0):
                if idx.max() > n:
                    raise LmError(f"vertex index {idx.max()} is beyond the {n} vertices")
                chosen = np.unique(idx - 1)
            else:
                raise LmError("vertex indices must be all positive or all negative")
        else:
            raise LmError(f"{labels[-1]} is neither logical nor vertex indices")
        sets.append(chosen)
    return sets[0], (sets[1] if len(sets) == 2 else None), ",".join(labels)


def design(lm, attributes: list[dict], contrasts: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """The design matrix (networks x columns) of a one-sided lm formula over
    the networks' attributes, and its column names as N() names them ("1"
    for the intercept)."""
    x, labels, offset = model_frame(lm, attributes, contrasts)
    if offset is not None:
        raise LmError("offset() is only supported in the linear models of N()")
    return x, labels


def _data(attributes: list[dict]) -> dict[str, np.ndarray]:
    names = sorted({k for a in attributes for k in a})
    return _arrays({k: [a.get(k) for a in attributes] for k in names})


def evaluate(expression, attributes: list[dict]) -> np.ndarray:
    """An R expression (``~`` optional) of the networks' attributes, one
    value per network (recycled, if shorter)."""
    text = str(expression).strip()
    text = text[1:] if text.startswith("~") else text
    value = np.asarray(_evaluate(_Parser(text).parse(), _data(attributes)))
    if value.ndim == 0:
        return np.full(len(attributes), value.item())
    if not value.size or len(attributes) % value.size:
        raise LmError(f"{expression!r} has {value.size} values for {len(attributes)} networks")
    return np.resize(value, len(attributes))


def network_subset(subset, attributes: list[dict]) -> np.ndarray:
    """The networks N()'s ``subset`` keeps, as a mask: an R expression of their
    attributes (``~n >= 4``), logical values (recycled) or 1-based indices
    (negative ones exclude)."""
    n = len(attributes)
    if isinstance(subset, str):
        text = subset.strip().lstrip("~")
        value = np.asarray(_evaluate(_Parser(text).parse(), _data(attributes)))
    else:
        value = np.asarray(subset)
    if value.dtype == bool or value.dtype == np.bool_:
        return np.resize(value, n) if value.size else np.zeros(n, dtype=bool)  # recycled, as R
    if value.dtype.kind in "if":
        idx = value.astype(int).ravel()
        if np.all(idx < 0):
            mask = np.ones(n, dtype=bool)
            mask[-idx - 1] = False
            return mask
        if np.all(idx > 0):
            if idx.max() > n:
                raise LmError(f"subset: network {idx.max()} is beyond the {n} networks")
            mask = np.zeros(n, dtype=bool)
            mask[idx - 1] = True
            return mask
        raise LmError("subset: network indices must be all positive or all negative")
    raise LmError(f"subset {subset!r} is neither logical nor network indices")


_CONTRASTS = ("contr.treatment", "contr.SAS", "contr.sum", "contr.helmert", "contr.poly")


def _contrast(kind, labels: list[str], name: str) -> tuple[np.ndarray, list[str]]:
    """R's contrast matrix (levels x columns) of a factor with these level
    labels, and the suffixes of its columns' names: contr.treatment (the
    default), contr.SAS, contr.sum, contr.helmert, contr.poly, or a matrix."""
    k = len(labels)
    if not isinstance(kind, str):
        m = np.asarray(kind, dtype=float)
        if m.ndim != 2 or m.shape[0] != k:
            raise LmError(f"the contrasts of {name} must have one row per level ({k})")
        return m, [str(j + 1) for j in range(m.shape[1])]
    if kind == "contr.treatment":
        return np.eye(k)[:, 1:], labels[1:]
    if kind == "contr.SAS":
        return np.eye(k)[:, :-1], labels[:-1]
    if kind == "contr.sum":
        return np.vstack([np.eye(k - 1), -np.ones(k - 1)]), [str(j + 1) for j in range(k - 1)]
    if kind == "contr.helmert":
        m = np.zeros((k, k - 1))
        for j in range(k - 1):
            m[:j + 1, j], m[j + 1, j] = -1.0, j + 1
        return m, [str(j + 1) for j in range(k - 1)]
    if kind == "contr.poly":
        # Orthonormal polynomials of the scores 1..k, as R's make.poly: Q
        # times the diagonal of R (whatever the signs QR picks), normalized.
        y = np.arange(1, k + 1) - (k + 1) / 2
        q, r = np.linalg.qr(np.vander(y, k, increasing=True))
        raw = q * np.diag(r)
        z = raw / np.sqrt((raw**2).sum(axis=0))
        names = [".L", ".Q", ".C"] + [f"^{j}" for j in range(4, k)]
        return z[:, 1:], names[:k - 1]
    raise LmError(f"unknown contrasts {kind!r} for {name}; use one of {', '.join(_CONTRASTS)} or a matrix")


def model_frame(lm, attributes: list[dict], contrasts: dict | None = None) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """The design matrix and column names of a linear model, and the sum of
    its offset() terms (None without any). Factors (and logical and character
    attributes) get treatment contrasts, or those of ``contrasts``, by term
    (``{"weekday": "contr.sum"}``), as R's ``contrasts.arg``."""
    text = "~1" if lm is None else str(lm).strip()
    text = text[1:] if text.startswith("~") else text
    if not text.strip():
        raise LmError("the linear model is empty")
    data = _data(attributes)
    intercept, terms, offset = True, [], None
    for sign, node in _terms(_Parser(text).parse()):
        if node[0] == "call" and node[1] == "offset":
            if sign < 0 or len(node[2]) != 1:
                raise LmError(f"bad offset term: {deparse(node)}")
            value = np.broadcast_to(np.asarray(_evaluate(node[2][0], data), dtype=float),
                                    (len(attributes),))
            if not np.all(np.isfinite(value)):
                raise LmError(f"{deparse(node)} is not finite for some networks")
            offset = value if offset is None else offset + value
        elif node[0] == "num" and node[1] in (0, 1):
            intercept = (node[1] == 1) == (sign > 0)
        elif node[0] == "binary" and node[1] in ("*", ":"):
            raise LmError(f"interactions are not supported in linear models: {deparse(node)}")
        elif sign < 0:
            raise LmError(f"only the intercept can be removed from a linear model: -{deparse(node)}")
        else:
            terms.append(node)
    n = len(attributes)
    columns, labels = ([np.ones(n)], ["1"]) if intercept else ([], [])
    full_levels = not intercept
    for node in terms:
        value = _evaluate(node, data)
        value = np.broadcast_to(np.asarray(value), (n,)) if np.ndim(value) == 0 else np.asarray(value)
        factor = node[0] == "call" and node[1] == "factor"
        if factor or value.dtype == bool or value.dtype == object:
            if value.dtype == object and any(v is None for v in value):
                raise LmError(f"{deparse(node)} is missing for some networks")
            levels = _levels(value)
            indicators = np.column_stack([(value == level).astype(float) for level in levels])
            name = deparse(node)
            if full_levels:
                matrix, suffixes = np.eye(len(levels)), [_level_label(v) for v in levels]
            else:
                kind = (contrasts or {}).get(name, "contr.treatment")
                matrix, suffixes = _contrast(kind, [_level_label(v) for v in levels], name)
            for column, suffix in zip((indicators @ matrix).T, suffixes):
                columns.append(column)
                labels.append(name + suffix)
            full_levels = False
        else:
            value = value.astype(float)
            if not np.all(np.isfinite(value)):
                raise LmError(f"{deparse(node)} is not finite for some networks")
            columns.append(value)
            labels.append(deparse(node))
    if not columns:
        raise LmError("the linear model has no columns")
    return np.column_stack(columns), labels, offset
