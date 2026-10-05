"""Parsing of R-style model formulas such as ``"edges + gwesp(0.5, fixed=TRUE)"``."""

from __future__ import annotations

import ast
import contextlib
import re

from .terms import BLOCK_OPERATORS, TERMS, Formula, Interaction

# R's constants, so formulas can be pasted from R.
_R_CONSTANTS = {"TRUE": True, "FALSE": False, "T": True, "F": False, "NA": None, "Inf": float("inf")}


class FormulaError(ValueError):
    """The formula can't be parsed."""


#: Multilayer terms' arguments of Layer Logic, kept as R text (as N()'s lm).
_LAYER_SP = ("despL", "espL", "ddspL", "dspL", "dnspL", "nspL", "dgwespL", "gwespL", "dgwdspL", "gwdspL",
             "dgwnspL", "gwnspL")
_LAYER_ARGUMENTS = {"L": ("Ls",), "CMBL": ("Ls",), "twostarL": ("Ls",), "mutualL": ("Ls",),
                    **dict.fromkeys(_LAYER_SP, ("Ls.path", "L.base"))}
#: The position of Ls, given without its name (the shared partner terms' are named).
_LAYER_POSITIONS = {"L": 1, "CMBL": 0, "twostarL": 0, "mutualL": 4, **dict.fromkeys(_LAYER_SP, -1)}

#: The terms formulas are parsed with (another registry, for valued networks).
_REGISTRY = [TERMS]


@contextlib.contextmanager
def use_terms(registry: dict):
    """Parse formulas with these terms (the valued networks' registry)."""
    _REGISTRY.append(registry)
    try:
        yield
    finally:
        _REGISTRY.pop()


def parse_formula(formula: str) -> Formula:
    """Parse a formula string into terms.

    The syntax is R's: terms separated by ``+``, with arguments in
    parentheses. A left-hand side (``"net ~ edges + mutual"``) is ignored;
    ``TRUE``/``FALSE``, ``NA``, ``c(2, 3)`` and ``2:3`` are accepted as in R,
    and so are interactions of dyad-independent terms (``a:b``, and ``a*b``
    for ``a + b + a:b``), R's argument names with dots (``sign.action=``), and
    the operators ``offset(term)``, ``F(~terms, ~filter)``, ``N(~terms,
    lm=~attributes)``, ``S(~terms, ~attributes)`` and tergm's ``Form()``,
    ``Persist()``, ``Diss()``, ``Cross()`` and ``Change()``. Nothing is
    evaluated: arguments must be literals.
    """
    text, models = _protect_linear_models(_strip_lhs(formula))
    rhs = _r_syntax(text)
    try:
        tree = ast.parse(rhs, mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"can't parse the formula {formula!r}: {e.msg}") from None
    return Formula(_terms(tree, formula, models))


def _arguments(text: str, open_at: int) -> tuple[list[tuple[int, int]], int]:
    """The spans of the top-level arguments of the call whose "(" is at
    `open_at`, and the position of its ")"."""
    depth, quote, spans, start = 0, None, [], open_at + 1
    for k in range(open_at, len(text)):
        ch = text[k]
        if quote:
            quote = None if ch == quote and text[k - 1] != "\\" else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                spans.append((start, k))
                return spans, k
        elif ch == "," and depth == 1:
            spans.append((start, k))
            start = k + 1
    raise FormulaError(f"unbalanced parentheses in {text!r}")


def _protect_linear_models(text: str) -> tuple[str, list[str]]:
    """The formula with the linear models of the block operators (their `lm`
    argument, or second positional one) replaced by placeholders, since they
    use R syntax that Python doesn't parse (such as `.NetworkID`), and the
    linear models' R text."""
    models: list[str] = []
    # The operators' arguments in R syntax: the second (or this keyword).
    keywords = {**dict.fromkeys(BLOCK_OPERATORS, ("lm", "subset", "offset", "weights")), "S": ("attrs",),
                "mm": ("attrs",), **_LAYER_ARGUMENTS}
    # The position of the first when it is given without its name.
    positions = {"mm": 0, **_LAYER_POSITIONS}
    pattern = re.compile(r"(?<![\w.])(" + "|".join(keywords) + r")\s*\(")
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        if text[:m.start()].count('"') % 2 or text[:m.start()].count("'") % 2:
            pos = m.end()  # inside a string
            continue
        spans, close = _arguments(text, m.end() - 1)
        positional = 0
        for start, stop in reversed(spans):
            argument = text[start:stop]
            keyword = re.match(r"\s*([A-Za-z_.][\w.]*)\s*=(?!=)", argument)
            name = keyword.group(1) if keyword is not None and keyword.group(1) in keywords[m.group(1)] \
                else None
            if keyword is None:
                positional = sum(1 for a, b in spans[:spans.index((start, stop))]
                                 if not re.match(r"\s*[A-Za-z_.][\w.]*\s*=(?!=)", text[a:b]))
                if positional == positions.get(m.group(1), 1):
                    name = keywords[m.group(1)][0]
            if name is not None:
                value = argument[keyword.end():] if keyword else argument
                models.append(value.strip())
                # The layer terms' stays positional (others may follow it), as its index.
                index = f"{len(models) - 1}" if keyword is None and m.group(1) in _LAYER_ARGUMENTS else \
                    f"{name}={len(models) - 1}"
                text = text[:start] + index + text[stop:]
        pos = m.end()
    return text, models


def _strip_lhs(formula: str) -> str:
    """The right-hand side of `formula`: what follows a top-level `~`, if any."""
    text = formula.strip()
    if text.startswith("~"):
        return text[1:].strip()
    depth, quote = 0, None
    for k, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "~" and depth == 0:
            return text[k + 1:].strip()
    return text


def _dotted(node: ast.expr) -> str | None:
    """An R name with dots (mean.age), which Python parses as attributes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _terms(node: ast.expr, formula: str, models: list[str] = ()) -> list:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _terms(node.left, formula, models) + _terms(node.right, formula, models)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.Mult)):
        # R's interactions: a:b (rewritten as a ** b) and a*b, for a + b + a:b.
        left, right = _terms(node.left, formula, models), _terms(node.right, formula, models)
        try:
            interaction = Interaction(left, right)
        except ValueError as e:
            raise FormulaError(f"{e}, in {formula!r}") from None
        return [interaction] if isinstance(node.op, ast.Pow) else [*left, *right, interaction]
    if isinstance(node, ast.Name):
        return [_make(node.id, [], {})]
    if isinstance(node, ast.Attribute) and _dotted(node):
        return [_make(_dotted(node), [], {})]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _dotted(node.func):
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords}
        return [_make(_dotted(node.func), args, kwargs)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "offset":
            if len(node.args) != 1 or node.keywords:
                raise FormulaError(f"offset() takes one term, in {formula!r}")
            inner = _terms(node.args[0], formula, models)
            if len(inner) != 1:
                raise FormulaError(f"offset() takes one term, in {formula!r}")
            return [_make("offset", inner, {})]
        if name == "F":
            return [_filter_term(node, formula, models)]
        if name in BLOCK_OPERATORS:
            return [_block_term(node, formula, models)]
        if name == "S":
            return [_subgraph_term(node, formula, models)]
        if name in _OPERATOR_NAMES or name in ("I", "For"):
            return _operator_terms(name, node, formula, models)
        if name == "EdgeAges":
            if len(node.args) != 1 or node.keywords:
                raise FormulaError(f"EdgeAges() takes one formula, in {formula!r}")
            inner = Formula(_terms(_one_sided(node.args[0], formula), formula, models))
            return [_make(name, [inner], {})]
        if name == "mm":
            kwargs = {k.arg: _literal(k.value) for k in node.keywords}
            if "attrs" not in kwargs or node.args:
                raise FormulaError(f"mm() takes the attributes' formula first, in {formula!r}")
            kwargs["attrs"] = models[kwargs["attrs"]]
            return [_make(name, [], kwargs)]
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords}
        if name in _LAYER_ARGUMENTS:  # the Layer Logic's R text
            for key in [k.replace(".", "_") for k in _LAYER_ARGUMENTS[name]]:
                if key in kwargs:
                    kwargs[key] = models[kwargs[key]]
            if 0 <= _LAYER_POSITIONS[name] < len(args):
                args[_LAYER_POSITIONS[name]] = models[args[_LAYER_POSITIONS[name]]]
        return [_make(name, args, kwargs)]
    raise FormulaError(f"can't parse {ast.unparse(node)!r} in the formula {formula!r}")


def _one_sided(node: ast.expr, formula: str) -> ast.expr:
    """The expression of an R one-sided formula, `~expr` (or a bare `expr`).

    Python binds `~` tighter than `+`, so `~a + b` parses as `(~a) + b`: the
    `~` is on the leftmost term."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return node.operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return ast.BinOp(left=_one_sided(node.left, formula), op=node.op, right=node.right)
    return node


def _block_term(node: ast.Call, formula: str, models: list[str]):
    name = node.func.id
    args = list(node.args)
    kwargs = {k.arg: k.value for k in node.keywords}
    if "formula" in kwargs:
        args.insert(0, kwargs.pop("formula"))
    if len(args) != 1:
        raise FormulaError(f"{name}() takes a formula and keyword arguments, in {formula!r}")
    terms = _terms(_one_sided(args[0], formula), formula, models)
    options = {}
    for key, value in kwargs.items():
        if key in ("lm", "subset", "offset", "weights"):
            options[key] = models[_literal(value)]
        else:
            options[key] = _literal(value)
    try:
        return TERMS[name](Formula(terms), **options)
    except TypeError as e:
        raise FormulaError(f"{name}: {e}") from None


def _subgraph_term(node: ast.Call, formula: str, models: list[str]):
    args = list(node.args)
    kwargs = {k.arg: k.value for k in node.keywords}
    if "formula" in kwargs:
        args.insert(0, kwargs.pop("formula"))
    if len(args) != 1 or set(kwargs) != {"attrs"}:
        raise FormulaError(f"S() takes a formula and the vertices' attributes, in {formula!r}")
    terms = _terms(_one_sided(args[0], formula), formula, models)
    try:
        return TERMS["S"](Formula(terms), models[_literal(kwargs["attrs"])])
    except (TypeError, ValueError) as e:
        raise FormulaError(f"S: {e}") from None


_OPERATOR_NAMES = ("Sum", "Prod", "Log", "Exp", "Symmetrize", "Label", "Passthrough", "Offset", "Curve",
                   "Parametrise", "Parametrize", "L")


def _formula_terms(node: ast.expr, formula: str, models) -> list:
    """The terms of a formula argument: one-sided (~terms), or a string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return list(parse_formula(node.value))
    if isinstance(node, (ast.List, ast.Tuple)) or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                                    and node.func.id in ("c", "list")):
        elements = node.elts if isinstance(node, (ast.List, ast.Tuple)) else node.args
        return [t for e in elements for t in _formula_terms(e, formula, models)]
    return _terms(_one_sided(node, formula), formula, models)


def _weighted_formulas(node: ast.expr, formula: str, models) -> list:
    """Sum()'s and Prod()'s formulas, each with its weights (R's left side) or None."""
    elements = node.args if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
        and node.func.id in ("list", "c") else [node]
    out = []
    for e in elements:
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.BitXor) \
                and isinstance(e.right, ast.UnaryOp) and isinstance(e.right.op, ast.Invert):
            out.append((_literal(e.left), Formula(_terms(e.right.operand, formula, models))))
        else:
            out.append((None, Formula(_formula_terms(e, formula, models))))
    return out


class _Substitute(ast.NodeTransformer):
    """For()'s placeholder, replaced by a value."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def visit_Name(self, node):
        return ast.Constant(self.value) if node.id == self.name else node


def _operator_terms(name: str, node: ast.Call, formula: str, models) -> list:
    """ergm's operators with formula arguments: I() and For() splice terms
    into the formula, the others are terms of a formula."""
    from ._operators import OPERATORS, AsIs

    args, kwargs = list(node.args), {k.arg: k.value for k in node.keywords}
    if name == "For":
        loops = [(k, _literal(v)) for k, v in kwargs.items()]
        if len(args) != 1 or not loops:
            raise FormulaError(f"For() takes a formula and one or more var = values, in {formula!r}")
        bodies = [args[0]]
        for var, values in loops:
            values = values if isinstance(values, list) else [values]
            bodies = [_Substitute(var, v).visit(ast.parse(ast.unparse(b), mode="eval").body)
                      for b in bodies for v in values]
        return [t for b in bodies for t in _formula_terms(b, formula, models)]
    if "formula" in kwargs:
        args.insert(0, kwargs.pop("formula"))
    if "formulas" in kwargs:
        args.insert(0, kwargs.pop("formulas"))
    if not args:
        raise FormulaError(f"{name}() takes a formula, in {formula!r}")
    if name == "I":
        if len(args) != 1 or kwargs:
            raise FormulaError(f"I() takes a formula, in {formula!r}")
        return _formula_terms(args[0], formula, models)

    def label(value):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "I":
            inner = _literal(value.args[0])
            return [AsIs(x) for x in inner] if isinstance(inner, list) else AsIs(inner)
        return _literal(value)

    rest = [label(a) for a in args[1:]]
    options = {k: label(v) for k, v in kwargs.items()}
    if name == "L":  # Ls, the Layer Logic's R text
        from ._layers import L

        layers = options.pop("Ls", rest[0] if rest else "~.")
        return [L(Formula(_formula_terms(args[0], formula, models)), models[layers] if isinstance(layers, int)
                  else layers)]
    if name in ("Sum", "Prod"):
        first = _weighted_formulas(args[0], formula, models)
    else:
        first = Formula(_formula_terms(args[0], formula, models))
    try:
        return [OPERATORS[name](first, *rest, **options)]
    except (TypeError, ValueError) as e:
        raise FormulaError(f"{name}: {e}") from None


def _filter_term(node: ast.Call, formula: str, models: list[str] = ()):
    args = list(node.args) + [k.value for k in node.keywords if k.arg in ("formula", "filter")]
    if len(args) != 2:
        raise FormulaError(f"F() takes a formula and a filter, in {formula!r}")
    terms = _terms(_one_sided(args[0], formula), formula, models)
    filter_node, negate = _one_sided(args[1], formula), False
    if isinstance(filter_node, ast.UnaryOp) and isinstance(filter_node.op, ast.USub):
        filter_node, negate = filter_node.operand, True  # R's `!`, rewritten as `-`
    filters = _terms(filter_node, formula, models)
    if len(filters) != 1:
        raise FormulaError(f"F(): the filter must be a single term, in {formula!r}")
    try:
        return TERMS["F"](Formula(terms), filters[0], negate)
    except (TypeError, ValueError) as e:
        raise FormulaError(str(e)) from None


def _r_syntax(text: str) -> str:
    """Rewrite R syntax that Python can't parse, outside quoted strings: integer
    ranges such as 2:4 become lists, other `:` (interactions) `**`, the
    negation `!` becomes `-`, the names degree1.5, idegree1.5 and odegree1.5
    lose their dot, and so do argument names (sign.action=), and lambda= is
    lambda_=."""
    parts = re.split(r"('[^']*'|\"[^\"]*\")", text)
    for i in range(0, len(parts), 2):
        # Backquoted names (list(`factor(n)` = ...)) become identifiers that _literal decodes.
        parts[i] = re.sub(r"`([^`]*)`", lambda m: _QUOTED + m[1].encode().hex(), parts[i])
        parts[i] = re.sub(
            r"(?<![\w.])(\d+)\s*:\s*(\d+)(?![\w.])",
            lambda m: str(list(range(int(m[1]), int(m[2]) + 1))),
            parts[i],
        )
        parts[i] = parts[i].replace(":", "**")
        parts[i] = re.sub(r"!(?!=)", "-", parts[i])
        parts[i] = re.sub(r"(?<![\w.])([io]?degree)1\.5(?![\w.])", r"\g<1>1_5", parts[i])
        parts[i] = re.sub(r"(?<![\w.])([A-Za-z]\w*)\.([A-Za-z]\w*)(?=\s*=(?!=))", r"\1_\2", parts[i])
        parts[i] = re.sub(r"(?<![\w.])lambda(?=\s*=(?!=))", "lambda_", parts[i])
        # Two-sided formulas (Sum(list(2 ~ edges))): the left side, an operand,
        # then "^", which Python parses, and the one-sided right side.
        parts[i] = re.sub(r"(?<=[\w)\]])(\s*)~", r"\1^ ~", parts[i])
        if i > 0:  # after a string, as 'sum' ~ degree(1:3)
            parts[i] = re.sub(r"^(\s*)~", r"\1^ ~", parts[i])
    return "".join(parts)


#: The prefix of backquoted R names, rewritten as identifiers.
_QUOTED = "_backquoted_"


def _unquoted(name: str) -> str:
    return bytes.fromhex(name[len(_QUOTED):]).decode() if name.startswith(_QUOTED) else name


def _literal(node: ast.expr):
    if isinstance(node, ast.Name) and node.id in _R_CONSTANTS:
        return _R_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        if number(value):
            return -value
        if isinstance(value, list) and all(map(number, value)):
            return [-v for v in value]  # R's -c(1, 3) and -(1:3)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
        if node.keywords and not node.args:  # R's named vector, c(a = 0, b = 1)
            return {_unquoted(k.arg): _literal(k.value) for k in node.keywords}
        return [_literal(a) for a in node.args]  # R's c(...)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "matrix":
        return _r_matrix(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list" and not node.args:
        # R's list(a = ...), as N()'s contrasts.
        return {_unquoted(k.arg): _literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "contr":
        return f"contr.{node.attr}"  # R's contrast functions, contr.sum...
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise FormulaError(f"term arguments must be literals, not {ast.unparse(node)!r}") from None


def _r_matrix(node: ast.Call) -> list:
    """R's matrix(values, nrow, ncol, byrow=FALSE), as a list of rows."""
    args = [_literal(a) for a in node.args]
    kwargs = {k.arg: _literal(k.value) for k in node.keywords}
    names = ["data", "nrow", "ncol", "byrow"]
    values = dict(zip(names, args)) | kwargs
    data = values.get("data")
    data = data if isinstance(data, list) else [data]
    nrow, ncol = values.get("nrow"), values.get("ncol")
    if nrow is None and ncol is None:
        nrow, ncol = len(data), 1
    nrow = nrow or -(-len(data) // ncol)
    ncol = ncol or -(-len(data) // nrow)
    data = [data[k % len(data)] for k in range(nrow * ncol)]  # R recycles values
    if values.get("byrow"):
        return [data[r * ncol:(r + 1) * ncol] for r in range(nrow)]
    return [[data[c * nrow + r] for c in range(ncol)] for r in range(nrow)]


def _make(name: str, args: list, kwargs: dict):
    registry = _REGISTRY[-1]
    if name == "offset" and name not in registry:
        registry = TERMS
    try:
        factory = registry[name]
    except KeyError:
        if name in ("memory", "delrecip", "timecov"):
            raise FormulaError(f"{name}() is a term of btergm(), of the networks before or of time") from None
        kind = "valued " if registry is not TERMS else ""
        raise FormulaError(f"unknown {kind}term {name!r}; available terms: {', '.join(registry)}") from None
    try:
        return factory(*args, **kwargs)
    except TypeError as e:
        raise FormulaError(f"{name}: {e}") from None
