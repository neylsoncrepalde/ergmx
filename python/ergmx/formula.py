"""Parsing of R-style model formulas such as ``"edges + gwesp(0.5, fixed=TRUE)"``."""

from __future__ import annotations

import ast
import re

from .terms import BLOCK_OPERATORS, TERMS, Formula

# R's constants, so formulas can be pasted from R.
_R_CONSTANTS = {"TRUE": True, "FALSE": False, "T": True, "F": False, "NA": None, "Inf": float("inf")}


class FormulaError(ValueError):
    """The formula can't be parsed."""


def parse_formula(formula: str) -> Formula:
    """Parse a formula string into terms.

    The syntax is R's: terms separated by ``+``, with arguments in
    parentheses. A left-hand side (``"net ~ edges + mutual"``) is ignored;
    ``TRUE``/``FALSE``, ``NA``, ``c(2, 3)`` and ``2:3`` are accepted as in R,
    and so are the operators ``offset(term)``, ``F(~terms, ~filter)``,
    ``N(~terms, lm=~attributes)`` and tergm's ``Form()``, ``Persist()``,
    ``Diss()``, ``Cross()`` and ``Change()``. Nothing is evaluated: arguments
    must be literals.
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
    pattern = re.compile(r"(?<![\w.])(" + "|".join(BLOCK_OPERATORS) + r")\s*\(")
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
            is_lm = keyword is not None and keyword.group(1) == "lm"
            if keyword is None:
                positional = sum(1 for a, b in spans[:spans.index((start, stop))]
                                 if not re.match(r"\s*[A-Za-z_.][\w.]*\s*=(?!=)", text[a:b]))
                is_lm = positional == 1
            if is_lm:
                value = argument[keyword.end():] if keyword else argument
                models.append(value.strip())
                text = text[:start] + f"lm={len(models) - 1}" + text[stop:]
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


def _terms(node: ast.expr, formula: str, models: list[str] = ()) -> list:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _terms(node.left, formula, models) + _terms(node.right, formula, models)
    if isinstance(node, ast.Name):
        return [_make(node.id, [], {})]
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
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords}
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
        if key == "lm":
            options["lm"] = models[_literal(value)]
        else:
            options[key] = _literal(value)
    try:
        return TERMS[name](Formula(terms), **options)
    except TypeError as e:
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
    ranges such as 2:4 become lists, and the negation `!` becomes `-`."""
    parts = re.split(r"('[^']*'|\"[^\"]*\")", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"(?<![\w.])(\d+)\s*:\s*(\d+)(?![\w.])",
            lambda m: str(list(range(int(m[1]), int(m[2]) + 1))),
            parts[i],
        )
        parts[i] = re.sub(r"!(?!=)", "-", parts[i])
    return "".join(parts)


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
        return [_literal(a) for a in node.args]  # R's c(...)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "matrix":
        return _r_matrix(node)
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
    try:
        factory = TERMS[name]
    except KeyError:
        raise FormulaError(f"unknown term {name!r}; available terms: {', '.join(TERMS)}") from None
    try:
        return factory(*args, **kwargs)
    except TypeError as e:
        raise FormulaError(f"{name}: {e}") from None
