"""Parsing of R-style model formulas such as ``"edges + gwesp(0.5, fixed=TRUE)"``."""

from __future__ import annotations

import ast
import re

from .terms import TERMS, Formula

# R's logical constants, so formulas can be pasted from R.
_R_CONSTANTS = {"TRUE": True, "FALSE": False, "T": True, "F": False}


class FormulaError(ValueError):
    """The formula can't be parsed."""


def parse_formula(formula: str) -> Formula:
    """Parse a formula string into terms.

    The syntax is R's: terms separated by ``+``, with arguments in
    parentheses. A left-hand side (``"net ~ edges + mutual"``) is ignored;
    ``TRUE``/``FALSE``, ``c(2, 3)`` and ``2:3`` are accepted as in R. Nothing
    is evaluated: arguments must be literals.
    """
    rhs = _integer_ranges(formula.split("~", 1)[-1].strip())
    try:
        tree = ast.parse(rhs, mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"can't parse the formula {formula!r}: {e.msg}") from None
    return Formula(_terms(tree, formula))


def _terms(node: ast.expr, formula: str) -> list:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _terms(node.left, formula) + _terms(node.right, formula)
    if isinstance(node, ast.Name):
        return [_make(node.id, [], {})]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords}
        return [_make(node.func.id, args, kwargs)]
    raise FormulaError(f"can't parse {ast.unparse(node)!r} in the formula {formula!r}")


def _integer_ranges(text: str) -> str:
    """Rewrite R's integer ranges such as 2:4 as lists, outside quoted strings."""
    parts = re.split(r"('[^']*'|\"[^\"]*\")", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"(?<![\w.])(\d+)\s*:\s*(\d+)(?![\w.])",
            lambda m: str(list(range(int(m[1]), int(m[2]) + 1))),
            parts[i],
        )
    return "".join(parts)


def _literal(node: ast.expr):
    if isinstance(node, ast.Name) and node.id in _R_CONSTANTS:
        return _R_CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "c":
        return [_literal(a) for a in node.args]  # R's c(...)
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise FormulaError(f"term arguments must be literals, not {ast.unparse(node)!r}") from None


def _make(name: str, args: list, kwargs: dict):
    try:
        factory = TERMS[name]
    except KeyError:
        raise FormulaError(f"unknown term {name!r}; available terms: {', '.join(TERMS)}") from None
    try:
        return factory(*args, **kwargs)
    except TypeError as e:
        raise FormulaError(f"{name}: {e}") from None
