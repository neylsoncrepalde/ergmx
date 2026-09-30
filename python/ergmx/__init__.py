"""ergmx: exponential-family random graph models (ERGMs) in Python, with a Rust core.

>>> import ergmx
>>> fit = ergmx.ergm(g, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)")  # doctest: +SKIP
>>> fit.summary()  # doctest: +SKIP
"""

from ._estimation import Control
from ._fit import ErgmFit, FitSummary
from ._simulate import ergm, simulate, summary_stats
from .formula import FormulaError, parse_formula
from .terms import (
    Formula,
    Term,
    absdiff,
    edges,
    gwesp,
    mutual,
    nodecov,
    nodefactor,
    nodematch,
    triangle,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Control",
    "ErgmFit",
    "FitSummary",
    "Formula",
    "FormulaError",
    "Term",
    "absdiff",
    "edges",
    "ergm",
    "gwesp",
    "mutual",
    "nodecov",
    "nodefactor",
    "nodematch",
    "parse_formula",
    "simulate",
    "summary_stats",
    "triangle",
]
