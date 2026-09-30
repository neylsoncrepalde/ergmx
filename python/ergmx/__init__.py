"""ergmx: exponential-family random graph models (ERGMs) in Python, with a Rust core.

>>> import ergmx
>>> fit = ergmx.ergm(g, "edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)")  # doctest: +SKIP
>>> fit.summary()  # doctest: +SKIP
"""

from ._estimation import Control
from ._fit import ErgmFit, FitSummary
from ._gof import GofResult, GofTable, gof
from ._simulate import ergm, simulate, summary_stats
from .formula import FormulaError, parse_formula
from .terms import (
    Formula,
    Term,
    absdiff,
    ctriple,
    edgecov,
    edges,
    gwdegree,
    gwdsp,
    gwesp,
    gwidegree,
    gwodegree,
    istar,
    kstar,
    mutual,
    nodecov,
    nodefactor,
    nodeicov,
    nodeifactor,
    nodematch,
    nodeocov,
    nodeofactor,
    ostar,
    triangle,
    ttriple,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Control",
    "ErgmFit",
    "FitSummary",
    "Formula",
    "FormulaError",
    "GofResult",
    "GofTable",
    "Term",
    "absdiff",
    "ctriple",
    "edgecov",
    "edges",
    "ergm",
    "gof",
    "gwdegree",
    "gwdsp",
    "gwesp",
    "gwidegree",
    "gwodegree",
    "istar",
    "kstar",
    "mutual",
    "nodecov",
    "nodefactor",
    "nodeicov",
    "nodeifactor",
    "nodematch",
    "nodeocov",
    "nodeofactor",
    "ostar",
    "parse_formula",
    "simulate",
    "summary_stats",
    "triangle",
    "ttriple",
]
