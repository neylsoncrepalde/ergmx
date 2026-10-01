"""Tables of fitted models side by side, as R's texreg: ``screenreg()`` (text),
``texreg()`` (LaTeX) and ``htmlreg()`` (HTML), plus Markdown."""

from __future__ import annotations

import html
import math
import re

import numpy as np

_SYMBOLS = ["***", "**", "*", "."]


def _pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan")


class ResultsTable:
    """Coefficients, standard errors and fit statistics of several models,
    as R's texreg tables. ``print()`` it (R's ``screenreg()``), or convert it
    with :meth:`to_latex` (``texreg()``), :meth:`to_html` (``htmlreg()``,
    also how notebooks show it) or :meth:`to_markdown`."""

    def __init__(self, fits, names=None, digits: int = 2, stars=(0.001, 0.01, 0.05),
                 include_aic: bool = True, include_bic: bool = True, include_loglik: bool = True,
                 rename: dict | None = None, omit: str | None = None):
        from ._fit import ErgmFit

        if len(fits) == 1 and isinstance(fits[0], (list, tuple)):
            fits = tuple(fits[0])
        if not fits or not all(isinstance(f, ErgmFit) for f in fits):
            raise TypeError("table() takes fitted models (ErgmFit), as arguments or a list")
        if len(stars) > len(_SYMBOLS) or list(stars) != sorted(stars):
            raise ValueError(f"stars must be up to {len(_SYMBOLS)} increasing p-value thresholds")
        self.model_names = list(names) if names is not None else [f"Model {k + 1}" for k in range(len(fits))]
        if len(self.model_names) != len(fits):
            raise ValueError(f"names: {len(fits)} models, {len(self.model_names)} names")
        self.digits, self.stars = int(digits), tuple(stars)
        rename = rename or {}
        # Coefficients: the union of the models' names, in order of appearance.
        self.rows: list[str] = []
        self.cells: list[dict[str, tuple]] = []
        for fit in fits:
            se = np.sqrt(np.diag(fit.cov))
            entries = {}
            for k, name in enumerate(fit.names):
                label = rename.get(name, name)
                if omit is not None and re.search(omit, name):
                    continue
                fixed = bool(fit._model.fixed[k])
                p = float("nan") if fixed else _pvalue(fit.params[k] / se[k])
                entries[label] = (float(fit.params[k]), None if fixed else float(se[k]), p)
                if label not in self.rows:
                    self.rows.append(label)
            self.cells.append(entries)
        self.gof_names = [name for name, include in (("AIC", include_aic), ("BIC", include_bic),
                                                     ("Log Likelihood", include_loglik)) if include]
        attribute = {"AIC": "aic", "BIC": "bic", "Log Likelihood": "loglik"}
        self.gof = [[getattr(f, attribute[g]) for g in self.gof_names] for f in fits]
        keep = [k for k, g in enumerate(self.gof_names) if any(row[k] is not None for row in self.gof)]
        self.gof_names = [self.gof_names[k] for k in keep]
        self.gof = [[row[k] for k in keep] for row in self.gof]

    # -- Cells --------------------------------------------------------------------------------

    def _symbols(self) -> list[str]:
        return _SYMBOLS[:len(self.stars)]

    def _stars_of(self, p: float) -> str:
        for cut, symbol in zip(self.stars, self._symbols()):
            if np.isfinite(p) and p < cut:
                return symbol
        return ""

    def _number(self, x: float | None) -> str:
        return "" if x is None or not np.isfinite(x) else f"{x:.{self.digits}f}"

    def _columns(self):
        """Per model: the coefficient, standard error and fit statistic cells,
        as (number, stars) pairs; None where the model has nothing."""
        out = []
        for entries, gof in zip(self.cells, self.gof):
            column = []
            for row in self.rows:
                if row in entries:
                    coef, se, p = entries[row]
                    column.append((self._number(coef), self._stars_of(p)))
                    column.append((f"({self._number(se)})" if se is not None else "", None))
                else:
                    column += [None, None]
            column += [(self._number(g), None) if g is not None else None for g in gof]
            out.append(column)
        return out

    def _row_names(self) -> list[str]:
        return [n for row in self.rows for n in (row, "")] + list(self.gof_names)

    def _legend(self, sep: str, fmt) -> str:
        return sep.join(fmt(symbol, cut) for cut, symbol in zip(self.stars, self._symbols()))

    # -- Formats ------------------------------------------------------------------------------

    def to_text(self) -> str:
        """The table as R's ``screenreg()`` prints it."""
        star_width = max(map(len, self._symbols()), default=0)
        columns = []
        for name, column in zip(self.model_names, self._columns()):
            # Coefficients are followed by their stars, padded to the widest.
            texts = []
            for c in column:
                if c is None:
                    texts.append(None)
                elif c[1] is not None:
                    texts.append(f"{c[0]} {c[1].ljust(star_width)}" if star_width else c[0])
                else:
                    texts.append(c[0])
            # Align on the decimal point, as texreg does.
            parts = [None if t is None or t == "" else t.split(".", 1) if "." in t else [t, None] for t in texts]
            left = max((len(p[0]) for p in parts if p is not None), default=0)
            right = max((len(p[1]) for p in parts if p is not None and p[1] is not None), default=0)
            dot = any(p is not None and p[1] is not None for p in parts)
            width = left + (1 + right if dot else 0)
            cells = []
            for p in parts:
                if p is None:
                    cells.append(" " * width)
                elif p[1] is None:
                    cells.append(p[0].rjust(left) + " " * (width - left))
                else:
                    cells.append(p[0].rjust(left) + "." + p[1].ljust(right))
            width = max(width, len(name))
            columns.append((name.ljust(width), [c.ljust(width) for c in cells]))
        names = self._row_names()
        name_width = max(map(len, names), default=0)
        total = name_width + sum(2 + len(h) for h, _ in columns)
        n_coef = 2 * len(self.rows)
        lines = ["=" * total, " " * name_width + "".join("  " + h for h, _ in columns), "-" * total]
        for r, name in enumerate(names):
            if r == n_coef and self.gof_names:
                lines.append("-" * total)
            lines.append(name.ljust(name_width) + "".join("  " + cells[r] for _, cells in columns))
        lines.append("=" * total)
        lines.append(self._legend("; ", lambda s, c: f"{s} p < {c:g}"))
        return "\n".join(lines) + "\n"

    def to_latex(self, caption: str = "Statistical models", label: str = "table:coefficients") -> str:
        """The table as R's ``texreg()`` writes it."""
        def escape(text: str) -> str:
            return re.sub(r"([#$%&_{}])", r"\\\1", text).replace("~", r"\textasciitilde{}") \
                .replace("^", r"\textasciicircum{}")

        columns = []
        for column in self._columns():
            cells = []
            for c in column:
                if c is None or c[0] == "":
                    cells.append("")
                elif c[1]:
                    cells.append(f"${c[0]}^{{{c[1]}}}$")
                else:
                    cells.append(f"${c[0]}$")
            width = max(map(len, cells), default=0)
            columns.append([c.ljust(width) for c in cells])
        names = [escape(n) for n in self._row_names()]
        name_width = max(map(len, names), default=0)
        n_coef = 2 * len(self.rows)
        lines = [r"\begin{table}", r"\begin{center}", r"\begin{tabular}{l " + " ".join("c" * len(columns)) + "}",
                 r"\hline", " & " + " & ".join(escape(n) for n in self.model_names) + r" \\", r"\hline"]
        for r, name in enumerate(names):
            if r == n_coef and self.gof_names:
                lines.append(r"\hline")
            lines.append(name.ljust(name_width) + " & " + " & ".join(c[r] for c in columns) + r" \\")
        legend = self._legend("; ", lambda s, c: f"$^{{{s}}}p<{c:g}$")
        lines += [r"\hline", rf"\multicolumn{{{len(columns) + 1}}}{{l}}{{\scriptsize{{{legend}}}}}",
                  r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{center}",
                  r"\end{table}"]
        return "\n".join(lines) + "\n"

    def to_html(self, caption: str = "Statistical models") -> str:
        """The table as R's ``htmlreg()`` writes it."""
        pad = 'style="padding-left: 5px;padding-right: 5px;"'

        def stars(s: str) -> str:
            return f'<sup class="texreg-stars">{"&#42;" * len(s) if set(s) == {"*"} else html.escape(s)}</sup>'

        def cell(c) -> str:
            if c is None or c[0] == "":
                return "&nbsp;"
            return c[0] + (stars(c[1]) if c[1] else "")

        columns = self._columns()
        names = self._row_names()
        n_coef = 2 * len(self.rows)
        lines = ['<table class="texreg" style="margin: 10px auto;border-collapse: collapse;border-spacing: '
                 '0px;caption-side: bottom;color: #000000;border-top: 2px solid #000000;">',
                 f"<caption>{html.escape(caption)}</caption>", "<thead>", "<tr>", f"<th {pad}>&nbsp;</th>"]
        lines += [f"<th {pad}>{html.escape(n)}</th>" for n in self.model_names]
        lines += ["</tr>", "</thead>", "<tbody>"]
        for r, name in enumerate(names):
            if r == 0 or (r == n_coef and self.gof_names):
                style = ' style="border-top: 1px solid #000000;"'
            elif r == len(names) - 1:
                style = ' style="border-bottom: 2px solid #000000;"'
            else:
                style = ""
            lines.append(f"<tr{style}>")
            lines.append(f"<td {pad}>{html.escape(name) if name else '&nbsp;'}</td>")
            lines += [f"<td {pad}>{cell(column[r])}</td>" for column in columns]
            lines.append("</tr>")
        legend = self._legend("; ", lambda s, c: f"{stars(s)}p &lt; {c:g}")
        lines += ["</tbody>", "<tfoot>", "<tr>",
                  f'<td style="font-size: 0.8em;" colspan="{len(columns) + 1}">{legend}</td>', "</tr>",
                  "</tfoot>", "</table>"]
        return "\n".join(lines) + "\n"

    def to_markdown(self) -> str:
        """The table in Markdown (a pipe table), with the legend below it."""
        def escape(text: str) -> str:
            return text.replace("|", r"\|").replace("*", r"\*").replace("~", r"\~")

        columns = self._columns()
        names = self._row_names()
        header = "| | " + " | ".join(escape(n) for n in self.model_names) + " |"
        lines = [header, "|---|" + "---:|" * len(columns)]
        for r, name in enumerate(names):
            cells = ["" if c[r] is None else escape(c[r][0] + (c[r][1] or "")) for c in columns]
            lines.append(f"| {escape(name)} | " + " | ".join(cells) + " |")
        legend = self._legend("; ", lambda s, c: f"{escape(s)} p < {c:g}")
        return "\n".join(lines) + "\n\n" + legend + "\n"

    def _repr_html_(self) -> str:
        return self.to_html()

    def __str__(self) -> str:
        return self.to_text()

    __repr__ = __str__


def table(*fits, names=None, digits: int = 2, stars=(0.001, 0.01, 0.05), include_aic: bool = True,
          include_bic: bool = True, include_loglik: bool = True, rename: dict | None = None,
          omit: str | None = None) -> ResultsTable:
    """A table of fitted models side by side, as R's texreg.

    ``print()`` it for R's ``screenreg()`` layout, or convert it with
    ``.to_latex()``, ``.to_html()`` (as notebooks show it) or ``.to_markdown()``.

    Parameters
    ----------
    *fits : ErgmFit
        The models, as arguments or one list.
    names : list of str, optional
        The models' column names ("Model 1", "Model 2"... by default).
    digits : int
        Decimals of the numbers.
    stars : tuple of float
        p-value thresholds of ``***``, ``**``, ``*`` (and ``.``).
    include_aic, include_bic, include_loglik : bool
        Rows of fit statistics (for models that have them).
    rename : dict, optional
        New names for coefficients, ``{"nodematch.Grade": "Same grade"}``;
        coefficients renamed to the same name share a row.
    omit : str, optional
        A regular expression: coefficients whose names match are left out.
    """
    return ResultsTable(fits, names, digits, stars, include_aic, include_bic, include_loglik, rename, omit)
