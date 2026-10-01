"""MCMC diagnostics of a Monte Carlo MLE fit, like R's ``mcmc.diagnostics()``."""

from __future__ import annotations

import math

import numpy as np

from ._estimation import autocorrelation_time, hotelling_pvalue, split_rhat


def _pvalue(z):
    return np.vectorize(lambda v: math.erfc(abs(v) / math.sqrt(2)))(z)


class McmcDiagnostics:
    """Diagnostics of the MCMC sample of the last Monte Carlo MLE iteration.

    The sample is of the model statistics, as deviations from the observed
    statistics: at the MLE, their mean should be zero. As in ergm, it was drawn
    at the coefficients before the final update, which corrects for the mean
    deviations. Print it for the tables, or call :meth:`plot` for trace and
    density plots.
    """

    def __init__(self, names: list[str], sample: np.ndarray, observed: np.ndarray,
                 interval: int | None = None):
        #: Names of the statistics.
        self.names = list(names)
        #: Sampled statistics minus the observed ones (chains x samples x statistics).
        self.deviations = np.asarray(sample, dtype=float) - np.asarray(observed, dtype=float)
        #: MCMC proposals between two samples.
        self.interval = interval

    @property
    def n_chains(self) -> int:
        return self.deviations.shape[0]

    @property
    def n_samples(self) -> int:
        """Samples per chain."""
        return self.deviations.shape[1]

    def _flat(self) -> np.ndarray:
        return self.deviations.reshape(-1, self.deviations.shape[-1])

    @property
    def mean(self) -> np.ndarray:
        return self._flat().mean(axis=0)

    @property
    def sd(self) -> np.ndarray:
        return self._flat().std(axis=0, ddof=1)

    @property
    def autocorrelation_time(self) -> np.ndarray:
        """Integrated autocorrelation time, in samples (1 for independent samples)."""
        return autocorrelation_time(self.deviations)

    @property
    def effective_size(self) -> np.ndarray:
        return self._flat().shape[0] / self.autocorrelation_time

    @property
    def naive_se(self) -> np.ndarray:
        """Standard error of the mean if the samples were independent."""
        return self.sd / np.sqrt(self._flat().shape[0])

    @property
    def timeseries_se(self) -> np.ndarray:
        """Standard error of the mean, accounting for autocorrelation."""
        return self.sd / np.sqrt(self.effective_size)

    @property
    def rhat(self) -> np.ndarray:
        """Split R-hat (`Gelman et al. 2013 <https://doi.org/10.1201/b16018>`__): about 1 when the chains agree; above
        1.01 to 1.1 suggests they have not mixed."""
        return split_rhat(self.deviations)

    @property
    def geweke(self) -> np.ndarray:
        """Geweke z-scores (chains x statistics): the mean of the first 10% of
        each chain against the last 50%, as in R's coda."""
        n = self.n_samples
        first, last = self.deviations[:, : max(2, n // 10)], self.deviations[:, n - n // 2:]
        z = np.empty((self.n_chains, len(self.names)))
        for c in range(self.n_chains):
            a, b = first[c : c + 1], last[c : c + 1]
            var_a = a[0].var(axis=0, ddof=1) * autocorrelation_time(a) / a.shape[1]
            var_b = b[0].var(axis=0, ddof=1) * autocorrelation_time(b) / b.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                z[c] = (a[0].mean(axis=0) - b[0].mean(axis=0)) / np.sqrt(var_a + var_b)
        return np.nan_to_num(z)

    @property
    def pvalue(self) -> float:
        """Hotelling's T^2 test that the mean deviations are zero, with the
        effective sample size."""
        return hotelling_pvalue(self.deviations, np.zeros(len(self.names)),
                                self.autocorrelation_time)

    def __str__(self) -> str:
        width = max(map(len, self.names))
        spacing = f", {self.interval} proposals apart" if self.interval else ""
        lines = [
            f"MCMC diagnostics of the last iteration: {self.n_chains} chains x "
            f"{self.n_samples} samples{spacing}",
            "",
            "Sample statistics, as deviations from the observed statistics:",
            "",
            f"{'':<{width}}  {'Mean':>9}  {'SD':>9}  {'Naive SE':>9}  {'Time-series SE':>14}  "
            f"{'Eff. size':>9}  {'R-hat':>6}",
        ]
        for i, name in enumerate(self.names):
            lines.append(
                f"{name:<{width}}  {self.mean[i]:9.3f}  {self.sd[i]:9.3f}  {self.naive_se[i]:9.4f}  "
                f"{self.timeseries_se[i]:14.4f}  {self.effective_size[i]:9.0f}  {self.rhat[i]:6.3f}"
            )
        lines += [
            "",
            "Are the sample statistics significantly different from the observed?",
            f"Hotelling's T^2 test p-value: {self.pvalue:.4f}. The largest mean deviation is "
            f"{np.max(np.abs(self.mean) / np.where(self.sd > 0, self.sd, 1)):.3f} SD; with large "
            "effective sizes, even negligible deviations are significant.",
            "",
            "Geweke z-scores (first 10% against last 50% of each chain):",
            "",
            f"{'':<{width}}  " + "  ".join(f"{'chain ' + str(c + 1):>8}" for c in range(self.n_chains)),
        ]
        z = self.geweke
        for i, name in enumerate(self.names):
            lines.append(f"{name:<{width}}  " + "  ".join(f"{z[c, i]:8.2f}" for c in range(self.n_chains)))
        flagged = int(np.sum(_pvalue(z) < 0.05))
        lines += ["", f"{flagged} of {z.size} Geweke z-scores have p < 0.05 "
                      f"(about {0.05 * z.size:.1f} expected by chance)."]
        worst = np.nanmax(self.rhat) if np.any(np.isfinite(self.rhat)) else np.nan
        if worst > 1.1:
            lines.append(f"R-hat up to {worst:.2f}: the chains disagree, so the MCMC has not mixed.")
        return "\n".join(lines)

    __repr__ = __str__

    def plot(self):
        """Trace and density of each statistic, one line per chain, like R's
        ``mcmc.diagnostics()`` plots. Needs matplotlib. Returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError('plotting needs matplotlib: install "ergmx[plot]"') from None
        p = len(self.names)
        fig, axes = plt.subplots(p, 2, figsize=(10, 2.2 * p), squeeze=False,
                                 gridspec_kw={"width_ratios": [3, 1]})
        for i, name in enumerate(self.names):
            trace, density = axes[i]
            values = self.deviations[:, :, i]
            bins = np.histogram_bin_edges(values, bins=30)
            for c in range(self.n_chains):
                trace.plot(values[c], linewidth=0.6, alpha=0.8)
                density.hist(values[c], bins=bins, histtype="step", density=True)
            trace.axhline(0, color="black", linewidth=1)
            density.axvline(0, color="black", linewidth=1)
            trace.set_ylabel(name, rotation=0, ha="right", fontsize=9)
            density.set_yticks([])
        axes[0, 0].set_title("Trace (deviation from observed)")
        axes[0, 1].set_title("Density")
        axes[-1, 0].set_xlabel("sample")
        fig.tight_layout()
        return fig
