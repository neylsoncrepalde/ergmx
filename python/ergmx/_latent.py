"""Latent space models of networks, as R's latentnet (``ergmm()``; Hoff,
Raftery and Handcock 2002; Handcock, Raftery and Tantrum 2007; Krivitsky,
Handcock, Raftery and Hoff 2009): the dyads are independent given each
vertex's position in a latent space, with

    eta_ij = L(Z_i, Z_j) + sum_k beta_k x_ij,k + s_i + r_j,

L the negative Euclidean distance (``euclidean``), its square
(``euclidean2``) or the inner product (``bilinear``), the positions normal
around 0 or around their cluster's mean, and random sender, receiver or
sociality effects. Fitted, as latentnet, by MCMC from a conditional
posterior mode, with the minimum Kullback-Leibler (MKL) positions, the
draws rotated to them and their clusters relabelled (Stephens' algorithm)."""

from __future__ import annotations

import dataclasses
import re
import warnings

import numpy as np

from . import _core
from ._network import Network, as_network
from .terms import Formula, Term

FAMILIES = {"Bernoulli": "Bernoulli", "Bernoulli.logit": "Bernoulli", "binomial": "binomial",
            "binomial.logit": "binomial", "Poisson": "Poisson", "Poisson.log": "Poisson", "normal": "normal",
            "normal.identity": "normal", "Gaussian": "normal", "Gaussian.identity": "normal"}

_EPS = np.finfo(float).eps


# -- Terms --------------------------------------------------------------------------------------


class _LatentTerm(Term):
    """A latent space term: the positions' dimension, clusters and priors."""

    effect = "euclidean"

    def __init__(self, d: int, G: int = 0, var_mul: float = 1 / 8, var=None, var_df_mul: float = 1,  # noqa: N803
                 var_df=None, mean_var_mul: float = 2, mean_var=None, pK_mul: float = 1, pK=None):  # noqa: N803
        if int(d) < 1:
            raise ValueError(f"{self.effect}(d=): the latent space's dimension, 1 or more")
        if int(G) < 0:
            raise ValueError(f"{self.effect}(G=): the number of clusters, 0 or more")
        self.d, self.G = int(d), int(G)
        self.var_mul, self.var, self.var_df_mul, self.var_df = var_mul, var, var_df_mul, var_df
        self.mean_var_mul, self.mean_var, self.pK_mul, self.pK = mean_var_mul, mean_var, pK_mul, pK
        if self.effect == "bilinear" and self.G > 0:
            warnings.warn("bilinear() with clusters: the bilinear latent space has no natural clusters", stacklevel=4)

    def prior(self, n: int) -> dict:
        """latentnet's default priors of the positions, from the network's size."""
        g = max(1, self.G)
        z_var = self.var if self.var is not None else self.var_mul * (n / g) ** (2 / self.d)
        return {
            "Z.var": float(z_var),
            "Z.mean.var": float(self.mean_var if self.mean_var is not None
                                else self.mean_var_mul * z_var * g ** (2 / self.d)),
            "Z.var.df": float(self.var_df if self.var_df is not None else self.var_df_mul * np.sqrt(n / g)),
            "Z.pK": float(self.pK if self.pK is not None else self.pK_mul * np.sqrt(n / g)),
        }

    def names(self, network):
        return []

    def __repr__(self) -> str:
        return f"{self.effect}(d={self.d}, G={self.G})"


class Euclidean(_LatentTerm):
    effect = "euclidean"


class Bilinear(_LatentTerm):
    effect = "bilinear"


class Euclidean2(_LatentTerm):
    effect = "euclidean2"


class _RandomEffect(Term):
    kind = "sender"

    def __init__(self, var: float = 1.0, var_df: float = 3.0):
        self.var, self.var_df = float(var), float(var_df)

    def names(self, network):
        return []

    def __repr__(self) -> str:
        return f"r{self.kind}()"


class RSender(_RandomEffect):
    kind = "sender"


class RReceiver(_RandomEffect):
    kind = "receiver"


class RSociality(_RandomEffect):
    kind = "sociality"


class _Covariate(Term):
    """A latentnet fixed effect: its covariate matrices (n x n) and names."""

    def __init__(self, mean: float = 0.0, var: float = 9.0):
        self.mean, self.var = mean, var

    def matrices(self, network: Network) -> tuple[list[np.ndarray], list[str]]:
        raise NotImplementedError

    def names(self, network):
        return self.matrices(network)[1]


class Intercept(_Covariate):
    """The intercept, as latentnet's ``1``, ``intercept`` or ``Intercept``."""

    def matrices(self, network):
        return [np.ones((network.n, network.n))], ["(Intercept)"]

    def __repr__(self) -> str:
        return "intercept"


def _attribute_matrix(values, kind: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if kind == "sender":
        return np.repeat(v[:, None], len(v), axis=1)
    if kind == "receiver":
        return np.repeat(v[None, :], len(v), axis=0)
    return v[:, None] + v[None, :]


class _NodeCovariate(_Covariate):
    """latentnet's sendercov, receivercov and socialitycov: x_ij = a_i, a_j
    or a_i + a_j; a factor gives one covariate per level but the first."""

    kind = "sender"

    def __init__(self, attrname: str, force_factor: bool = False, mean=0.0, var=9.0):
        super().__init__(mean, var)
        self.attr, self.force_factor = attrname, bool(force_factor)

    def matrices(self, network):
        values = network.attribute(self.attr)
        name = f"{self.kind}cov.{self.attr}"
        numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)
        if numeric and not self.force_factor:
            return [_attribute_matrix(values, self.kind)], [name]
        levels = sorted({str(v) for v in values})
        out, names = [], []
        for level in levels[1:]:
            out.append(_attribute_matrix([str(v) == level for v in values], self.kind))
            names.append(f"{name}.{level}")
        return out, names

    def __repr__(self) -> str:
        return f"{self.kind}cov({self.attr!r})"


class SenderCov(_NodeCovariate):
    kind = "sender"


class ReceiverCov(_NodeCovariate):
    kind = "receiver"


class SocialityCov(_NodeCovariate):
    kind = "sociality"


class LatentCov(_Covariate):
    """latentnet's latentcov(x, attrname): a covariate matrix, a graph
    attribute's name, or a network (its adjacency, or its edge attribute
    ``attrname``)."""

    def __init__(self, x, attrname: str | None = None, mean=0.0, var=9.0):
        super().__init__(mean, var)
        self.x, self.attrname = x, attrname

    def matrices(self, network):
        x = self.x
        label = self.attrname if self.attrname is not None else (x if isinstance(x, str) else "x")
        if isinstance(x, str):
            if x not in network.graph_attributes:
                raise KeyError(f"latentcov: no graph attribute {x!r}")
            x = network.graph_attributes[x]
        if not isinstance(x, np.ndarray) and hasattr(x, "get_adjacency"):
            x = np.array(x.get_adjacency(attribute=self.attrname).data if self.attrname else x.get_adjacency().data,
                         dtype=float)
        x = np.asarray(x, dtype=float)
        if x.shape != (network.n, network.n):
            raise ValueError(f"latentcov: an {network.n} x {network.n} matrix")
        return [x], [f"latentcov.{label}"]

    def __repr__(self) -> str:
        return "latentcov()"


LATENT_TERMS = {
    "euclidean": Euclidean, "bilinear": Bilinear, "euclidean2": Euclidean2,
    "rsender": RSender, "rreceiver": RReceiver, "rsociality": RSociality,
    "intercept": Intercept, "Intercept": Intercept, "latentcov": LatentCov,
    "sendercov": SenderCov, "receivercov": ReceiverCov, "socialitycov": SocialityCov,
}


# -- The model ----------------------------------------------------------------------------------


@dataclasses.dataclass
class LatentModel:
    """A latent space model bound to a network: what the sampler, the
    estimates and the post-fit tools need."""

    network: Network
    formula: str
    response: str | None
    family: str
    fam_par: dict
    latent: str  # "none", "euclidean", "bilinear", "euclidean2"
    d: int
    G: int
    coef_names: list[str]
    X: np.ndarray  # p x n x n
    y: np.ndarray  # n x n (0 where unobserved)
    trials: np.ndarray  # n x n (binomial)
    observed: np.ndarray  # n x n bool: directed i != j, undirected i > j
    sender: bool
    receiver: bool
    sociality: bool
    eff_sender: np.ndarray
    eff_receiver: np.ndarray
    prior: dict
    intercept: bool
    core: object = dataclasses.field(repr=False)

    n = property(lambda self: self.network.n)
    p = property(lambda self: len(self.coef_names))
    directed = property(lambda self: self.network.directed)
    dispersion = property(lambda self: self.family == "normal")

    def eta(self, par: dict) -> np.ndarray:
        """The linear predictor of every dyad (n x n)."""
        n = self.n
        eta = np.zeros((n, n))
        z = par.get("Z")
        if self.d > 0 and z is not None:
            eta += latent_effect(self.latent, np.asarray(z, dtype=float))
        beta = np.asarray(par.get("beta", []), dtype=float)
        if len(beta):
            eta += np.tensordot(beta, self.X, axes=1)
        if self.sociality:
            s = np.asarray(par["sociality"], dtype=float)
            eta += s[:, None] + s[None, :]
        else:
            if self.sender:
                eta += np.asarray(par["sender"], dtype=float)[:, None]
            if self.receiver:
                eta += np.asarray(par["receiver"], dtype=float)[None, :]
        return eta

    def expected(self, eta: np.ndarray) -> np.ndarray:
        """E[Y | eta]."""
        if self.family == "Bernoulli":
            return 1 / (1 + np.exp(-eta))
        if self.family == "binomial":
            return self.trials / (1 + np.exp(-eta))
        if self.family == "Poisson":
            return np.exp(eta)
        return eta

    def core_par(self, par: dict) -> dict:
        """A configuration as the Rust core takes it."""
        n, d, g = self.n, self.d, self.G
        out = {"beta": [float(b) for b in np.atleast_1d(par.get("beta", []))]}
        if d > 0:
            out["z"] = np.ascontiguousarray(np.asarray(par["Z"], dtype=float).reshape(n, d))
            out["z_var"] = [float(v) for v in np.atleast_1d(par["Z.var"])]
            if g > 0:
                out["z_mean"] = np.ascontiguousarray(np.asarray(par["Z.mean"], dtype=float).reshape(g, d))
                out["z_k"] = [int(k) - 1 for k in np.atleast_1d(par["Z.K"])]
                out["z_pk"] = [float(v) for v in np.atleast_1d(par["Z.pK"])]
            else:
                out["z_mean"] = np.zeros((0, d))
        else:
            out["z"] = np.zeros((n, 0))
            out["z_mean"] = np.zeros((0, 0))
            out["z_var"] = []
        if self.sociality:
            out["sender"] = [float(v) for v in par["sociality"]]
            out["sender_var"] = float(par["sociality.var"])
        else:
            if self.sender:
                out["sender"] = [float(v) for v in par["sender"]]
                out["sender_var"] = float(par["sender.var"])
            if self.receiver:
                out["receiver"] = [float(v) for v in par["receiver"]]
                out["receiver_var"] = float(par["receiver.var"])
        if self.dispersion:
            out["dispersion"] = float(par["dispersion"])
        return out

    def from_core(self, core: dict) -> dict:
        """A configuration from the Rust core's, with latentnet's names."""
        par = {"beta": np.asarray(core["beta"], dtype=float)}
        if self.d > 0:
            par["Z"] = np.asarray(core["z"], dtype=float)
            par["Z.var"] = np.asarray(core["z_var"], dtype=float)
            if self.G > 0:
                par["Z.mean"] = np.asarray(core["z_mean"], dtype=float)
                par["Z.K"] = np.asarray(core["z_k"], dtype=int) + 1
                par["Z.pK"] = np.asarray(core["z_pk"], dtype=float)
        if self.sociality:
            par["sociality"], par["sociality.var"] = np.asarray(core["sender"]), float(core["sender_var"])
        else:
            if self.sender:
                par["sender"], par["sender.var"] = np.asarray(core["sender"]), float(core["sender_var"])
            if self.receiver:
                par["receiver"], par["receiver.var"] = np.asarray(core["receiver"]), float(core["receiver_var"])
        if self.dispersion:
            par["dispersion"] = float(core["dispersion"])
        return par

    def lp(self, par: dict) -> dict:
        """latentnet's log-densities of a configuration: lpY (with its
        constants), lpZ, lpbeta, lpRE, lpLV, lpREV and lpdispersion."""
        return dict(self.core.lp(self.core_par(par)))


def latent_effect(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "bilinear":
        return z @ z.T
    sq = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    return -sq if kind == "euclidean2" else -np.sqrt(sq)


def _observed(network: Network) -> np.ndarray:
    """The dyads of the likelihood: directed i != j, undirected i > j, but
    the missing; bipartite networks, the dyads between the modes."""
    n = network.n
    obs = ~np.eye(n, dtype=bool) if network.directed else np.tril(np.ones((n, n), dtype=bool), -1)
    if network.bipartite:
        obs &= network.mode[:, None] != network.mode[None, :]
    for i, j in np.asarray(network.missing, dtype=np.int64).reshape(-1, 2):
        obs[i, j] = False
        if not network.directed:
            obs[j, i] = False
    return obs


def _ergm_covariates(network: Network, term) -> tuple[list[np.ndarray], list[str]]:
    """An ergm term as dyadic covariates: its change statistics of adding
    each dyad's tie to the observed network (both ways if undirected)."""
    from ._model import bind

    model = bind(network, Formula([term]))
    x, _, pairs = model.core.mple_data(model.network.edges)
    n, k = network.n, x.shape[1]
    out = np.zeros((k, n, n))
    for c in range(k):
        out[c, pairs[:, 0], pairs[:, 1]] = x[:, c]
        if not network.directed:
            out[c, pairs[:, 1], pairs[:, 0]] = x[:, c]
    if not term.dyad_independent:
        warnings.warn(f"{term!r} is dyad-dependent: as latentnet, its covariates are its change statistics on the "
                      "observed network, and the likelihood becomes a pseudo-likelihood", stacklevel=5)
    return list(out), list(model.stat_names)


def _shift_directions(X: np.ndarray, kind: str) -> np.ndarray:
    """latentnet's directions of the random effects' shifts in the joint
    proposal: each covariate's row (sender), column (receiver) or both
    means, orthogonalized (Gram-Schmidt, not normalized); ones if none."""
    n = X.shape[1] if X.ndim == 3 and len(X) else 0
    if not len(X):
        return None
    rows = {"sender": X.mean(axis=2), "receiver": X.mean(axis=1), "sociality": X.mean(axis=2) + X.mean(axis=1)}[kind]
    out = rows.copy()
    for k1 in range(1, len(out)):
        for k2 in range(k1):
            utu = out[k2] @ out[k2]
            if np.isclose(utu, 0, atol=1.5e-8):
                break
            out[k1] = out[k1] - (out[k2] @ out[k1]) / utu * out[k2]
    keep = np.mean(np.abs(out), axis=1) > 1.5e-8
    del n
    return out[keep]


def _parse(formula: str):
    """The formula's terms (latentnet's and ergm's) and whether it has an intercept."""
    from .formula import parse_formula, use_terms
    from .terms import TERMS

    text = str(formula).strip()
    text = text.split("~", 1)[1] if "~" in text else text
    intercept = True
    for pattern in (r"\s*-\s*1\b", r"^\s*0\s*\+\s*", r"\s*\+\s*0\b"):
        if re.search(pattern, text):
            intercept = False
            text = re.sub(pattern, "", text)
    if text.strip() in ("", "1"):
        return [], intercept
    with use_terms({**TERMS, **LATENT_TERMS, "1": Intercept}):
        terms = list(parse_formula(text))
    return terms, intercept


def bind_latent(network, formula, *, response=None, family: str = "Bernoulli", fam_par=None, prior=None,
                bipartite=None) -> LatentModel:
    """A latent space model of a network."""
    from ._valued import valued_network

    if family not in FAMILIES:
        raise ValueError(f"family: one of {', '.join(FAMILIES)}, not {family!r}")
    family = FAMILIES[family]
    fam_par = dict(fam_par or {})
    if response is not None:
        net, triples = valued_network(network, response, bipartite)
    else:
        net = as_network(network, bipartite)
        triples = np.column_stack([net.edges, np.ones(len(net.edges))]) if len(net.edges) else np.zeros((0, 3))
    if net.combined:
        raise ValueError("latent space models take a single network")
    if family != "Bernoulli" and response is None:
        raise ValueError(f"the {family} family needs response=, the edge attribute with the values")
    n = net.n
    observed = _observed(net)
    y = np.zeros((n, n))
    t = np.asarray(triples, dtype=float).reshape(-1, 3)
    i, j = t[:, 0].astype(int), t[:, 1].astype(int)
    y[i, j] = t[:, 2]
    if not net.directed:
        y[j, i] = t[:, 2]
    y = np.where(observed, y, 0.0)
    if family == "Bernoulli" and np.any((y != 0) & (y != 1)):
        raise ValueError("the Bernoulli family takes 0/1 values; use the binomial, Poisson or normal family")
    trials = np.zeros((n, n))
    if family == "binomial":
        if "trials" not in fam_par:
            raise ValueError("the binomial family needs fam_par={'trials': ...}")
        trials = np.broadcast_to(np.asarray(fam_par["trials"], dtype=float), (n, n)).copy()
        if np.any(y[observed] > trials[observed]):
            raise ValueError("some values exceed their number of trials")
    if family == "normal" and not {"prior_var", "prior_var_df"} <= set(fam_par):
        raise ValueError("the normal family needs fam_par={'prior_var': ..., 'prior_var_df': ...}, the prior of its "
                         "variance")
    terms, intercept = _parse(formula)
    latent_terms = [t for t in terms if isinstance(t, _LatentTerm)]
    if len(latent_terms) > 1:
        raise ValueError("a latent space model has one latent space term")
    effects = {t.kind: t for t in terms if isinstance(t, _RandomEffect)}
    if "sociality" in effects and ({"sender", "receiver"} & set(effects)):
        raise ValueError("rsociality() is for models without rsender() and rreceiver()")
    if not net.directed and ({"sender", "receiver"} & set(effects)):
        raise ValueError("rsender() and rreceiver() are for directed networks: use rsociality()")
    # The binary network the covariates and starting values read: the ties with values.
    binary = dataclasses.replace(net, edges=net.edges)
    X, names, means, variances = [], [], [], []
    for term in terms:
        if isinstance(term, (_LatentTerm, _RandomEffect)):
            continue
        if isinstance(term, _Covariate):
            mats, labels = term.matrices(binary)
            mean, var = term.mean, term.var
        else:
            mats, labels = _ergm_covariates(binary, term)
            mean, var = 0.0, 9.0
        for m, label in zip(mats, labels):
            m = np.asarray(m, dtype=float).copy()
            if not net.directed:  # symmetric, from the upper triangle, as latentnet
                lower = np.tril_indices(n, -1)
                m[lower] = m.T[lower]
            X.append(m)
            names.append(label)
        means += list(np.resize(np.asarray(mean, dtype=float), len(labels)))
        variances += list(np.resize(np.asarray(var, dtype=float), len(labels)))
    if intercept and not any(np.all(m[observed] == m[observed][0]) and m[observed][0] != 0 for m in X):
        X.insert(0, np.ones((n, n)))
        names.insert(0, "(Intercept)")
        means.insert(0, 0.0)
        variances.insert(0, 9.0)
    X = np.array(X).reshape(len(names), n, n)
    pr: dict = {"beta.mean": np.array(means, dtype=float), "beta.var": np.array(variances, dtype=float)}
    user = dict(prior or {})
    if user.pop("adjust_beta_var", True):
        for k in range(len(names)):
            square = np.mean(X[k][observed] ** 2)
            if square > 0:
                pr["beta.var"][k] = pr["beta.var"][k] / square
    d = G = 0
    kind = "none"
    if latent_terms:
        lt = latent_terms[0]
        d, G, kind = lt.d, lt.G, lt.effect
        pr.update(lt.prior(n))
    for k, term in effects.items():
        pr[f"{k}.var"], pr[f"{k}.var.df"] = term.var, term.var_df
    if family == "normal":
        pr["dispersion"], pr["dispersion.df"] = float(fam_par["prior_var"]), float(fam_par["prior_var_df"])
    for key, value in user.items():  # the user's priors, in latentnet's names (Z.var, beta.var...)
        pr[key.replace("_", ".") if key.replace("_", ".") in _PRIOR_NAMES else key] = value
    pr["beta.mean"] = np.resize(np.asarray(pr["beta.mean"], dtype=float), len(names))
    pr["beta.var"] = np.resize(np.asarray(pr["beta.var"], dtype=float), len(names))
    sender, receiver, sociality = "sender" in effects, "receiver" in effects, "sociality" in effects
    ones = np.ones((1, n))
    eff_s = (_shift_directions(X, "sociality" if sociality else "sender") if len(names) else ones) \
        if sender or sociality else np.zeros((0, n))
    eff_r = (_shift_directions(X, "receiver") if len(names) else ones) if receiver else np.zeros((0, n))
    core_prior = {"beta_mean": list(pr["beta.mean"]), "beta_var": list(pr["beta.var"])}
    if d > 0:
        core_prior.update({"Z_var": pr["Z.var"], "Z_mean_var": pr["Z.mean.var"], "Z_var_df": pr["Z.var.df"],
                           "Z_pK": pr["Z.pK"]})
    if sociality:
        core_prior.update({"sender_var": pr["sociality.var"], "sender_var_df": pr["sociality.var.df"]})
    if sender:
        core_prior.update({"sender_var": pr["sender.var"], "sender_var_df": pr["sender.var.df"]})
    if receiver:
        core_prior.update({"receiver_var": pr["receiver.var"], "receiver_var_df": pr["receiver.var.df"]})
    if family == "normal":
        core_prior.update({"dispersion": pr["dispersion"], "dispersion_df": pr["dispersion.df"]})
    core = _core.LatentModel(kind, family, d, G, net.directed, np.ascontiguousarray(y), np.ascontiguousarray(trials),
                             np.ascontiguousarray(observed), np.ascontiguousarray(X), sender, receiver, sociality,
                             np.ascontiguousarray(eff_s), np.ascontiguousarray(eff_r), core_prior)
    return LatentModel(net, str(formula), response, family, fam_par, kind, d, G, names, X, y, trials, observed,
                       sender, receiver, sociality, eff_s, eff_r, pr, intercept, core)


_PRIOR_NAMES = {"Z.var", "Z.mean.var", "Z.var.df", "Z.pK", "beta.mean", "beta.var", "sender.var", "sender.var.df",
                "receiver.var", "receiver.var.df", "sociality.var", "sociality.var.df"}


# -- Log-densities and their gradients, for the optimizations -----------------------------------

_LN_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sclinvchisq(x, df, scale):
    from scipy.special import gammaln

    return 0.5 * df * np.log(0.5 * df * scale) - gammaln(0.5 * df) - (0.5 * df + 1) * np.log(x) - df * scale / (2 * x)


def _d_sclinvchisq(x, df, scale):
    return -(0.5 * df + 1) / x + df * scale / (2 * x * x)


def _lnorm(x, mean, var):
    return -_LN_SQRT_2PI - 0.5 * np.log(var) - (x - mean) ** 2 / (2 * var)


class _Objective:
    """The log-likelihood (up to its constants) or the conditional log
    posterior (given the clusters), over a subset of the parameters, packed
    as a vector, with their gradients: latentnet's find.mle and find.mpe."""

    def __init__(self, model: LatentModel, start: dict, free: list[str], posterior: bool, y=None):
        self.m, self.free, self.posterior = model, free, posterior
        self.y = model.y if y is None else y
        self.base = {k: (np.array(v, dtype=float) if not np.isscalar(v) else float(v)) for k, v in start.items()}
        self.shapes = {k: np.shape(self.base[k]) for k in free}

    def pack(self, par: dict) -> np.ndarray:
        return np.concatenate([np.ravel(np.asarray(par[k], dtype=float)) for k in self.free]) if self.free \
            else np.zeros(0)

    def unpack(self, x: np.ndarray) -> dict:
        par, at = dict(self.base), 0
        for k in self.free:
            size = int(np.prod(self.shapes[k])) if self.shapes[k] else 1
            v = x[at:at + size]
            par[k] = v.reshape(self.shapes[k]) if self.shapes[k] else float(v[0])
            at += size
        return par

    def bounds(self) -> list:
        lower = np.sqrt(_EPS)
        out = []
        for k in self.free:
            size = int(np.prod(self.shapes[k])) if self.shapes[k] else 1
            bound = (lower, None) if k in ("Z.var", "sender.var", "receiver.var", "sociality.var", "dispersion") \
                else (None, None)
            out += [bound] * size
        return out

    def value_grad(self, x: np.ndarray):
        m, par = self.m, self.unpack(x)
        eta = m.eta(par)
        obs, y = m.observed, self.y
        disp = par.get("dispersion", 1.0)
        if m.family == "Bernoulli":
            ll = np.where(obs, y * eta - _softplus(eta), 0.0)
            w = y - 1 / (1 + np.exp(-eta))
        elif m.family == "binomial":
            ll = np.where(obs, y * eta - m.trials * _softplus(eta), 0.0)
            w = y - m.trials / (1 + np.exp(-eta))
        elif m.family == "Poisson":
            ll = np.where(obs, y * eta - np.exp(eta), 0.0)
            w = y - np.exp(eta)
        else:
            ll = np.where(obs, -(y - eta) ** 2 / disp / 2 - np.log(disp) / 2, 0.0)
            w = (y - eta) / disp
        w = np.where(obs, w, 0.0)
        value = ll.sum()
        grad = {}
        if "beta" in self.free:
            grad["beta"] = np.tensordot(m.X, w, axes=([1, 2], [0, 1])) if m.p else np.zeros(0)
        if "Z" in self.free:
            z, s = np.asarray(par["Z"]), w + w.T
            if m.latent == "bilinear":
                grad["Z"] = s @ z
            else:
                diff = z[:, None, :] - z[None, :, :]
                if m.latent == "euclidean":
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    a = np.divide(s, dist, out=np.zeros_like(s), where=dist > 0)
                    grad["Z"] = -np.einsum("ij,ijk->ik", a, diff)
                else:
                    grad["Z"] = -2 * np.einsum("ij,ijk->ik", s, diff)
        if "sociality" in self.free:
            grad["sociality"] = w.sum(axis=1) + w.sum(axis=0)
        if "sender" in self.free:
            grad["sender"] = w.sum(axis=1)
        if "receiver" in self.free:
            grad["receiver"] = w.sum(axis=0)
        if "dispersion" in self.free:
            r = np.where(obs, (y - eta) ** 2, 0.0)
            grad["dispersion"] = np.sum(r) / (2 * disp ** 2) - obs.sum() / (2 * disp)
        if self.posterior:
            value += self._prior(par, grad)
        g = np.concatenate([np.ravel(grad.get(k, np.zeros(self.shapes[k] or 1))) for k in self.free]) \
            if self.free else np.zeros(0)
        return value, g

    def _prior(self, par: dict, grad: dict) -> float:
        m, pr = self.m, self.m.prior
        total = 0.0

        def add(key, value):
            grad[key] = grad.get(key, 0) + value

        if m.p:
            beta = np.asarray(par["beta"])
            total += np.sum(_lnorm(beta, pr["beta.mean"], pr["beta.var"]))
            if "beta" in self.free:
                add("beta", -(beta - pr["beta.mean"]) / pr["beta.var"])
        if m.d > 0:
            z = np.asarray(par["Z"])
            var = np.atleast_1d(np.asarray(par["Z.var"], dtype=float))
            if m.G > 0:
                k = np.asarray(par["Z.K"], dtype=int) - 1
                mean = np.asarray(par["Z.mean"], dtype=float)
                mu, v = mean[k], var[k][:, None]
            else:
                mu, v = 0.0, var[0]
            total += np.sum(_lnorm(z, mu, v))
            if "Z" in self.free:
                add("Z", -(z - mu) / v)
            if "Z.var" in self.free:
                dv = -0.5 / v + (z - mu) ** 2 / (2 * v * v)
                if m.G > 0:
                    gv = np.zeros(m.G)
                    np.add.at(gv, k, dv.sum(axis=1))
                else:
                    gv = np.array([dv.sum()])
                add("Z.var", gv + _d_sclinvchisq(var, pr["Z.var.df"], pr["Z.var"]))
            total += np.sum(_sclinvchisq(var, pr["Z.var.df"], pr["Z.var"]))
            if m.G > 0:
                total += np.sum(_lnorm(mean, 0.0, pr["Z.mean.var"]))
                if "Z.mean" in self.free:
                    gm = np.zeros_like(mean)
                    np.add.at(gm, k, (z - mu) / v)
                    add("Z.mean", gm - mean / pr["Z.mean.var"])
        for kind in ("sender", "receiver", "sociality"):
            if getattr(m, kind):
                e, v = np.asarray(par[kind]), float(par[f"{kind}.var"])
                total += np.sum(_lnorm(e, 0.0, v)) + _sclinvchisq(v, pr[f"{kind}.var.df"], pr[f"{kind}.var"])
                if kind in self.free:
                    add(kind, -e / v)
                if f"{kind}.var" in self.free:
                    add(f"{kind}.var", np.sum(-0.5 / v + e * e / (2 * v * v))
                        + _d_sclinvchisq(v, pr[f"{kind}.var.df"], pr[f"{kind}.var"]))
        if m.dispersion:
            disp = float(par["dispersion"])
            total += _sclinvchisq(disp, pr["dispersion.df"], pr["dispersion"])
            if "dispersion" in self.free:
                add("dispersion", _d_sclinvchisq(disp, pr["dispersion.df"], pr["dispersion"]))
        return float(total)


def _optimize(model: LatentModel, start: dict, free: list[str], posterior: bool, y=None, maxit: int = 100) -> dict:
    """L-BFGS-B, as latentnet's (maxit, factr 1e7, 5 corrections), maximizing the objective."""
    from scipy.optimize import minimize

    obj = _Objective(model, start, [k for k in free if k in start], posterior, y)
    if not obj.free:
        return dict(start)
    res = minimize(lambda x: tuple(-v for v in obj.value_grad(x)), obj.pack(start), jac=True, method="L-BFGS-B",
                   bounds=obj.bounds(), options={"maxiter": maxit, "maxcor": 5, "ftol": 1e7 * _EPS, "gtol": 0.0})
    return obj.unpack(res.x)


_LLK_NAMES = ["beta", "Z", "sender", "receiver", "sociality", "dispersion"]
_ALL_NAMES = ["beta", "Z", "sender", "receiver", "sociality", "Z.var", "Z.mean", "sender.var", "receiver.var",
              "sociality.var", "dispersion"]


def find_mle(model: LatentModel, start: dict, y=None, maxit: int = 100, given=()) -> dict:
    """The log-likelihood's maximum over the coefficients, positions, random
    effects and dispersion (but `given`), from `start` (latentnet's find.mle)."""
    return _optimize(model, start, [k for k in _LLK_NAMES if k not in given], False, y, maxit)


def find_mpe(model: LatentModel, start: dict, maxit: int = 100) -> dict:
    """The log posterior's maximum given the clusters (latentnet's find.mpe)."""
    return _optimize(model, start, _ALL_NAMES, True, None, maxit)


def _same(a: dict, b: dict) -> bool:
    """R's isTRUE(all.equal()) of two configurations."""
    for k in a:
        x, y = np.asarray(a[k], dtype=float), np.asarray(b.get(k, np.nan), dtype=float)
        if x.shape != y.shape:
            return False
        diff = np.abs(x - y)
        scale = np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 1.0
        if np.mean(diff) / scale >= 1.5e-8:
            return False
    return True


# -- Clusters: latentnet's mbc.VII.EM -----------------------------------------------------------


def _kmeans(z: np.ndarray, g: int, rng, starts: int = 15):
    """k-means with k-means++ starts, the best of `starts` (R's kmeans(nstart=15))."""
    best = None
    for _ in range(starts):
        centers = z[rng.choice(len(z), 1)]
        for _ in range(1, g):
            d2 = np.min(((z[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            total = d2.sum()
            pick = rng.choice(len(z), p=d2 / total) if total > 0 else rng.integers(len(z))
            centers = np.vstack([centers, z[pick]])
        for _ in range(100):
            labels = np.argmin(((z[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            new = np.array([z[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(g)])
            if np.allclose(new, centers):
                break
            centers = new
        withinss = np.array([((z[labels == k] - centers[k]) ** 2).sum() for k in range(g)])
        if best is None or withinss.sum() < best[2].sum():
            best = (centers, labels, withinss)
    return best


def mbc_vii_em(g: int, z: np.ndarray, rng, resume: dict | None = None, maxit: int = 200, maxstarts: int = 15) -> dict:
    """A mixture of g spherical normals with their own variances, by EM
    (latentnet's mbc.VII.EM): means, variances, probabilities, the most
    probable cluster of each point (1 to g), and the log-likelihood."""
    n, d = z.shape
    tol = np.sqrt(_EPS)
    out = None
    for attempt in range(1, maxstarts + 1):
        if resume is not None and attempt <= 2:
            mean, var, pk = (np.array(resume["Z.mean"], dtype=float), np.array(resume["Z.var"], dtype=float),
                             np.array(resume["Z.pK"], dtype=float))
        elif g > 1:
            centers, labels, withinss = _kmeans(z, g, rng)
            sizes = np.bincount(labels, minlength=g).astype(float)
            mean, var, pk = centers, withinss / np.maximum(sizes, 1), sizes / sizes.sum()
            var = np.where(var <= tol, np.sum((z - z.mean(axis=0)) ** 2) / max(n - d, 1), var)
        else:
            mean = z.mean(axis=0)[None]
            var, pk = np.array([np.sum((z - mean) ** 2) / max(n - d, 1)]), np.array([1.0])
        try:
            for _ in range(maxit):
                old = np.concatenate([mean.ravel(), var, pk])
                logp = (np.log(pk)[None] - 0.5 * d * np.log(2 * np.pi * var)[None]
                        - ((z[:, None, :] - mean[None]) ** 2).sum(-1) / (2 * var[None]))
                resp = np.exp(logp - logp.max(axis=1, keepdims=True))
                resp /= resp.sum(axis=1, keepdims=True)
                pk = resp.mean(axis=0)
                mean = (resp.T @ z) / (pk[:, None] * n)
                var = np.array([np.sum(resp[:, k] * ((z - mean[k]) ** 2).mean(axis=1)) / resp[:, k].sum()
                                for k in range(g)])
                new = np.concatenate([mean.ravel(), var, pk])
                if np.mean(np.abs(new - old)) / max(np.mean(np.abs(old)), 1e-300) < tol:
                    break
            labels = np.argmax(resp, axis=1) + 1
        except (FloatingPointError, ValueError, ZeroDivisionError):
            continue
        mean, var = np.nan_to_num(mean), np.nan_to_num(var)
        if np.min(var) <= 0 or np.max(var) / np.min(var) > _EPS ** -0.5:
            continue
        logp = (np.log(pk)[None] - 0.5 * d * np.log(2 * np.pi * var)[None]
                - ((z[:, None, :] - mean[None]) ** 2).sum(-1) / (2 * var[None]))
        top = logp.max(axis=1, keepdims=True)
        llk = float(np.sum(top.ravel() + np.log(np.exp(logp - top).sum(axis=1))))
        out = {"Z.mean": mean, "Z.var": var, "Z.pK": pk, "Z.K": labels, "Z.pZK": resp, "llk": llk}
        break
    if out is None:
        out = {"Z.mean": np.zeros((g, d)), "Z.var": np.ones(g), "Z.pK": np.full(g, 1 / g),
               "Z.K": np.ones(n, dtype=int), "Z.pZK": np.full((n, g), 1 / g), "llk": np.inf}
    return out


# -- Starting values (latentnet's ergmm.initvals) -----------------------------------------------


def _geodesics(a: np.ndarray) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n = len(a)
    d = shortest_path(csr_matrix(a.astype(float)), directed=False, unweighted=True)
    d[~np.isfinite(d)] = n
    return d


def _cmdscale(dist: np.ndarray, k: int) -> np.ndarray:
    n = len(dist)
    j = np.eye(n) - 1 / n
    b = -0.5 * j @ (dist ** 2) @ j
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:k]
    values, vectors = values[order], vectors[:, order]
    if np.any(values <= 0):
        warnings.warn("some of the multidimensional scaling's eigenvalues are not positive: their positions are 0",
                      stacklevel=4)
    return vectors * np.sqrt(np.maximum(values, 0))[None]


def initial_values(model: LatentModel, rng, maxit: int = 100) -> dict:
    """latentnet's starting values: classical scaling of the geodesic
    distances, clusters by EM, the coefficients and random effects from the
    densities, then the conditional posterior mode."""
    from scipy.special import logit

    n, d, g = model.n, model.d, model.G
    obs = model.observed
    ym = np.where(obs, model.y, np.nan)
    ym01 = np.where(obs, (model.y > np.nanmean(ym)).astype(float), np.nan)
    par: dict = {}
    if d > 0:
        a = np.nan_to_num(ym01) > 0
        a = a | a.T
        z = _cmdscale(_geodesics(a), d)
        keep = np.ones(n, dtype=bool)
        if n > d + 1:
            cov = np.cov(z, rowvar=False).reshape(d, d)
            try:
                keep = np.einsum("ij,jk,ik->i", z, np.linalg.pinv(cov), z) < 20
            except np.linalg.LinAlgError:
                pass
        z[~keep] = 0
        par["Z"] = z
        if g > 0:
            em = mbc_vii_em(g, z[keep], rng)
            labels = np.full(n, int(np.argmax(np.bincount(em["Z.K"], minlength=g + 1)[1:]) + 1))
            labels[keep] = em["Z.K"]
            par["Z.K"] = labels
            par["Z.pK"] = np.bincount(labels, minlength=g + 1)[1:] / n
            par["Z.var"] = np.array([np.var(z[keep & (labels == k + 1)].ravel(), ddof=1)
                                     if np.sum(keep & (labels == k + 1)) * d > 1 else model.prior["Z.var"]
                                     for k in range(g)])
            par["Z.mean"] = np.array([z[keep & (labels == k + 1)].mean(axis=0) if np.any(keep & (labels == k + 1))
                                      else np.zeros(d) for k in range(g)])
        else:
            par["Z.var"] = np.array([np.var(z[keep].ravel(), ddof=1)])
    beta = np.asarray(model.prior["beta.mean"], dtype=float).copy()
    if model.intercept and model.p and model.coef_names[0] == "(Intercept)":
        density = np.clip(np.nanmean(ym01), 1e-6, 1 - 1e-6)
        dist = np.sqrt(((par["Z"][:, None] - par["Z"][None]) ** 2).sum(-1)).mean() if d > 0 else 0.0
        beta[0] = logit(density) + dist
    par["beta"] = beta

    def bayes_prop(values, axis):
        known = ~np.isnan(values)
        return (np.nansum(values, axis=axis) + 1) / (known.sum(axis=axis) + 2)

    if model.sociality:
        s = logit((bayes_prop(ym01, 1) + bayes_prop(ym01, 0)) / 2)
        par["sociality"], par["sociality.var"] = s - s.mean(), float(np.var(s, ddof=1))
    if model.sender:
        s = logit(bayes_prop(ym01, 1))
        par["sender"], par["sender.var"] = s - s.mean(), float(np.var(s, ddof=1))
    if model.receiver:
        r = logit(bayes_prop(ym01, 0))
        par["receiver"], par["receiver.var"] = r - r.mean(), float(np.var(r, ddof=1))
    if model.dispersion:
        par["dispersion"] = 1.0
    for _ in range(maxit):
        old = {k: np.copy(v) for k, v in par.items()}
        par = find_mpe(model, par, maxit)
        if g > 0:
            em = mbc_vii_em(g, np.asarray(par["Z"]), rng,
                            resume={"Z.mean": par["Z.mean"], "Z.var": par["Z.var"], "Z.pK": par["Z.pK"]})
            par["Z.K"] = em["Z.K"]
            par["Z.pK"] = np.bincount(par["Z.K"], minlength=g + 1)[1:] / n
        if _same(old, par):
            break
    return par


# -- The MCMC, with latentnet's tuning ----------------------------------------------------------


@dataclasses.dataclass
class LatentControl:
    """The MCMC's settings, as latentnet's control.ergmm(): draws, burn-in
    and thinning per chain, the chains (in parallel threads), the proposals'
    initial sizes, and their tuning by `pilot_runs` runs of the burn-in."""

    sample_size: int = 4000
    burnin: int = 10000
    interval: int = 10
    n_chains: int = 4
    z_delta: float = 0.6
    re_delta: float = 0.6
    group_deltas: float = 0.4
    pilot_runs: int = 4
    pilot_factor: float = 0.8
    pilot_discard_first: float = 0.5
    target_acc_rate: float = 0.234
    backoff_threshold: float = 0.05
    backoff_factor: float = 0.2
    mle_maxit: int = 100


@dataclasses.dataclass
class _Tuning:
    z: float
    re: float
    group: np.ndarray
    pilot_factor: float
    acf: np.ndarray


def _initial_tuning(model: LatentModel, control: LatentControl) -> _Tuning:
    m = model.core.group_dim
    v = [1 / np.sqrt(np.mean(x[model.observed] ** 2)) for x in model.X]
    if model.d > 0:
        v.append(0.05)
    v += [1 / max(model.p, 1)] * (m - len(v))
    return _Tuning(control.z_delta, control.re_delta, np.diag(control.group_deltas * np.asarray(v) * 2 / (1 + m)),
                   control.pilot_factor, np.ones(m))


def _extended(model: LatentModel, draws: dict) -> np.ndarray:
    """The draws' coefficients, log-scale of the positions and shifts of the
    random effects: what the joint proposal moves (latentnet's get.beta.ext)."""
    n = model.n
    cols = [draws["beta"]] if model.p else []
    if model.d > 0:
        cols.append(np.log(np.mean(np.sqrt((draws["z"] ** 2).sum(-1)), axis=1))[:, None])
    if model.sender or model.sociality:
        cols.append(draws["sender"] @ model.eff_sender.T / n)
    if model.receiver:
        cols.append(draws["receiver"] @ model.eff_receiver.T / n)
    if model.dispersion:
        cols.append(np.log(draws["dispersion"])[:, None])
    return np.column_stack(cols) if cols else np.zeros((len(draws["lpY"]), 0))


def _lag1(x: np.ndarray) -> np.ndarray:
    """Each column's Yule-Walker AR(1) coefficient: its lag-1 autocorrelation."""
    c = x - x.mean(axis=0)
    denominator = np.sum(c * c, axis=0)
    return np.divide(np.sum(c[1:] * c[:-1], axis=0), denominator, out=np.zeros(c.shape[1]), where=denominator > 0)


def _retune(model: LatentModel, draws: dict, t: _Tuning, control: LatentControl) -> _Tuning:
    """latentnet's get.sample.deltas: the proposals' sizes from a pilot run."""
    s = len(draws["lpY"])
    use = slice(int(np.ceil(s * control.pilot_discard_first)) - 1, s)
    has_vertex = model.d > 0 or model.sender or model.receiver or model.sociality
    z_rate = float(np.mean(draws["Z_rate"][use])) if has_vertex else 0.5
    group_rate = float(np.mean(draws["group_rate"][use]))
    ext = _extended(model, draws)
    group, pilot_factor, acf = t.group, t.pilot_factor, t.acf
    if ext.shape[1]:
        ar = np.clip(_lag1(ext), -0.99, 0.99)
        eff = 1 / (1 - ar)
        acf = acf * eff / np.exp(np.mean(np.log(eff)))
        sigma = np.atleast_2d(np.cov(ext, rowvar=False)) * np.outer(acf, acf)
        pilot_factor = pilot_factor * group_rate / control.target_acc_rate
        try:
            group = np.linalg.cholesky(sigma).T * pilot_factor
        except np.linalg.LinAlgError:
            raise RuntimeError("the latent space model's MCMC did not mix: its pilot run's coefficients are "
                               "degenerate") from None
    scale = z_rate / control.target_acc_rate
    return _Tuning(t.z * scale, t.re * scale, group, pilot_factor, acf)


def _backoff(model: LatentModel, draws: dict, t: _Tuning, control: LatentControl):
    """latentnet's backoff.check: smaller (larger) proposals if a pilot run
    accepted almost nothing (almost everything). Returns the tuning and
    whether it backed off."""
    bt = control.backoff_threshold if control.backoff_threshold <= 0.5 else 1 - control.backoff_threshold
    bf = control.backoff_factor
    z, re_, group, factor, changed = t.z, t.re, t.group, t.pilot_factor, False
    if model.d > 0 or model.sender or model.receiver or model.sociality:
        if np.any(draws["Z_rate"] < bt):
            z, re_, changed = z * bf, re_ * bf, True
        elif np.any(draws["Z_rate"] > 1 - bt):
            z, re_, changed = z / bf, re_ / bf, True
    if model.core.group_dim:
        rate = np.mean(draws["group_rate"])
        if rate < bt:
            group, factor, changed = group * bf, factor * bf, True
        elif rate > 1 - bt:
            group, factor, changed = group / bf, factor / bf, True
    return _Tuning(z, re_, group, factor, t.acf), changed


def run_mcmc(model: LatentModel, start: dict, control: LatentControl, rng) -> tuple[list[dict], list[_Tuning]]:
    """The chains' draws (one dict each), after the tuned burn-in."""
    k = control.n_chains
    if control.sample_size % k:
        raise ValueError(f"sample_size ({control.sample_size}) must be a multiple of n_chains ({k})")
    seeds = lambda: [int(s) for s in rng.integers(0, 2**63, k)]  # noqa: E731
    states = [model.core_par(start)] * k
    tunings = [_initial_tuning(model, control)] * k
    if control.burnin > 0:
        runs = max(control.pilot_runs, 1)
        size = control.burnin / runs / control.interval
        if size != int(size) or size < 1:
            raise ValueError("burnin / max(pilot_runs, 1) / interval must be a whole number")
        for _ in range(runs):
            pending = list(range(k))
            for _attempt in range(50):
                out = model.core.run([states[c] for c in pending], [(tunings[c].z, tunings[c].re,
                                     np.ascontiguousarray(tunings[c].group)) for c in pending], int(size),
                                     control.interval, seeds()[:len(pending)])
                retry = []
                for c, draws in zip(pending, out):
                    states[c] = dict(draws["last"])
                    if control.pilot_runs:
                        tunings[c], changed = _backoff(model, draws, tunings[c], control)
                        if changed:
                            retry.append(c)
                            continue
                        tunings[c] = _retune(model, draws, tunings[c], control)
                pending = retry
                if not pending:
                    break
    out = model.core.run(states, [(t.z, t.re, np.ascontiguousarray(t.group)) for t in tunings],
                         control.sample_size // k, control.interval, seeds())
    return [dict(o) for o in out], tunings


# -- Post-processing: labels, MKL, clusters of the MKL positions, rotations ----------------------


def _stack(chains: list[dict]) -> dict:
    """The chains' draws, one after another, with latentnet's names."""
    keys = ["beta", "z", "z_k", "z_mean", "z_var", "z_pk", "sender", "receiver", "sender_var", "receiver_var",
            "dispersion", "lpY", "lpZ", "lpbeta", "lpRE", "lpLV", "lpREV", "lpdispersion", "Z_rate", "group_rate"]
    return {k: np.concatenate([np.asarray(c[k]) for c in chains]) for k in keys}


def _sample(model: LatentModel, stacked: dict) -> dict:
    """The draws as latentnet's sample: beta, Z, Z.K (1 to G)... per draw."""
    out = {"beta": stacked["beta"], "lpY": stacked["lpY"], "lpZ": stacked["lpZ"], "lpbeta": stacked["lpbeta"],
           "lpRE": stacked["lpRE"], "lpLV": stacked["lpLV"], "lpREV": stacked["lpREV"],
           "lpdispersion": stacked["lpdispersion"], "Z.rate": stacked["Z_rate"], "beta.rate": stacked["group_rate"]}
    if model.d > 0:
        out["Z"], out["Z.var"] = stacked["z"], stacked["z_var"]
        if model.G > 0:
            out["Z.mean"], out["Z.K"], out["Z.pK"] = stacked["z_mean"], stacked["z_k"] + 1, stacked["z_pk"]
    if model.sociality:
        out["sociality"], out["sociality.var"] = stacked["sender"], stacked["sender_var"]
    else:
        if model.sender:
            out["sender"], out["sender.var"] = stacked["sender"], stacked["sender_var"]
        if model.receiver:
            out["receiver"], out["receiver.var"] = stacked["receiver"], stacked["receiver_var"]
    if model.dispersion:
        out["dispersion"] = stacked["dispersion"]
    return out


def draw(sample: dict, s: int) -> dict:
    """Draw s of a sample, as a configuration."""
    return {k: v[s] for k, v in sample.items() if not k.startswith("lp") and not k.endswith(".rate")}


def _membership(sample: dict) -> np.ndarray:
    """Each draw's probabilities of each vertex's cluster, given its parameters (draws x n x G)."""
    z, mean, var, pk = sample["Z"], sample["Z.mean"], sample["Z.var"], sample["Z.pK"]
    d = z.shape[2]
    logp = (np.log(pk)[:, None, :] - 0.5 * d * np.log(2 * np.pi * var)[:, None, :]
            - ((z[:, :, None, :] - mean[:, None, :, :]) ** 2).sum(-1) / (2 * var[:, None, :]))
    p = np.exp(logp - logp.max(axis=2, keepdims=True))
    return p / p.sum(axis=2, keepdims=True)


def label_switch(sample: dict, reference: np.ndarray, maxit: int = 100) -> tuple[dict, np.ndarray]:
    """Stephens' (2000) relabelling, as latentnet's klswitch: each draw's
    clusters permuted to minimize the Kullback-Leibler divergence of its
    membership probabilities from their mean Q, until no draw changes.
    Returns the relabelled sample and Q (n x G)."""
    from scipy.optimize import linear_sum_assignment

    g = sample["Z.mean"].shape[1]
    q = np.full((len(reference), g), 1 / (2 * g))
    q[np.arange(len(reference)), np.asarray(reference) - 1] = (1 + 1 / g) / 2
    p = _membership(sample)
    logp = np.log(np.maximum(p, 1e-300))
    perms = np.tile(np.arange(g), (len(p), 1))
    for it in range(maxit):
        changed = False
        logq = np.log(q)
        for s in range(len(p)):
            # cost[new, old]: the divergence of the old label's probabilities from Q's new label.
            cost = (p[s] * logp[s]).sum(0)[None, :] - logq.T @ p[s]
            _, cols = linear_sum_assignment(cost)
            if not np.array_equal(cols, perms[s]):
                perms[s], changed = cols, True
        if not changed and it > 0:
            break
        q = np.mean(np.take_along_axis(p, perms[:, None, :], axis=2), axis=0)
    out = dict(sample)
    out["Z.mean"] = np.take_along_axis(sample["Z.mean"], perms[:, :, None], axis=1)
    out["Z.var"] = np.take_along_axis(sample["Z.var"], perms, axis=1)
    out["Z.pK"] = np.take_along_axis(sample["Z.pK"], perms, axis=1)
    inverse = np.argsort(perms, axis=1)
    out["Z.K"] = np.take_along_axis(inverse, sample["Z.K"] - 1, axis=1) + 1
    return out, q


def _centre(model: LatentModel, par: dict) -> dict:
    """latentnet's .scale.ergmm.model: the positions centred (on their
    clusters' mean means), and the random effects' means moved into the
    intercept (twice the sociality effects' mean: eta has s_i + s_j)."""
    par = dict(par)
    if model.d > 0 and model.latent != "bilinear":
        centre = np.mean(par["Z.mean"], axis=0) if "Z.mean" in par else np.mean(par["Z"], axis=0)
        par["Z"] = par["Z"] - centre
        if "Z.mean" in par:
            par["Z.mean"] = par["Z.mean"] - centre
    if model.intercept and model.p and model.coef_names[0] == "(Intercept)":
        shift = 0.0
        for kind, mul in (("sociality", 2.0), ("sender", 1.0), ("receiver", 1.0)):
            if kind in par:
                mean = float(np.mean(par[kind]))
                shift += mul * mean
                par[kind] = par[kind] - mean
        beta = np.array(par["beta"], dtype=float)
        beta[0] += shift
        par["beta"] = beta
    return par


def procrustes_rotation(z: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The rotation (or reflection) R that brings z closest to the reference: U V' of svd(z_c' ref_c)."""
    a = z - z.mean(axis=0)
    b = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    return u @ vt


def _lp_continuous(model: LatentModel, ey: np.ndarray, eta: np.ndarray, dispersion: float = 1.0) -> float:
    obs = model.observed
    if model.family == "Bernoulli":
        v = ey * eta - _softplus(eta)
    elif model.family == "binomial":
        v = ey * eta - model.trials * _softplus(eta)
    elif model.family == "Poisson":
        v = ey * eta - np.exp(eta)
    else:
        v = -(ey - eta) ** 2 / dispersion / 2 - np.log(dispersion) / 2
    return float(np.sum(v[obs]))


def posterior_expected(model: LatentModel, sample: dict) -> np.ndarray:
    """The posterior mean of E[Y_ij]: the mean over the draws of each draw's expected values."""
    total = np.zeros((model.n, model.n))
    for s in range(len(sample["lpY"])):
        total += model.expected(model.eta(draw(sample, s)))
    return total / len(sample["lpY"])


def find_mkl(model: LatentModel, sample: dict, maxit: int = 100) -> dict:
    """The minimum Kullback-Leibler estimate (Shortreed, Handcock and Hoff
    2006): the configuration whose expected values are closest to the
    posterior's mean expected values, from the closest draw."""
    ey = posterior_expected(model, sample)
    scores = [_lp_continuous(model, ey, model.eta(draw(sample, s)), sample["dispersion"][s] if model.dispersion
                             else 1.0) for s in range(len(sample["lpY"]))]
    best = int(np.argmax(scores))
    mkl = {k: v for k, v in draw(sample, best).items() if k in _LLK_NAMES}
    y = np.where(model.observed, ey, 0.0)
    for _ in range(maxit):
        old = mkl
        mkl = find_mle(model, mkl, y=y, maxit=maxit)
        if _same(old, mkl):
            break
    mkl = _centre(model, mkl)
    for k in ("Z.K", "Z.pK"):
        if k in sample:
            mkl[k] = sample[k][best]
    return mkl


def bayes_mbc(g: int, z: np.ndarray, prior: dict, rng, reference=None, samples: int = 2000, interval: int = 10,
              burnin: int = 500) -> dict:
    """latentnet's bayesmbc: the Bayesian mixture of spherical normals of
    fixed positions (the MKL's), by Gibbs sampling, relabelled; posterior
    means of the clusters' means and variances, each vertex's cluster
    probabilities and most probable cluster."""
    n, d = z.shape
    em = mbc_vii_em(g, z, rng)
    mean, var, pk, labels = em["Z.mean"], em["Z.var"], em["Z.pK"], em["Z.K"] - 1
    out = {"Z.mean": [], "Z.var": [], "Z.pK": [], "Z.K": []}
    for sweep in range(burnin + samples * interval):
        logp = (np.log(pk)[None] - 0.5 * d * np.log(2 * np.pi * var)[None]
                - ((z[:, None, :] - mean[None]) ** 2).sum(-1) / (2 * var[None]))
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        labels = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, g - 1)
        counts = np.bincount(labels, minlength=g)
        pk = rng.dirichlet(counts + prior["Z.pK"])
        sq = np.array([((z[labels == k] - mean[k]) ** 2).sum() for k in range(g)])
        df = counts * d + prior["Z.var.df"]
        var = (prior["Z.var"] * prior["Z.var.df"] + sq) / rng.chisquare(df)
        sums = np.array([z[labels == k].sum(axis=0) for k in range(g)])
        shrink = counts + var / prior["Z.mean.var"]
        mean = sums / shrink[:, None] + np.sqrt(var / shrink)[:, None] * rng.standard_normal((g, d))
        if sweep >= burnin and (sweep - burnin + 1) % interval == 0:
            out["Z.mean"].append(mean)
            out["Z.var"].append(var)
            out["Z.pK"].append(pk)
            out["Z.K"].append(labels + 1)
    sample = {k: np.array(v) for k, v in out.items()}
    sample["Z"] = np.broadcast_to(z, (samples, n, d))
    sample, _ = label_switch(sample, em["Z.K"] if reference is None else reference)
    counts = np.stack([np.bincount(sample["Z.K"][:, i] - 1, minlength=g) for i in range(n)]) / samples
    return {"Z.mean": sample["Z.mean"].mean(axis=0), "Z.var": sample["Z.var"].mean(axis=0),
            "Z.K": np.argmax(counts, axis=1) + 1, "Z.pZK": counts,
            "Z.pK": np.bincount(sample["Z.K"].ravel() - 1, minlength=g) / sample["Z.K"].size}


def rotate_sample(sample: dict, reference: np.ndarray) -> dict:
    """Each draw's positions (and clusters' means) rotated onto the reference."""
    out = dict(sample)
    z = np.array(sample["Z"])
    mean = np.array(sample["Z.mean"]) if "Z.mean" in sample else None
    for s in range(len(z)):
        r = procrustes_rotation(z[s], reference)
        z[s] = z[s] @ r
        if mean is not None:
            mean[s] = mean[s] @ r
    out["Z"] = z
    if mean is not None:
        out["Z.mean"] = mean
    return out


# -- Fitting ------------------------------------------------------------------------------------

TOFIT = ("mcmc", "mkl", "mkl.mbc", "procrustes", "klswitch", "pmode", "mle")


def ergmm(network, formula, *, response=None, family: str = "Bernoulli", fam_par=None, prior=None,
          control: LatentControl | None = None, seed=None, tofit=("mcmc", "mkl", "mkl.mbc", "procrustes", "klswitch"),
          bipartite=None, **control_args) -> ErgmmFit:
    """Fit a latent space model of a network, as R's latentnet ``ergmm()``.

    Parameters
    ----------
    network : igraph.Graph or networkx.Graph
        The network; bipartite networks need ``bipartite=``. Edges with a
        true ``na`` attribute are missing dyads.
    formula : str
        In latentnet's syntax: a latent space term, ``euclidean(d, G=0)``
        (the negative distance between the positions), ``euclidean2(d, G)``
        (its square) or ``bilinear(d, G)`` (their inner product), with ``G``
        clusters of positions; random effects ``rsender``, ``rreceiver``
        (directed) or ``rsociality``; covariates (``latentcov(x)``,
        ``sendercov(attr)``, ``receivercov(attr)``, ``socialitycov(attr)``,
        or any ergm term, by its change statistics); and an intercept,
        unless ``- 1``: ``"euclidean(d=2, G=3) + rreceiver + nodematch('group')"``.
    response, family, fam_par
        For valued networks: the edge attribute with the values, the family
        (``"Bernoulli"``, ``"binomial"`` with ``fam_par={"trials": m}``,
        ``"Poisson"``, ``"normal"`` with ``fam_par={"prior_var": s2,
        "prior_var_df": nu}``, the scaled inverse chi-squared prior of its
        variance).
    prior : dict, optional
        latentnet's priors to override, by name: ``{"Z.var": 2, "beta.var":
        [4, 4]}``... (``adjust_beta_var=False`` keeps the coefficients'
        variances at 9 instead of dividing them by the covariates' mean
        square).
    control : LatentControl, optional
        The MCMC's settings; or their names as keyword arguments
        (``sample_size=4000``, ``burnin=10000``, ``interval=10``,
        ``n_chains=4``...).
    tofit : tuple
        The estimates: latentnet's ``"mcmc"``, ``"mkl"`` (minimum
        Kullback-Leibler positions), ``"mkl.mbc"`` (their clusters),
        ``"procrustes"`` (the draws rotated onto them), ``"klswitch"`` (the
        clusters relabelled), and ``"pmode"`` (the conditional posterior
        mode) and ``"mle"`` (the maximum likelihood estimate).
    """
    unknown = set(tofit) - set(TOFIT)
    if unknown:
        raise ValueError(f"tofit: some of {', '.join(TOFIT)}, not {sorted(unknown)}")
    tofit = set(tofit)
    if "mle" in tofit:
        tofit.add("pmode")
    if tofit & {"mkl", "mkl.mbc", "procrustes", "klswitch"}:
        tofit.add("mcmc")
    if "mkl.mbc" in tofit:
        tofit.add("mkl")
    control = dataclasses.replace(control or LatentControl(), **control_args)
    rng = np.random.default_rng(seed)
    model = bind_latent(network, formula, response=response, family=family, fam_par=fam_par, prior=prior,
                        bipartite=bipartite)
    start = initial_values(model, rng, control.mle_maxit)
    fit = ErgmmFit(model, control, start)
    if "mcmc" in tofit:
        chains, fit.tunings = run_mcmc(model, start, control, rng)
        fit.chain_breaks = np.cumsum([len(c["lpY"]) for c in chains])
        fit.mcmc_mle = max((model.from_core(c["best_y"]) | {"lpY": c["best_y"]["lpY"]} for c in chains),
                           key=lambda p: p["lpY"])
        best_post = [(c["best_post"], sum(c["best_post"][k] for k in ("lpY", "lpZ", "lpbeta", "lpRE", "lpLV", "lpREV",
                                                                          "lpdispersion"))) for c in chains]
        fit.mcmc_pmode = model.from_core(max(best_post, key=lambda b: b[1])[0])
        sample = _sample(model, _stack(chains))
        if "klswitch" in tofit and model.G > 1:
            sample, fit.Q = label_switch(sample, fit.mcmc_pmode["Z.K"])
        fit.sample = sample
    if "pmode" in tofit:
        candidates = [start] + ([fit.mcmc_pmode] if fit.mcmc_pmode is not None else [])
        best = None
        for c in candidates:
            pm = _pmode_loop(model, c, rng, control.mle_maxit)
            score = sum(model.lp(pm).values())
            if best is None or score > best[1]:
                best = (pm, score)
        fit.pmode = _centre(model, best[0])
    if "mle" in tofit:
        candidates = [start] + ([fit.mcmc_mle] if fit.mcmc_mle is not None else [])
        best = None
        for c in candidates:
            mle = {k: v for k, v in c.items() if k in _LLK_NAMES}
            for _ in range(control.mle_maxit):
                old = mle
                mle = find_mle(model, mle, maxit=control.mle_maxit)
                if _same(old, mle):
                    break
            score = model.lp(mle | {k: c[k] for k in c if k not in mle})["lpY"]
            if best is None or score > best[1]:
                best = (mle, score)
        fit.mle = _centre(model, best[0])
        fit.mle["lpY"] = best[1]
    if "mkl" in tofit:
        fit.mkl = find_mkl(model, fit.sample, control.mle_maxit)
        if "mkl.mbc" in tofit and model.G > 0 and model.d > 0:
            reference = np.argmax(fit.Q, axis=1) + 1 if fit.Q is not None else fit.mcmc_pmode["Z.K"]
            fit.mkl["mbc"] = bayes_mbc(model.G, np.asarray(fit.mkl["Z"]), model.prior, rng, reference)
    if "procrustes" in tofit and model.d > 0 and fit.sample is not None:
        reference = fit.mkl["Z"] if fit.mkl is not None else fit.mcmc_pmode["Z"]
        fit.sample = rotate_sample(fit.sample, np.asarray(reference))
    return fit


def _pmode_loop(model: LatentModel, start: dict, rng, maxit: int) -> dict:
    par = dict(start)
    for _ in range(maxit):
        old = par
        par = find_mpe(model, par, maxit)
        if model.G > 1:
            em = mbc_vii_em(model.G, np.asarray(par["Z"]), rng,
                            resume={"Z.mean": par["Z.mean"], "Z.var": par["Z.var"], "Z.pK": par["Z.pK"]})
            par["Z.K"] = em["Z.K"]
            par["Z.pK"] = np.bincount(par["Z.K"], minlength=model.G + 1)[1:] / model.n
        if _same(old, par):
            break
    return par


class ErgmmFit:
    """A latent space model's fit (:func:`ergmx.ergmm`): the posterior draws
    (relabelled and rotated onto the MKL positions), the starting values,
    and the point estimates (``mkl``, and ``pmode`` and ``mle`` if fitted),
    as dicts of latentnet's names (``beta``, ``Z``, ``Z.K``, ``Z.mean``,
    ``Z.var``, ``sender``...)."""

    def __init__(self, model: LatentModel, control: LatentControl, start: dict):
        self.model, self.control, self.start = model, control, start
        self.sample: dict | None = None
        self.mkl: dict | None = None
        self.pmode: dict | None = None
        self.mle: dict | None = None
        self.mcmc_mle: dict | None = None
        self.mcmc_pmode: dict | None = None
        #: Each vertex's probabilities of the (relabelled) clusters, from Stephens' algorithm.
        self.Q: np.ndarray | None = None
        self.tunings = None
        self.chain_breaks = None

    @property
    def names(self) -> list[str]:
        return list(self.model.coef_names)

    @property
    def coef(self) -> dict[str, float]:
        """The coefficients' posterior means."""
        self._need_sample()
        return dict(zip(self.names, self.sample["beta"].mean(axis=0).tolist()))

    def _need_sample(self):
        if self.sample is None:
            raise ValueError("the model was fitted without MCMC (tofit without 'mcmc')")

    @property
    def pmean(self) -> dict:
        """The posterior means of the parameters; with clusters, each
        vertex's probabilities of the clusters (``Z.pZK``) and the most
        probable (``Z.K``)."""
        self._need_sample()
        out = {k: v.mean(axis=0) for k, v in self.sample.items() if k != "Z.K"}
        if "Z.K" in self.sample:
            g = self.model.G
            counts = np.stack([np.bincount(self.sample["Z.K"][:, i] - 1, minlength=g)
                               for i in range(self.model.n)])
            out["Z.pZK"] = counts / len(self.sample["Z.K"])
            out["Z.K"] = np.argmax(counts, axis=1) + 1
        return out

    def bic(self, eff_obs="ties") -> dict:
        """latentnet's BIC: of the likelihood (the coefficients refitted given
        the MKL positions and random effects), the positions (a mixture of
        spherical normals, by EM, if clustered) and the random effects;
        ``eff_obs`` is the effective number of observations: the ties (the
        nonzero observed dyads), the dyads, the actors, or a number."""
        if self.mkl is None:
            raise ValueError("the BIC needs the MKL estimate (tofit with 'mkl')")
        m, mkl = self.model, self.mkl
        values = m.y[m.observed]
        count = {"ties": np.count_nonzero(values), "dyads": len(values), "actors": m.n}
        n_obs = count[eff_obs] if isinstance(eff_obs, str) else float(eff_obs)
        given = [k for k in ("Z", "sender", "receiver", "sociality") if k in mkl]
        cond = find_mle(m, {k: mkl[k] for k in _LLK_NAMES if k in mkl}, maxit=self.control.mle_maxit, given=given)
        lp_y = m.lp(_completed(m, cond))["lpY"]
        out = {"Y": -2 * lp_y + m.p * np.log(n_obs)}
        z = np.asarray(mkl["Z"]) if m.d > 0 else None
        if m.d > 0:
            if m.G > 0:
                rng = np.random.default_rng(0)
                for empty in range(m.G):
                    g = m.G - empty
                    llk = mbc_vii_em(g, z, rng)["llk"]
                    if np.isfinite(llk):
                        if empty:
                            warnings.warn(f"BIC: treating {empty} clusters as empty", stacklevel=2)
                        break
                out["Z"] = -2 * llk + (g - 1 + m.d * g + g) * np.log(m.n)
            else:
                sd = np.sqrt(np.mean(np.asarray(cond["Z"]) ** 2) * m.d)
                out["Z"] = -2 * np.sum(_lnorm(z, 0.0, sd ** 2)) + np.log(m.n * m.d)
        for kind in ("sender", "receiver", "sociality"):
            if getattr(m, kind):
                e = np.asarray(cond[kind])
                out[kind] = -2 * np.sum(_lnorm(e, 0.0, np.mean(e ** 2))) + np.log(m.n)
        out["overall"] = sum(out.values())
        return out

    def summary(self, point_est=None, quantiles=(0.025, 0.975), bic_eff_obs="ties") -> ErgmmSummary:
        """The coefficients' posterior means and quantiles, the MKL (and MLE)
        estimates, and the BIC, printed like R's ``summary(ergmm)``."""
        return ErgmmSummary(self, point_est, quantiles, bic_eff_obs)

    def predict(self, type="post") -> np.ndarray:
        """Each dyad's expected value (its tie probability, for binary
        networks), n x n: the posterior mean over the draws (``"post"``), or
        at an estimate (``"mkl"``, ``"pmean"``, ``"start"``, ``"pmode"``,
        ``"mle"``, a draw's number or a configuration)."""
        m = self.model
        if isinstance(type, str) and type == "post":
            self._need_sample()
            return posterior_expected(m, self.sample)
        if isinstance(type, dict):
            par = type
        elif isinstance(type, (int, np.integer)):
            self._need_sample()
            par = draw(self.sample, int(type))
        else:
            par = {"mkl": self.mkl, "pmean": self.pmean if self.sample is not None else None, "start": self.start,
                   "pmode": self.pmode, "mle": self.mle}.get(type)
            if par is None:
                raise ValueError(f"predict(type=): no {type!r} estimate in this fit")
        return m.expected(m.eta(par))

    def simulate(self, nsim: int = 1, *, seed=None, output: str = "network"):
        """Networks from the posterior: each from a draw chosen at random, its
        dyads independent given it (``output="adjacency"``: the n x n matrices
        of values)."""
        self._need_sample()
        m = self.model
        rng = np.random.default_rng(seed)
        s = len(self.sample["lpY"])
        out = []
        for _ in range(nsim):
            par = draw(self.sample, int(rng.integers(s)))
            eta = m.eta(par)
            if m.family == "Bernoulli":
                y = (rng.random(eta.shape) < 1 / (1 + np.exp(-eta))).astype(float)
            elif m.family == "binomial":
                y = rng.binomial(m.trials.astype(int), 1 / (1 + np.exp(-eta))).astype(float)
            elif m.family == "Poisson":
                y = rng.poisson(np.exp(eta)).astype(float)
            else:
                y = rng.normal(eta, np.sqrt(par["dispersion"]))
            y = np.where(m.observed, y, 0.0)
            if not m.directed:
                y = np.tril(y, -1) + np.tril(y, -1).T
            out.append(y if output == "adjacency" else self._graph(y))
        return out

    def _graph(self, y: np.ndarray):
        from ._network import to_graph

        m = self.model
        ii, jj = np.nonzero(np.tril(y, -1) if not m.directed else y)
        edges = np.column_stack([ii, jj]).astype(np.uint32) if m.directed else np.column_stack([jj, ii]).astype(np.uint32)
        g = to_graph(m.network, edges)
        if m.response is not None or m.family != "Bernoulli":
            values = y[ii, jj].tolist()
            name = m.response or "weight"
            if hasattr(g, "es"):
                g.es[name] = values
            else:
                import networkx as nx

                nodes = list(g)
                nx.set_edge_attributes(g, {(nodes[a], nodes[b]) if m.directed else (nodes[b], nodes[a]): v
                                           for a, b, v in zip(ii, jj, values)}, name)
        return g

    def gof(self, nsim: int = 100, *, stats=None, seed=None):
        """Goodness of fit, as latentnet's gof(): the degree (in- and
        out-degree), edgewise shared partner and geodesic distance
        distributions of networks simulated from the posterior against the
        network's (the dyads with values, for valued networks)."""
        from ._gof import GofResult, GofTable, _labels, _many

        m = self.model
        if stats is None:
            stats = ["idegree", "odegree", "espartners", "distance"] if m.directed else \
                ["degree", "espartners", "distance"]
        sims = self.simulate(nsim, seed=seed, output="adjacency")

        def edges(y):
            ii, jj = np.nonzero(y if m.directed else np.tril(y, -1))
            return np.column_stack([ii, jj]).astype(np.uint32)

        observed = edges(np.where(m.observed | m.observed.T, m.y, 0.0))
        lists = [edges(y) for y in sims]
        tables = {s: GofTable(s, _labels(m.n, s), _many(m.n, m.directed, [observed], s)[0],
                              _many(m.n, m.directed, lists, s)) for s in stats}
        return GofResult(tables, nsim)

    def mcmc_diagnostics(self, vertex: int = 0) -> LatentDiagnostics:
        """Diagnostics of the draws, as latentnet's mcmc.diagnostics(): the
        log-likelihood, the coefficients, a vertex's positions and random
        effects, across the chains."""
        self._need_sample()
        s = self.sample
        names, cols = ["lpY"], [s["lpY"]]
        names += self.names
        cols += [s["beta"][:, k] for k in range(self.model.p)]
        if self.model.d > 0:
            names += [f"Z.{vertex + 1}.{c + 1}" for c in range(self.model.d)]
            cols += [s["Z"][:, vertex, c] for c in range(self.model.d)]
        for kind in ("sender", "receiver", "sociality"):
            if kind in s:
                names.append(f"{kind}.{vertex + 1}")
                cols.append(s[kind][:, vertex])
        if "dispersion" in s:
            names.append("dispersion")
            cols.append(s["dispersion"])
        values = np.column_stack(cols)
        chains = np.split(values, self.chain_breaks[:-1])
        length = min(len(c) for c in chains)
        return LatentDiagnostics(names, np.stack([c[:length] for c in chains]), np.zeros(len(names)),
                                 self.control.interval)

    def plot(self, what="mkl", ax=None, labels: bool = False, **style):
        """The positions (MKL, by default; ``"pmean"``, ``"start"``...), the
        ties, the clusters' means and their standard deviations' circles,
        and each vertex coloured by its most probable cluster. Needs
        matplotlib."""
        import matplotlib.pyplot as plt

        m = self.model
        if m.d < 1:
            raise ValueError("plot(): the model has no latent space")
        par = {"mkl": self.mkl, "pmean": self.pmean if self.sample is not None else None, "start": self.start,
               "pmode": self.pmode, "mle": self.mle}.get(what) if isinstance(what, str) else what
        if par is None:
            raise ValueError(f"plot(what=): no {what!r} estimate in this fit")
        z = np.asarray(par["Z"])
        if m.d == 1:
            z = np.column_stack([z[:, 0], np.zeros(m.n)])
        elif m.d > 2:
            u, sv, _ = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
            z = u[:, :2] * sv[:2]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        y = np.where(m.observed | m.observed.T, m.y, 0)
        for i, j in zip(*np.nonzero(y if m.directed else np.tril(y, -1))):
            ax.plot(z[[i, j], 0], z[[i, j], 1], color="0.75", lw=0.6, zorder=1)
        colours = None
        if m.G > 0:
            clusters = self.mkl.get("mbc") if what == "mkl" and self.mkl is not None and "mbc" in self.mkl else \
                (self.pmean if self.sample is not None else None)
            if clusters is not None:
                k = np.asarray(clusters["Z.K"]) - 1
                colours = plt.cm.tab10(k % 10)
                if m.d == 2:
                    for g in range(m.G):
                        mu, sd = np.asarray(clusters["Z.mean"])[g], np.sqrt(np.asarray(clusters["Z.var"])[g])
                        ax.add_patch(plt.Circle(mu, sd, fill=False, color=plt.cm.tab10(g % 10), lw=1))
                        ax.plot(*mu, "+", color=plt.cm.tab10(g % 10), ms=10)
        ax.scatter(z[:, 0], z[:, 1], c=colours if colours is not None else "C0", s=style.pop("s", 40), zorder=2,
                   edgecolors="k", linewidths=0.5, **style)
        if labels:
            names = m.network.attributes.get("name") or [str(i) for i in range(m.n)]
            for (a, b), name in zip(z, names):
                ax.annotate(str(name), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_title(f"{what.upper() if isinstance(what, str) else ''} positions")
        return ax


def _completed(model: LatentModel, par: dict) -> dict:
    """A configuration with the variances and clusters it lacks (which lpY doesn't read)."""
    out = dict(par)
    g = model.G
    if model.d > 0:
        out.setdefault("Z.var", np.ones(max(g, 1)))
        if g > 0:
            out.setdefault("Z.mean", np.zeros((g, model.d)))
            out.setdefault("Z.K", np.ones(model.n, dtype=int))
            out.setdefault("Z.pK", np.full(g, 1 / g))
    for kind in ("sender", "receiver", "sociality"):
        if getattr(model, kind):
            out.setdefault(f"{kind}.var", 1.0)
    return out


class LatentDiagnostics:
    """Diagnostics of a latent space model's draws: means, standard
    deviations, standard errors, effective sizes, R-hat across the chains
    and Geweke z-scores."""

    def __init__(self, names, sample, observed, interval):
        from ._diagnostics import McmcDiagnostics

        self._d = McmcDiagnostics(names, sample, observed, interval)
        self.names = self._d.names

    def __getattr__(self, name):
        return getattr(self._d, name)

    def __str__(self) -> str:
        d = self._d
        width = max(map(len, d.names))
        lines = [f"MCMC diagnostics: {d.n_chains} chains x {d.n_samples} draws, {d.interval} iterations apart", "",
                 f"{'':<{width}}  {'Mean':>9}  {'SD':>9}  {'Naive SE':>9}  {'Time-series SE':>14}  "
                 f"{'Eff. size':>9}  {'R-hat':>6}"]
        for i, name in enumerate(d.names):
            lines.append(f"{name:<{width}}  {d.mean[i]:9.3f}  {d.sd[i]:9.3f}  {d.naive_se[i]:9.4f}  "
                         f"{d.timeseries_se[i]:14.4f}  {d.effective_size[i]:9.0f}  {d.rhat[i]:6.3f}")
        return "\n".join(lines)

    __repr__ = __str__

    def plot(self, **options):
        return self._d.plot(**options)


class ErgmmSummary:
    """A latent space model's summary, printed like R's summary(ergmm)."""

    def __init__(self, fit: ErgmmFit, point_est, quantiles, bic_eff_obs):
        self.fit = fit
        self.point_est = list(point_est) if point_est is not None else \
            (["mle"] if fit.mle is not None else []) + (["pmean", "mkl"] if fit.sample is not None else [])
        self.quantiles = tuple(quantiles)
        self.bic = fit.bic(bic_eff_obs) if fit.mkl is not None and bic_eff_obs is not None else None

    def table(self) -> dict:
        """The posterior means' table: estimates, quantiles and 2 min(Pr(>0), Pr(<0))."""
        beta = self.fit.sample["beta"]
        q = np.quantile(beta, self.quantiles, axis=0)
        p0 = 2 * np.minimum(np.mean(beta <= 0, axis=0), np.mean(beta >= 0, axis=0))
        return {"Estimate": beta.mean(axis=0), **{f"{100 * x:g}%": q[k] for k, x in enumerate(self.quantiles)},
                "2*min(Pr(>0),Pr(<0))": p0}

    def __str__(self) -> str:
        fit, m = self.fit, self.fit.model
        lines = ["", "==========================", "Summary of model fit", "==========================", "",
                 f"Formula:   {m.formula}", f"Attribute: {m.response or 'edges'}", f"Model:     {m.family}"]
        names = fit.names
        width = max([len(n) for n in names] + [8])
        if "pmean" in self.point_est and fit.sample is not None:
            c = fit.control
            lines.append(f"MCMC sample of size {len(fit.sample['lpY'])} ({c.n_chains} chains), draws are {c.interval} "
                         f"iterations apart, after burnin of {c.burnin} iterations.")
            if m.p:
                t = self.table()
                head = list(t)
                lines += ["Covariate coefficients posterior means:",
                          f"{'':<{width}}  " + "  ".join(f"{h:>9}" for h in head)]
                for k, name in enumerate(names):
                    p = t[head[-1]][k]
                    cells = [f"{t[h][k]:9.4f}" for h in head[:-1]] + [f"{'< 2.2e-16' if p == 0 else f'{p:.4g}':>9}"]
                    lines.append(f"{name:<{width}}  " + "  ".join(cells))
            pm = fit.pmean
            for key, label in (("dispersion", "Dispersion parameter"), ("sender.var", "Sender effect variance"),
                               ("receiver.var", "Receiver effect variance"),
                               ("sociality.var", "Sociality effect variance")):
                if key in pm:
                    lines.append(f"{label}: {float(pm[key]):.4f}.")
        if "mle" in self.point_est and fit.mle is not None and m.p:
            lines += ["", "Covariate coefficients MLE:"]
            lines += [f"{n:<{width}}  {b:9.4f}" for n, b in zip(names, fit.mle["beta"])]
        if self.bic is not None:
            b = self.bic
            lines += ["", f"Overall BIC:        {b['overall']:.4f}", f"Likelihood BIC:     {b['Y']:.4f}"]
            if "Z" in b:
                lines.append(f"Latent space/clustering BIC:     {b['Z']:.4f}")
            for kind in ("sender", "receiver", "sociality"):
                if kind in b:
                    lines.append(f"{kind.capitalize()} effect BIC:     {b[kind]:.4f}")
        if "mkl" in self.point_est and fit.mkl is not None and m.p:
            lines += ["", "Covariate coefficients MKL:", f"{'':<{width}}  {'Estimate':>9}"]
            lines += [f"{n:<{width}}  {b:9.4f}" for n, b in zip(names, np.atleast_1d(fit.mkl["beta"]))]
        return "\n".join(lines)

    __repr__ = __str__
