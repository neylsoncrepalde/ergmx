//! Rust core of ergmx: networks, change statistics and the MCMC sampler.

mod durational;
mod gof;
mod layers;
mod multilevel;
mod network;
mod partners;
mod rng;
mod sampler;
mod space;
mod terms;
mod userterm;
mod valued;
mod vocab;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use network::Network;
use rng::Rng;
use space::{Bounds, Preserve};
use terms::{Layout, TermSpec};

type EdgeArray<'py> = Bound<'py, PyArray2<u32>>;

create_exception!(_core, DensityGuardError, PyRuntimeError, "A simulated network exceeded max_edges.");
type MpleData<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<u32>>);
type MpleTable<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
/// A dynamic simulation's networks, statistics (time steps x statistics) and MCMC steps.
type SeriesData<'py> = (Vec<EdgeArray<'py>>, Bound<'py, PyArray2<f64>>, Vec<u64>, Vec<(u32, u32, u32)>);
/// Degree bounds per vertex: (min_out, max_out, min_in, max_in).
type DegreeBounds = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);
/// bd(attribs=): the number of classes, the vertices x classes membership
/// matrix and the bounds (min_out, max_out, min_in, max_in), each vertices x classes.
type ClassBoundsData = (usize, Vec<bool>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn to_edges(array: &PyReadonlyArray2<u32>) -> PyResult<Vec<(u32, u32)>> {
    let a = array.as_array();
    if a.ncols() != 2 {
        return Err(PyValueError::new_err("edges must be an array of shape (m, 2)"));
    }
    Ok(a.rows().into_iter().map(|r| (r[0], r[1])).collect())
}

fn to_array<'py>(py: Python<'py>, edges: &[(u32, u32)]) -> EdgeArray<'py> {
    let flat = edges.iter().flat_map(|&(i, j)| [i, j]).collect();
    Array2::from_shape_vec((edges.len(), 2), flat).unwrap().into_pyarray(py)
}

/// The sample space of a model: which dyads may change, bounds on degrees,
/// and whether degrees are preserved.
#[pyclass(name = "Space", module = "ergmx._core", frozen)]
struct PySpace {
    space: space::Space,
}

#[pymethods]
impl PySpace {
    /// The dyads that may change: those within a group (`groups`, one per
    /// vertex, negative for none) and, if bipartite, between the modes
    /// (`modes`, true for the first mode's vertices), but the fixed ones
    /// (`fixed_mask`, n x n row-major, and the pairs `fixed`); if `only` is
    /// given, only those of its pairs. `bounds` are (min_out, max_out, min_in,
    /// max_in) per vertex, and `class_bounds` those of bd(attribs=); `preserve`
    /// is "", "degrees", "odegrees", "idegrees", "edges", "b1degrees",
    /// "b2degrees" (these two with `first_mode`, the vertices of the first
    /// mode), "degreedist", "odegreedist" or "idegreedist".
    #[new]
    #[pyo3(signature = (n, directed, groups = None, modes = None, fixed_mask = None, fixed = None, only = None, bounds = None, preserve = "", first_mode = None, class_bounds = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: u32,
        directed: bool,
        groups: Option<PyReadonlyArray1<i64>>,
        modes: Option<PyReadonlyArray1<bool>>,
        fixed_mask: Option<PyReadonlyArray1<bool>>,
        fixed: Option<PyReadonlyArray2<u32>>,
        only: Option<PyReadonlyArray2<u32>>,
        bounds: Option<DegreeBounds>,
        preserve: &str,
        first_mode: Option<Vec<bool>>,
        class_bounds: Option<ClassBoundsData>,
    ) -> PyResult<Self> {
        let preserve = match preserve {
            "" => Preserve::Nothing,
            "degrees" => Preserve::Degrees,
            "odegrees" => Preserve::OutDegrees,
            "idegrees" => Preserve::InDegrees,
            "edges" => Preserve::Edges,
            "b1degrees" => Preserve::FirstModeDegrees,
            "b2degrees" => Preserve::SecondModeDegrees,
            "degreedist" => Preserve::DegreeDist,
            "odegreedist" => Preserve::OutDegreeDist,
            "idegreedist" => Preserve::InDegreeDist,
            other => return Err(PyValueError::new_err(format!("unknown degree constraint {other:?}"))),
        };
        if !directed && matches!(preserve, Preserve::OutDegrees | Preserve::InDegrees | Preserve::OutDegreeDist | Preserve::InDegreeDist) {
            return Err(PyValueError::new_err("odegrees, idegrees, odegreedist and idegreedist need a directed network"));
        }
        let mode_degrees = matches!(preserve, Preserve::FirstModeDegrees | Preserve::SecondModeDegrees);
        if mode_degrees != first_mode.is_some() || first_mode.as_ref().is_some_and(|m| m.len() != n as usize) {
            return Err(PyValueError::new_err("b1degrees and b2degrees need the first mode's vertices"));
        }
        let classes = match class_bounds {
            None => None,
            Some((k, attribs, min_out, max_out, min_in, max_in)) => {
                let size = n as usize * k;
                if [attribs.len(), min_out.len(), max_out.len(), min_in.len(), max_in.len()].iter().any(|&l| l != size) {
                    return Err(PyValueError::new_err("bd(attribs=): the matrices must be vertices x classes"));
                }
                Some(space::ClassBounds { k, attribs, min_out, max_out, min_in, max_in })
            }
        };
        let bounds = match (bounds, classes) {
            (None, None) => None,
            (Some((min_out, max_out, min_in, max_in)), classes) => {
                if [&min_out, &max_out, &min_in, &max_in].iter().any(|b| b.len() != n as usize) {
                    return Err(PyValueError::new_err("degree bounds need one value per vertex"));
                }
                Some(Bounds { min_out, max_out, min_in, max_in, classes })
            }
            (None, classes) => {
                let (none, all) = (vec![0; n as usize], vec![u32::MAX; n as usize]);
                Some(Bounds { min_out: none.clone(), max_out: all.clone(), min_in: none, max_in: all, classes })
            }
        };
        let pairs = |a: &Option<PyReadonlyArray2<u32>>| -> PyResult<Option<Vec<(u32, u32)>>> {
            a.as_ref().map(|a| {
                let a = a.as_array();
                if a.ncols() != 2 {
                    return Err(PyValueError::new_err("dyads must be two columns of vertices"));
                }
                Ok(a.rows().into_iter().map(|r| (r[0], r[1])).collect())
            }).transpose()
        };
        let (fixed, only) = (pairs(&fixed)?.unwrap_or_default(), pairs(&only)?);
        let restriction = space::Restriction {
            groups: groups.as_ref().map(|g| g.as_slice()).transpose()?,
            modes: modes.as_ref().map(|m| m.as_slice()).transpose()?,
            fixed_mask: fixed_mask.as_ref().map(|f| f.as_slice()).transpose()?,
            fixed: &fixed,
            only: only.as_deref(),
        };
        let mut space = space::Space::new(n, directed, restriction, bounds, preserve).map_err(PyValueError::new_err)?;
        space.first_mode = first_mode;
        Ok(Self { space })
    }

    /// Number of dyads that may change.
    #[getter]
    fn n_free(&self) -> u64 {
        self.space.n_free()
    }
}

/// An ERGM for networks with `n` vertices: its terms, ready to compute
/// statistics and sample networks. For networks combined from several,
/// `blocks` are the (first vertex, size) of each, and `prev` their previous
/// networks (edges numbered within the block) for tergm's operators.
#[pyclass(name = "Model", module = "ergmx._core", frozen)]
struct PyModel {
    model: terms::Model,
    n: u32,
    directed: bool,
}

impl PyModel {
    fn network(&self, edges: &PyReadonlyArray2<u32>) -> PyResult<Network> {
        Network::from_edges(self.n, self.directed, &to_edges(edges)?).map_err(PyValueError::new_err)
    }
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (n, directed, terms, blocks = None, prev = None, callbacks = None))]
    fn new(
        n: u32,
        directed: bool,
        terms: Vec<TermSpec>,
        blocks: Option<Vec<(u32, u32)>>,
        prev: Option<Vec<PyReadonlyArray2<u32>>>,
        callbacks: Option<Vec<Py<PyAny>>>,
    ) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err("networks need at least 2 vertices"));
        }
        let layout = blocks.map(|b| Layout::new(&b, n)).transpose().map_err(PyValueError::new_err)?;
        let prev = match (&layout, prev) {
            (_, None) => Vec::new(),
            (None, Some(_)) => return Err(PyValueError::new_err("previous networks need blocks")),
            (Some(layout), Some(prev)) => {
                if prev.len() != layout.len() {
                    return Err(PyValueError::new_err("need one previous network per block"));
                }
                prev.iter()
                    .enumerate()
                    .map(|(k, edges)| Network::from_edges(layout.size(k), directed, &to_edges(edges)?)
                        .map_err(PyValueError::new_err))
                    .collect::<PyResult<Vec<_>>>()?
            }
        };
        let model = userterm::with_callbacks(callbacks.unwrap_or_default(), || {
            terms::Model::new(n, directed, &terms, layout, prev)
        })
        .map_err(PyValueError::new_err)?;
        Ok(Self { model, n, directed })
    }

    #[getter]
    fn n_stats(&self) -> usize {
        self.model.n_stats()
    }

    /// Statistics of the network with these edges.
    fn summary(&self, edges: PyReadonlyArray2<u32>) -> PyResult<Vec<f64>> {
        Ok(self.model.summary(&self.network(&edges)?))
    }

    /// Change statistics (dyads x statistics), ties (0 or 1) and the dyads
    /// themselves (dyads x 2) of every free dyad (every dyad, without a space).
    #[pyo3(signature = (edges, space = None))]
    fn mple_data<'py>(
        &self,
        py: Python<'py>,
        edges: PyReadonlyArray2<u32>,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<MpleData<'py>> {
        let net = self.network(&edges)?;
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let (x, y, pairs) = py.detach(|| self.model.mple_data(net, space));
        let x = Array2::from_shape_vec((y.len(), self.model.n_stats()), x).unwrap();
        let pairs = Array2::from_shape_vec((y.len(), 2), pairs.into_iter().flat_map(|(i, j)| [i, j]).collect()).unwrap();
        Ok((x.into_pyarray(py), Array1::from_vec(y).into_pyarray(py), pairs.into_pyarray(py)))
    }

    /// The MPLE data compressed: the distinct rows of change statistics,
    /// with their response (0 or 1) and number of dyads.
    #[pyo3(signature = (edges, space = None))]
    fn mple_table<'py>(
        &self,
        py: Python<'py>,
        edges: PyReadonlyArray2<u32>,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<MpleTable<'py>> {
        let net = self.network(&edges)?;
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let (x, y, w) = py.detach(|| self.model.mple_table(net, space));
        let x = Array2::from_shape_vec((y.len(), self.model.n_stats()), x).unwrap();
        Ok((x.into_pyarray(py), Array1::from_vec(y).into_pyarray(py), Array1::from_vec(w).into_pyarray(py)))
    }

    /// Runs one Markov chain from each starting network, in parallel threads.
    ///
    /// Returns the sampled statistics (chains x samplesize x statistics), the
    /// last network of each chain and, if `keep_networks`, every sampled network.
    /// `triadic_weight` is the share of triadic proposals. If a network gets more
    /// than `max_edges` edges, the chains stop and DensityGuardError is raised.
    /// `chain_thetas`, one coefficient vector per chain, replaces `theta`.
    /// `space` restricts the networks sampled; the starting networks must be in it.
    #[pyo3(signature = (starts, theta, burnin, interval, samplesize, seed, keep_networks = false, triadic_weight = 0.0, max_edges = None, chain_thetas = None, space = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        starts: Vec<PyReadonlyArray2<u32>>,
        theta: Vec<f64>,
        burnin: u64,
        interval: u64,
        samplesize: usize,
        seed: u64,
        keep_networks: bool,
        triadic_weight: f64,
        max_edges: Option<usize>,
        chain_thetas: Option<Vec<Vec<f64>>>,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Vec<EdgeArray<'py>>, Vec<Vec<EdgeArray<'py>>>)> {
        let thetas = chain_thetas.unwrap_or_else(|| vec![theta; starts.len()]);
        if thetas.len() != starts.len() {
            return Err(PyValueError::new_err("need one coefficient vector per chain"));
        }
        if let Some(bad) = thetas.iter().find(|t| t.len() != self.model.n_stats()) {
            return Err(PyValueError::new_err(format!(
                "expected {} coefficients, got {}",
                self.model.n_stats(),
                bad.len()
            )));
        }
        if interval == 0 || starts.is_empty() {
            return Err(PyValueError::new_err("need interval > 0 and at least one chain"));
        }
        if !(0.0..1.0).contains(&triadic_weight) {
            return Err(PyValueError::new_err("triadic_weight must be in [0, 1)"));
        }
        let max_edges = max_edges.unwrap_or(usize::MAX);
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let proposal = sampler::Proposal { triadic_weight, max_edges, space };
        let nets = starts.iter().map(|s| self.network(s)).collect::<PyResult<Vec<_>>>()?;
        // Independent streams: nearby seeds must not share chains.
        let mut seeder = Rng::new(seed);
        let seeds: Vec<u64> = nets.iter().map(|_| seeder.next_u64()).collect();
        let chains: Vec<sampler::Chain> = py.detach(|| {
            nets.into_par_iter()
                .zip(seeds)
                .zip(&thetas)
                .map(|((net, chain_seed), theta)| {
                    let mut rng = Rng::new(chain_seed);
                    let (model, proposal) = (&self.model, &proposal);
                    sampler::run_chain(
                        model, proposal, net, theta, burnin, interval, samplesize, &mut rng, keep_networks,
                    )
                })
                .collect()
        });

        if chains.iter().any(|c| c.exceeded) {
            return Err(DensityGuardError::new_err(format!(
                "a simulated network has more than {max_edges} edges"
            )));
        }
        let shape = (chains.len(), samplesize, self.model.n_stats());
        let stats = chains.iter().flat_map(|c| c.stats.iter().copied()).collect();
        let stats = Array3::from_shape_vec(shape, stats).unwrap().into_pyarray(py);
        let last = chains.iter().map(|c| to_array(py, c.last.edges())).collect();
        let networks = chains
            .iter()
            .map(|c| c.networks.iter().map(|e| to_array(py, e)).collect())
            .collect();
        Ok((stats, last, networks))
    }

    /// Simulated annealing (ergm's SAN) from the network `start`: the
    /// statistics `targeted` (their indices) towards `target`, with `offsets`
    /// (statistic, coefficient) and the weights `weights` (targeted x
    /// targeted, row-major), at temperature `tau`, for `nsteps` proposals and
    /// `samplesize` samples. Returns the last network, and for each sample the
    /// deviations from the targets and the sums of the proposed changes
    /// (samples x targeted, each).
    #[pyo3(signature = (start, targeted, target, offsets, weights, tau, nsteps, samplesize, seed, triadic_weight = 0.0, space = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn san<'py>(
        &self,
        py: Python<'py>,
        start: PyReadonlyArray2<u32>,
        targeted: Vec<usize>,
        target: Vec<f64>,
        offsets: Vec<(usize, f64)>,
        weights: Vec<f64>,
        tau: f64,
        nsteps: u64,
        samplesize: usize,
        seed: u64,
        triadic_weight: f64,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<(EdgeArray<'py>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let p = self.model.n_stats();
        let q = targeted.len();
        if target.len() != q || weights.len() != q * q {
            return Err(PyValueError::new_err("need one target per targeted statistic, and q x q weights"));
        }
        if targeted.iter().chain(offsets.iter().map(|(k, _)| k)).any(|&k| k >= p) {
            return Err(PyValueError::new_err(format!("statistics are numbered 0 to {}", p.saturating_sub(1))));
        }
        if !(0.0..1.0).contains(&triadic_weight) || tau < 0.0 {
            return Err(PyValueError::new_err("triadic_weight must be in [0, 1), and tau non-negative"));
        }
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let proposal = sampler::Proposal { triadic_weight, max_edges: usize::MAX, space };
        let net = self.network(&start)?;
        let settings = sampler::SanSettings {
            targeted: &targeted,
            target: &target,
            offsets: &offsets,
            weights: &weights,
            tau,
            nsteps,
            samplesize,
        };
        let result = py.detach(|| sampler::run_san(&self.model, &proposal, net, &settings, &mut Rng::new(seed)));
        let rows = result.deviations.len() / q.max(1);
        let deviations = Array2::from_shape_vec((rows, q), result.deviations).unwrap().into_pyarray(py);
        let proposed = Array2::from_shape_vec((rows, q), result.proposed).unwrap().into_pyarray(py);
        Ok((to_array(py, result.last.edges()), deviations, proposed))
    }

    /// Runs `slices` time steps of a model with tergm's operators, from each
    /// starting network (one replication each, in parallel threads): at each
    /// step, the previous networks are the networks at its start, and the
    /// chain runs until the number of dyads that changed stops growing, as
    /// tergm's MCMC.burnin.min, .max, .pval and .add (`min_steps`...). For
    /// durational terms, `ages` are each start's ties' ages, as (i, j, age)
    /// rows (missing ones are 1). Returns, for each run, each step's network,
    /// the statistics, each step's number of proposals and, with durational
    /// terms, the last network's ties' ages.
    #[pyo3(signature = (starts, theta, slices, seed, min_steps = 1000, max_steps = 100_000, pval = 0.5, add = 1.0, triadic_weight = 0.0, max_edges = None, space = None, ages = None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate_series<'py>(
        &self,
        py: Python<'py>,
        starts: Vec<PyReadonlyArray2<u32>>,
        theta: Vec<f64>,
        slices: usize,
        seed: u64,
        min_steps: u64,
        max_steps: u64,
        pval: f64,
        add: f64,
        triadic_weight: f64,
        max_edges: Option<usize>,
        space: Option<PyRef<'py, PySpace>>,
        ages: Option<Vec<Vec<(u32, u32, u32)>>>,
    ) -> PyResult<Vec<SeriesData<'py>>> {
        let Some(layout) = self.model.layout() else {
            return Err(PyValueError::new_err("dynamic simulation needs a model with blocks"));
        };
        let ages = ages.unwrap_or_else(|| vec![Vec::new(); starts.len()]);
        if ages.len() != starts.len() {
            return Err(PyValueError::new_err("need the ages of each start's ties"));
        }
        let block_ages: Vec<_> = ages.iter().map(|rows| layout.block_ages(self.directed, rows)).collect();
        if theta.len() != self.model.n_stats() {
            return Err(PyValueError::new_err(format!(
                "expected {} coefficients, got {}",
                self.model.n_stats(),
                theta.len()
            )));
        }
        if !(0.0..1.0).contains(&triadic_weight) || min_steps == 0 || max_steps < min_steps {
            return Err(PyValueError::new_err("need triadic_weight in [0, 1) and 0 < min_steps <= max_steps"));
        }
        let max_edges = max_edges.unwrap_or(usize::MAX);
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let proposal = sampler::Proposal { triadic_weight, max_edges, space };
        let rule = sampler::StepRule { min: min_steps, max: max_steps, pval, add };
        let nets = starts.iter().map(|s| self.network(s)).collect::<PyResult<Vec<_>>>()?;
        let mut seeder = Rng::new(seed);
        let seeds: Vec<u64> = nets.iter().map(|_| seeder.next_u64()).collect();
        let runs: Vec<sampler::Series> = py.detach(|| {
            nets.into_par_iter()
                .zip(seeds)
                .zip(block_ages)
                .map(|((net, chain_seed), ages)| {
                    let mut rng = Rng::new(chain_seed);
                    sampler::run_series(&self.model, &proposal, net, &theta, slices, &rule, &mut rng, ages)
                })
                .collect()
        });
        if runs.iter().any(|r| r.exceeded) {
            return Err(DensityGuardError::new_err(format!(
                "a simulated network has more than {max_edges} edges"
            )));
        }
        let p = self.model.n_stats();
        Ok(runs
            .into_iter()
            .map(|r| {
                let stats = Array2::from_shape_vec((r.steps.len(), p), r.stats).unwrap().into_pyarray(py);
                let ages = layout.joined_ages(&r.ages);
                (r.networks.iter().map(|e| to_array(py, e)).collect(), stats, r.steps, ages)
            })
            .collect())
    }
}

/// A goodness-of-fit distribution ("espartners", "dspartners" or
/// "distance") of each network with these edge lists, in parallel threads:
/// networks x values.
#[pyfunction]
fn gof_distribution<'py>(
    py: Python<'py>,
    n: u32,
    directed: bool,
    edge_lists: Vec<PyReadonlyArray2<u32>>,
    stat: &str,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    use rayon::prelude::*;
    let f: fn(&network::Network) -> Vec<u64> = match stat {
        "espartners" => gof::espartners,
        "dspartners" => gof::dspartners,
        "distance" => gof::distances,
        other => return Err(PyValueError::new_err(format!("unknown distribution {other:?}"))),
    };
    let nets = edge_lists
        .iter()
        .map(|e| {
            let pairs: Vec<(u32, u32)> = e.as_array().rows().into_iter().map(|r| (r[0], r[1])).collect();
            network::Network::from_edges(n, directed, &pairs).map_err(PyValueError::new_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let rows: Vec<Vec<u64>> = py.detach(|| nets.par_iter().map(f).collect());
    let width = rows.first().map_or(0, Vec::len);
    let flat: Vec<u64> = rows.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((flat.len() / width.max(1), width), flat).unwrap().into_pyarray(py))
}

/// A valued network's dyads with nonzero values, as rows (i, j, value).
fn to_triples(values: &PyReadonlyArray2<f64>) -> PyResult<Vec<(u32, u32, f64)>> {
    let a = values.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err("valued networks are three columns: i, j and the value"));
    }
    Ok(a.rows().into_iter().map(|r| (r[0] as u32, r[1] as u32, r[2])).collect())
}

fn triples_array<'py>(py: Python<'py>, triples: &[(u32, u32, f64)]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = triples.iter().flat_map(|&(i, j, v)| [i as f64, j as f64, v]).collect();
    Array2::from_shape_vec((triples.len(), 3), flat).unwrap().into_pyarray(py)
}

/// A valued reference measure from its name and parameters (for "StdNormal",
/// the standard deviation of its proposals' steps), checking `p0`.
fn wt_reference(reference: &(String, Vec<f64>), p0: Option<f64>) -> PyResult<valued::Reference> {
    if p0.is_some_and(|p| !(0.0..1.0).contains(&p)) {
        return Err(PyValueError::new_err("p0 must be in [0, 1)"));
    }
    Ok(match (reference.0.as_str(), reference.1.as_slice()) {
        ("Poisson", []) => valued::Reference::Poisson,
        ("Geometric", []) => valued::Reference::Geometric,
        ("Binomial", [trials]) => valued::Reference::Binomial(*trials),
        ("DiscUnif", [a, b]) => valued::Reference::DiscUnif(*a, *b),
        ("Unif", [a, b]) => valued::Reference::Unif(*a, *b),
        ("StdNormal", [sd]) if *sd > 0.0 => valued::Reference::StdNormal(*sd),
        (other, _) => return Err(PyValueError::new_err(format!("unknown reference {other:?}"))),
    })
}

type WtSimulation<'py> = (Bound<'py, PyArray3<f64>>, Vec<Bound<'py, PyArray2<f64>>>, Vec<Vec<Bound<'py, PyArray2<f64>>>>);

/// A valued ERGM (ergm.count) for networks with `n` vertices: its terms
/// (`dyads` is the number of dyads, for the terms that count the dyads
/// with a value), to compute statistics and sample valued networks.
#[pyclass(name = "WtModel", module = "ergmx._core", frozen)]
struct PyWtModel {
    model: valued::WtModel,
    n: u32,
    directed: bool,
}

#[pymethods]
impl PyWtModel {
    #[new]
    fn new(n: u32, directed: bool, terms: Vec<TermSpec>, dyads: f64) -> PyResult<Self> {
        if n < 3 {
            return Err(PyValueError::new_err("valued networks need at least 3 vertices"));
        }
        let model = valued::WtModel::new(n, directed, &terms, dyads).map_err(PyValueError::new_err)?;
        Ok(Self { model, n, directed })
    }

    #[getter]
    fn n_stats(&self) -> usize {
        self.model.n_stats()
    }

    /// Statistics of the valued network with these nonzero values (rows i, j, value).
    fn summary(&self, values: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
        let net = valued::WtNetwork::from_values(self.n, self.directed, &to_triples(&values)?)
            .map_err(PyValueError::new_err)?;
        Ok(self.model.summary(&net))
    }

    /// Runs one Markov chain from each starting valued network, in parallel
    /// threads, with the reference measure `reference` ("Poisson",
    /// "Geometric", "Binomial" with the trials, "DiscUnif" or "Unif" with
    /// its bounds, "StdNormal" with the standard deviation of its proposals'
    /// steps) and ergm.count's DiscTNT proposal (`p0`, the share of jumps of
    /// a nonzero dyad to 0) or Disc (`p0=None`; continuous references
    /// ignore `p0`). Returns the statistics
    /// (chains x samples x statistics), each chain's last network, and the
    /// sampled networks if `keep_networks`.
    #[pyo3(signature = (starts, theta, burnin, interval, samplesize, seed, reference, p0 = Some(0.2), keep_networks = false, max_nonzero = None, chain_thetas = None, space = None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        starts: Vec<PyReadonlyArray2<f64>>,
        theta: Vec<f64>,
        burnin: u64,
        interval: u64,
        samplesize: usize,
        seed: u64,
        reference: (String, Vec<f64>),
        p0: Option<f64>,
        keep_networks: bool,
        max_nonzero: Option<usize>,
        chain_thetas: Option<Vec<Vec<f64>>>,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<WtSimulation<'py>> {
        let reference = wt_reference(&reference, p0)?;
        let thetas = chain_thetas.unwrap_or_else(|| vec![theta; starts.len()]);
        if thetas.len() != starts.len() || thetas.iter().any(|t| t.len() != self.model.n_stats()) {
            return Err(PyValueError::new_err(format!("need {} coefficients per chain", self.model.n_stats())));
        }
        if interval == 0 || starts.is_empty() {
            return Err(PyValueError::new_err("need interval > 0 and at least one chain"));
        }
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let proposal = valued::WtProposal { reference, p0, space, max_nonzero: max_nonzero.unwrap_or(usize::MAX) };
        let nets = starts
            .iter()
            .map(|s| valued::WtNetwork::from_values(self.n, self.directed, &to_triples(s)?).map_err(PyValueError::new_err))
            .collect::<PyResult<Vec<_>>>()?;
        let mut seeder = Rng::new(seed);
        let seeds: Vec<u64> = nets.iter().map(|_| seeder.next_u64()).collect();
        let chains: Vec<valued::WtChain> = py.detach(|| {
            nets.into_par_iter()
                .zip(seeds)
                .zip(&thetas)
                .map(|((net, chain_seed), theta)| {
                    let mut rng = Rng::new(chain_seed);
                    valued::run_wt_chain(&self.model, &proposal, net, theta, burnin, interval, samplesize, &mut rng, keep_networks)
                })
                .collect()
        });
        if chains.iter().any(|c| c.exceeded) {
            return Err(DensityGuardError::new_err("a simulated network has too many nonzero dyads"));
        }
        let shape = (chains.len(), samplesize, self.model.n_stats());
        let stats = chains.iter().flat_map(|c| c.stats.iter().copied()).collect();
        let stats = Array3::from_shape_vec(shape, stats).unwrap().into_pyarray(py);
        let last = chains.iter().map(|c| triples_array(py, &c.last.triples())).collect();
        let networks = chains.iter().map(|c| c.networks.iter().map(|t| triples_array(py, t)).collect()).collect();
        Ok((stats, last, networks))
    }

    /// Simulated annealing (ergm's SAN) of the valued network `start`, as
    /// `Model.san`, with the valued proposals of `simulate`. Returns the last
    /// network's nonzero values, and for each sample the deviations from the
    /// targets and the sums of the proposed changes.
    #[pyo3(signature = (start, targeted, target, offsets, weights, tau, nsteps, samplesize, seed, reference, p0 = Some(0.2), space = None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn san<'py>(
        &self,
        py: Python<'py>,
        start: PyReadonlyArray2<f64>,
        targeted: Vec<usize>,
        target: Vec<f64>,
        offsets: Vec<(usize, f64)>,
        weights: Vec<f64>,
        tau: f64,
        nsteps: u64,
        samplesize: usize,
        seed: u64,
        reference: (String, Vec<f64>),
        p0: Option<f64>,
        space: Option<PyRef<'py, PySpace>>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let p = self.model.n_stats();
        let q = targeted.len();
        if target.len() != q || weights.len() != q * q {
            return Err(PyValueError::new_err("need one target per targeted statistic, and q x q weights"));
        }
        if targeted.iter().chain(offsets.iter().map(|(k, _)| k)).any(|&k| k >= p) {
            return Err(PyValueError::new_err(format!("statistics are numbered 0 to {}", p.saturating_sub(1))));
        }
        if tau < 0.0 {
            return Err(PyValueError::new_err("tau must be non-negative"));
        }
        let reference = wt_reference(&reference, p0)?;
        let all = space::Space::unconstrained(self.n, self.directed);
        let space = space.as_ref().map_or(&all, |s| &s.space);
        let proposal = valued::WtProposal { reference, p0, space, max_nonzero: usize::MAX };
        let net = valued::WtNetwork::from_values(self.n, self.directed, &to_triples(&start)?).map_err(PyValueError::new_err)?;
        let settings = sampler::SanSettings {
            targeted: &targeted,
            target: &target,
            offsets: &offsets,
            weights: &weights,
            tau,
            nsteps,
            samplesize,
        };
        let result = py.detach(|| valued::run_wt_san(&self.model, &proposal, net, &settings, &mut Rng::new(seed)));
        let rows = result.deviations.len() / q.max(1);
        let deviations = Array2::from_shape_vec((rows, q), result.deviations).unwrap().into_pyarray(py);
        let proposed = Array2::from_shape_vec((rows, q), result.proposed).unwrap().into_pyarray(py);
        Ok((triples_array(py, &result.last.triples()), deviations, proposed))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gof_distribution, m)?)?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyWtModel>()?;
    m.add_class::<userterm::NetView>()?;
    m.add_class::<PySpace>()?;
    m.add("DensityGuardError", m.py().get_type::<DensityGuardError>())
}
