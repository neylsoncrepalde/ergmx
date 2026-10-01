//! Rust core of ergmx: networks, change statistics and the MCMC sampler.

mod network;
mod rng;
mod sampler;
mod space;
mod terms;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use network::Network;
use rng::Rng;
use space::{Bounds, Preserve};
use terms::TermSpec;

type EdgeArray<'py> = Bound<'py, PyArray2<u32>>;

create_exception!(_core, DensityGuardError, PyRuntimeError, "A simulated network exceeded max_edges.");
type MpleData<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
/// Degree bounds per vertex: (min_out, max_out, min_in, max_in).
type DegreeBounds = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

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
    /// `free` is an n x n row-major boolean mask of the dyads that may change
    /// (upper triangle if undirected); `bounds` are (min_out, max_out, min_in,
    /// max_in) per vertex; `preserve` is "", "degrees", "odegrees" or "idegrees".
    #[new]
    #[pyo3(signature = (n, directed, free = None, bounds = None, preserve = ""))]
    fn new(
        n: u32,
        directed: bool,
        free: Option<PyReadonlyArray1<bool>>,
        bounds: Option<DegreeBounds>,
        preserve: &str,
    ) -> PyResult<Self> {
        let preserve = match preserve {
            "" => Preserve::Nothing,
            "degrees" => Preserve::Degrees,
            "odegrees" => Preserve::OutDegrees,
            "idegrees" => Preserve::InDegrees,
            other => return Err(PyValueError::new_err(format!("unknown degree constraint {other:?}"))),
        };
        if !directed && matches!(preserve, Preserve::OutDegrees | Preserve::InDegrees) {
            return Err(PyValueError::new_err("odegrees and idegrees need a directed network"));
        }
        let bounds = match bounds {
            None => None,
            Some((min_out, max_out, min_in, max_in)) => {
                if [&min_out, &max_out, &min_in, &max_in].iter().any(|b| b.len() != n as usize) {
                    return Err(PyValueError::new_err("degree bounds need one value per vertex"));
                }
                Some(Bounds { min_out, max_out, min_in, max_in })
            }
        };
        let mask = free.as_ref().map(|f| f.as_slice()).transpose()?;
        let space = space::Space::new(n, directed, mask, bounds, preserve).map_err(PyValueError::new_err)?;
        Ok(Self { space })
    }

    /// Number of dyads that may change.
    #[getter]
    fn n_free(&self) -> u64 {
        self.space.n_free()
    }
}

/// An ERGM for networks with `n` vertices: its terms, ready to compute
/// statistics and sample networks.
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
    fn new(n: u32, directed: bool, terms: Vec<TermSpec>) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err("networks need at least 2 vertices"));
        }
        let model = terms::Model::new(n, directed, &terms).map_err(PyValueError::new_err)?;
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

    /// Change statistics (dyads x statistics) and ties (0 or 1) of every free
    /// dyad (every dyad, without a space).
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
        let (x, y) = py.detach(|| self.model.mple_data(net, space));
        let x = Array2::from_shape_vec((y.len(), self.model.n_stats()), x).unwrap();
        Ok((x.into_pyarray(py), Array1::from_vec(y).into_pyarray(py)))
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
        let mut master = Rng::new(seed);
        let seeds: Vec<u64> = nets.iter().map(|_| master.next_u64()).collect();
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
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PySpace>()?;
    m.add("DensityGuardError", m.py().get_type::<DensityGuardError>())
}
