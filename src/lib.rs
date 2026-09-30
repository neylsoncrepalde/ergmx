//! Rust core of ergmx: networks, change statistics and the MCMC sampler.

mod network;
mod rng;
mod sampler;
mod terms;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use network::Network;
use rng::Rng;
use terms::TermSpec;

type EdgeArray<'py> = Bound<'py, PyArray2<u32>>;
type MpleData<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

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

    /// Change statistics (dyads x statistics) and ties (0 or 1) of every dyad.
    fn mple_data<'py>(
        &self,
        py: Python<'py>,
        edges: PyReadonlyArray2<u32>,
    ) -> PyResult<MpleData<'py>> {
        let net = self.network(&edges)?;
        let (x, y) = py.detach(|| self.model.mple_data(&net));
        let x = Array2::from_shape_vec((y.len(), self.model.n_stats()), x).unwrap();
        Ok((x.into_pyarray(py), Array1::from_vec(y).into_pyarray(py)))
    }

    /// Runs one Markov chain from each starting network, in parallel threads.
    ///
    /// Returns the sampled statistics (chains x samplesize x statistics), the
    /// last network of each chain and, if `keep_networks`, every sampled network.
    /// `triadic_weight` is the share of triadic proposals (undirected networks only).
    #[pyo3(signature = (starts, theta, burnin, interval, samplesize, seed, keep_networks = false, triadic_weight = 0.0))]
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
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Vec<EdgeArray<'py>>, Vec<Vec<EdgeArray<'py>>>)> {
        if theta.len() != self.model.n_stats() {
            return Err(PyValueError::new_err(format!(
                "expected {} coefficients, got {}",
                self.model.n_stats(),
                theta.len()
            )));
        }
        if interval == 0 || starts.is_empty() {
            return Err(PyValueError::new_err("need interval > 0 and at least one chain"));
        }
        if !(0.0..1.0).contains(&triadic_weight) || (triadic_weight > 0.0 && self.directed) {
            return Err(PyValueError::new_err(
                "triadic_weight must be in [0, 1), and 0 for directed networks",
            ));
        }
        let proposal = sampler::Proposal { triadic_weight };
        let nets = starts.iter().map(|s| self.network(s)).collect::<PyResult<Vec<_>>>()?;
        // Independent streams: nearby seeds must not share chains.
        let mut master = Rng::new(seed);
        let seeds: Vec<u64> = nets.iter().map(|_| master.next_u64()).collect();
        let chains: Vec<sampler::Chain> = py.detach(|| {
            nets.into_par_iter()
                .zip(seeds)
                .map(|(net, chain_seed)| {
                    let mut rng = Rng::new(chain_seed);
                    let (model, theta, proposal) = (&self.model, &theta, &proposal);
                    sampler::run_chain(
                        model, proposal, net, theta, burnin, interval, samplesize, &mut rng, keep_networks,
                    )
                })
                .collect()
        });

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
    m.add_class::<PyModel>()
}
