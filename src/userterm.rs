//! Terms written in Python: a dyad-independent term's changes, tabulated
//! once for every dyad ("dyadtable"), or a dyad-dependent term's change
//! function, called back at each proposal ("python") with a read-only view
//! of the network, valid only during the call. Callbacks hold the GIL, so
//! they serialize the chains: they are for trying statistics out.

use std::cell::{Cell, RefCell};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::network::Network;
use crate::terms::Term;

thread_local! {
    /// The Python terms of the model being built, by index (`PyModel::new`).
    static CALLBACKS: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` (building a model) with these Python terms available to `build`.
pub fn with_callbacks<T>(callbacks: Vec<Py<PyAny>>, f: impl FnOnce() -> T) -> T {
    CALLBACKS.with(|c| *c.borrow_mut() = callbacks);
    let out = f();
    CALLBACKS.with(|c| c.borrow_mut().clear());
    out
}

/// A dyad-independent term from its table: the change of each statistic on
/// adding the tie i -> j (i < j if undirected), at (i n + j) p + k.
struct DyadTable {
    n: usize,
    p: usize,
    directed: bool,
    table: Vec<f64>,
}

impl Term for DyadTable {
    fn n_stats(&self) -> usize {
        self.p
    }

    fn change(&self, _: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let (a, b) = if !self.directed && i > j { (j, i) } else { (i, j) };
        let at = (a as usize * self.n + b as usize) * self.p;
        out.iter_mut().zip(&self.table[at..at + self.p]).for_each(|(o, t)| *o = sign * t);
    }
}

/// A read-only view of the network, for a Python term's change function:
/// valid during the call only.
#[pyclass(name = "NetworkView", module = "ergmx._core", unsendable)]
pub struct NetView {
    net: *const Network,
    valid: Cell<bool>,
}

impl NetView {
    fn get(&self) -> PyResult<&Network> {
        if !self.valid.get() {
            return Err(PyRuntimeError::new_err("the network view is only valid during the change() call"));
        }
        // SAFETY: the pointer is valid while `valid` is true: during the call that created the view.
        Ok(unsafe { &*self.net })
    }
}

#[pymethods]
impl NetView {
    /// The number of vertices.
    #[getter]
    fn n(&self) -> PyResult<u32> {
        Ok(self.get()?.n())
    }

    #[getter]
    fn directed(&self) -> PyResult<bool> {
        Ok(self.get()?.directed())
    }

    /// The number of ties.
    #[getter]
    fn n_edges(&self) -> PyResult<usize> {
        Ok(self.get()?.n_edges())
    }

    /// Whether i -> j (i -- j) is a tie.
    fn has_edge(&self, i: u32, j: u32) -> PyResult<bool> {
        Ok(self.get()?.has_edge(i, j))
    }

    /// The vertices i is tied to (i's out-neighbours, if directed).
    fn neighbours(&self, i: u32) -> PyResult<Vec<u32>> {
        Ok(self.get()?.out_neighbours(i).to_vec())
    }

    fn out_neighbours(&self, i: u32) -> PyResult<Vec<u32>> {
        Ok(self.get()?.out_neighbours(i).to_vec())
    }

    fn in_neighbours(&self, i: u32) -> PyResult<Vec<u32>> {
        Ok(self.get()?.in_neighbours(i).to_vec())
    }

    /// i's degree (out-degree, if directed).
    fn degree(&self, i: u32) -> PyResult<usize> {
        Ok(self.get()?.out_neighbours(i).len())
    }

    fn in_degree(&self, i: u32) -> PyResult<usize> {
        Ok(self.get()?.in_neighbours(i).len())
    }

    /// The ties, as (i, j) pairs.
    fn edges(&self) -> PyResult<Vec<(u32, u32)>> {
        Ok(self.get()?.edges().to_vec())
    }
}

/// A dyad-dependent term written in Python: its `_change(view, i, j, adding)`
/// returns the change of its statistics on toggling (i, j).
struct PyTerm {
    callback: Py<PyAny>,
    p: usize,
    empty: Vec<f64>,
}

impl Term for PyTerm {
    fn n_stats(&self) -> usize {
        self.p
    }

    fn change(&self, net: &Network, i: u32, j: u32, sign: f64, out: &mut [f64]) {
        let result: PyResult<Vec<f64>> = Python::attach(|py| {
            let view = Py::new(py, NetView { net: net as *const Network, valid: Cell::new(true) })?;
            let value = self.callback.call_method1(py, "_change", (view.clone_ref(py), i, j, sign > 0.0));
            view.borrow(py).valid.set(false);
            value?.extract::<Vec<f64>>(py)
        });
        match result {
            Ok(values) if values.len() == self.p => out.copy_from_slice(&values),
            Ok(values) => panic!("a Python term's change() returned {} values for {} statistics", values.len(), self.p),
            Err(e) => panic!("a Python term's change() raised: {e}"),
        }
    }

    fn empty(&self, _n: u32, _directed: bool, out: &mut [f64]) {
        out.iter_mut().zip(&self.empty).for_each(|(o, e)| *o += e);
    }
}

/// A term written in Python, from its spec: "dyadtable" (reals: the table;
/// ints: [p]) or "python" (ints: [the callback's index, p]; reals: the
/// statistics of the empty network).
pub fn build(n: usize, directed: bool, name: &str, reals: &[f64], ints: &[i64]) -> Result<Box<dyn Term>, String> {
    match name {
        "dyadtable" => {
            let p = *ints.first().ok_or("dyadtable: missing the number of statistics")? as usize;
            if reals.len() != n * n * p {
                return Err(format!("dyadtable: {} values for {n} x {n} dyads and {p} statistics", reals.len()));
            }
            Ok(Box::new(DyadTable { n, p, directed, table: reals.to_vec() }))
        }
        "python" => {
            let (index, p) = match ints {
                [index, p] => (*index as usize, *p as usize),
                _ => return Err("python: bad parameters".into()),
            };
            let callback = CALLBACKS
                .with(|c| Python::attach(|py| c.borrow().get(index).map(|o| o.clone_ref(py))))
                .ok_or("python: the term's object is missing")?;
            let empty = if reals.is_empty() { vec![0.0; p] } else { reals.to_vec() };
            if empty.len() != p {
                return Err("python: the empty network's statistics have the wrong length".into());
            }
            Ok(Box::new(PyTerm { callback, p, empty }))
        }
        _ => Err(format!("unknown term {name:?}")),
    }
}
