# Installation

`ergmx` needs Python 3.9 or newer. There are no prebuilt wheels yet, so it is
built from source, which needs a Rust compiler:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust, from https://rustup.rs
git clone https://github.com/neylsoncrepalde/ergmx.git
cd ergmx
pip install ".[igraph]"
```

`pip` compiles the Rust core with [maturin](https://www.maturin.rs), which
takes about a minute the first time.

## Optional dependencies

`ergmx` itself only needs NumPy and SciPy. Networks come from a graph library,
and plots need matplotlib:

| Extra | Installs | For |
|---|---|---|
| `igraph` | python-igraph | `igraph.Graph` networks and {mod}`ergmx.datasets` |
| `networkx` | networkx | `networkx.Graph` / `DiGraph` networks |
| `plot` | matplotlib | `gof(...).plot()` and `mcmc_diagnostics().plot()` |
| `test` | pytest and all of the above | the test suite |
| `docs` | Sphinx and its extensions | this documentation |

For example, `pip install ".[igraph,plot]"`.

## Development

```bash
python -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop --release --extras test   # build the Rust core, install ergmx
.venv/bin/python -m pytest                          # Python tests
cargo test --release                                # Rust tests
```

Run `maturin develop --release` again after changing Rust code. Without
`--release` the core is built without optimizations, and the MCMC is much
slower.

To build this documentation:

```bash
.venv/bin/maturin develop --release --extras docs
.venv/bin/sphinx-build -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

Then open `docs/_build/html/index.html`.
