# Installation

`ergmx` is a Python package with a core written in Rust. How you install it
decides whether you need Rust:

| You install | You need | Rust compiler |
|---|---|---|
| A release from PyPI | [uv](https://docs.astral.sh/uv/) (or pip) | No: the compiled core is inside the wheel |
| From GitHub, or on a platform without a wheel | uv (or pip) and a C linker | Yes, but maturin downloads it if you don't have it |
| For development | uv, rustup and a C linker | Yes, the version pinned in `rust-toolchain.toml` |

`ergmx` needs Python 3.11 or newer. uv installs Python for you if needed.
For what Rust, cargo and maturin are, and why `ergmx` uses them, see
[](under-the-hood.md).

## Install uv

[uv](https://docs.astral.sh/uv/) manages Python versions, environments and
packages. On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Homebrew (`brew install uv`), pipx and other options are in
[uv's installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Install ergmx from PyPI

Released versions of `ergmx` come as *wheels*: one per platform, with the Rust
core already compiled, for Linux (x86-64 and ARM, glibc and musl), macOS
(Intel and Apple Silicon) and Windows (x86-64). Each wheel serves every
Python from 3.11 on. Installing one needs no compiler.

In a uv project (a folder with a `pyproject.toml`, created with `uv init`):

```bash
uv add "ergmx[igraph,plot]"
```

In any environment:

```bash
uv pip install "ergmx[igraph,plot]"     # or: pip install "ergmx[igraph,plot]"
```

To run a script without creating an environment:

```bash
uv run --with "ergmx[igraph]" python my_analysis.py
```

### Optional dependencies

`ergmx` itself only needs NumPy and SciPy. Networks come from a graph library,
and plots need matplotlib. Pick the extras you use:

| Extra | Installs | For |
|---|---|---|
| `igraph` | python-igraph | `igraph.Graph` networks and {mod}`ergmx.datasets` |
| `networkx` | networkx | `networkx.Graph` / `DiGraph` networks |
| `plot` | matplotlib | `gof(...).plot()` and `mcmc_diagnostics().plot()` |
| `pandas` | pandas | `.to_frame()` of fits, predictions and tables of results |

## From GitHub

```bash
uv add "ergmx[igraph,plot] @ git+https://github.com/neylsoncrepalde/ergmx"
# or, in any environment:
uv pip install "ergmx[igraph,plot] @ git+https://github.com/neylsoncrepalde/ergmx"
```

Add `@v0.3.1` (a tag), `@main` or a commit after the URL to pick a version.
This builds `ergmx` from source, as does installing on a platform without a
wheel: uv runs [maturin](https://www.maturin.rs), the build tool of the Rust
core, which compiles it with optimizations. The first build takes under a
minute on a recent computer; uv caches the result. It needs a Rust compiler, which maturin
downloads if you don't have one (see [](#the-rust-toolchain)).

## The Rust toolchain

Building `ergmx` from source needs two things:

1. **A C linker**, which Rust uses to produce the compiled library. Neither
   uv nor maturin installs it:
   - macOS: the Xcode Command Line Tools, `xcode-select --install`;
   - Linux: a C compiler, such as `sudo apt install build-essential` (Debian,
     Ubuntu) or `sudo dnf install gcc` (Fedora);
   - Windows: the
     [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
     with the "Desktop development with C++" workload.
2. **Rust 1.88 or newer**, installed by you or downloaded by maturin.

### Rust installed by maturin

If no `cargo` command is found, maturin downloads a private Rust toolchain
before building, with
[puccinialin](https://github.com/konstin/puccinialin). It takes about half a
minute and 500 MB, in your user cache folder (`~/Library/Caches/puccinialin`
on macOS, `~/.cache/puccinialin` on Linux, under `%LOCALAPPDATA%` on
Windows), and is reused by later builds. It is not added to your `PATH`, so it
doesn't interfere with anything else. Delete the folder to remove it.

To forbid the download, so that a build without Rust fails instead, set
`MATURIN_NO_INSTALL_RUST=1`.

### Rust installed with rustup

For development, or to keep a single Rust installation, install
[rustup](https://rustup.rs), Rust's official installer. On macOS and Linux:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On Windows, download and run
[rustup-init.exe](https://win.rustup.rs/x86_64), after the Visual Studio Build
Tools. Then open a new terminal, so that `cargo` is on your `PATH`.

The repository pins the Rust version in `rust-toolchain.toml` (1.98.1, with
clippy and rustfmt), so that every build, and every lint, is the same. In the
repository, install that version with:

```bash
rustup toolchain install
```

rustup then uses it automatically inside the repository, and your default
Rust everywhere else. The oldest Rust that builds `ergmx` is 1.88, set as
`rust-version` in `Cargo.toml`.

## Development

Clone the repository and let uv set up everything else:

```bash
git clone https://github.com/neylsoncrepalde/ergmx.git
cd ergmx
rustup toolchain install    # the pinned Rust
uv sync                     # Python, dependencies, and ergmx built from source
uv run pytest               # the Python tests
```

`uv sync`:

- installs the Python of `.python-version` (3.14) if needed, and creates
  `.venv`;
- installs the exact versions in `uv.lock`, including the `dev` dependency
  group (pytest, igraph, networkx, matplotlib, pandas, maturin);
- builds the Rust core with optimizations and installs `ergmx` in editable
  mode: changes to the Python code apply immediately.

After changing Rust code, nothing else is needed: `uv run` notices the change
(through `cache-keys` in `pyproject.toml`) and rebuilds the core, in a few
seconds, before running the command. Use `uv run` for everything, or run
`uv sync` before using `.venv` directly.

The Rust checks run with cargo:

```bash
cargo clippy --release --all-targets -- -D warnings
cargo test --release
```

`cargo test` links the Rust unit tests to the Python library of the `python3`
on your `PATH` (set `PYO3_PYTHON` to choose another); CI uses the Python of
`actions/setup-python`.

To test another Python in a temporary environment, leaving `.venv` alone, or
the oldest dependencies that `ergmx` allows:

```bash
uv run --isolated --python 3.11 pytest
uv venv --python 3.11 /tmp/lowest
uv pip install --python /tmp/lowest --resolution lowest-direct . \
    "pytest>=8" "igraph>=0.11.5" "networkx>=3.2" "matplotlib>=3.9" "pandas>=2.0"
/tmp/lowest/bin/python -m pytest
```

### Building the documentation

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

Then open `docs/_build/html/index.html`. Every example runs during the build,
so it takes under a minute.

### Comparing with R

The reference results in `tests/data/` and the benchmark come from R scripts.
To regenerate them, install R with the ergm, igraph and jsonlite packages, and
run from the repository:

```bash
Rscript scripts/r_reference.R && Rscript scripts/r_gof_reference.R
Rscript benchmarks/benchmark.R && uv run python benchmarks/benchmark.py
```

## How the Rust is bundled

The Rust code in `src/` is a library built with [PyO3](https://pyo3.rs), which
exposes it to Python as the module `ergmx._core`. maturin compiles it into a
single file, `_core.abi3.so` (`_core.pyd` on Windows), and places it inside the
`ergmx` package, next to the Python code in `python/ergmx/`. A wheel is that
package, zipped: installing it copies the compiled file, and Python imports
it like any module. Nothing is compiled on the user's machine.

The core uses Python's stable ABI (*abi3*): it only calls the part of Python's
C interface that doesn't change between versions, so one compiled file works
on Python 3.11, 3.12, 3.13, 3.14 and later.

To build the wheel for your platform and the source distribution:

```bash
uv build
```

They are written to `dist/`. The source distribution has the Rust sources and
`rust-toolchain.toml`; installing it compiles them, as installing from GitHub
does.

The `Release` workflow (`.github/workflows/release.yml`) builds the wheels of
every platform on GitHub's machines, installs each one it can run (Linux
x86-64, macOS and Windows) on Python 3.11 and 3.14 in an environment without
Rust or the source code, and fits a model with it.

### Publishing a release

When a GitHub release is published, the `Release` workflow uploads the wheels
and the source distribution to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts
this repository's workflow, so no password or token is stored. It needs a
one-time setup:

1. On PyPI, under *Your account* > *Publishing*, add a *pending publisher*:
   project name `ergmx`, owner `neylsoncrepalde`, repository `ergmx`, workflow
   `release.yml`, environment `pypi`.
2. On GitHub, optionally, protect the `pypi` environment (*Settings* >
   *Environments*) so that publishing needs your approval.

Then, for each release: update the version in `pyproject.toml` and
`Cargo.toml` and the changelog, commit, and create a release on GitHub with a
tag such as `v0.3.1`. To build and test the wheels without publishing, run the
workflow by hand from the *Actions* tab.

## Troubleshooting

`Rust not found, installing into a temporary directory`
: maturin is downloading Rust, as above. Install rustup to use your own, or
  set `MATURIN_NO_INSTALL_RUST=1` to fail instead.

`linker 'cc' not found`, `xcrun: error`, `link.exe not found`
: The C linker is missing: see [](#the-rust-toolchain).

rustup says the toolchain of `rust-toolchain.toml` is not installed
: Run `rustup toolchain install` in the repository.

Changes to the Rust code don't show up
: The core is rebuilt by `uv run` and `uv sync`, not by running `.venv`'s Python
  directly. Run `uv sync`, or force a rebuild with
  `uv sync --reinstall-package ergmx`.

Fits are much slower than the benchmarks
: The core was built without optimizations, for example by
  `maturin develop` without `--release`. Rebuild it with
  `uv sync --reinstall-package ergmx`.
