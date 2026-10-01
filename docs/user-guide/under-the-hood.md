---
file_format: mystnb
kernelspec:
  name: python3
---

# Under the hood

*An optional page: you don't need any of this to use `ergmx`.*

`ergmx` is a Python package, but the part that does the heavy work is written
in [Rust](https://www.rust-lang.org), compiled ahead of time to machine code.
This page explains why, what Rust brings, and what the tools you meet when
installing from source (cargo, rustup, PyO3, maturin) do.

## Why not just Python?

Fitting a dyad-dependent ERGM means simulating networks by Markov chain Monte
Carlo (MCMC). Each step of the chain is tiny:

1. pick a dyad to toggle (add the tie if it is absent, remove it otherwise);
2. compute how the model's statistics would change, the *change statistics*:
   for `triangle`, the number of partners the two vertices share;
3. accept or reject the toggle, with a probability that depends on that change.

A single fit takes tens of millions of these steps: a fit of
faux.mesa.high's five-term model runs about 30 million for the estimate, and
37 million more for its log-likelihood. And each step depends on the network the
previous one left, so they can't be done all at once.

That is what makes Python slow here. NumPy is fast because it hands a whole
array to compiled code, which loops over it without returning to Python. A
chain of steps that each depend on the last has no array to hand over: the
loop must run in Python, one step at a time, and each Python operation costs
tens of nanoseconds of interpretation (looking up names, checking types,
creating objects) before any arithmetic happens.

Here is the same sampler, for `edges + triangle` with random toggles, in plain
Python:

```{code-cell} ipython3
import math
import random
import time

from ergmx import datasets

mesa = datasets.load("faux.mesa.high")
n, theta = mesa.vcount(), (-4.6, 0.2)  # edges, triangle


def python_sampler(steps, seed=1):
    rng = random.Random(seed)
    neighbors = [set(mesa.neighbors(v)) for v in range(n)]
    for _ in range(steps):
        i, j = rng.sample(range(n), 2)
        tie = j in neighbors[i]
        sign = -1 if tie else 1
        change = sign * (theta[0] + theta[1] * len(neighbors[i] & neighbors[j]))
        if change >= 0 or rng.random() < math.exp(change):
            if tie:
                neighbors[i].discard(j)
                neighbors[j].discard(i)
            else:
                neighbors[i].add(j)
                neighbors[j].add(i)


steps = 200_000
start = time.perf_counter()
python_sampler(steps)
python_ns = (time.perf_counter() - start) / steps * 1e9
print(f"Python: {python_ns:.0f} ns per step")
```

and `ergmx`'s Rust sampler on the same model, on one thread:

```{code-cell} ipython3
import ergmx

steps = 5_000_000
start = time.perf_counter()
ergmx.simulate(mesa, "edges + triangle", theta, burnin=steps, interval=1,
               triadic_weight=0, output="stats")
rust_ns = (time.perf_counter() - start) / steps * 1e9
print(f"Rust: {rust_ns:.0f} ns per step, {python_ns / rust_ns:.0f} times faster")
fit_steps = 67e6  # a fit of faux.mesa.high's model, with its log-likelihood
print(f"The steps of a fit: {fit_steps * python_ns / 1e9:.0f} s at Python's speed, "
      f"{fit_steps * rust_ns / 1e9:.1f} s at Rust's")
```

The numbers are those of the computer that built this page, and understate
the difference: `ergmx`'s sampler does more than this bare loop on each step
(it handles any combination of terms, keeps every statistic up to date and
uses better proposals), and models with terms such as gwesp make each step
more work in both languages. `ergmx` also splits the steps
across parallel chains, so a fit takes a fraction of the single-thread
time.

## Why Rust?

Several languages compile to machine code as fast as C's. A benchmark of this
sampler on a 1,000-vertex network, during the design of `ergmx`, gave:

| Implementation | Nanoseconds per step |
|---|---|
| NumPy (one step at a time, on an adjacency matrix) | 1,300 |
| Pure Python (neighbor sets, as above) | 445 |
| [Numba](https://numba.pydata.org) (Python compiled at run time) | 28 |
| C | 19 |
| Rust | 19 |

Rust is as fast as C, and adds what made it the choice for `ergmx`:

- **Memory safety without a garbage collector.** The compiler checks, before
  the program runs, that no memory is used after being freed and that no two
  threads change the same data at once. Bugs that crash C programs, or worse,
  silently corrupt results, don't compile.
- **Parallelism that is safe by construction.** `ergmx` runs its MCMC chains in
  parallel threads; the compiler guarantees they can't interfere. Python's
  global interpreter lock (GIL) is released while they run, so they use every
  CPU core.
- **Tools.** One build tool, cargo, and a package registry,
  [crates.io](https://crates.io), for libraries, and PyO3 and maturin, which
  make a Rust library a Python package with little code.

Numba would also have been a reasonable choice, but it compiles when the
program runs, is slower when the model's settings aren't known at compile time
(28 ns rather than 20 here), and doesn't give the same guarantees for threads.

## What runs where

`ergmx` keeps in Rust only what runs millions of times, and everything else in
Python, where it is easier to read and change:

| Python (`python/ergmx/`) | Rust (`src/`) |
|---|---|
| Formulas and terms: names, arguments, attributes | The network, as sorted neighbor lists |
| The estimation: MPLE, contrastive divergence, Monte Carlo MLE | Change statistics of every term |
| Log-likelihoods, standard errors, diagnostics | The MCMC sampler and its proposals |
| Summaries, goodness of fit, plots | Parallel chains |

They meet in a few large calls: once per iteration, Python asks the Rust core
to run its chains (millions of steps) and gets back an array of the sampled
statistics. Crossing between the two languages costs microseconds, which is
nothing once per iteration, and would be too much once per step.

Speed also comes from the algorithms, which the language only makes cheap:
change statistics update only what a toggle changes, sorted neighbor lists
make shared partners a merge of two short lists, and triadic proposals help
the chains explore clustered networks. The proposal matters as much as the
language: with plain tie/no-tie proposals, R's ergm takes 128 s on
faux.magnolia.high, against 14 s with its triadic default (both without the
log-likelihood).

## The tools

If you know Python's tools, Rust's have a counterpart for each:

| Rust | What it does | Python counterpart |
|---|---|---|
| `rustc` | the compiler: Rust source to machine code | (the interpreter) |
| [cargo](https://doc.rust-lang.org/cargo/) | builds the code and downloads its dependencies | pip and a build tool |
| [crates.io](https://crates.io) | the registry of Rust libraries, *crates* | PyPI |
| `Cargo.toml`, `Cargo.lock` | the package's settings and exact dependencies | `pyproject.toml`, `uv.lock` |
| [rustup](https://rustup.rs) | installs and switches Rust versions | `uv python install`, pyenv |
| `rust-toolchain.toml` | the Rust version a project uses | `.python-version` |

`ergmx`'s Rust core depends on four crates: [PyO3](https://pyo3.rs), the
bridge to Python; [numpy](https://github.com/PyO3/rust-numpy), to exchange
arrays without copying; [rayon](https://github.com/rayon-rs/rayon), for the
parallel chains; and rustc-hash, a fast hash function for its tables.

**PyO3** turns Rust into a Python module. A Rust structure marked
`#[pyclass]` becomes a Python class, and its methods Python methods; PyO3
converts the arguments (lists, NumPy arrays, numbers) between the two
languages, and turns Rust errors into Python exceptions. The core is the
module `ergmx._core`; you never use it directly.

**maturin** is the bridge to Python packaging. `pyproject.toml` names it as
the *build backend*, the tool that `uv`, pip or `uv build` call to turn the
source into a wheel, as they would call setuptools or hatchling for a pure
Python package:

```toml
[build-system]
requires = ["maturin>=1.9,<2"]
build-backend = "maturin"
```

When asked to build, maturin runs cargo with optimizations, takes the
compiled library (`_core.abi3.so`, or `_core.pyd` on Windows), puts it in the
`ergmx` folder next to the Python code, and zips them into a wheel. If cargo is
missing, it downloads Rust first.

## From source code to `import ergmx`

```{mermaid}
flowchart LR
    A["uv add ergmx"] --> B{"A wheel for<br>this platform?"}
    B -- yes --> C["download it<br>(compiled core inside)"]
    B -- no --> D["maturin"] --> E["cargo, rustc<br>(compile with optimizations)"] --> F["wheel"]
    C --> G["install: copy files"]
    F --> G
    G --> H["import ergmx"]
```

Published versions take the top path: the compiling was done once, on
GitHub's computers, for each platform. Building from source takes the bottom
one, on your computer.

The optimized build is set in `Cargo.toml`: *link-time optimization* lets the
compiler optimize across the whole program, inlining the change statistics
into the sampler's loop, at the cost of a slower build. An unoptimized
("debug") build, which cargo makes by default for development, is about five
times slower on faux.mesa.high's model; when uv or pip build `ergmx`,
maturin always optimizes.

```toml
[profile.release]
codegen-units = 1
lto = "fat"
```

Finally, the compiled core uses Python's *stable ABI*: it only calls the part
of Python's C interface that is promised not to change between versions. That
is why one wheel per platform serves Python 3.11, 3.12, 3.13, 3.14 and later
versions.
