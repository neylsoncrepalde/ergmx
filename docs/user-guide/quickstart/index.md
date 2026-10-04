# Quick start

Seven complete analyses, each from the data to a table of results, to read
in a few minutes and adapt to your own networks:

- **An ERGM**: the friendships of a high school. Do students befriend others
  like them, and the friends of their friends?
- **A multilevel ERGM**: researchers, their laboratories, and the ties
  within and between the two levels, simulated from a known model and drawn
  with [multinets](https://github.com/neylsoncrepalde/multinets-py).
- **A valued ERGM**: the number of contexts in which the members of a karate
  club interact. Who interacts more, and how are the counts dispersed?
- **An egocentric ERGM**: a survey of half the students of a high school,
  each naming their friends. Does it recover the whole network's model?
- **A Bayesian ERGM**: the liking among the novices of a monastery. What does
  the posterior distribution say, and what does a prior add?
- **A temporal ERGM**: the same novices at three times. Which ties form, which
  persist, and where was the process heading?
- **A bipartite ERGM**: the Southern Women and the events they attended. Do
  the women split into groups, beyond their activity and the events'
  popularity?

Each step links to the page of the user guide that covers it in depth. To
run them, add `ergmx` with igraph and matplotlib to a uv project (created
with `uv init`), and multinets for the second
(see [Installation](../installation.md)):

```bash
uv add "ergmx[igraph,plot]"
uv add "multinets @ git+https://github.com/neylsoncrepalde/multinets-py"
```

Then run a script with `uv run python analysis.py`.

```{toctree}
:maxdepth: 1

ergm
multilevel
valued
egocentric
bayesian
temporal
bipartite
```
